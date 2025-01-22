import numpy as np
import time
from hitpoly.utils.constants import *


class ForceFieldFileReader:
    def __init__(
        self,
        molecule,
        path,
    ) -> None:
        self.molecule = molecule
        self.smiles = molecule.smiles[0]

        self.read_file(path)

    def read_file(self, path):
        prop_list = (
            "Masses",
            "Pair Coeffs",
            "Bond Coeffs",
            "Angle Coeffs",
            "Dihedral Coeffs",
            "Improper Coeffs",
            "Atoms",
            "Bonds",
            "Angles",
            "Dihedrals",
            "Impropers",
        )
        self.mol_dic = {}
        self.mol_dic[self.smiles] = {}
        param_dict = {}
        for i in prop_list:
            param_dict[i] = []

        writing = None
        writer = False
        with open(f"{path}", "r+") as f:
            lines = f.readlines()
            for i in lines:
                if i[:-1].startswith(prop_list):
                    writing = i[:-1].lstrip().rstrip()
                    writer = True
                    continue
                if writer:
                    if i == "\n" and param_dict[writing]:
                        writer = False
                        continue
                    if i == "\n":
                        continue
                    param_dict[writing].append([float(j) for j in i[:-1].split()])

        masses_file = {}
        for i in param_dict["Masses"]:
            masses_file[int(i[0])] = i[1]

        self.mol_dic[self.smiles]["Masses"] = masses_file

        bond_params_file = {}
        for i in np.array(param_dict["Bond Coeffs"]):
            bond_params_file[int(i[0])] = i[1:]
        bonds_file = np.array(param_dict["Bonds"])

        self.mol_dic[self.smiles]["Bond Coeffs"] = bond_params_file
        self.mol_dic[self.smiles]["Bonds"] = bonds_file

        angle_params_file = {}
        for i in np.array(param_dict["Angle Coeffs"]):
            param = i[1:]
            param[-1] = param[-1] * np.pi / 180
            angle_params_file[int(i[0])] = param
        angles_file = np.array(param_dict["Angles"])

        self.mol_dic[self.smiles]["Angle Coeffs"] = angle_params_file
        self.mol_dic[self.smiles]["Angles"] = angles_file

        dihedral_params_file = {}
        for i in np.array(param_dict["Dihedral Coeffs"]):
            dihedral_params_file[int(i[0])] = i[1:]
        dihedrals_file = np.array(param_dict["Dihedrals"])

        self.mol_dic[self.smiles]["Dihedral Coeffs"] = dihedral_params_file
        self.mol_dic[self.smiles]["Dihedrals"] = dihedrals_file

        improper_params_file = {}
        for i in np.array(param_dict["Improper Coeffs"]):
            improper_params_file[int(i[0])] = i[1:]
        impropers_file = np.array(param_dict["Impropers"])

        self.mol_dic[self.smiles]["Improper Coeffs"] = improper_params_file
        self.mol_dic[self.smiles]["Impropers"] = impropers_file

        lj_file = {}
        for i in param_dict["Pair Coeffs"]:
            lj_file[int(i[0])] = i[1:]

        pair_params_file = {}
        for i in np.array(param_dict["Atoms"]):
            p1 = i[3]
            p2 = lj_file[i[2]]
            pair_params_file[int(i[0])] = [p1, p2[0], p2[1]]
        self.mol_dic[self.smiles]["Atoms"] = param_dict["Atoms"]
        self.mol_dic[self.smiles]["Pair Coeffs"] = pair_params_file

        atom_indeces = []
        for i in param_dict["Atoms"]:
            atom_name = (
                MASS_TO_ELEMENT.get(self.mol_dic[self.smiles]["Masses"][i[2]])
                or MASS_TO_ELEMENT[
                    min(
                        MASS_TO_ELEMENT.keys(),
                        key=lambda k: abs(
                            k - self.mol_dic[self.smiles]["Masses"][i[2]]
                        ),
                    )
                ]
            )
            atom_indeces.append(ELEMENT_TO_NUM[atom_name])
        self.mol_dic[self.smiles]["Atomic nums"] = atom_indeces

        self.order_dict = {}
        k = 0
        idx = 0
        startTime = int(round(time.time() * 1000))
        while idx < len(self.molecule.atomic_nums):
            for ind, mol_name in enumerate(self.mol_dic.keys()):
                mol_nums = self.mol_dic[mol_name]["Atomic nums"]
                if self.molecule.atomic_nums[idx : idx + len(mol_nums)] == mol_nums:
                    self.order_dict[k] = list(self.mol_dic.keys())[ind]
                    k += 1
                    idx += len(mol_nums)
                if int(round(time.time() * 1000)) - startTime > 5000:
                    raise RuntimeError(
                        f"Runtime for parametrizing {mol_name} exceeded 5s"
                    )

        # Checking if the ordering worked
        new_indeces = []
        for key, val in self.order_dict.items():
            new_indeces.append(self.mol_dic[val]["Atomic nums"])
        new_indeces = [item for sublist in new_indeces for item in sublist]

        assert new_indeces == self.molecule.atomic_nums

        pair_params_file = []
        atom_names_file = []
        for key, val in self.order_dict.items():
            for atom in self.mol_dic[val]["Atoms"]:
                atom_names_file.append(
                    ELEMENT_TO_NUM[
                        MASS_TO_ELEMENT.get(
                            self.mol_dic[val]["Masses"][atom[2]],
                            MASS_TO_ELEMENT[
                                min(
                                    MASS_TO_ELEMENT.keys(),
                                    key=lambda k: abs(
                                        k - self.mol_dic[val]["Masses"][atom[2]]
                                    ),
                                )
                            ],
                        )
                    ]
                )
                pair_params_file.append(self.mol_dic[val]["Pair Coeffs"][atom[0]])
        self.pair_params_file = torch.tensor(pair_params_file)
        assert self.molecule.pair_params.shape == self.pair_params_file.shape
        assert self.molecule.atomic_nums == atom_names_file

    def load_pair_params(self, charges=True, charge_scaling=None):
        """
        Args:
            charges (bool, optional): If True then loads charges from file,
                otherwise just loads the LJ parameters
        """
        for ind, i in enumerate(self.pair_params_file):
            if charges:
                if charge_scaling:
                    i = i * charge_scaling
                self.molecule.pair_params[ind] = i
            else:
                self.molecule.pair_params[ind, 1:] = i[1:]

    def load_bond_params(self):
        bond_counter = 0
        for key, val in self.order_dict.items():
            if self.mol_dic[val]["Bonds"].size != 0:
                self.mol_dic[val]["Bonds"][:, -2:] = (
                    self.mol_dic[val]["Bonds"][:, -2:] + bond_counter
                )

            for i in self.mol_dic[val]["Bonds"]:
                for ind, j in enumerate(self.molecule.bonds.numpy()):
                    if ((i[2:] - 1).astype(int) == j).all() or (
                        (i[2:][::-1] - 1).astype(int) == j
                    ).all():
                        self.molecule.bond_params[ind] = torch.tensor(
                            self.mol_dic[val]["Bond Coeffs"][int(i[1])]
                        )
            if self.mol_dic[val]["Bonds"].size != 0:
                bond_counter += self.mol_dic[val]["Bonds"][:, -2:].max()
            else:
                bond_counter += 1

    def load_angle_params(self):
        angle_counter = 0
        for key, val in self.order_dict.items():
            if self.mol_dic[val]["Angles"].size != 0:
                self.mol_dic[val]["Angles"][:, -3:] = (
                    self.mol_dic[val]["Angles"][:, -3:] + angle_counter
                )

            for i in self.mol_dic[val]["Angles"]:
                for ind, j in enumerate(self.molecule.angles.numpy()):
                    if ((i[2:] - 1).astype(int) == j).all() or (
                        (i[2:][::-1] - 1).astype(int) == j
                    ).all():
                        self.molecule.angle_params[ind] = torch.tensor(
                            self.mol_dic[val]["Angle Coeffs"][int(i[1])]
                        )
            if self.mol_dic[val]["Angles"].size != 0:
                angle_counter += self.mol_dic[val]["Angles"][:, -3:].max()
            else:
                angle_counter += 1

    def load_dihedral_params(self):
        dihedral_counter = 0
        for key, val in self.order_dict.items():
            if self.mol_dic[val]["Dihedrals"].size != 0:
                self.mol_dic[val]["Dihedrals"][:, -4:] = (
                    self.mol_dic[val]["Dihedrals"][:, -4:] + dihedral_counter
                )

            for i in self.mol_dic[val]["Dihedrals"]:
                for ind, j in enumerate(self.molecule.dihedrals.numpy()):
                    if ((i[2:] - 1).astype(int) == j).all() or (
                        (i[2:][::-1] - 1).astype(int) == j
                    ).all():
                        self.molecule.dihedral_params[ind] = torch.tensor(
                            self.mol_dic[val]["Dihedral Coeffs"][int(i[1])]
                        )
            if self.mol_dic[val]["Dihedrals"].size != 0:
                dihedral_counter += self.mol_dic[val]["Dihedrals"][:, -4:].max()
            else:
                dihedral_counter += 1

    def load_improper_params(self):
        improper_counter = 0
        for key, val in self.order_dict.items():
            if self.mol_dic[val]["Impropers"].size != 0:
                self.mol_dic[val]["Impropers"][:, -4:] = (
                    self.mol_dic[val]["Impropers"][:, -4:] + improper_counter
                )

            for i in self.mol_dic[val]["Impropers"]:
                for ind, j in enumerate(self.molecule.impropers.numpy()):
                    if ((i[2:] - 1).astype(int) == j).all() or (
                        (i[2:][::-1] - 1).astype(int) == j
                    ).all():
                        self.molecule.improper_params[ind][1] = torch.tensor(
                            [self.mol_dic[val]["Improper Coeffs"][int(i[1])][0]]
                        )
                    # Sometimes the ordering of the impropers can be wonky
                    #  so this is added as an extra check
                    elif i[3] == j[0]:
                        if np.isin(i[3:], j[1:]):
                            self.molecule.improper_params[ind][1] = torch.tensor(
                                [self.mol_dic[val]["Improper Coeffs"][int(i[1])][0]]
                            )
            if self.mol_dic[val]["Impropers"].size != 0:
                improper_counter += self.mol_dic[val]["Impropers"][:, -4:].max()
            else:
                improper_counter += 1
