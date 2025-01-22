from typing import List, Dict, Union
import copy
import itertools
import torch
import numpy as np
import pandas as pd
import random
from typing import Iterator
import re
from multiprocessing import Pool
import os, sys

from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit.Chem import rdMolDescriptors, GetFormalCharge, rdmolfiles
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import EditableMol
from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
from hitpoly.utils.constants import (
    ELEMENT_TO_NUM,
    SIGMA_ATOMS,
    EPSILON_ATOMS,
    CHARGES,
    D_CHARGES,
    D_ALPHA,
    D_THOLE,
    NUM_TO_ELEMENT,
    COUL_CONST,
)
from hitpoly.utils.args import hitpolyArgs
from hitpoly.utils.geometry_calc import (
    get_theta,
    build_bonds,
    build_angles,
    build_dihedrals,
    build_impropers,
    build_pairs,
    get_bonds_in_angles,
    get_bonds_in_impropers,
    get_angles_in_dihedrals,
)
from hitpoly.data.initialize_discrete_params import initialize_discrete_params
from openmm.app import PDBFile


class TopologyBuilder:
    """
    Inherits stuff from chemprop data object, so that can be used fully in the pipeline
    Builds the angles, bonds, dihedrals, pairwise and improper interactions for a smiles
    """

    def __init__(
        self,
        smiles: List[str],
        train_args: hitpolyArgs,
        species: int = None,
        geom_ids: List[int] = None,
        num_atoms: int = None,
        load_geoms: bool = True,
        lj_from_file: bool = False,
        mol_path: str = None,
    ):
        self.smiles = smiles
        self.species_id = species
        self.geom_ids = geom_ids
        self.train_args = train_args
        # Adding the loading of LJ params and geometries here to save runtime on the cost of memory usage
        self.build_graph()
        if not train_args.discrete_flag:
            self.initialize_ff_params()
        if lj_from_file:
            self.lj_params = self.get_lj_params(train_args.geom_path)
        else:
            self.lj_params = self.lj_from_lookup()  # Write this, populate simple table

    def build_graph(self):
        molecule = Chem.MolFromSmiles(self.smiles[0])

        molecule = Chem.AddHs(molecule)

        neighbors = self.get_neighbors(molecule)  # molecule is now a rdkit molecule

        self.atomic_nums = [x.GetAtomicNum() for x in molecule.GetAtoms()]
        # what is the element of each atom in the same order as atom_idxs
        self.atom_idxs = list(range(len(self.atomic_nums)))
        self.atom_num = len(self.atom_idxs)  # how many atoms are in this molecule

        if (
            "." in self.smiles[0]  # and not self.train_args.final
        ):  # Means the smiles consists of multiple components
            neighbors = self.reorder_cluster_atoms(neighbors, self.smiles[0])
        if len(neighbors) > 1:
            self.bonds = build_bonds(self, neighbors)
            self.angles = build_angles(self, neighbors)
            self.dihedrals = build_dihedrals(self, neighbors)
            self.impropers = build_impropers(self, self.bonds, neighbors)
            self.pairs, self.f_1_4 = build_pairs(
                self,
                self.bonds,
                self.angles,
                self.dihedrals,
                self.train_args.exclude_dihedrals_in_pairs,
            )

        else:
            self.bonds = torch.tensor([])
            self.angles = torch.tensor([])
            self.dihedrals = torch.tensor([])
            self.impropers = torch.tensor([])
            self.pairs = torch.tensor([])

    def reorder_encoding(self, encode):
        """
        Reordering the encoding of atoms coming from chemprop
        """
        if "." not in self.smiles[0]:
            self.encode = encode
            return
        encode_separate = []
        begin_ind = 0
        end_ind = len(self.atom_numbers_true)
        for ind, i in enumerate(self.atomic_nums_separate):
            if (
                1 not in i
            ):  # Checking if there are any hydrogens in the molecule and rearanging properly
                # Because chemprop puts Hs at the end of the cluster embedding
                encode_separate.append(encode.narrow(0, begin_ind, len(i)))
                begin_ind += len(i)
            else:
                encode_separate.append(encode.narrow(0, begin_ind, i.index(1)))
                encode_separate[ind] = torch.cat(
                    (
                        encode_separate[ind],
                        encode.narrow(
                            0, end_ind - (len(i) - i.index(1)), len(i) - i.index(1)
                        ),
                    )
                )
                begin_ind += i.index(1)
                end_ind -= len(i) - i.index(1)

        encode_reorder = []
        for _, val in self.order_dict.items():
            encode_reorder.append(encode_separate[val])
        self.encode = torch.cat([i for i in encode_reorder])

    def reorder_cluster_atoms(self, neighbors, smiles):
        # Retreive atomic order from a random parsed geometry
        geom_id = random.choice(self.geom_ids)
        self.atom_numbers_true = []
        with open(
            f"{self.train_args.geom_path}/{self.species_id}/{geom_id}.xyz", "r"
        ) as f:
            lines = f.readlines()
            for i in lines[2:]:
                self.atom_numbers_true.append(ELEMENT_TO_NUM[i[:2].strip()])
        split_smiles = smiles.split(".")
        mols = [Chem.MolFromSmiles(i) for i in split_smiles]
        mols = [Chem.AddHs(i) for i in mols]
        self.atomic_nums_separate = [
            [x.GetAtomicNum() for x in i.GetAtoms()] for i in mols
        ]
        hydrogen_dict = {}
        for ind, i in enumerate(self.atomic_nums_separate):
            if 1 in i:
                hydrogen_dict[ind] = (i.index(1), len(i) - i.index(1))
            else:
                hydrogen_dict[ind] = (len(i), 0)

        # Compare and reorder the smiles so that both match
        self.order_dict = {}
        k = 0
        idx = 0
        while idx < len(self.atom_numbers_true):
            for ind, sub_list in enumerate(self.atomic_nums_separate):
                if self.atom_numbers_true[idx : idx + len(sub_list)] == sub_list:
                    self.order_dict[k] = ind
                    k += 1
                    idx += len(sub_list)
        smiles_reordered = []
        for _, val in self.order_dict.items():
            smiles_reordered.append(split_smiles[val])

        # Create atomic numbers, atomic indicies, neighbors and bonds in the new, correct order
        mols = [Chem.MolFromSmiles(i) for i in smiles_reordered]
        mols = [Chem.AddHs(i) for i in mols]
        self.atomic_nums = [x.GetAtomicNum() for i in mols for x in i.GetAtoms()]
        self.atom_idxs = [x for x, _ in enumerate(self.atomic_nums)]
        x = 0
        self.atom_idxs_separate = []
        self.formal_charges_separate = []
        for i in mols:
            idxs = []
            for j in i.GetAtoms():
                idxs.append(x)
                x += 1
            self.atom_idxs_separate.append(idxs)

            self.formal_charges_separate += [GetFormalCharge(i)]

        # Fixing the indicies of the separate mols
        neighbors_separate = [
            [[x.GetIdx() for x in y.GetNeighbors()] for y in mol.GetAtoms()]
            for mol in mols
        ]
        curr_len_neigh = len(neighbors_separate[0])
        for ind_1, neigh in enumerate(neighbors_separate[1:]):
            for ind_2, atom in enumerate(neigh):
                if atom:
                    for ind_3, each in enumerate(atom):
                        neighbors_separate[ind_1 + 1][ind_2][ind_3] += curr_len_neigh
            curr_len_neigh += len(neighbors_separate[ind_1 + 1])

        neighbors = []
        for i in neighbors_separate:
            neighbors.extend(i)

        return neighbors

    def initialize_ff_params(self):
        """
        Initializes the force field parameters for calculating
        forces and energies on the geometries

        If the train_args argument warm_start is set to true, then the first round of
        parameters are initialized to data from a lammps file that has to be specified.
        """
        if self.train_args.ff_type == "opls":
            self.train_args.bond_output = 2
            self.train_args.angle_output = 2
            self.train_args.dihedral_output = 4
            self.train_args.improper_output = 4
            self.train_args.pair_output = (
                1  # The LJ parameters are taken from tabulated data
            )
            # it has been seen (by Pablo) that predicting them
            # from DFT calculations is unfeasible (to test more)
        else:
            raise ValueError(f"{self.train_args.ff_type} not available, only 'opls'")

        # Linear basis parameterization 2 pred params, 2 boundaries
        if self.bonds.numel():
            self.bond_params = torch.empty(
                self.bonds.shape[0], self.train_args.bond_output
            ).to(self.train_args.device)
        else:
            self.bond_params = torch.zeros((0, 0)).to(self.train_args.device)

        # Linear basis parameterization 2 pred params, 2 boundaries
        if self.angles.numel():
            self.angle_params = torch.empty(
                self.angles.shape[0], self.train_args.angle_output
            ).to(self.train_args.device)
        else:
            self.angle_params = torch.zeros((0, 0)).to(self.train_args.device)

        if self.dihedrals.numel():
            self.dihedral_params = torch.empty(
                self.dihedrals.shape[0], self.train_args.dihedral_output
            ).to(self.train_args.device)
        else:
            self.dihedral_params = torch.zeros((0, 0)).to(self.train_args.device)

        if self.impropers.numel():
            self.improper_params = torch.zeros(
                self.impropers.shape[0], self.train_args.improper_output
            ).to(self.train_args.device)
        else:
            self.improper_params = torch.zeros((0, 0)).to(self.train_args.device)
        # Since only charges are prediced from Pairnet
        self.pair_params = torch.empty(
            self.atom_num, 3 * self.train_args.pair_output
        ).to(self.train_args.device)
        # Pair params are ordered as charge, epsilon, sigma

        # Since only charges are prediced from Pairnet
        self.true_charge = torch.empty(self.atom_num, 1).to(self.train_args.device)

    def get_neighbors(self, molecule):
        return [[x.GetIdx() for x in y.GetNeighbors()] for y in molecule.GetAtoms()]

    def get_lj_params(self, path):
        """
        Retreiving the LJ parameters for the prediction and simulation
        from the geometry specific folder
        """
        params = (
            pd.read_csv(f"{path}/{self.species_id}/atom_parameters.csv")
            .reset_index()
            .to_numpy()
        )

        if not (params[:, 0] == self.atom_idxs).all():
            raise ValueError(
                f"The indexes for {self.smiles} (spec_id - {self.species_id}) is not the same as read from atom_parameters file"
            )
        if not (params[:, 2] == self.atomic_nums).all():
            raise ValueError(
                f"The atomic nums for {self.smiles} (spec_id - {self.species_id}) is not the same as read from atom_parameters file"
            )

        return torch.tensor(
            params[:, -3:].astype("float32")
        )  # Loads the last 3 elements of the file - charge, epsilon, sigma

    def lj_from_lookup(self):
        """
        Building the LJ parameters from a lookup table.
        """
        sigmas = torch.tensor([SIGMA_ATOMS[i] for i in self.atomic_nums]).view(-1, 1)
        epsilons = torch.tensor([EPSILON_ATOMS[i] for i in self.atomic_nums]).view(
            -1, 1
        )
        charges = torch.tensor([CHARGES[i] for i in self.atomic_nums]).view(-1, 1)
        return torch.cat([charges, epsilons, sigmas], axis=1).type(torch.float32)


class TopologyDataset(Dataset):
    """
    TopologyDataset inherits from torch Dataset

    The discete atom types and forcefield parameters are set here if
    hitpolyModel().train_args.discrete_flag = True
    """

    def __init__(
        self,
        data: List[TopologyBuilder],
        train_args=hitpolyArgs,
    ):
        print("Created a TopologyDataset object with address: " + str(self.__repr__()))
        self.data = data
        self._batch_graph = None
        self.train_args = train_args
        if train_args.discrete_flag:
            all_smiles = []
            new_atomic_nums, new_bonds = [], []
            prev_num_atoms = 0
            for molecule in self.data:
                """
                to get the new neighbors, you have to make sure the indices of each bond are referencing atoms from the newer
                molecule, instead of all of them starting at index = 0
                """
                all_smiles = all_smiles + molecule.smiles[0].split(".")
                new_atomic_nums = new_atomic_nums + molecule.atomic_nums
                new_bonds = new_bonds + (molecule.bonds + prev_num_atoms).tolist()
                prev_num_atoms = prev_num_atoms + molecule.atom_num
            new_bonds = np.array(new_bonds)
            neighbors = [
                new_bonds[
                    np.where(new_bonds == cur)[0], ~np.where(new_bonds == cur)[1]
                ].tolist()
                for cur in range(len(new_atomic_nums))
            ]
            self.atomic_nums = new_atomic_nums
            # print("combined smiles:", ".".join(np.unique(np.array(all_smiles))))
            self.define_discrete_topology_types(neighbors)
            self.set_molecule_topology_types()
            if self.train_args.init_params:
                initialize_discrete_params(self, self.train_args)

    def define_discrete_topology_types(self, neighbors):
        """
        Define all bond/angle/etc types for the full set of possible chemistries in training data
        saved into the dataset first. These will be referenced to populate the individual molecule types
        """
        self.atom_type = self.set_atom_type(neighbors)
        self.num_node_types = self.atom_type.max().item() + 1
        self.bonds = build_bonds(self, neighbors)
        self.angles = build_angles(self, neighbors)
        self.dihedrals = build_dihedrals(self, neighbors)
        self.impropers = build_impropers(self, self.bonds, neighbors)

        self.bond_type = self.set_bond_type()
        self.angle_type = self.set_angle_type()
        self.dihedral_type = self.set_dihedral_type()
        self.improper_type = self.set_improper_type()

    def set_molecule_topology_types(self):
        """
        iterate through all the molecules in the dataset and assign their atom environments based on the
        available environments from the dataset. This is needed because all the parameters have to be referenced from
        ONE base location (dataset.params) in order for backpropagation to work.
        """
        for molecule in self.data:
            # each molecule is a TopologyBuilder object
            neighbors = [
                molecule.bonds[
                    np.where(molecule.bonds.cpu() == i)[0],
                    ~np.where(molecule.bonds.cpu() == i)[1],
                ].tolist()
                for i in range(molecule.atom_num)
            ]
            molecule.atom_type = self.set_atom_type(neighbors, molecule=molecule)
            molecule.num_node_types = (
                self.atom_type.max().item() + 1
            )  # always use the full possible node types for onehot encoding

            molecule.bond_type = self.set_bond_type(molecule=molecule)
            molecule.angle_type = self.set_angle_type(molecule=molecule)
            molecule.dihedral_type = self.set_dihedral_type(molecule=molecule)

            molecule.improper_type = self.set_improper_type(molecule=molecule)

    def set_atom_type(self, neighbors, molecule=None):
        """
        r = one hot encoding of atomic number
        neighbors = indices of atoms in each edge
        d = number of edges connected to each node
        """
        if molecule is not None:
            molecule.unique_values = {}
            atomic_nums = molecule.atomic_nums
            r = torch.vstack([self.atom_one_hot[cur] for cur in atomic_nums])
            d = np.array([len(cur) for cur in neighbors])
            molecule.atom_degrees = d
        else:  # set atom types for the full dataset that will be used to set types for individual molecules
            self.unique_values = {}
            self.atom_one_hot = {}
            atomic_nums = self.atomic_nums
            unique_values_orig, one_hot_mapping_orig = torch.tensor(atomic_nums).unique(
                dim=0, return_inverse=True
            )
            r = torch.eye(unique_values_orig.shape[0])[one_hot_mapping_orig].to(
                self.train_args.device
            )
            d = np.array([len(cur) for cur in neighbors])
            self.atom_degrees = d
            # save the translation between the full atomic number and the full one hot encoding to be used by individual molecules
            for cur in range(len(atomic_nums)):
                self.atom_one_hot[atomic_nums[cur]] = r[cur]
        for rad in range(self.train_args.discrete_neighborhood_depth + 1):
            if rad != 0:
                # the message is in the direction (atom 1 -> atom 0) for each edge, so the message is the current atom label of atom1
                # the messages from each incoming atom are then split by receiving atom,
                # so the messages that are going into a particular atom are all grouped together
                messages = [r[cur] for cur in neighbors]
                # the messages incoming to each atom are then added together to enforce permutation invariance
                messages = [messages[n].sum(0) for n in range(len(d))]
                messages = torch.stack(messages)
                # the message is then appended to the current state to remember order of messages
                r = torch.cat([r, messages], dim=1)
            cur_unique_values, one_hot_mapping = r.unique(dim=0, return_inverse=True)
            if molecule is not None and rad not in molecule.unique_values.keys():
                molecule.unique_values[rad] = cur_unique_values
            elif molecule is None and rad not in self.unique_values.keys():
                self.unique_values[rad] = cur_unique_values
            index = self.index_of2(
                r, self.unique_values[rad]
            )  # index based on the full atomic_num options saved in the dataset
            r = (
                torch.eye(len(self.unique_values[rad]))
                .to(torch.long)
                .to(r.device)[index]
            )
            if len(torch.nonzero((r.sum(1) == 0).to(torch.long), as_tuple=False)) != 0:
                raise Exception("Unrecognized graph neighborhood.")
        return torch.nonzero(r)[:, 1].to(self.train_args.device)

    def index_of2(self, input, source):
        source, sorted_index, inverse = np.unique(
            source.tolist(), return_index=True, return_inverse=True, axis=0
        )
        index = (
            torch.cat([torch.tensor(source).to(self.train_args.device), input])
            .unique(sorted=True, return_inverse=True, dim=0)[1][-len(input) :]
            .to(self.train_args.device)
        )
        try:
            index = torch.tensor(sorted_index).to(self.train_args.device)[index]
        except:
            raise NameError("error in one-hot encoding")
        return index

    def set_bond_type(self, molecule=None):
        """
        For set_<n body interaction>_type() functions, there are two modes.
        set_<>_type() is first used with molecule set to None. This assigns bond types to all the
        bonds in the dataset, which can include multiple molecules at the same time. This ensures that
        a bond between two atoms of the same type will always have the same type and parameters, even if
        they are in different molecules.

        After the full dataset bonds have been assigned, this function is called again by passing in each
        molecule (topology builder) in the dataset to assign the bond types of each molecule
        according to the full available types in the dataset
        """
        if self.bonds.shape[0] == 0:
            return torch.tensor([])
        # node_types is a list of the type of each node
        # top_types contains the node types of each node inside this topology
        if molecule is not None:
            # in this case, you use the atom (and their environments) from the molecule
            # to assign the types of bond in the molecule
            node_types = molecule.atom_type.view(-1)
            if len(molecule.bonds) == 0:
                return torch.tensor([])
            top_types = node_types[molecule.bonds]
        else:
            # in this case, you use the dataset (self) atom types, which contains the atomic environments
            # from all molecules at the same time
            node_types = self.atom_type.view(-1)
            if len(self.bonds) == 0:
                return torch.tensor([])
            top_types = node_types[self.bonds]

        top_types = (
            torch.eye(self.num_node_types)
            .to(self.train_args.device)[top_types]
            .to(
                torch.long,
            )
        )  # always use the full possible node types from the dataset for onehot encoding
        # create a diagonal square array whose dimension is the number of node types (one hot encoding of the possible node types)
        # then changes top_types to be these one-hot-encoding representations fo the different possible node types by indexing [top_types]
        # sums the two one-hot encodings to allow for permutation invariance in bond-types
        top_types = top_types.sum(1)
        # top_types is now the integer index of the distinct two-hot encodings (sum of 2 one-hot encodings),
        # while self.unique_values[type_key] now holds the distinct two-hot encodings...
        # so like dataset.topologies['bond'].unique_values['type'][0] will have the 9th and 12th index be 1 and
        # the rest 0 if the bond contains node type 9 and 12!
        cur_unique_values = top_types.unique(dim=0, return_inverse=True)[0]
        if molecule is not None:
            molecule.bond_unique_values = cur_unique_values
        else:
            self.bond_unique_values = cur_unique_values
        top_types = self.index_of2(input=top_types, source=self.bond_unique_values)
        return top_types

    def set_angle_type(self, molecule=None):
        """
        Same logic as set_bond_type. This is first called with molecule=None to assign angle types
        to all the possible angles in the entire dataset. Then, it is called again with each molecule in the dataset
        (for molecule in dataset.data) to assign the angle types according to the parent dataset.
        Also, since bond types have now been assigned, the code will reorder the atom indices within each angle
        to ensure permutation invariance if cross-terms between angles and bonds are needed (for class 2 force fields)
        """
        if self.angles.shape[0] == 0:
            return torch.tensor([])
        elif molecule is not None and molecule.angles.shape[0] == 0:
            return torch.tensor([])

        # change angle indices to maintain canonical ordering for cross-terms
        if molecule is not None:
            bonds_in_angles = get_bonds_in_angles(molecule)
            bond_types = molecule.bond_type
            all_bond_types = bond_types[bonds_in_angles].squeeze()
            mask = all_bond_types[:, 0] > all_bond_types[:, 1]
            molecule.angles[mask] = molecule.angles[mask].flip(dims=(1,))
        else:
            bonds_in_angles = get_bonds_in_angles(self)
            bond_types = self.bond_type
            all_bond_types = bond_types[bonds_in_angles].squeeze()
            mask = all_bond_types[:, 0] > all_bond_types[:, 1]
            self.angles[mask] = self.angles[mask].flip(dims=(1,))

        # actually set angle types
        if molecule is not None:
            node_types = molecule.atom_type.view(-1)
            """
            this is of shape [#angles, 3] where the top_type is the atom type of each atom in each angle"""
            top_types = node_types[molecule.angles]
            node_degrees = molecule.atom_degrees[molecule.angles.cpu()]
        else:
            node_types = self.atom_type.view(-1)
            if len(self.angles) == 0:
                return torch.tensor([])
            top_types = node_types[self.angles]
            node_degrees = self.atom_degrees[self.angles.cpu()]
        # cos_theta = get_theta(xyz, self.angles)[1]
        top_types = (
            torch.eye(self.num_node_types)
            .to(device=self.train_args.device)[top_types]
            .to(torch.long)
        )  # always use the full possible node types for onehot encoding
        # only do the permutation invariance on the outer nodes; the inner node is unique to each angle type
        top_types = torch.cat([top_types[:, 1], top_types[:, [0, 2]].sum(1)], dim=1)
        # TODO: need to separate the atom types of linear PF6- angles and perpendicular PF6- angles, also for 5-coordinated species
        # if 6 in node_degrees or 5 in node_degrees:
        #     if 6 in node_degrees:
        #         #find all the angles with the potential linear/nonlinear separation (only occurs in 6-coordinated shapes
        #         hybridization_mask = node_degrees[:,1] == 6
        #     else:
        #         #find all the angles with the potential linear/nonlinear separation (only occurs in 5-coordinated shapes
        #         hybridization_mask = node_degrees[:,1] == 5
        #     #find all the angles that are close to linear
        #     angle_mask = torch.isclose(cos_theta.abs(), torch.ones_like(cos_theta), rtol=0.3).view(-1)
        #     #create a mask that contains indexes when both above conditions are true
        #     mask = torch.logical_and(hybridization_mask, angle_mask).view(-1,1).expand(top_types.shape)
        #     #add "1" to the one-hot-encodings generated in top_types to ENSURE that the new types (for linear angles
        #     #in 6-coordinated molecules) are distinct from any possible outcome of one-hot-encoding from using "torch.eye" above
        #     top_types = torch.where(mask, top_types+1, top_types)
        cur_unique_values = top_types.unique(dim=0, return_inverse=True)[0]
        if molecule is not None:
            molecule.angle_unique_values = cur_unique_values
        else:
            self.angle_unique_values = cur_unique_values
        top_types = self.index_of2(input=top_types, source=self.angle_unique_values)
        return top_types

    def set_dihedral_type(self, molecule=None):
        """
        Same logic as set_bond_type. This is first called with molecule=None to assign dihedral types
        to all the possible dihedrals in the entire dataset. Then, it is called again with each molecule in the dataset
        (for molecule in dataset.data) to assign the dihedral types according to the parent dataset.
        Also, since angle types have now been assigned, the code will reorder the atom indices within each dihedral
        to ensure permutation invariance if cross-terms containing dihedrals are needed (for class 2 force fields)
        """
        if self.dihedrals.shape[0] == 0:
            return torch.tensor([])
        if molecule is not None and molecule.dihedrals.shape[0] == 0:
            return torch.tensor([])

        # change dihderal indices to maintain canonical ordering for cross-terms
        if molecule is not None:
            angles_in_dihedrals = get_angles_in_dihedrals(molecule)
            angle_types = molecule.angle_type
            all_angle_types = angle_types[angles_in_dihedrals].squeeze()
            mask = all_angle_types[:, 0] > all_angle_types[:, 1]
            molecule.dihedrals[mask] = molecule.dihedrals[mask].flip(dims=(1,))
        else:
            angles_in_dihedrals = get_angles_in_dihedrals(self)
            angle_types = self.angle_type
            all_angle_types = angle_types[angles_in_dihedrals].squeeze()
            mask = all_angle_types[:, 0] > all_angle_types[:, 1]
            self.dihedrals[mask] = self.dihedrals[mask].flip(dims=(1,))

        # actually set dihedral types
        if molecule is not None:
            node_types = molecule.atom_type.view(-1)
            top_types = node_types[molecule.dihedrals]
        else:
            node_types = self.atom_type.view(-1)
            top_types = node_types[self.dihedrals]
        top_types = (
            torch.eye(self.num_node_types)
            .to(device=self.train_args.device)[top_types]
            .to(torch.long)
        )
        # top_types after the stack is of dimension (# topologies, 2, 2*num_node_types)
        top_types = torch.stack(
            [
                torch.cat([top_types[:, 1], top_types[:, 0]], dim=1),
                torch.cat([top_types[:, 2], top_types[:, 3]], dim=1),
            ],
            dim=1,
        )
        # pair_types is this stack but now the 0-1 and 3-2 bonds are all in sequence instead of being stacked together in the 1st dimension
        # pair_types shape is now (2*(# topologies), 2*num_node_types))
        pair_types = top_types.view(-1, top_types.shape[2])
        cur_unique_pair_values, cur_unique_pair_types = pair_types.unique(
            dim=0, return_inverse=True
        )
        if molecule is not None:
            molecule.pair_unique_values = cur_unique_pair_values
        else:
            self.pair_unique_values = cur_unique_pair_values
        pair_types = self.index_of2(input=pair_types, source=self.pair_unique_values)
        pair_types = pair_types.view(-1, 2)

        if molecule is not None:
            molecule.num_pair_types = pair_types.max().item() + 1
        else:
            self.num_pair_types = pair_types.max().item() + 1
        top_types = (
            torch.eye(self.num_pair_types)
            .to(device=self.train_args.device)[pair_types]
            .to(torch.long)
        )
        top_types = top_types.sum(1)
        cur_unique_values = top_types.unique(dim=0, return_inverse=True)[0]
        if molecule is not None:
            molecule.dihedral_unique_values = cur_unique_values
        else:
            self.dihedral_unique_values = cur_unique_values
        top_types = self.index_of2(input=top_types, source=self.dihedral_unique_values)
        return top_types

    def set_improper_type(self, molecule=None):
        """
        Same logic as set_bond_type. This is first called with molecule=None to assign improper types
        to all the possible impropers in the entire dataset. Then, it is called again with each molecule in the dataset
        (for molecule in dataset.data) to assign the improper types according to the parent dataset.
        Permutation invariance for class 2 is based on bond-types of internal bonds
        """
        if self.impropers.shape[0] == 0:
            return torch.tensor([])
        elif molecule is not None and molecule.impropers.shape[0] == 0:
            return torch.tensor([])

        # rearrange for cross terms:
        if molecule is not None:
            bond_nodes = molecule.impropers[:, [0, 1, 0, 2, 0, 3]].view(-1, 3, 2)
            bonds = get_bonds_in_impropers(molecule)
            bond_types = molecule.bond_type[bonds].squeeze()
        else:
            bond_nodes = self.impropers[:, [0, 1, 0, 2, 0, 3]].view(-1, 3, 2)
            bonds = get_bonds_in_impropers(self)
            bond_types = self.bond_type[bonds].squeeze()
        min_inx = torch.argmin(bond_types, dim=1)
        """(cur-1).abs() makes it so that if all three bond types are the same, the middle index will always be chosen
        regardless of if torch.argmax(torch.tensor([1,1,1])) returns 0 or 2."""
        max_inx = torch.tensor(
            [
                (cur - 1).abs() if cur == min_inx[i] else cur
                for (i, cur) in enumerate(torch.argmax(bond_types, dim=1))
            ]
        ).to(device=min_inx.device)
        mask = torch.ones_like(bonds).scatter_(
            1, torch.stack((min_inx, max_inx)).transpose(1, 0), 0.0
        )
        mid_inx = (
            torch.ones([bonds.shape[0], 1], dtype=torch.int64, device=min_inx.device)
            * torch.arange(3).to(device=min_inx.device)
        )[mask.bool()]

        """rearrange the indices of impropers to ensure permutation invariance. This is done by sorting
        the bond types and then reordering i,k, and l. the central atom (self.impropers[:,1]) should NEVER be 
        resorted since it must be second according to LAMMPS conventions - https://docs.lammps.org/improper_harmonic.html"""
        """doing (range(), min_inx, 1) in order to get the index of the atom that is in the minimum-type-bond 
        that ISNT the central atom which is in place 0 for each bond_node"""
        impropers_i = bond_nodes[(range(bond_nodes.shape[0]), min_inx, 1)]
        impropers_k = bond_nodes[(range(bond_nodes.shape[0]), mid_inx, 1)]
        impropers_l = bond_nodes[(range(bond_nodes.shape[0]), max_inx, 1)]
        if molecule is not None:
            molecule.impropers = torch.stack(
                (molecule.impropers[:, 0], impropers_i, impropers_k, impropers_l), dim=1
            )
        else:
            self.impropers = torch.stack(
                (self.impropers[:, 0], impropers_i, impropers_k, impropers_l), dim=1
            )

        # actually determine the improper types
        # column 0 of improper_types is the center node of the improper
        if molecule is not None:
            node_types = molecule.atom_type.view(-1)
            if len(molecule.impropers) == 0:
                return torch.tensor([]).view(-1)
            improper_types = node_types[molecule.impropers]
        else:
            node_types = self.atom_type.view(-1)
            if len(self.impropers) == 0:
                return torch.tensor([]).view(-1)
            improper_types = node_types[self.impropers]
        if len(improper_types) == 0:
            return torch.tensor([])
        top_types = (
            torch.eye(self.num_node_types)
            .to(device=self.train_args.device)[improper_types]
            .to(torch.long)
        )
        top_types = top_types.sum(1)
        cur_unique_values = top_types.unique(dim=0, return_inverse=True)[0]
        if molecule is not None:
            molecule.improper_unique_values = cur_unique_values
        else:
            self.improper_unique_values = cur_unique_values
        top_types = self.index_of2(input=top_types, source=self.improper_unique_values)
        return top_types

    def batch_graph(self) -> List[TopologyBuilder]:
        self._batch_graph = [self.data]
        return self._batch_graph

    def __getitem__(self, item) -> Union[TopologyBuilder, List[TopologyBuilder]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                if a slice is provided.
        """
        return self.data[item]

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)


def construct_molecule_batch(data: List[TopologyBuilder]) -> List[TopologyBuilder]:
    data = TopologyDataset(data)

    data.batch_graph()
    return data


class TopologyDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(
        self,
        dataset: TopologyDataset,
        batch_size: int = 50,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0

        super(TopologyDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=construct_molecule_batch,
        )
