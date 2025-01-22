import json
import torch
from hitpoly.utils.constants import *


def lammps_writer(path, bonds, angles, dihedrals, impropers, pairs, molecule):
    # Adding plus one to start enumerating at 1!!
    atom_idxs = torch.tensor(molecule.atom_idxs) + 1
    atom_types = molecule.atomic_nums
    bond_idxs = molecule.bonds + 1
    angle_idxs = molecule.angles + 1
    dihedral_idxs = molecule.dihedrals + 1
    improper_idxs = molecule.impropers + 1
    geometry = molecule.geometries[0]
    x_min, x_max = geometry[:, 0].min(), geometry[:, 0].max() * 4
    y_min, y_max = geometry[:, 1].min(), geometry[:, 1].max() * 4
    z_min, z_max = geometry[:, 2].min(), geometry[:, 2].max() * 4

    with open(path, "w+") as f:
        f.write(
            f"LAMMPS force field file for {molecule.smiles[0]}, created by hitpoly\n\n"
        )
        f.write(f"{len(pairs)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n")
        f.write(f"{len(dihedrals)} dihedrals\n")
        f.write(f"{len(impropers)} impropers\n\n")
        f.write(f"{len(atom_types)} atom types\n")
        f.write(f"{len(bond_idxs)} bond types\n")
        f.write(f"{len(angle_idxs)} angle types\n")
        f.write(f"{len(dihedral_idxs)} dihedral types\n")
        f.write(f"{len(improper_idxs)} improper types\n\n")
        f.write(f"{x_min} {x_max} xlo xhi\n")
        f.write(f"{y_min} {y_max} ylo yhi\n")
        f.write(f"{z_min} {z_max} zlo zhi\n\n")

        f.write("Masses \n\n")
        for i, atom in zip(atom_idxs, atom_types):
            mass = ELEMENT_TO_MASS[NUM_TO_ELEMENT[atom]]
            f.write(f"{i} {mass} # {atom} \n")
        f.write("\n")
        f.write("Pair Coeffs \n\n")
        for i, pair in enumerate(pairs):
            f.write(f"{i + 1} {pair[1]} {pair[2]} \n")
        f.write("\n")
        f.write("Bond Coeffs \n\n")
        for i, (bond_f, bond_r) in enumerate(zip(bonds[:, 0], bonds[:, 1])):
            f.write(f"{i + 1} {bond_f} {bond_r} \n")
        f.write("\n")
        f.write("Angle Coeffs \n\n")
        for i, (angle_f, angle_t) in enumerate(zip(angles[:, 0], angles[:, 1])):
            f.write(f"{i + 1} {angle_f} {angle_t} \n")
        f.write("\n")
        f.write("Dihedral Coeffs \n\n")
        for i, dihedral in enumerate(dihedrals):
            f.write(
                f"{i + 1} {dihedral[0]} {dihedral[1]} {dihedral[2]} {dihedral[3]} \n"
            )
        f.write("\n")
        f.write("Improper Coeffs \n\n")
        for i, imporper in enumerate(impropers):
            # Here it's a bit confusing, but I'm saving the
            # improper coefficients from the lammps file as if
            # it was an openmm xml file with 4 improper 'k' coeffs
            # but actually only one of them is a value for the CVFF
            # improper angle, and it's in the 1st position
            f.write(f"{i + 1} {imporper[1]}  -1   2\n")
        f.write("\n")
        f.write("Atoms \n\n")
        for i, (charge, xyz) in enumerate(zip(pairs, geometry)):
            f.write(f"{i + 1} 1 {i + 1} {charge[0]} {xyz[0]} {xyz[1]} {xyz[2]} \n")
        f.write("\n")
        f.write("Bonds \n\n")
        for i, bond in enumerate(bond_idxs):
            f.write(f"{i + 1} {i + 1} {bond[0]} {bond[1]} \n")
        f.write("\n")
        f.write("Angles \n\n")
        for i, angle in enumerate(angle_idxs):
            f.write(f"{i + 1} {i + 1} {angle[0]} {angle[1]} {angle[2]} \n")
        f.write("\n")
        f.write("Dihedrals \n\n")
        for i, dih in enumerate(dihedral_idxs):
            f.write(f"{i + 1} {i + 1} {dih[0]} {dih[1]} {dih[2]} {dih[3]} \n")
        f.write("\n")
        f.write("Impropers \n\n")
        for i, imp in enumerate(improper_idxs):
            f.write(f"{i + 1} {i + 1} {imp[0]} {imp[1]} {imp[2]} {imp[3]} \n")


def json_dump(arguments, path):
    dump = {}
    for ind, i in enumerate(arguments):
        if i.startswith("--"):
            if i == arguments[-1]:
                dump[i] = ""
            elif arguments[ind + 1].startswith("--"):
                dump[i] = ""
            else:
                dump[i] = arguments[ind + 1]
    with open(f"{path}/hyperparameters.json", "w") as f:
        json.dump(dump, f)


def json_read(path):
    with open(f"{path}/hyperparameters.json", "r") as f:
        load = json.load(f)
    arguments = []
    for key, val in load.items():
        if not val:
            arguments.append(key)
        else:
            arguments.append(key)
            arguments.append(val)
    return arguments


def loss_writer(path, loss):
    with open(f"{path}/loss.txt", "+w") as f:
        for i, l in enumerate(loss):
            f.write(f"{i} \t {l} \n")
