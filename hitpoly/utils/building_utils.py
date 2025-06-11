from ffnet.writers.box_builder import create_long_smiles, get_atom_count, get_mol_mass


def calculate_box_numbers(
    smiles,
    repeats,
    concentration=None,
    polymer_count=None,
    solv_atom_count=None,
    end_Cs=True,
    salt_smiles=None,
):
    long_smiles, _ = create_long_smiles(smiles, repeats=repeats, add_end_Cs=end_Cs)
    print(
        f"Polymer chain has {get_atom_count(long_smiles)} atoms, should be around 800 to not brake."
    )
    if not concentration and not polymer_count:
        return

    if not salt_smiles:
        salt_smiles = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "[Li+]"]

    print(
        "Amount of atoms in box",
        get_atom_count(long_smiles, salt_smiles, polymer_count, concentration),
    )

    print(
        "Repeat unit molecular mass",
        get_mol_mass(smiles.replace("[Cu]", "").replace("[Au]", "")),
    )

  # molality
    print(
        "Molality", concentration / (get_mol_mass(long_smiles) * polymer_count) * 1000
    )


    # Li:solv_atoms
    print(
        "Li:SolvAtom ratio",
        (concentration / (repeats * polymer_count * solv_atom_count)),
    )
  
    return long_smiles


def salt_string_to_values(ffnet_path, salt_string):

    cation = salt_string.split(".")[0]
    anion = salt_string.split(".")[1]

    
    salt_path = f"{ffnet_path}/data/pdb_files"
    if cation == "Na":
        salt_smiles = ["[Na+]"]
        salt_paths = [
            f"{salt_path}/geometry_file_Na.pdb",
        ]
        salt_data_paths = [
            f"{ffnet_path}/data/forcefield_files/lammps_Na_q100.data",
        ]
    elif cation == "Li":
        salt_smiles = ["[Li+]"]
        salt_paths = [
            f"{salt_path}/geometry_file_Li.pdb",
        ]
        salt_data_paths = [
            f"{ffnet_path}/data/forcefield_files/lammps_Li_q100.data",
        ]
    else:
        raise ValueError(f"Cation {cation} not supported")
    
    if anion == "TFSI":
        salt_smiles.append("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F")
        salt_paths.append(f"{salt_path}/geometry_file_TFSI.pdb")
        salt_data_paths.append(f"{ffnet_path}/data/forcefield_files/lammps_TFSI_q100.data")
        ani_name_rdf = ["N,O"]
    elif anion == "PF6":
        salt_smiles.append("F[P-](F)(F)(F)(F)F")
        salt_paths.append(f"{salt_path}/geometry_file_PF6.pdb")
        salt_data_paths.append(f"{ffnet_path}/data/forcefield_files/lammps_PF6_q100.data")
        ani_name_rdf = ["P,F"]
    elif anion == 'FSI':
        salt_smiles.append("[N-](S(=O)(=O)F)S(=O)(=O)F")
        salt_paths.append(f"{salt_path}/geometry_file_FSI.pdb")
        salt_data_paths.append(f"{ffnet_path}/data/forcefield_files/lammps_FSI_q100.data")
        ani_name_rdf = ["N,O"]
    else:
        raise ValueError(f"Anion {anion} not supported")

    return salt_smiles, salt_paths, salt_data_paths, ani_name_rdf
