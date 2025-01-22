from hitpoly.writers.box_builder import *


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
    # print(
    #     "Li:SolvAtom ratio",
    #     (concentration / (repeats * polymer_count * solv_atom_count)),
    # )

    return long_smiles
