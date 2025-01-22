import argparse
import os
import time
import shutil
import uuid
from hitpoly.writers.box_builder import *
from hitpoly.simulations.gromacs_writer import GromacsWriter
from distutils.dir_util import copy_tree
import sys

sys.setrecursionlimit(5000)


def run(
    results_path,
    final_path,
    smiles="[Cu]COC[Au]",
    ligpargen_repeats=3,
    repeats=60,
    charge_scale=0.75,
    polymer_count=25,
    concentration: list = [100, 100],
    poly_name="PEO",
    charges="LPG",
    add_end_Cs=True,
    hitpoly_path=None,
    salt_smiles=None,
    salt_paths=None,
    salt_data_paths=None,
    lit_charges_save_path=None,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    box_multiplier=1,
    enforce_generation=False,
    salt=True,
    simu_type="conductivity",
    simu_temp=415,
):
    packmol_path = f"/home/{os.getlogin()}/packages/packmol"
    if not hitpoly_path:
        hitpoly_path = f"/home/{os.getlogin()}/HiTPoly"

    if charges == "LIT" or charges == "DFT":
        lit_charges_save_path = f"/home/{os.getlogin()}/HiTPoly/data/forcefield_files"

    if salt:
        if not salt_smiles and not salt_paths and not salt_data_paths:
            salt_path = f"{hitpoly_path}/data/pdb_files"
            salt_smiles = ["[Li+]", "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"]
            salt_paths = [
                f"{salt_path}/geometry_file_Li.pdb",
                f"{salt_path}/geometry_file_TFSI.pdb",
            ]
            salt_data_paths = [
                f"{hitpoly_path}/data/forcefield_files/lammps_Li_q100.data",
                f"{hitpoly_path}/data/forcefield_files/lammps_TFSI_q100.data",
            ]
        elif not salt_smiles or not salt_paths or not salt_data_paths:
            raise ValueError(
                "Must have an input all of these: salt_smiles, salt_paths and salt_data_paths or none."
            )
    else:
        salt_paths = []
        salt_data_paths = []
        salt_smiles = []

    date_name = time.strftime("%y%m%d_H%HM%M")
    conc_name = int(np.array(concentration).mean())
    extra_naming = f"{poly_name}_N{repeats}_T{simu_temp}_C{conc_name}_q0{int(charge_scale * 100)}_q{charges}_"
    if not salt:
        extra_naming = f"{poly_name}_N{repeats}_PC{polymer_count}_q{charges}_"
    folder_name = f"{extra_naming}{date_name}"
    save_path = f"{results_path}/{folder_name}"
    ligpargen_path = f"{save_path}/ligpargen"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isdir(ligpargen_path):
        os.makedirs(ligpargen_path)

    print(
        f"Creating LigParGen parameter files for {smiles} with {ligpargen_repeats} repeats "
    )
    print(f" at {ligpargen_path}")
    mol_inital, smiles_initial = create_ligpargen(
        smiles=smiles,
        repeats=ligpargen_repeats,
        add_end_Cs=add_end_Cs,
        ligpargen_path=ligpargen_path,
        hitpoly_path=hitpoly_path,
        reaction=reaction,
        product_index=product_index,
        platform="local",
    )

    if ligpargen_repeats == repeats:
        long_smiles = smiles
        filename = "polymer_conformation.pdb"
        shutil.move(f"{ligpargen_path}/PLY.pdb", f"{save_path}/{filename}")
        minimize = False
    else:
        long_smiles, repeats = create_long_smiles(
            smiles,
            repeats=repeats,
            add_end_Cs=add_end_Cs,
            reaction=reaction,
            product_index=product_index,
        )

    print(f"Creating conformer files for monomer {smiles}, with {repeats} repeats")
    print(f"at path {save_path}")
    if not ligpargen_repeats == repeats:
        filename, mol, minimize = create_conformer_pdb(
            save_path,
            long_smiles,
            name="polymer_conformation",
            enforce_generation=enforce_generation,
        )
    print(f"Saved conformer pdb.")

    r, atom_names, atoms, bonds_typed = generate_atom_types(mol_inital, 2)

    mol_long = Chem.MolFromSmiles(long_smiles)
    mol_long = Chem.AddHs(mol_long)
    r_long, atom_names_long, atoms_long, bonds_typed_long = generate_atom_types(
        mol_long, 2
    )

    assert r.shape[-1] == r_long.shape[-1]

    param_dict = generate_parameter_dict(ligpargen_path, atom_names, atoms, bonds_typed)

    if minimize:
        minimize_polymer(
            short_smiles=smiles_initial,
            save_path=save_path,
            long_smiles=long_smiles,
            atoms_long=atoms_long,
            atoms_short=atoms,
            atom_names_short=atom_names,
            atom_names_long=atom_names_long,
            param_dict=param_dict,
            lit_charges_save_path=lit_charges_save_path,
            charges=charges,
            poly_name=poly_name,
        )

    create_box_and_ff_files(
        short_smiles=smiles_initial,
        save_path=save_path,
        long_smiles=long_smiles,
        filename=filename,
        polymer_count=polymer_count,
        concentration=concentration,
        packmol_path=packmol_path,
        atoms_long=atoms_long,
        atoms_short=atoms,
        atom_names_short=atom_names,
        atom_names_long=atom_names_long,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        charge_scale=charge_scale,
        salt_smiles=salt_smiles,
        salt_paths=salt_paths,
        salt_data_paths=salt_data_paths,
        box_multiplier=box_multiplier,
        salt=salt,
        poly_name=poly_name,
    )

    simu_fold_name = f"{date_name}_{str(uuid.uuid1(4))[:5]}_T{simu_temp}"
    gromacs_save_path = f"{save_path}/{simu_fold_name}"

    print(f"Creating equilibration files at {save_path}")

    if not os.path.isdir(gromacs_save_path):
        os.makedirs(gromacs_save_path)

    file_names = os.listdir(save_path + "/gromacs")
    for file_name in file_names:
        shutil.copy(os.path.join(save_path + "/gromacs", file_name), gromacs_save_path)

    gromacs = GromacsWriter(
        save_path=gromacs_save_path,
        overall_save_path=f"{'/'.join(final_path.split('/')[-2:])}/{folder_name}/{simu_fold_name}",
    )
    if simu_type.lower() == "conductivity":
        gromacs.equil_and_prod_balsara(
            prod_run_time=300,
            simu_temperature=simu_temp,
            analysis=True,
            image_name=f"{extra_naming}T{simu_temp}",
        )
    elif simu_type.lower() == "tg":
        gromacs.tg_simulations(
            prod_run_time=10,
            start_temperature=500,
            end_temperature=100,
            temperature_step=20,
        )

    copy_tree(save_path, f"{final_path}/{folder_name}")
    print(f"Directory successfully coppied to {final_path}/{folder_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxbuilder for OpenMM simulations")
    parser.add_argument(
        "-p", "--results_path", help="Path to the where the new directory to be created"
    )
    parser.add_argument(
        "-s",
        "--smiles_path",
        help="Smiles of the polymer to be created, of the form [Cu]*[Au]",
    )
    parser.add_argument(
        "-lr",
        "--lpg_repeats",
        help="How many repeats to be created for the small n-mer polymer",
        default="1",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        help="How many repeats to be created for the polymer",
        default="1",
    )
    parser.add_argument(
        "-pc",
        "--polymer_count",
        help="How many polymer chains or molecules to be packed",
    )
    parser.add_argument(
        "-c",
        "--concentration",
        help="Amount of LiTFSI pairs in the solution",
        default="0",
    )
    parser.add_argument(
        "-pn",
        "--poly_name",
        help="Extra name for the folder where all the files be put",
    )
    parser.add_argument(
        "-cs",
        "--charge_scale",
        help="To what value the charges of the salts be scaled",
        default="1",
    )
    parser.add_argument(
        "-ct", "--charge_type", help="What type of charges to select", default="LPG"
    )
    parser.add_argument(
        "-ecs",
        "--end_carbons",
        help="When creating polymer if end carbons should be added",
        default="False",
    )
    parser.add_argument(
        "-f",
        "--hitpoly_path",
        help="Path towards the HiTPoly folder",
        default="None",
    )
    parser.add_argument(
        "-ff",
        "--final_path",
        help="Path to where the folders should be transfered",
        default="None",
    )
    parser.add_argument(
        "-react",
        "--reaction",
        help="Reaction that creates the polymer",
        default="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    )
    parser.add_argument(
        "-pi",
        "--product_index",
        help="Product index which to use for the smarts reaction",
        default="0",
    )
    parser.add_argument(
        "-box",
        "--box_multiplier",
        help="PBC box size multiplier for packmol, poylmers 1, other molecules 4-10",
        default="1",
    )
    parser.add_argument(
        "-conf",
        "--enforce_generation",
        help="Whether to force rdkit to create a conformation",
        default="False",
    )
    parser.add_argument(
        "--simu_type",
        help="What type of simulation to perform, options [conductivity, tg]}",
        default="conductivity",
    )
    parser.add_argument(
        "--salt", help="Should salt be added to the simulation", default="True"
    )
    parser.add_argument("--temperature", help="Simulation temperature", default="430")

    args = parser.parse_args()
    if args.end_carbons == "false" or args.end_carbons == "False":
        add_end_Cs = False
    else:
        add_end_Cs = True

    if args.hitpoly_path == "None":
        args.hitpoly_path = None
    if args.final_path == "None":
        args.final_path = None
    if args.enforce_generation == "False":
        args.enforce_generation = False
    else:
        args.enforce_generation = True
    if args.salt == "False":
        args.salt = False
    else:
        args.salt = True

    with open(args.smiles_path, "r") as f:
        lines = f.readlines()
        smiles = lines[0]

    run(
        results_path=args.results_path,
        final_path=args.final_path,
        smiles=smiles,
        ligpargen_repeats=int(args.lpg_repeats),
        repeats=int(args.repeats),
        poly_name=args.poly_name,
        charge_scale=float(args.charge_scale),
        polymer_count=int(args.polymer_count),
        concentration=[int(args.concentration), int(args.concentration)],
        charges=args.charge_type,
        add_end_Cs=add_end_Cs,
        reaction=args.reaction,
        product_index=int(args.product_index),
        box_multiplier=float(args.box_multiplier),
        enforce_generation=args.enforce_generation,
        simu_type=args.simu_type,
        salt=args.salt,
        simu_temp=int(args.temperature),
    )
