import os
import sys

sys.setrecursionlimit(5000)
import time
import math
import torch
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
import string as STRING
import itertools
from openbabel import pybel
import subprocess
from typing import Union

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

from hitpoly.writers.file_writers import json_read
from hitpoly.utils.args import hitpolyArgs
from hitpoly.data.builder import TopologyBuilder, TopologyDataset
from hitpoly.data.ff_file_reader import ForceFieldFileReader
from hitpoly.utils.constants import NUM_TO_ELEMENT, ELEMENT_TO_NUM
from hitpoly.simulations.openmm_scripts import (
    equilibrate_polymer,
    equilibrate_system_1,
    equilibrate_system_2,
    prod_run_nvt,
)

from openmm.app import PDBFile, ForceField
#Test
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
from openmm.app import PDBFile
import ast

def write_atom_names_rdf_from_pdb(pdb_path, output_path="atom_names_rdf.txt"):
    pdb = PDBFile(pdb_path)
    atom_lines = []

    for atom in pdb.topology.atoms():
        atom_lines.append(f"{atom.element.symbol}, {atom.residue.name}, {atom.residue.id}")

    with open(output_path, "w") as f:
        for line in atom_lines:
            f.write(line + "\n")

    print(f"Created atom_names_rdf.txt at {output_path}")

def label_polymer_atoms_with_substructure(polymer_smiles_list, monomer_smiles, cap_smiles="C"):
    # Handle input
    if isinstance(polymer_smiles_list, list):
        assert len(polymer_smiles_list) == 1, "Expected a list with one polymer SMILES"
        polymer_smiles = polymer_smiles_list[0]
    else:
        polymer_smiles = polymer_smiles_list

    polymer = Chem.AddHs(Chem.MolFromSmiles(polymer_smiles))
    labels = [3] * polymer.GetNumAtoms()  # Default label: other

    # Label caps as 2
    cap = Chem.AddHs(Chem.MolFromSmiles(cap_smiles))
    for match in polymer.GetSubstructMatches(cap):
        for idx in match:
            labels[idx] = 2

    # Strip [Cu] and [Au] directly in place before matching
    monA_smiles = monomer_smiles[0].replace("[Cu]", "").replace("[Au]", "")
    monB_smiles = monomer_smiles[1].replace("[Cu]", "").replace("[Au]", "")

    # Label monomer A as 0
    monA = Chem.AddHs(Chem.MolFromSmiles(monA_smiles))
    for match in polymer.GetSubstructMatches(monA):
        for idx in match:
            if labels[idx] == 3:
                labels[idx] = 0

    # Label monomer B as 1
    monB = Chem.AddHs(Chem.MolFromSmiles(monB_smiles))
    for match in polymer.GetSubstructMatches(monB):
        for idx in match:
            if labels[idx] == 3:
                labels[idx] = 1

    return labels


def create_oligomers(smiles, polymerization_mode, arms=3):
    try:
        # Validate input
        if len(smiles) == 0 or len(smiles) > 10:
            raise ValueError("The smiles list must contain between 1 and 10 monomers.")

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            raise ValueError("One or more SMILES strings could not be converted to valid RDKit molecules.")
        
        rxn = AllChem.ReactionFromSmarts('[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]')
        products_dict = {}
        products_list = []

        if polymerization_mode in ['random', 'alternating']:
            labels = ['A', 'B', 'C'][:len(mols)]
            combinations = list(itertools.product(labels, repeat=3))
            mapping = {labels[i]: mols[i] for i in range(len(mols))}
            patterns = {''.join(combo): [mapping[c] for c in combo] for combo in combinations}

        elif polymerization_mode == 'block':
            valid_combinations = ['AAA', 'BBB', 'CCC', 'AAB', 'ABB', 'BAA', 'BBA', 'AAC', 'ACC', 'CAA', 'CCA', 'BBC', 'BCC', 'CBB', 'CCB']
            mapping = {'A': mols[0], 'B': mols[1] if len(mols) > 1 else mols[0], 'C': mols[2] if len(mols) > 2 else mols[0]}
            patterns = {combo: [mapping[c] for c in combo] for combo in valid_combinations}

        elif polymerization_mode == 'star':
            if len(mols) != 1:
                raise ValueError("Star polymerization requires exactly one repeat unit SMILES.")
            monomer = smiles[0].replace("[Cu]", "").replace("[Au]", "")
            patterns = {
                'AAA': f"{monomer}{monomer}{monomer}",
                'Core': f"C({')C('.join([monomer * 2] * arms)})"
            }

        elif polymerization_mode == 'homopolymer':
            if len(mols) == 1:
                # Single homopolymer mode — default behavior
                patterns = {
                    'AAA': [mols[0], mols[0], mols[0]]
                }
            else:
                #Blend mode: generate AAA, BBB, CCC for each monomer
                labels = [chr(65 + i) for i in range(len(mols))]  # A, B, C...
                patterns = {label * 3: [mol, mol, mol] for label, mol in zip(labels, mols)}

        else:
            raise ValueError(f"Unknown polymerization mode: {polymerization_mode}")

        # Reaction logic
        for pattern_name, pattern in patterns.items():
            new_mol = pattern[0] if isinstance(pattern, list) else pattern
            if isinstance(pattern, list):
                for reactant in pattern[1:]:
                    results = rxn.RunReactants((new_mol, reactant))
                    if results and len(results[0]) == 1:
                        new_mol = results[0][0]
                    else:
                        break
                pattern_smiles = Chem.MolToSmiles(new_mol, canonical=True)
            else:
                pattern_smiles = pattern  # already a SMILES string

            # Replace placeholders with carbon
            pattern_smiles = pattern_smiles.replace("[Cu]", "C").replace("[Au]", "C")
            products_dict[pattern_name] = pattern_smiles
            products_list.append(pattern_smiles)

        return products_dict, products_list

    except Exception as e:
        print(f"An error occurred in create_oligomers: {e}")
        return None, None



def create_long_smiles(
    save_path=".",
    smiles=[],
    fractions=[],
    total_repeats=10,
    add_end_Cs=True,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    polymerization_mode='homopolymer',
    num_blocks="None",
    arms="None",
    write_log=True,
    log_filename=None
):
    try:
        if len(smiles) == 0 or len(fractions) == 0:
            raise ValueError("The 'smiles' and 'fractions' lists cannot be empty.")
        if len(fractions) != len(smiles):
            raise ValueError("The 'fractions' list must have the same length as the 'smiles' list.")
        if sum(fractions) != 1.0:
            raise ValueError("The sum of the fractions must equal 1.0.")

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            raise ValueError("One or more SMILES strings could not be converted to valid RDKit molecules.")

        if polymerization_mode == 'star':
            if len(smiles) != 1:
                raise ValueError("Star polymerization requires exactly one repeat unit SMILES.")
            monomer = smiles[0]
            repeat_unit = monomer.replace("[Cu]", "").replace("[Au]", "")
            arm_structure = repeat_unit * total_repeats
            core = "C(" + ")C(".join([arm_structure] * arms) + ")"
            final_smiles = core
            monomer_counts = {monomer: arms * total_repeats}

            try:
                final_mol = Chem.MolFromSmiles(final_smiles)
                final_mol_with_h = Chem.AddHs(final_mol)
                total_atoms = final_mol_with_h.GetNumAtoms()
            except Exception as e:
                print(f"Warning: Could not calculate atom count for star polymer. Error: {e}")
                total_atoms = 0

        else:
            rxn = AllChem.ReactionFromSmarts(reaction)
            new_mol = mols[0]  # always start with the first smiles
            repeated_mols = []

            if polymerization_mode == 'homopolymer':
                repeated_mols = [mols[0]] * (total_repeats - 1)
                monomer_counts = {Chem.MolToSmiles(mols[0]): total_repeats}

            elif polymerization_mode == 'alternating':
                repeated_mols = [mols[i % len(mols)] for i in range(total_repeats)]

                monomer_counts = {}
                for m in mols:
                    count = repeated_mols.count(m)
                    if m == mols[0]:
                        count += 1
                    monomer_counts[Chem.MolToSmiles(m)] = count

            elif polymerization_mode == 'random':
                monomer_counts_int = [int(total_repeats * f) for f in fractions]
                for i in range(total_repeats - sum(monomer_counts_int)):
                    monomer_counts_int[i % len(monomer_counts_int)] += 1
                for count, mol in zip(monomer_counts_int, mols):
                    repeated_mols.extend([mol] * count)
                random.shuffle(repeated_mols)
                monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}

            elif polymerization_mode == 'block':
                if num_blocks == "None" or not isinstance(num_blocks, int):
                    raise ValueError("'num_blocks' must be set to an integer for block copolymerization.")

                fractions_equal = all(abs(f - fractions[0]) < 1e-6 for f in fractions)
                blocks_per_type = num_blocks // len(smiles)

                if fractions_equal:
                    # total_repeats must be divisible by (blocks_per_type * len(smiles))
                    monomers_per_type = (total_repeats // len(smiles))
                    # make it divisible by blocks_per_type
                    monomers_per_block = monomers_per_type // blocks_per_type
                    if monomers_per_block == 0:
                        monomers_per_block = 1  # fallback if too small
                    monomers_per_type = monomers_per_block * blocks_per_type
                    total_repeats_adjusted = monomers_per_type * len(smiles)
                    print(f"Adjusted total_repeats from {total_repeats} → {total_repeats_adjusted} for even blocks.")
                    total_repeats = total_repeats_adjusted

                    monomer_counts_int = [monomers_per_type] * len(smiles)
                else:
                    # original logic
                    monomer_counts_int = [int(total_repeats * f) for f in fractions]
                    for i in range(total_repeats - sum(monomer_counts_int)):
                        monomer_counts_int[i % len(monomer_counts_int)] += 1

                # Compute block sizes
                block_sizes_per_type = []
                for count in monomer_counts_int:
                    size = max(1, count // blocks_per_type)
                    block_sizes_per_type.append(size)

                repeated_mols = []
                block_sizes_and_monomers = []
                for b in range(num_blocks):
                    monomer_idx = b % len(smiles)
                    mol = mols[monomer_idx]
                    size = block_sizes_per_type[monomer_idx]
                    repeated_mols.extend([mol] * size)
                    block_sizes_and_monomers.append((size, mol))

                monomer_counts = {
                    Chem.MolToSmiles(m): repeated_mols.count(m)
                    for m in mols
                }
            else:
                raise ValueError(f"Unknown polymerization mode: {polymerization_mode}")

            # REMOVE FIRST MONOMER JUST BEFORE REACTING
            if polymerization_mode in ["alternating", "random", "block"]:
                repeated_mols = repeated_mols[1:]
                # Recompute actual counts after removing first monomer
                monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}
                # Add one for initial mol
                first_smiles = Chem.MolToSmiles(mols[0])
                monomer_counts[first_smiles] += 1

                total_count = sum(monomer_counts.values())
                if total_count < total_repeats:
                    leftovers = total_repeats - total_count
                    for i in range(leftovers):
                        m = mols[i % len(mols)]
                        monomer_counts[Chem.MolToSmiles(m)] += 1


            for reactant_mol in repeated_mols:
                results = rxn.RunReactants((new_mol, reactant_mol))
                if results and len(results[0]) == 1:
                    new_mol = results[product_index][0]
                else:
                    break

        # Create SMILES from RDKit molecule
        final_smiles = Chem.MolToSmiles(new_mol)

        # Replace reactive placeholders with terminal methyl groups, for example
        if add_end_Cs:
            final_smiles = final_smiles.replace("[Cu]", "C").replace("[Au]", "C")

        # Canonicalize the final SMILES after all modifications
        final_mol = Chem.MolFromSmiles(final_smiles)
        if final_mol:
            final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
            final_mol_with_h = Chem.AddHs(final_mol)
            total_atoms = final_mol_with_h.GetNumAtoms()
        else:
            final_smiles = ""
            total_atoms = 0

        print(f"Generated SMILES ({polymerization_mode}): {final_smiles}")
        print(f"Atom count (with Hs): {total_atoms}")

        # Only write a log if a filename is explicitly passed
        if write_log and log_filename:
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(save_path, log_filename)
            with open(output_path, "w") as file:
                file.write("Polymerization Log\n")
                file.write(f"Architecture: {polymerization_mode}\n")
                if polymerization_mode == 'block':
                    file.write(f"Block sizes and monomers: {[(size, Chem.MolToSmiles(mol)) for size, mol in block_sizes_and_monomers]}\n")
                file.write(f"Monomer counts: {monomer_counts}\n")
                file.write(f"Final SMILES: {final_smiles}\n")
                file.write(f"Total number of atoms: {total_atoms}\n")
            print(f"Saved polymer log to {output_path}")

        return final_smiles, total_repeats

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0

def create_polymer_for_estimation(
    save_path=".",
    smiles=[],
    fractions=[],
    total_repeats=10,
    add_end_Cs=True,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    polymerization_mode='homopolymer',
    num_blocks="None",
    arms="None",
    write_log=True,
    log_filename=None
):
    try:
        if len(smiles) == 0 or len(fractions) == 0:
            raise ValueError("The 'smiles' and 'fractions' lists cannot be empty.")
        if len(fractions) != len(smiles):
            raise ValueError("The 'fractions' list must have the same length as the 'smiles' list.")
        if sum(fractions) != 1.0:
            raise ValueError("The sum of the fractions must equal 1.0.")

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            raise ValueError("One or more SMILES strings could not be converted to valid RDKit molecules.")

        if polymerization_mode == 'star':
            if len(smiles) != 1:
                raise ValueError("Star polymerization requires exactly one repeat unit SMILES.")
            monomer = smiles[0]
            repeat_unit = monomer.replace("[Cu]", "").replace("[Au]", "")
            arm_structure = repeat_unit * total_repeats
            core = "C(" + ")C(".join([arm_structure] * arms) + ")"
            final_smiles = core
            monomer_counts = {monomer: arms * total_repeats}

            try:
                final_mol = Chem.MolFromSmiles(final_smiles)
                final_mol_with_h = Chem.AddHs(final_mol)
                total_atoms = final_mol_with_h.GetNumAtoms()
            except Exception as e:
                print(f"Warning: Could not calculate atom count for star polymer. Error: {e}")
                total_atoms = 0

        else:
            rxn = AllChem.ReactionFromSmarts(reaction)
            new_mol = mols[0] #always start with the first smiles
            repeated_mols = []

            if polymerization_mode == 'homopolymer':
                repeated_mols = [mols[0]] * (total_repeats - 1)
                monomer_counts = {Chem.MolToSmiles(mols[0]): total_repeats}

            elif polymerization_mode == 'alternating':
                repeated_mols = [mols[i % len(mols)] for i in range(total_repeats)]
                print(f"LENGTH REPEATED MOLS {len(repeated_mols)}")
                repeated_mols = repeated_mols[1:]
                print(f"LENGTH AGAIN {len(repeated_mols)}")
                monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}

            elif polymerization_mode == 'random':
                monomer_counts_int = [int(total_repeats * f) for f in fractions]
                for i in range(total_repeats - sum(monomer_counts_int)):
                    monomer_counts_int[i % len(monomer_counts_int)] += 1
                for count, mol in zip(monomer_counts_int, mols):
                    repeated_mols.extend([mol] * count)
                random.shuffle(repeated_mols)
                monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}

            elif polymerization_mode == 'block':
                monomer_counts_int = [int(total_repeats * f) for f in fractions]
                for i in range(total_repeats - sum(monomer_counts_int)):
                    monomer_counts_int[i % len(monomer_counts_int)] += 1
                blocks_per_type = num_blocks // len(smiles)
                block_sizes = [monomer_counts_int[i] // blocks_per_type for i in range(len(smiles))]
                block_sizes_and_monomers = [(block_sizes[b % len(mols)], mols[b % len(mols)]) for b in range(num_blocks)]
                for size, mol in block_sizes_and_monomers:
                    repeated_mols.extend([mol] * size)
                monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}

            else:
                raise ValueError(f"Unknown polymerization mode: {polymerization_mode}")

            for reactant_mol in repeated_mols:
                results = rxn.RunReactants((new_mol, reactant_mol))
                if results and len(results[0]) == 1:
                    new_mol = results[product_index][0]
                else:
                    break

        # Create SMILES from RDKit molecule
        final_smiles = Chem.MolToSmiles(new_mol)

        # Replace reactive placeholders with terminal methyl groups, for example
        if add_end_Cs:
            final_smiles = final_smiles.replace("[Cu]", "C").replace("[Au]", "C")

        # Canonicalize the final SMILES after all modifications
        final_mol = Chem.MolFromSmiles(final_smiles)
        if final_mol:
            final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
            final_mol_with_h = Chem.AddHs(final_mol)
            total_atoms = final_mol_with_h.GetNumAtoms()
        else:
            final_smiles = ""
            total_atoms = 0

        return final_smiles, total_repeats

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0


def get_mol_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Sum up the atomic masses of all atoms in the molecule
    mass = 0
    for atom in mol.GetAtoms():
        mass += atom.GetMass()

    return mass
    
def get_atom_count(
    smiles,
    fractions=None,  # If it's a list of SMILES, fractions are required
    salt_smiles=[],
    poly_count=1,
    salt_count=1,
):
    num_atoms = 0

    # Check if it's a list of SMILES (for monomers)
    if fractions:
        if len(smiles) != len(fractions):
            raise ValueError("The 'smiles' and 'fractions' lists must have the same length.")

        # Step 1: Calculate the atom count for the polymer (multiple monomers)
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)  # Add implicit hydrogens
            num_atoms += mol.GetNumAtoms() * fractions[i] * poly_count
    else:
        # If it's a single SMILES string, just calculate the atom count directly
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Add implicit hydrogens
        num_atoms += mol.GetNumAtoms() * poly_count

    # Step 2: Calculate the atom count for salts (if any)
    for i in salt_smiles:
        mol = Chem.MolFromSmiles(i)
        mol = Chem.AddHs(mol)
        num_atoms += mol.GetNumAtoms() * salt_count

    return num_atoms


def create_ligpargen(smiles, ligpargen_path, hitpoly_path, mol_filename, output_prefix, platform):
    # Directly use the provided SMILES string to create the molecule
    mol_initial = Chem.MolFromSmiles(smiles)
        
    # Add hydrogens and 3D conformation
    mol_initial = Chem.AddHs(mol_initial, addCoords=True)
    initial_confs = AllChem.EmbedMultipleConfs(
        mol_initial,
        numConfs=1,
        maxAttempts=1000,
        boxSizeMult=5,
        useRandomCoords=False,
    )
    if len(initial_confs) == 0:
        initial_confs = AllChem.EmbedMultipleConfs(
            mol_initial,
            numConfs=1,
            maxAttempts=20000,
            boxSizeMult=5,
            useRandomCoords=True,
        )


    # Save molecule to mol file
    mol_filepath = os.path.join(ligpargen_path, mol_filename)
    with open(mol_filepath, "w+") as mol_file:
        mol_file.write(Chem.MolToMolBlock(mol_initial))
    
    if platform == "local":
        os.chdir(ligpargen_path)
        command = f"LigParGen -m {mol_filename} -o 0 -c 0 -r {output_prefix} -d . -l"
        subprocess.run(command, shell=True)
        os.chdir(hitpoly_path)
    elif platform == "supercloud":
        supercloud_ligpargen(ligpargen_path, mol_filename, output_prefix)
    elif platform == "perlmutter":
        perlmutter_ligpargen(ligpargen_path, mol_filename, output_prefix)
    elif platform == "engaging":
        engaging_ligpargen(ligpargen_path)
    
    return mol_initial, smiles

def supercloud_ligpargen(ligpargen_path, mol_filename, output_prefix):

    ligpargen = os.environ.get("LigParGen")
    print(f"LigParGen path: {ligpargen}")  # Debugging line
    print(f"Molecule filename: {mol_filename}")  # Debugging line
    print(f"Output prefix: {output_prefix}")  # Debugging line
    
    with open(f"{ligpargen_path}/run.sh", "w") as f:
        f.write("#!/bin/bash" + "\n")
        f.write("#SBATCH --job-name=ligpargen" + "\n")
        f.write("#SBATCH --partition=xeon-p8" + "\n")
        f.write("#SBATCH --nodes=1" + "\n")
        f.write("#SBATCH --ntasks-per-node=1" + "\n")
        f.write("#SBATCH --cpus-per-task=1" + "\n")
        f.write("#SBATCH --time=2:00:00" + "\n")
        f.write("\n")
        f.write("# Load modules" + "\n")
        f.write("source /etc/profile" + "\n")
        f.write("source $HOME/.bashrc" + "\n")
        f.write("conda activate ligpargen" + "\n")
        f.write("export PYTHONPATH=$HOME/ligpargen:$PYTHONPATH" + "\n")
        f.write("cwd=$(pwd)" + "\n")
        f.write(f"cd {ligpargen_path}" + "\n")
        f.write(f"{ligpargen} -m {mol_filename} -o 0 -c 0 -r {output_prefix} -d . -l\n")
        f.write("cd $cwd" + "\n")
    command = f"sbatch {ligpargen_path}/run.sh"
    subprocess.run(command, shell=True)
    # Wait for the output file (with output_prefix) to be generated
    t0 = time.time()
    expected_output_file = os.path.join(ligpargen_path, f"{output_prefix}.xml")
    while True:
        if os.path.exists(expected_output_file):
            time.sleep(2)
            print(f"Output file {expected_output_file} found.")
            break
        elif time.time() - t0 > 300:  # Timeout after 5 minutes
            print(f"Timeout: {expected_output_file} not found within the time limit.")
            break
        else:
            time.sleep(10)


def perlmutter_ligpargen(ligpargen_path, mol_filename, output_prefix):
    ligpargen = os.environ.get("LigParGen")
    print(f"LigParGen path: {ligpargen}")  # Debugging line
    print(f"Molecule filename: {mol_filename}")  # Debugging line
    print(f"Output prefix: {output_prefix}")  # Debugging line
    with open(f"{ligpargen_path}/run.sh", "w") as f:
        f.write("#!/bin/bash" + "\n")
        f.write("#SBATCH --job-name=ligpargen" + "\n")
        f.write("#SBATCH --account=m5068" + "\n")
        f.write("#SBATCH -C cpu" + "\n")
        f.write("#SBATCH --qos=debug" + "\n")
        f.write("#SBATCH --nodes=1" + "\n")
        f.write("#SBATCH --ntasks-per-node=1" + "\n")
        f.write("#SBATCH --cpus-per-task=1" + "\n")
        f.write("#SBATCH --time=0:30:00" + "\n")
        f.write("\n")
        f.write("# Load modules" + "\n")
        # f.write("source /etc/profile" + "\n")
        f.write("source $HOME/.bashrc" + "\n")
        f.write("source activate htvs" + "\n")
        f.write("cwd=$(pwd)" + "\n")
        f.write(f"cd {ligpargen_path}" + "\n")
        f.write(f"{ligpargen} -m {mol_filename} -o 0 -c 0 -r {output_prefix} -d . -l\n")
        f.write("cd $cwd" + "\n")
    command = f"sbatch {ligpargen_path}/run.sh"
    subprocess.run(command, shell=True)
    # Wait for the output file (with output_prefix) to be generated
    t0 = time.time()
    expected_output_file = os.path.join(ligpargen_path, f"{output_prefix}.xml")
    while True:
        if os.path.exists(expected_output_file):
            time.sleep(2)
            print(f"Output file {expected_output_file} found.")
            break
        elif time.time() - t0 > 300:  # Timeout after 5 minutes
            print(f"Timeout: {expected_output_file} not found within the time limit.")
            break
        else:
            time.sleep(10)


def engaging_ligpargen(ligpargen_path):
    ligpargen = os.environ.get("LigParGen")
    with open(f"{ligpargen_path}/run.sh", "w") as f:
        f.write("#!/bin/bash" + "\n")
        f.write("#SBATCH --job-name=ligpargen" + "\n")
        f.write("#SBATCH --partition=sched_mit_rafagb" + "\n")
        f.write("#SBATCH --nodes=1" + "\n")
        f.write("#SBATCH --ntasks=16" + "\n")
        f.write("#SBATCH --time=04:00:00" + "\n")
        f.write("source $HOME/.bashrc" + "\n")
        f.write("source activate htvs" + "\n")
        f.write("cwd=$(pwd)" + "\n")
        f.write(f"cd {ligpargen_path}" + "\n")
        f.write(f"{ligpargen} -m poly.mol -o 0 -c 0 -r PLY -d . -l" + "\n")
        f.write("cd $cwd" + "\n")
    command = f"sbatch {ligpargen_path}/run.sh"
    print(f"ligpargenpath: {ligpargen_path}")
    subprocess.run(command, shell=True)
    t0 = time.time()
    while True:
        if os.path.exists("PLY.xml"):
            time.sleep(2)
            break
        elif time.time() - t0 > 300:
            break
        else:
            time.sleep(10)


def create_conformer_pdb(
    path_to_save,
    smiles,
    name=None,
    enforce_generation=False,
):
    #Generate the current date for the filename
    day_time = time.strftime("%y%m%d")

    #Generate a name for the PDB file if not provided
    if not name:
        name = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))

    file_name = f"{name}.pdb"

    #Initialize minimize variable
    minimize = True

    #Try to generate a conformer using RDKit's ETKDG method with useRandomCoords
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    try:
        #Create ETKDG embedding parameters
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True  # Allow random coordinates for better chances with large molecules
        
        #Attempt to embed the molecule with the custom parameters
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            raise ValueError("RDKit ETKDG embedding failed")
        AllChem.UFFOptimizeMolecule(mol)
        print("Successfully generated 3D conformer using RDKit ETKDG with random coordinates")
    except Exception as e:
        print(f"RDKit ETKDG failed: {e}")
        mol = None

    #If RDKit fails, fall back to Pybel
    if mol is None:
        print("Attempting to generate 3D conformer using Pybel...")
        minimize = True
        conf = False
        opt_steps = 1
        while not conf:
            pybel_mol = pybel.readstring("smi", smiles)
            pybel_mol.make3D(steps=opt_steps)
            pybel_mol.write("pdb", f"{path_to_save}/{file_name}", overwrite=True)
            try:
                print(f"Attempting to generate Pybel conf with {opt_steps} steps")
                temp_mol = Chem.MolFromPDBFile(f"{path_to_save}/{file_name}", removeHs=False)
                if temp_mol is None:
                    raise ValueError("Failed to create a valid molecule from PDB file.")

                # Check minimal distance between atoms
                X = temp_mol.GetConformer(0).GetPositions()
                a = euclidean_distances(X, X)
                np.fill_diagonal(a, np.inf)
                if a.min() > 0.5:
                    conf = True
                    minimize = False

                    Chem.rdDepictor.Compute2DCoords(mol)
                    conformation = mol.GetConformer()
                    for i in range(mol.GetNumAtoms()):
                        assert pybel_mol.atoms[i].atomicnum == mol.GetAtoms()[i].GetAtomicNum()
                        x, y, z = (
                            pybel_mol.atoms[i].coords[0],
                            pybel_mol.atoms[i].coords[1],
                            pybel_mol.atoms[i].coords[2],
                        )
                        conformation.SetAtomPosition(i, Point3D(x, y, z))

                    writer = Chem.PDBWriter(f"{path_to_save}/{file_name}")
                    writer.write(mol, confId=mol.GetConformer().GetId())
                    writer.close()
                    print(f"Successfully generated Pybel 3D conformer with {opt_steps} steps")
            except Exception as e:
                print(f"Failed with {opt_steps} steps: {e}")
                opt_steps *= 10
                if opt_steps > 10000:
                    print("Catastrophic failure, skipping this molecule.")
                    return None, None, None

    #Save the conformer to a PDB file if successful
    if mol is not None:
        writer = Chem.PDBWriter(f"{path_to_save}/{file_name}")
        writer.write(mol)
        writer.close()
        return file_name, mol, minimize

    return None, None, None  #If all attempts fail, return None

def create_packmol_input_file(
    poly_paths: list,
    salt_paths: list,
    output_path: str,
    output_name: str,
    salt_smiles: list = None,
    polymer_count: Union[int, float, list] = 25,
    salt_concentrations: list = [100, 100],
    tolerance: float = 5.0,
    box_multiplier: int = 3,
    random_seed: int = -1,
    salt: bool = True,
):
    """
    Creates a Packmol input file for polymer/salt systems.

    Args:
        poly_paths: list of PDB files for polymers
        salt_paths: list of PDB files for salts
        output_path: path to save the Packmol .inp file
        output_name: name of output structure file
        salt_smiles: list of salt SMILES strings used to infer charge
        polymer_count: int or list of ints — number of chains per polymer
        salt_concentrations: list of ints — count of each salt
        tolerance: Packmol tolerance
        box_multiplier: multiplier on volume
        random_seed: unused but preserved for future use
        salt: whether to include salts
    """

    # === Handle scalar or list polymer counts ===
    if isinstance(polymer_count, (int, float)):
        polymer_count = [int(polymer_count)] * len(poly_paths)

    if len(polymer_count) != len(poly_paths):
        raise ValueError("Length of polymer_count must match number of polymer files")

    print(f"POLYMER PDBs: {poly_paths}")
    print(f"POLYMER COUNTS: {polymer_count}")

    # === Estimate volume ===
    total_vol = 0.0
    for idx, path in enumerate(poly_paths):
        mol = Chem.MolFromPDBFile(path, removeHs=False)
        conf = mol.GetConformer()
        radius = distance_matrix(conf.GetPositions(), conf.GetPositions()).max() / 2
        volume = 4 / 3 * np.pi * radius**3
        total_vol += volume * polymer_count[idx]
        print(f"🧪 Polymer {idx}: radius = {radius:.2f} Å, volume = {volume:.2f} Å³ x {polymer_count[idx]}")

    total_vol *= box_multiplier
    box_radi = (3 / (4 * np.pi) * total_vol) ** (1 / 3)
    print(f"Box radius: {box_radi:.2f} Å (after scaling)")

    # === Determine salt placement ===
    cation_paths, anion_paths, cation_conc, anion_conc = [], [], [], []

    if salt:
        if len(salt_paths) > 2 and salt_smiles:
            for ind, (path, smi) in enumerate(zip(salt_paths, salt_smiles)):
                net_charge = smi.count("+") - smi.count("-")
                if net_charge == 1 and salt_concentrations[ind] > 0:
                    cation_paths.append(path)
                    cation_conc.append(salt_concentrations[ind])
                elif net_charge == -1 and salt_concentrations[ind] > 0:
                    anion_paths.append(path)
                    anion_conc.append(salt_concentrations[ind])
        elif len(salt_paths) == 2:
            if salt_concentrations[0] > 0:
                cation_paths = [salt_paths[0]]
                cation_conc = [salt_concentrations[0]]
            if salt_concentrations[1] > 0:
                anion_paths = [salt_paths[1]]
                anion_conc = [salt_concentrations[1]]
        else:
            print("Warning: Salt input incomplete or ambiguous — no salts will be packed.")
    else:
        print("Salt packing disabled.")

    print(f"CATIONS: {cation_paths} x {cation_conc}")
    print(f"ANIONS: {anion_paths} x {anion_conc}")

    # === Write packmol input file ===
    with open(output_path, "w+") as f:
        f.write(f"tolerance {tolerance}\n")
        f.write(f"filetype pdb\n")
        f.write(f"output {output_name}\n\n")

        for idx, path in enumerate(poly_paths):
            f.write(f"structure {path}\n")
            f.write(f"  number {polymer_count[idx]}\n")
            f.write(f"  inside box 0 0 0 {box_radi:.2f} {box_radi:.2f} {box_radi:.2f}\n")
            f.write("end structure\n\n")

        for idx, path in enumerate(anion_paths):
            f.write(f"structure {path}\n")
            f.write(f"  number {anion_conc[idx]}\n")
            f.write(f"  inside box 0 0 0 {box_radi:.2f} {box_radi:.2f} {box_radi:.2f}\n")
            f.write("end structure\n\n")

        for idx, path in enumerate(cation_paths):
            f.write(f"structure {path}\n")
            f.write(f"  number {cation_conc[idx]}\n")
            f.write(f"  inside box 0 0 0 {box_radi:.2f} {box_radi:.2f} {box_radi:.2f}\n")
            f.write("end structure\n\n")

    print("Packmol input file created.")

def run_packmol(save_path, packmol_path):
    command = f"{packmol_path} < {save_path}/packmol.inp"
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print(f"Error executing packmol")
        print(f"STDOUT: {process.stdout}")
        print(f"STDERR: {process.stderr}")
    else:
        print("Run packmol successfully.")
        print(f"STDOUT: {process.stdout}")


def generate_parameter_dict(save_path, output_prefix, atom_names, atoms, bonds_typed):
    forceFieldFiles = [f"{save_path}/{output_prefix}.xml"]
    forcefield = ForceField(*forceFieldFiles)

    # Finding the correct residue template name
    template_name = [name for name in forcefield._templates.keys() if name.startswith("poly")][0]

    # Checking if the atoms have the same order and if the bonds are the same!
    atoms_ff = [i.element._symbol for i in forcefield._templates[template_name].atoms]
    assert atoms_ff == atoms, f"Mismatch between forcefield atoms {atoms_ff} and input atoms {atoms}"

    bond_list = [list(i) for i in forcefield._templates[template_name].bonds]
    same_bonds = 0
    for i in bond_list:
        for j in bonds_typed:
            if set(i) == set(j):
                same_bonds += 1
    assert same_bonds == len(bond_list), f"Mismatch in bonds: {same_bonds} vs {len(bond_list)}"
    print("LigParGen and RDkit atoms and bonds are the same")

    param_dict = {
        "atoms": {},
        "bonds": {},
        "angles": {},
        "dihedrals": {},
        "impropers": {},
    }

    type2atomdict = {}
    atom2typedict = {}
    atomff2atom1hotdict = {}
    atom1hotdict2atomff = {}

    # Mapping atom names to their types and ensuring charges are stored correctly
    for ind, atom in enumerate(forcefield._templates[template_name].atoms):
        atom_name = atom.name
        atom_type = atom.type
        other_type = atom_names[ind]
        element = atom.element._symbol
        charge = forcefield._forces[3].params.paramsForType[atom_type]["charge"]
        sigma = forcefield._forces[3].params.paramsForType[atom_type]["sigma"]
        epsilon = forcefield._forces[3].params.paramsForType[atom_type]["epsilon"]

        if other_type not in param_dict["atoms"]:
            param_dict["atoms"][other_type] = {
                "type": atom_type,
                "element": element,
                "charges": [charge],
                "sigma": sigma,
                "epsilon": epsilon,
            }
        else:
            param_dict["atoms"][other_type]["charges"].append(charge)

        type2atomdict[atom_type] = atom_name
        atom2typedict[atom_name] = atom_type
        atomff2atom1hotdict[atom_name] = other_type
        atom1hotdict2atomff[other_type] = atom_name

    # Bonds
    for ind, (t1, t2) in enumerate(zip(forcefield._forces[0].types1, forcefield._forces[0].types2)):
        a1 = type2atomdict[list(t1)[0]]
        a2 = type2atomdict[list(t2)[0]]
        b_name = f"bond_{ind}"
        param_dict["bonds"][b_name] = {
            "ff_idx": [a1, a2],
            "other_idx": [atomff2atom1hotdict[a1], atomff2atom1hotdict[a2]],
            "length": forcefield._forces[0].length[ind],
            "k": forcefield._forces[0].k[ind],
        }

    # Angles
    for ind, (t1, t2, t3) in enumerate(zip(forcefield._forces[1].types1, forcefield._forces[1].types2, forcefield._forces[1].types3)):
        a1 = type2atomdict[list(t1)[0]]
        a2 = type2atomdict[list(t2)[0]]
        a3 = type2atomdict[list(t3)[0]]
        b_name = f"angle_{ind}"
        param_dict["angles"][b_name] = {
            "ff_idx": [a1, a2, a3],
            "other_idx": [atomff2atom1hotdict[a1], atomff2atom1hotdict[a2], atomff2atom1hotdict[a3]],
            "angle": forcefield._forces[1].angle[ind],
            "k": forcefield._forces[1].k[ind],
        }

    # Dihedrals
    for ind, dih in enumerate(forcefield._forces[2].proper):
        a1 = type2atomdict[list(dih.types1)[0]]
        a2 = type2atomdict[list(dih.types2)[0]]
        a3 = type2atomdict[list(dih.types3)[0]]
        a4 = type2atomdict[list(dih.types4)[0]]
        b_name = f"dihedral_{ind}"
        param_dict["dihedrals"][b_name] = {
            "ff_idx": [a1, a2, a3, a4],
            "other_idx": [atomff2atom1hotdict[a1], atomff2atom1hotdict[a2], atomff2atom1hotdict[a3], atomff2atom1hotdict[a4]],
            "periodicity": dih.periodicity,
            "phase": dih.phase,
            "k": dih.k,
        }

    # Impropers
    for ind, dih in enumerate(forcefield._forces[2].improper):
        a1 = type2atomdict[list(dih.types1)[0]]
        a2 = type2atomdict[list(dih.types2)[0]]
        a3 = type2atomdict[list(dih.types3)[0]]
        a4 = type2atomdict[list(dih.types4)[0]]
        b_name = f"improper_{ind}"
        param_dict["impropers"][b_name] = {
            "ff_idx": [a1, a2, a3, a4],
            "other_idx": [atomff2atom1hotdict[a1], atomff2atom1hotdict[a2], atomff2atom1hotdict[a3], atomff2atom1hotdict[a4]],
            "periodicity": dih.periodicity,
            "phase": dih.phase,
            "k": dih.k,
        }

    return param_dict


def create_combined_param_dict(smiles_list, ligpargen_path, hitpoly_path, platform):
    import os
    import time

    # Initialize combined parameter dictionary
    combined_param_dict = {
        "atoms": {},
        "bonds": {},
        "angles": {},
        "dihedrals": {},
        "impropers": {},
    }

    atoms_short = []
    atom_names_short = []

    # Define a dictionary to map atomic numbers to atomic symbols
    number_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        # Add more elements as needed
    }

    # Create the combined dataset with all oligomers
    train_args = hitpolyArgs()
    train_args.discrete_flag = True  # Set the discrete_flag to True
    train_molecule_data = [
        TopologyBuilder(
            smiles=[smiles], train_args=train_args, load_geoms=False, lj_from_file=False
        )
        for smiles in smiles_list
    ]
    train_dataset = TopologyDataset(data=train_molecule_data, train_args=train_args)

    # Process each oligomer
    for smiles_index, smiles in enumerate(smiles_list):
        output_prefix = f"poly_{smiles_index+1}"
        mol_filename = os.path.join(ligpargen_path, f"{output_prefix}.mol")

        # Create Initial Molecule with LigParGen
        mol_initial, smiles_initial = create_ligpargen(
            smiles=smiles,
            ligpargen_path=ligpargen_path,
            hitpoly_path=hitpoly_path,
            mol_filename=mol_filename,
            output_prefix=output_prefix,
            platform=platform
        )
        print(f"Initial molecule created with SMILES: {smiles_initial}")

        # Extract atom types and other information from the combined dataset
        molecule = train_molecule_data[smiles_index]
        atom_names = molecule.atom_type.tolist()
        atoms = molecule.atomic_nums
        bonds_typed = molecule.bonds.tolist()
        angles = molecule.angles.tolist()

        # Convert angles to atom types
        angles_atom_types = [
            [
                molecule.atom_type[angle[0]].item(),
                molecule.atom_type[angle[1]].item(),
                molecule.atom_type[angle[2]].item()
            ]
            for angle in molecule.angles
        ]    

        # Print the angles with atom types for debugging
        #print(f"Angles with atom types for initial molecule: {angles_atom_types}")

        # Convert atomic numbers to atomic symbols
        atoms = [number_to_symbol[number] for number in atoms]

        # Generate Parameter Dictionary
        param_file_path = os.path.join(ligpargen_path, f"{output_prefix}.xml")
        if os.path.exists(param_file_path):
            param_dict = generate_parameter_dict(ligpargen_path, output_prefix, atom_names, atoms, bonds_typed)
            print(f"Parameter dictionary generated for oligomer {smiles}")

            # Combine parameters into the combined_param_dict
            for key in combined_param_dict.keys():
                for sub_key, sub_value in param_dict[key].items():
                    if key == "atoms":
                        if sub_key in combined_param_dict[key]:
                            if "charges" not in combined_param_dict[key][sub_key]:
                                combined_param_dict[key][sub_key]["charges"] = []
                            combined_param_dict[key][sub_key]["charges"].extend(sub_value["charges"])
                            combined_param_dict[key][sub_key]["sigma"] = sub_value["sigma"]
                            combined_param_dict[key][sub_key]["epsilon"] = sub_value["epsilon"]
                        else:
                            combined_param_dict[key][sub_key] = sub_value
                    else:
                        unique_key = f"{key[:-1]}_{'_'.join(map(str, sub_value['other_idx']))}"
                        combined_param_dict[key][unique_key] = sub_value

        else:
            print(f"Parameter file for oligomer {smiles} not found. Skipping.")
            continue

        # Append to atoms and atom names
        atoms_short.extend(atoms)
        atom_names_short.extend(atom_names)

    # Ensure unique atom names
    atom_names_short = list(set(atom_names_short))
    atoms_short = list(set(atoms_short))

    return combined_param_dict, atoms_short, atom_names_short

def atom_name_reindexing(mol_dict):
    import string as STRING
    import itertools

    digits = STRING.digits
    letters = STRING.ascii_uppercase

    # Expanded index list generation
    index_list = []

    # 1. Single digit (1-9)
    for i in range(1, 10):
        index_list.append(str(i))

    # 2. Two-digit combinations (00-99)
    for i in itertools.product(digits, repeat=2):
        index_list.append("".join(i))

    # 3. Letter-digit combinations (A0-Z9)
    for i in itertools.product(letters, digits):
        index_list.append("".join(i))

    # 4. Two-letter combinations (AA-ZZ)
    for i in itertools.product(letters, repeat=2):
        index_list.append("".join(i))

    # 5. Three-letter combinations (AAA-ZZZ)
    for i in itertools.product(letters, repeat=3):
        index_list.append("".join(i))

    # 6. Letter-digit-letter combinations (A0A-Z9Z)
    for i in itertools.product(letters, digits, letters):
        index_list.append("".join(i))

    # 7. Three-digit combinations (000-999)
    for i in itertools.product(digits, repeat=3):
        index_list.append("".join(i))

    # ** New: Add dynamic expansion beyond current naming schemes **
    def expand_index_list():
        """Generator to expand naming options dynamically if the list is exhausted."""
        for length in range(4, 10):  # Start with 4-character combinations
            for combo in itertools.product(letters + digits, repeat=length):
                yield "".join(combo)

    # Debug: Print the generated index list and its length
    print(f"Total pre-generated indices: {len(index_list)}")

    atom_types_list = []  # list of lists where each sublist has 4 elements
    # name, class, element, mass
    atom_dict = {}

    # Debug: Print the initial mol_dict keys and a brief summary
    print(f"Initial mol_dict keys: {list(mol_dict.keys())}")
    for key, mol in mol_dict.items():
        print("IN ATOM REINDEXING")
        print(f"Processing molecule key: {key}")

        for ind, atom in enumerate(mol["mol_pdb"]._residues[0]._atoms):
            # Debug: Print initial atom details
            # print(f"Initial Atom: {atom}, Name: {atom.name}, Element: {atom.element.symbol}")

            atom_name = atom.name[len(atom.element.symbol):]
            if atom.element.symbol not in atom_dict:
                atom_dict[atom.element.symbol] = atom_name
            else:
                prev_atom_ind = index_list.index(atom_dict[atom.element.symbol])

                # Check if the next index is within bounds
                try:
                    if index_list.index(atom.name[len(atom.element.symbol):]) <= prev_atom_ind:
                        if prev_atom_ind + 1 < len(index_list):
                            atom_name = index_list[prev_atom_ind + 1]
                        else:
                            # Dynamically expand the index list if exhausted
                            new_suffix = next(expand_index_list())
                            index_list.append(new_suffix)
                            atom_name = new_suffix

                        full_name = atom.element.symbol + atom_name
                        for i in mol["mol_pdb"]._residues:
                            i._atoms[ind].name = full_name
                        atom_dict[atom.element.symbol] = atom_name
                    else:
                        atom_dict[atom.element.symbol] = atom_name
                except ValueError:
                    # Handle case where atom.name doesn't match the current index_list
                    atom_name = index_list[0]  # Reset to first valid name
                    full_name = atom.element.symbol + atom_name
                    atom_dict[atom.element.symbol] = atom_name

            # Debug: Print updated atom details
            # print(f"Updated Atom Name: {atom.element.symbol + atom_name}")

            # Verification Step: Print to verify the atom name update
            # print(f"Verifying update: Atom {ind} is now named {mol['mol_pdb']._residues[0]._atoms[ind].name}")

            atom_types_list.append(
                [atom.name, atom.name, atom.element.symbol, atom.element.mass._value]
            )

    # Debug: Print final atom_dict and atom_types_list
    print(f"Final atom_dict: {atom_dict}")
    print(f"Final atom_types_list: {atom_types_list}")

    return atom_dict, atom_types_list



def param_scaling_openmm(topology, scaling=False):
    # scaling mol dict params from lammps to openmm
    topology.bond_params_openmm = topology.bond_params.clone()
    topology.angle_params_openmm = topology.angle_params.clone()
    topology.dihedral_params_openmm = topology.dihedral_params.clone()
    topology.improper_params_openmm = topology.improper_params.clone()
    topology.pair_params_openmm = topology.pair_params.clone()

    if scaling:
        if topology.bond_params.numel():
            topology.bond_params_openmm[:, 0] = (
                topology.bond_params_openmm[:, 0] * 2 * 4.184 * 100
            )
            topology.bond_params_openmm[:, 1] = topology.bond_params_openmm[:, 1] / 10
        if topology.angle_params.numel():
            topology.angle_params_openmm[:, 0] = (
                topology.angle_params_openmm[:, 0] * 2 * 4.184
            )
        if topology.dihedral_params.numel():
            topology.dihedral_params_openmm = (
                topology.dihedral_params_openmm / 2 * 4.184
            )
        if topology.improper_params.numel():
            topology.improper_params_openmm = topology.improper_params_openmm * 4.184
        topology.pair_params_openmm[:, 1] = (
            topology.pair_params_openmm[:, 1] * 4.184
        )  # Epsilon
        topology.pair_params_openmm[:, 2] = (
            topology.pair_params_openmm[:, 2] / 10
        )  # Sigma

    return topology

def creating_ff_and_resid_files(mol_dict, atom_types_list, index=None):
    ff_file = []
    ff_file.append("<ForceField>")
    ff_file.append("<AtomTypes>")
    
    # Debugging atom types
    print("Atom Types List:")
    for a_name, a_class, a_elem, a_mass in atom_types_list:
        string = f'<Type name="{a_name}" class="{a_class}" element="{a_elem}" mass="{a_mass}" />'
        ff_file.append(string)
    
    ff_file.append("</AtomTypes>")
    ff_file.append("<Residues>")

    if index is None:
        name_iterables = []
        ani_ind = 1
        cat_ind = 1
        ply_ind = 1
        for key in mol_dict.keys():
            minuses = key.count("-")
            pluses = key.count("+")
            if pluses - minuses == -1:
                name_iterables.append(f"AN{ani_ind}")
                ani_ind += 1
            elif pluses - minuses == 1:
                name_iterables.append(f"CA{cat_ind}")
                cat_ind += 1
            else:
                name_iterables.append(f"P{ply_ind}")
                ply_ind += 1
    else:
        name_iterables = []
        ani_ind = 1
        cat_ind = 1
        for key in mol_dict.keys():
            minuses = key.count("-")
            pluses = key.count("+")
            if pluses - minuses == -1:
                name_iterables.append(f"AN{ani_ind}")
                ani_ind += 1
            elif pluses - minuses == 1:
                name_iterables.append(f"CA{cat_ind}")
                cat_ind += 1
            else:
                name_iterables.append(f"P{index+1}")

    
    # Debugging name iterables
    print(f"Name Iterables: {name_iterables}")
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        total_atoms = len(mol['mol_pdb']._residues[0]._atoms) if hasattr(mol["mol_pdb"], "_residues") and len(mol["mol_pdb"]._residues) > 0 else 0
        print(f"Residue Name: {n}, Corresponding Molecule: {key}, Total number of atoms in mol_pdb for this molecule: {total_atoms}")

    for key, mol in mol_dict.items():
        if hasattr(mol['mol_pdb'], '_residues'):
            print(f"Molecule {key}: Total residues = {len(mol['mol_pdb']._residues)}")
            for residue in mol['mol_pdb']._residues:
                print(f"Residue {residue.id}: Total atoms = {len(residue._atoms)}")
        else:
            print(f"Error: No residues found in mol_pdb for molecule {key}")

    
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        ff_file.append(f'<Residue name="{n}">')  # mol["mol_pdb"].id
        
        # Debugging atom and charge assignment
        print(f"Processing molecule {key} with residue name {n}")
        
        # Check if residues exist in mol_pdb
        if not hasattr(mol["mol_pdb"], "_residues") or len(mol["mol_pdb"]._residues) == 0:
            print(f"Error: No residues found in PDB for molecule {key}")
            continue
        
        total_atoms = len(mol['mol_pdb']._residues[0]._atoms)
        print(f"Total atoms in molecule: {total_atoms}")
        
        for ind, (atom_pdb, charge) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params[:, 0].detach().numpy(),
            )
        ):
            string = f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />' 
            ff_file.append(string)
            #print(f"Added atom {atom_pdb.name} with charge {charge}")
        
        # Debugging bonds
        for bond in mol["mol_hitpoly"].bonds.numpy():
            total_atoms = len(mol["mol_pdb"]._residues[0]._atoms)

            # Check if bond indices are within range
            if bond[0] >= total_atoms or bond[1] >= total_atoms:
                print(f"Error: Bond indices {bond[0]} or {bond[1]} are out of range.")
            else:
                string = f'<Bond from="{bond[0]}" to="{bond[1]}" />' ##figure out how to make this the same as the other dorcefield file
                ff_file.append(string)
        
        ff_file.append("</Residue>")
    
    ff_file.append("</Residues>")
    
    # Harmonic Bond Force
    ff_file.append("<HarmonicBondForce>")
    for key, mol in mol_dict.items():
        print(f"Processing Harmonic Bonds for molecule {key}")
        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            if bond[0] >= len(mol["mol_pdb"]._residues[0]._atoms) or bond[1] >= len(mol["mol_pdb"]._residues[0]._atoms):
                print(f"Error: Bond indices out of range for harmonic bond in molecule {key}")
                continue
            
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f'<Bond k="{param[0]}" length="{param[1]}" class1="{atom1}" class2="{atom2}" />'
            ff_file.append(string)
            #print(f"Added Harmonic Bond between {atom1} and {atom2} with k={param[0]} and length={param[1]}")
    ff_file.append("</HarmonicBondForce>")
    
    # Harmonic Angle Force
    ff_file.append("<HarmonicAngleForce>")
    for key, mol in mol_dict.items():
        print(f"Processing Harmonic Angles for molecule {key}")
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            if (
                angle[0] >= len(mol["mol_pdb"]._residues[0]._atoms) or
                angle[1] >= len(mol["mol_pdb"]._residues[0]._atoms) or
                angle[2] >= len(mol["mol_pdb"]._residues[0]._atoms)
            ):
                print(f"Error: Angle indices out of range for harmonic angle in molecule {key}")
                continue
            
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string1 = f'<Angle k="{param[0]}" angle="{param[1]}" '
            string2 = f'class1="{atom1}" class2="{atom2}" class3="{atom3}" />'
            ff_file.append(string1 + string2)
            #print(f"Added Angle between {atom1}, {atom2}, and {atom3}")
    ff_file.append("</HarmonicAngleForce>")
    
    # Periodic Torsion Force
    ff_file.append("<PeriodicTorsionForce>")
    for key, mol in mol_dict.items():
        print(f"Processing Torsion for molecule {key}")
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            if (
                dihedral[0] >= len(mol["mol_pdb"]._residues[0]._atoms) or
                dihedral[1] >= len(mol["mol_pdb"]._residues[0]._atoms) or
                dihedral[2] >= len(mol["mol_pdb"]._residues[0]._atoms) or
                dihedral[3] >= len(mol["mol_pdb"]._residues[0]._atoms)
            ):
                #print(f"Error: Torsion indices out of range for periodic torsion in molecule {key}")
                continue
            
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            string1 = f'<Proper k1="{param[0]}" k2="{param[1]}" k3="{param[2]}" k4="{param[3]}" '
            string2 = (
                f'periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" '
            )
            string3 = 'phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793" '
            string4 = f'class1="{atom1}" class2="{atom2}" class3="{atom3}" class4="{atom4}" />'
            ff_file.append(string1 + string2 + string3 + string4)
            #print(f"Added Torsion between {atom1}, {atom2}, {atom3}, and {atom4}")
    ff_file.append("</PeriodicTorsionForce>")
    
    ff_file.append('<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">')
    # ff_file.append('<UseAttributeFromResidue name="charge" />')
    for key, mol in mol_dict.items():
        print(f"Processing Nonbonded Forces for molecule {key}")
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            string = f'<Atom charge="{param[0]}" epsilon="{param[1]}" sigma="{param[2]}" type="{atom_pdb.name}" />'
            ff_file.append(string)
            #print(f"Added Nonbonded Force for atom {atom_pdb.name} with charge {param[0]}, epsilon {param[1]}, sigma {param[2]}")
    ff_file.append("</NonbondedForce>")
    ff_file.append("</ForceField>")
    
    ff_file_resid = []
    ff_file_resid.append("<Residues>")
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        ff_file_resid.append(f'<Residue name="{n}">') 
        print(f"Processing Residue for {n}")
        for ind, (atom_pdb, charge) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params[:, 0].detach().numpy(),
            )
        ):
            string = f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />'
            ff_file_resid.append(string)
            #print(f"Added Residue Atom {atom_pdb.name}")
        
        for bond in mol["mol_hitpoly"].bonds.numpy():
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f'<Bond from="{atom1}" to="{atom2}" />'
            ff_file_resid.append(string)
        
        ff_file_resid.append("</Residue>")
    ff_file_resid.append("</Residues>")

    return ff_file, ff_file_resid, name_iterables



def creating_ff_and_resid_files_gromacs(
    mol_dict,
    atom_types_list,
    save_path,
    pdb_file,
    polymer_count,
    salt_count,
):
    name_iterables = []
    ani_ind = 1
    cat_ind = 1
    ply_ind = 1
    for key in mol_dict.keys():
        minuses = key.count("-")
        pluses = key.count("+")
        if pluses - minuses == -1:
            name_iterables.append(f"AN{ani_ind}")
            ani_ind += 1
        elif pluses - minuses == 1:
            name_iterables.append(f"CA{cat_ind}")
            cat_ind += 1
        else:
            name_iterables.append(f"P{ply_ind}")
            ply_ind += 1

    atom_ind = 0
    atom_ind_name_dict = {}
    
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            atom_ind_name_dict[a_name] = atom_ind + 1
            atom_ind += 1

            
    ff_files_separate = []
    atom_ind = 0
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        print(f"Processing bonds for molecule: {n} ({key})")
        print(f"Name: {n}")
        
        # Access and print bond parameters before entering the bond loop
        bond_params = mol["mol_hitpoly"].bond_params_openmm.detach().numpy()
        bonds = mol["mol_hitpoly"].bonds.detach().numpy()
        print(f"Bonds for molecule {n} ({key}): {bonds}")
        print(f"Bond Params: {bond_params}")
        
        ff_file = []
        ff_file.append(";")
        ff_file.append("; GENERATED BY hitpoly")
        ff_file.append("; Jurgis Ruza @ MIT ")
        ff_file.append(";")
        ff_file.append("[ moleculetype ]")
        ff_file.append("; Name               nrexcl")
        ff_file.append(f"{n:>3}{3:>20}")
        ff_file.append("[ atoms ]")

        # I've no idea what this cgnr is and what does it mean
        ff_file.append(
            ";   nr       type  resnr residue  atom   cgnr     charge       mass"
        )
        cgnr = 1
        temp_atom_dict = {}
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            if atom_ind + 1 == 32 * cgnr:
                cgnr += 1
            string = f"{ind+1:>6}{a_name:>11}{resid_ind+1:>6}{n:>7}"
            string1 = f"{a_name:>6}{str(cgnr):>7}{str(round(param[0],6)):>11}{str(round(a_mass,3)):>11}"
            ff_file.append(string + string1)
            temp_atom_dict[a_name] = ind + 1
            atom_ind += 1
            
            # Print details of the atom in the PDB file
            print(f"Initial Atom: {atom_pdb}, Name: {atom_pdb.name}, Element: {atom_pdb.element.symbol}")
            print(f"Updated Atom Name: {a_name}")

        ff_file.append("[ bonds ]")
        ff_file.append(";  ai    aj  funct            c0            c1")
        
        # Debugging print
        print(f"temp_atom_dict for molecule {n}: {temp_atom_dict}")
        #print(f"initial to updated atom dict {initial_to_updated_atom_name}")

        print(f"Number of atoms in mol_pdb for {n}: {len(mol['mol_pdb']._residues[0]._atoms)}")
        print(f"Bond information for molecule {n}: {mol['mol_hitpoly'].bonds.detach().numpy()}")


        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            print(f"Molecule: {n}, Key: {key}")
            print(f"Bond: {bond}, ATOM 1: {atom1}, ATOM 2: {atom2}")
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>6}{1:>6}"
            string1 = f"{str(round(param[1],4)):>12} {str(round(param[0],3)):>11}"

            ff_file.append(string + string1)

        ff_file.append("[ angles ]")
        ff_file.append(";  ai    aj    ak funct            c0            c1")
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>6}{temp_atom_dict[atom3]:>6}"
            string1 = f"{1:>6}{str(round(param[1]*57.29578,3)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)

        ff_file.append("[ dihedrals ]")
        ff_file.append("; IMPROPER DIHEDRAL ANGLES ")
        ff_file.append(
            ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
        )
        ff_file.append("")
        ff_file.append("[ dihedrals ]")
        ff_file.append("; PROPER DIHEDRAL ANGLES")
        ff_file.append(
            ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
        )
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            # The *2 multiplication comes from the openmm-gromacs dihedral difference
            C0 = round(
                param[1] + 1 / 2 * (param[0] + param[2]) * 2, 3
            )  # C0 = F2 + 1/2 (F1+F3)
            C1 = round(1 / 2 * (-param[0] + 3 * param[2]) * 2, 3)  # C1 = 1/2 (-F1+3F3)
            C2 = round(-param[1] + 4 * param[3] * 2, 3)  # C2 = -F2 + 4F4
            C3 = round(-2 * param[2] * 2, 3)  # C3 = -2F3
            C4 = round(-4 * param[3] * 2, 3)  # C4 = -4F4
            C5 = round(0, 3)  # C5 = 0
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>5}{temp_atom_dict[atom3]:>5}"
            string1 = f"{temp_atom_dict[atom4]:>5}{3:>8}{C0:>12}{C1:>8}{C2:>8}{C3:>8}{C4:>8}{C5:>8}"
            ff_file.append(string + string1)
        ff_file.append("")
        ff_files_separate.append(ff_file)

    ff_file = []
    ff_file.append(";")
    ff_file.append("; GENERATED BY hitpoly")
    ff_file.append("; Jurgis Ruza @ MIT ")
    ff_file.append(";")
    ff_file.append("[ atomtypes ]")
    ff_file.append("; name  at.num  mass  charge particletype   sigma    epsilon")
    atom_ind = 0
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            string = f"{a_name:>10}{ELEMENT_TO_NUM[a_elem]:>6}{round(a_mass,4):>11}{str(round(param[0],6)):>10}"
            string1 = (
                f'{"A":>5}{str(round(param[2],6)):>15}{str(round(param[1],6)):>14}'
            )
            ff_file.append(string + string1)
            atom_ind += 1

    ff_file.append("[ bondtypes ]")
    ff_file.append(";  i    j  funct            b0            kb")
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f"{atom1:>5}{atom2:>6}{1:>6}"
            string1 = f"{str(round(param[1],4)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)
    ff_file.append("[ angletypes ]")
    ff_file.append(";  i    j    k funct            theta            k0")
    for key, mol in mol_dict.items():
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string = f"{atom1:>5}{atom2:>6}{atom3:>6}"
            string1 = f"{1:>6}{str(round(param[1]*57.29578,3)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)

    ff_file.append("[ dihedraltypes ]")
    ff_file.append("; IMPROPER DIHEDRAL ANGLES ")
    ff_file.append(
        ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
    )
    ff_file.append("")
    ff_file.append("[ dihedraltypes ]")
    ff_file.append("; PROPER DIHEDRAL ANGLES")
    ff_file.append(
        ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
    )
    for key, mol in mol_dict.items():
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            # The *2 multiplication comes from the openmm-gromacs dihedral difference
            C0 = round(
                param[1] + 1 / 2 * (param[0] + param[2]) * 2, 3
            )  # C0 = F2 + 1/2 (F1+F3)
            C1 = round(1 / 2 * (-param[0] + 3 * param[2]) * 2, 3)  # C1 = 1/2 (-F1+3F3)
            C2 = round(-param[1] + 4 * param[3] * 2, 3)  # C2 = -F2 + 4F4
            C3 = round(-2 * param[2] * 2, 3)  # C3 = -2F3
            C4 = round(-4 * param[3] * 2, 3)  # C4 = -4F4
            C5 = round(0, 3)  # C5 = 0
            string = f"{atom1:>5}{atom2:>5}{atom3:>5}{atom4:>5}"
            string1 = f"{3:>8}{C0:>12}{C1:>8}{C2:>8}{C3:>8}{C4:>8}{C5:>8}"
            ff_file.append(string + string1)
    ff_file.append("")

    box_file = []
    box_file.append("Simulation box")
    box_file.append(f"{len(pdb_file._positions[0])}")
    with open(f"{save_path}/packed_box.pdb", "r") as f:
        lines = f.readlines()
        for i in lines:
            if i.startswith("HETATM"):
                atom_ind = int(i[6:12])
                resid_ind = i[22:26]
                resid = i[12:].split()[1]
                atom = i[12:].split()[0]
                x = round(float(i[28:].split()[0]) / 10, 3)
                y = round(float(i[28:].split()[1]) / 10, 3)
                z = round(float(i[28:].split()[2]) / 10, 3)
                box_file.append(
                    f"{resid_ind:>5}{resid}{atom:>7}{atom_ind:>5}{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                )
                
    # Check for periodic box vectors
    max_box_dimension = pdb_file.positions.max().max()._value
    pbc = round(max_box_dimension, 5)
    box_file.append(f"{pbc:>10}{pbc:>9}{pbc:>10}")
    box_file.append("")

    # Assume polymer_count and salt_count are already passed as lists
    all_counts = polymer_count + salt_count

    topology_file = [] 
    topology_file.append("[ defaults ]")
    topology_file.append(
        "; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ"
    )
    topology_file.append("1               3               yes             0.5     0.5")
    topology_file.append("")
    topology_file.append('#include "./force_field.itp"')
    for i in name_iterables:
        topology_file.append(f'#include "./{i}.top"')
    topology_file.append("")
    topology_file.append("[ system ]")
    topology_file.append("; Name")
    topology_file.append("OPV")
    topology_file.append("")
    topology_file.append("[ molecules ]")
    topology_file.append("; Compound        #mols")
    for name, count in zip(name_iterables, all_counts):
        topology_file.append(f"{name:<18}{count}")

    return ff_files_separate, box_file, name_iterables, ff_file, topology_file

def write_openmm_files(
    save_path,
    pdb_file,
    ff_file,
    ff_file_resid,
    name_iterables,
    polymer=False,
    packed_name=None,
    index=None,
):
    # Use the provided packed_name or default to the existing naming convention
    if packed_name is None:
        packed_name = "polymer_conformation" if polymer else "packed_box"

    if index is None:
        print(f"Writing Force field files at {save_path}")
        with open(f"{save_path}/force_field.xml", "w") as f:
            for i in ff_file:
                f.write(i + "\n")

        with open(f"{save_path}/force_field_resids.xml", "w") as f:
            for i in ff_file_resid:
                f.write(i + "\n")
    else:
        print(f"Writing Force field files at {save_path}")
        with open(f"{save_path}/force_field_{index}.xml", "w") as f:
            for i in ff_file:
                f.write(i + "\n")

        with open(f"{save_path}/force_field_resids_{index}.xml", "w") as f:
            for i in ff_file_resid:
                f.write(i + "\n")
        
        # Use a unique residue name based on the index
        unique_name = f"P{index + 1}"
        name_iterables = [unique_name]  # Replace `name_iterables` with a list containing the unique name


    # Processing chains and renaming residues
    for chain_index, (chain, name) in enumerate(zip(pdb_file.topology._chains, name_iterables), start=1):
        print(f"Processing chain {chain_index} with name {name}")
        for residue_index, residue in enumerate(chain._residues, start=1):
            print(f"Renaming residue {residue_index} in chain {chain_index} from {residue.name} to {name}")
            chain._residues[residue_index - 1].name = name  # Assign new residue name

    print("Saving the pdb file with the packed polymer.")
    with open(f"{save_path}/{packed_name}.pdb", "w") as f:
        pdb_file.writeFile(
            topology=pdb_file.topology,
            positions=pdb_file.positions,
            file=f,
        )

    print("Rewriting the packed box file while removing the connectivity records (CONECT).")
    with open(f"{save_path}/{packed_name}.pdb", "r") as f:
        lines = f.readlines()
        new_lines = []
        for line_index, line in enumerate(lines):
            if "CONECT" not in line:
                new_lines.append(line)
                #print(f"Writing line {line_index + 1}: {line.strip()}")  # Debug statement for each line

    with open(f"{save_path}/{packed_name}.pdb", "w") as f:
        for line in new_lines:
            f.write(line)


def write_gromacs_files(
    save_path,
    ff_files,
    ff_file,
    box_file,
    name_iterables,
    topology_file,
):
    print(f"Writing Force field files at {save_path}/gromacs")
    
    # Check if directory exists and create if not
    if not os.path.isdir(save_path + "/gromacs"):
        os.makedirs(save_path + "/gromacs")
    else:
        print(f"Directory {save_path}/gromacs already exists.")

    # Debug statements for force field files
    print(f"Force field files content (first 10 lines): {ff_file[:10]}")
    
    with open(f"{save_path}/gromacs/force_field.itp", "w") as f:
        for i in ff_file:
            f.write(i + "\n")

    # Debug statements for topology file
    print(f"Topology file content (first 10 lines): {topology_file[:10]}")
    print(f"Full topology file content:")
    for line in topology_file:
        print(line)
    
    with open(f"{save_path}/gromacs/packed_box.top", "w") as f:
        for i in topology_file:
            f.write(i + "\n")

    # Debug statements for individual FF files
    for file, name in zip(ff_files, name_iterables):
        print(f"Writing individual FF file: {name}.top")
        print(f"File content (first 10 lines): {file[:10]}")
        with open(f"{save_path}/gromacs/{name}.top", "w") as f:
            for i in file:
                f.write(i + "\n")

    # Debug statements for .gro file
    print(f"Box file content (first 10 lines): {box_file[:10]}")
    
    with open(f"{save_path}/gromacs/packed_box.gro", "w") as f:
        for i in box_file:
            f.write(i + "\n")

    # Final confirmation
    if os.path.exists(f"{save_path}/gromacs/packed_box.gro"):
        print(f".gro file successfully created at {save_path}/gromacs/packed_box.gro")
    else:
        print(f"Failed to create .gro file at {save_path}/gromacs/packed_box.gro")

    # Debug statement for molecule counts
    print("Molecule counts in the topology file:")
    with open(f"{save_path}/gromacs/packed_box.top", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "[ molecules ]" in line:
                start_idx = lines.index(line) + 2
                break
        for line in lines[start_idx:]:
            if line.strip():
                print(line.strip())


def assign_lpg_params(
    short_smiles_list,
    atoms_long_list,
    atom_names_long_list,
    atoms_short,
    atom_names_short,
    train_dataset,
    param_dict,
    lit_charges_save_path,
    charges,
    htvs_path,
    htvs_details,
):
    for i, (atoms_long, atom_names_long) in enumerate(zip(atoms_long_list, atom_names_long_list)):
        test_atoms = [NUM_TO_ELEMENT[j] for j in train_dataset[i].atomic_nums]
        print("IN ASSIGN LPG")
        print(f"TEST ATOMS {test_atoms}")
        print(f"ATOMS LONG {atoms_long}")
        assert test_atoms == atoms_long

        for ind, j in enumerate(train_dataset[i].pair_params):
            for key, val in param_dict["atoms"].items():
                if key == atom_names_long[ind]:
                    pairs = [sum(val["charges"]) / len(val["charges"]), val["epsilon"], val["sigma"]]
                    train_dataset[i].pair_params[ind] = torch.tensor(pairs)
                    #print(f"Assigned pair parameters for {atom_names_long[ind]}: {pairs}")  # Print assigned pair parameters
                    break

        if charges == "DFT":
            print("Loading HTVS to access charges")
            combined_charges = []
            combined_htvs_atoms = []
            for short_smiles in short_smiles_list:
                char_mean, char_std, htvs_atoms = get_charges_htvs(
                    smiles=short_smiles, htvs_path=htvs_path, htvs_details=htvs_details
                )
                combined_charges.extend(char_mean)
                combined_htvs_atoms.extend(htvs_atoms)

            assert [
                "".join([j for j in k if not j.isdigit()]) for k in atom_names_short
            ] == combined_htvs_atoms

            df = pd.DataFrame({"charge": combined_charges, "names": atom_names_short})

            charges_dict = {}
            for k in df.groupby("names", as_index=False).mean().to_dict("records"):
                charges_dict[k["names"]] = k["charge"]

        elif charges == "LPG":
            charge_dict_temp = {}
            for key, val in param_dict["atoms"].items():
                if key not in charge_dict_temp.keys():
                    charge_dict_temp[key] = []
                    charge_dict_temp[key].append(sum(val["charges"]) / len(val["charges"]))
                else:
                    charge_dict_temp[key].append(sum(val["charges"]) / len(val["charges"]))

            charges_dict = {}
            for key, val in charge_dict_temp.items():
                charges_dict[key] = np.array(val).mean()

        elif charges == "LIT":
            charges_list = []
            with open(f"{lit_charges_save_path}/charges_lit.csv", "r") as f:
                lines = f.readlines()
                try:
                    for l, a in zip(lines, atoms_short):
                        if l.split(",")[1] == a:
                            charges_list.append(float(l.split(",")[0]))
                except:
                    for l, a in zip(lines[1:], atoms_short):
                        if l.split(",")[1] == a:
                            charges_list.append(float(l.split(",")[0]))
            assert len(charges_list) == len(atoms_short)

            df = pd.DataFrame({"charge": charges_list, "names": atom_names_short})

            charges_dict = {}
            for l in df.groupby("names", as_index=False).mean().to_dict("records"):
                charges_dict[l["names"]] = l["charge"]

        for ind, j in enumerate(train_dataset[i].pair_params):
            atom_name = atom_names_long[ind]
            if atom_name not in charges_dict:
                print(f"Warning: {atom_name} not found in charges_dict. Using default value.")
                train_dataset[i].pair_params[ind, 0] = 0.0  # Default value
            else:
                train_dataset[i].pair_params[ind, 0] = charges_dict[atom_name]
            #print(f"Assigned charge for {atom_name}: {train_dataset[i].pair_params[ind, 0]}")  # Print assigned charge

        mean = train_dataset[i].pair_params[:, 0].mean()
        smallest_charge = train_dataset[i].pair_params[:, 0].abs().min()
        sum_char = train_dataset[i].pair_params[:, 0].sum()
        print(
            f"Mean charge: {mean:.4}, Sum of charges: {sum_char}, Smallest charge: {smallest_charge}"
        )
        if train_dataset[i].pair_params[:, 0].sum().abs() * 50 > 1e-3:
            charges_all = train_dataset[i].pair_params[:, 0]
            pol_pos = charges_all[charges_all > 0].sum()
            pol_neg = charges_all[charges_all < 0].sum()
            pol_sc = torch.mean(torch.tensor((pol_pos, torch.abs(pol_neg))))
            charges_all[charges_all > 0] = (charges_all[charges_all > 0] / pol_pos) * pol_sc
            charges_all[charges_all < 0] = (
                charges_all[charges_all < 0] / torch.abs(pol_neg)
            ) * pol_sc
            train_dataset[i].pair_params[:, 0] = charges_all
            mean = train_dataset[i].pair_params[:, 0].mean()
            smallest_charge = train_dataset[i].pair_params[:, 0].abs().min()
            sum_char = train_dataset[i].pair_params[:, 0].sum()
            print("Charges have been rescaled")
            print(
                f"Mean charge: {mean:.4}, Sum of charges: {sum_char}, Smallest charge: {smallest_charge}"
            )
        if train_dataset[i].pair_params[:, 0].sum().abs() * 10 > 1e-3:
            raise ValueError("CHARGES HAVE NOT BEEN PROPERLY RESCALED")
        
        charge_bonds = {}
        for key, val in param_dict["bonds"].items():
            charge_bonds[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = [val["k"], val["length"]]

        bonds_added = 0
        for ind, j in enumerate(train_dataset[i].bonds):
            bonds_temp = [atom_names_long[j[0]], atom_names_long[j[1]]]
            bond_added = False
            for key, val in param_dict["bonds"].items():
                if set(val["other_idx"]) == set(bonds_temp):
                    bond_p = [val["k"], val["length"]]
                    train_dataset[i].bond_params[ind] = torch.tensor(bond_p)
                    bond_added = True
                    bonds_added += 1
                    #print(f"Assigned bond parameters for {bonds_temp}: {bond_p}")  # Print assigned bond parameters
                    break
            if not bond_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in bonds_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in bonds_temp[::-1]]
                )
                if charge_temp1 in charge_bonds.keys():
                    bond_p = charge_bonds[charge_temp1]
                elif charge_temp2 in charge_bonds.keys():
                    bond_p = charge_bonds[charge_temp2]
                train_dataset[i].bond_params[ind] = torch.tensor(bond_p)
                bonds_added += 1
                #print(f"Assigned bond parameters for {bonds_temp} (from charges): {bond_p}")  # Print assigned bond parameters

        print(f"Bonds added: {bonds_added}")
        print(f"Bonds expected: {len(train_dataset[i].bonds)}")
        assert len(train_dataset[i].bonds) == bonds_added, f"Number of bonds added {bonds_added} does not match expected {len(train_dataset[i].bonds)}"

        charge_angles = {}
        for key, val in param_dict["angles"].items():
            charge_angles[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = [val["k"], val["angle"]]
        
        angles_added = 0
        for ind, j in enumerate(train_dataset[i].angles):
            angles_temp = [
                atom_names_long[j[0]],
                atom_names_long[j[1]],
                atom_names_long[j[2]],
            ]
            angle_added = False
            for key, val in param_dict["angles"].items():
                if val["other_idx"] == angles_temp or val["other_idx"] == angles_temp[::-1]:
                    angles_p = [val["k"], val["angle"]]
                    train_dataset[i].angle_params[ind] = torch.tensor(angles_p)
                    angle_added = True
                    angles_added += 1
                    #print(f"Assigned angle parameters for {angles_temp}: {angles_p}")  # Print assigned angle parameters
                    break
            if not angle_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict.get(c, 0.0), 3)) for c in angles_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict.get(c, 0.0), 3)) for c in angles_temp[::-1]]
                )
                if charge_temp1 in charge_angles.keys():
                    angles_p = charge_angles[charge_temp1]
                elif charge_temp2 in charge_angles.keys():
                    angles_p = charge_angles[charge_temp2]
                train_dataset[i].angle_params[ind] = torch.tensor(angles_p)
                angles_added += 1
                #print(f"Assigned angle parameters for {angles_temp} (from charges): {angles_p}")  # Print assigned angle parameters

        print(f"Angles added: {angles_added}")
        print(f"Angles expected: {len(train_dataset[i].angles)}")
        assert len(train_dataset[i].angles) == angles_added, f"Number of angles added {angles_added} does not match expected {len(train_dataset[i].angles)}"

        charge_dihs = {}
        for key, val in param_dict["dihedrals"].items():
            charge_dihs[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = val["k"]

        dihedrals_added = 0
        for ind, j in enumerate(train_dataset[i].dihedrals):
            dihedrals_temp = [
                atom_names_long[j[0]],
                atom_names_long[j[1]],
                atom_names_long[j[2]],
                atom_names_long[j[3]],
            ]
            dih_added = False
            for key, val in param_dict["dihedrals"].items():
                if (
                    val["other_idx"] == dihedrals_temp
                    or val["other_idx"] == dihedrals_temp[::-1]
                ):
                    dihedral_p = val["k"]
                    train_dataset[i].dihedral_params[ind] = torch.tensor(dihedral_p)
                    dih_added = True
                    dihedrals_added += 1
                    #print(f"Assigned dihedral parameters for {dihedrals_temp}: {dihedral_p}")  # Print assigned dihedral parameters
                    break

            if not dih_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict.get(c, 0.0), 3)) for c in dihedrals_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict.get(c, 0.0), 3)) for c in dihedrals_temp[::-1]]
                )
                if charge_temp1 in charge_dihs.keys():
                    dihedral_p = charge_dihs[charge_temp1]
                elif charge_temp2 in charge_dihs.keys():
                    dihedral_p = charge_dihs[charge_temp2]
                train_dataset[i].dihedral_params[ind] = torch.tensor(dihedral_p)
                dihedrals_added += 1
                #print(f"Assigned dihedral parameters for {dihedrals_temp} (from charges): {dihedral_p}")  # Print assigned dihedral parameters

        print(f"Dihedrals added: {dihedrals_added}")
        print(f"Dihedrals expected: {len(train_dataset[i].dihedrals)}")
    
        assert len(train_dataset[i].dihedrals) == dihedrals_added, f"Number of dihedrals added {dihedrals_added} does not match expected {len(train_dataset[i].dihedrals)}"

        improp_indices = []
        for ind, j in enumerate(train_dataset[i].impropers):
            impropers_temp = [
                atom_names_long[j[0]],
                atom_names_long[j[1]],
                atom_names_long[j[2]],
                atom_names_long[j[3]],
            ]
            for key, val in param_dict["impropers"].items():
                if (
                    val["other_idx"] == impropers_temp
                    or val["other_idx"] == impropers_temp[::-1]
                ):
                    improper_p = val["k"]
                    train_dataset[i].improper_params[ind] = torch.tensor(improper_p)
                    improp_indices.append(ind)
                    #print(f"Assigned improper parameters for {impropers_temp}: {improper_p}")  # Print assigned improper parameters
                    break
                elif val["other_idx"][0] == impropers_temp[0]:
                    if np.isin(val["other_idx"][1:], impropers_temp[1:]).all():
                        improper_p = val["k"]
                        #print(f"Added partial improper: {impropers_temp} using params: {improper_p}")
                        train_dataset[i].improper_params[ind] = torch.tensor(improper_p)
                        improp_indices.append(ind)
                        break
        train_dataset[i].impropers = train_dataset[i].impropers[improp_indices]
        train_dataset[i].improper_params = train_dataset[i].improper_params[improp_indices]

    return train_dataset

def minimize_polymer(
    save_path,
    short_smiles,
    long_smiles,
    atoms_long,
    atom_names_long,
    atoms_short,
    atom_names_short,
    param_dict,
    lit_charges_save_path,
    charges,
    htvs_path,
    htvs_details,
):
    # Loop through each set of long SMILES, atom data, and atom names
    for i, (smiles, atoms, atom_names) in enumerate(zip(long_smiles, atoms_long, atom_names_long)):
        # Construct the unique path for each random chain's PDB file
        pdb_path = f"{save_path}"
        packed_name = f"polymer_conformation_{i}"
        
        # Load force field parameters for the polymer
        train_dataset = load_hitpoly_params(
            [smiles],  # Handle a single SMILES at a time
            [],
        )
        
        print(f"Atoms Long ")
        # Assign LigParGen parameters to the polymer
        train_dataset = assign_lpg_params(
            short_smiles_list=[short_smiles],
            atoms_long_list=[atoms],
            atom_names_long_list=[atom_names],
            atoms_short=atoms_short,
            atom_names_short=atom_names_short,
            train_dataset=train_dataset,
            param_dict=param_dict,
            lit_charges_save_path=lit_charges_save_path,
            charges=charges,
            htvs_path=htvs_path,
            htvs_details=htvs_details,
        )

        # Load the PDB file and create molecule dictionary from it
        mol_dict, pdb_file = load_pdb_create_mol_dict(pdb_path, train_dataset, polymer=True, packed_name = packed_name)

        # Print some information about the structure
        print(f"Number of atoms: {pdb_file.topology.getNumAtoms()}")
        print(f"Number of residues: {pdb_file.topology.getNumResidues()}")
        print(f"Number of chains: {pdb_file.topology.getNumChains()}")

        # Reindex atom names for consistency
        atom_dict, atom_types_list = atom_name_reindexing(mol_dict)

        # ONLY SCALE THE SALTS NOT THE POLYMERS!!!!! or code will FAIL
        total_molecules = len(mol_dict)
        for idx, mol_key in enumerate(mol_dict.keys()):
            # Only scale for the last two molecules in the dictionary
            if idx >= total_molecules - 2:
                lammps_openmm_scaling = True
            else:
                lammps_openmm_scaling = False

            # Print the molecule being processed
            print(f"Processing molecule: {mol_key} (Scaling: {lammps_openmm_scaling})")

            # Apply scaling based on the condition above
            mol_dict[mol_key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[mol_key]["mol_hitpoly"], lammps_openmm_scaling
        )

        # Create force field and residue files for the polymer
        ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
            mol_dict, atom_types_list, index = i,
        )
        
        # Write the OpenMM-compatible files
        write_openmm_files(
            save_path, pdb_file, ff_file, ff_file_resid, name_iterables, polymer=True, packed_name = packed_name, index = i,
        )

        print(f"Temporary polymer force field built for conformation {i}!")

        # Perform energy minimization to relax the polymer structure
        equilibrate_polymer(
            save_path=save_path,
            mol_dict=mol_dict, #test
            packed_name=packed_name,
            index = i,
        )

        print(f"Polymer structure {i} has been minimized and saved at {pdb_path}")

    print("All polymer conformations have been minimized.")

def load_hitpoly_params(
    poly_smiles: list,
    salt_smiles: list,
    salt_data_paths: list = [],
    charge_scale: float = None,
    poly_count: int = 1,
):
    smiles = poly_smiles + salt_smiles
    train_args = hitpolyArgs()

    train_molecule_data = [
        TopologyBuilder(
            smiles=[i], train_args=train_args, load_geoms=False, lj_from_file=False
        )
        for i in smiles
    ]

    train_dataset = TopologyDataset(data=train_molecule_data)

    for i, name in zip(train_dataset[poly_count:], salt_data_paths):
        ff_read = ForceFieldFileReader(
            i,
            name,
        )
        ff_read.load_pair_params()
        ff_read.load_bond_params()
        ff_read.load_angle_params()
        ff_read.load_dihedral_params()

    if charge_scale:
        for i in train_dataset[1:]:
            i.pair_params[:, 0] = i.pair_params[:, 0] * charge_scale

    for i in train_dataset:
        print(f"Charges for {i.smiles[0]} - ", i.pair_params[:, 0].sum())

    return train_dataset

def load_pdb_create_mol_dict(
    save_path,
    train_dataset,
    polymer=False,
    packed_name=None,
):
    # Use the provided packed_name or default to the existing naming convention
    if packed_name is None:
        packed_name = "polymer_conformation" if polymer else "packed_box"

    pdb_file = PDBFile(f"{save_path}/{packed_name}.pdb")
    mol_dict = {}

    #cation and tfsi even if single ion conductor tfsi isnt in the box but doesnt get matched so doesnt matter leave a 2
    salt_smiles_length = 2 

    # Split the training dataset into polymer molecules and salt molecules
    polymer_train_data = train_dataset[:-salt_smiles_length]
    salt_train_data = train_dataset[-salt_smiles_length:]

    print(f"Total chains in PDB file: {len(list(pdb_file.topology.chains()))}")
    print(f"Total polymer molecules in dataset: {len(polymer_train_data)}")
    print(f"Total salt molecules in dataset: {len(salt_train_data)}")

    # First, handle the polymer chains, which should already be in order
    for idx, mol in enumerate(pdb_file.topology.chains()):
        if idx < len(polymer_train_data):
            smiles = polymer_train_data[idx].smiles[0]
            mol_dict[smiles] = {}
            mol_dict[smiles]["mol_hitpoly"] = polymer_train_data[idx]
            mol_dict[smiles]["mol_pdb"] = mol
            
            # Debug print for chain assignment
            print(f"Assigning chain index {idx} to polymer SMILES: {smiles}")
            
            # Perform atomic number check
            pdb_atoms = [atom.element.atomic_number for atom in mol._residues[0].atoms()]
            rdkit_atoms = polymer_train_data[idx].atomic_nums
            
            if not np.array_equal(np.array(rdkit_atoms), np.array(pdb_atoms)):
                print(f"Warning: Atomic numbers mismatch for polymer SMILES: {smiles}")
                print(f"RDKit atomic numbers: {rdkit_atoms}")
                print(f"PDB atomic numbers: {pdb_atoms}")
            else:
                print(f"Match confirmed for polymer SMILES: {smiles} with chain index {idx}")
        else:
            print(f"All polymers processed, moving on to salt molecules.")
            break  # Break out of loop after handling polymers

    # Next, match the salt molecules by atomic numbers
    salt_pdb_chains = list(pdb_file.topology.chains())[-salt_smiles_length:]  # Get the last salt_smiles_length chains from PDB

    for salt_pdb_mol in salt_pdb_chains:
        pdb_atoms = [atom.element.atomic_number for atom in salt_pdb_mol._residues[0].atoms()]
        matched = False

        # Try to match the PDB chain to an RDKit salt molecule based on atomic numbers
        for salt_train_mol in salt_train_data:
            rdkit_atoms = salt_train_mol.atomic_nums
            smiles = salt_train_mol.smiles[0]

            if np.array_equal(np.array(rdkit_atoms), np.array(pdb_atoms)):
                mol_dict[smiles] = {}
                mol_dict[smiles]["mol_hitpoly"] = salt_train_mol
                mol_dict[smiles]["mol_pdb"] = salt_pdb_mol
                print(f"Match confirmed for salt SMILES: {smiles} with PDB atomic numbers: {pdb_atoms}")
                matched = True
                break
        
        if not matched:
            print(f"Warning: No matching salt molecule found for PDB chain with atomic numbers: {pdb_atoms}")

    # Print final summary of the matching
    if mol_dict:
        print(f"All molecules matched successfully. Total matched molecules: {len(mol_dict)}")
    else:
        raise ValueError("No matches were found between the RDKit molecules and the PDB chains.")

    return mol_dict, pdb_file

def get_concentration_from_molality(
    save_path,
    molality,
    polymer_count,
    monomers,
    fractions,
    add_end_Cs,
    polymerization_mode,
    num_blocks,
    polymer_chain_length=1100,
    arms=None,
    blend_mode=False
):
    print(f"Inside get_concentration_from_molality - SMILES: {monomers}, Fractions: {fractions}")

    def create_polymer_for_estimation(repeat_units):
        smiles, _ = create_long_smiles(
            save_path=save_path,
            smiles=monomers,
            fractions=fractions,
            total_repeats=repeat_units,
            add_end_Cs=add_end_Cs,
            polymerization_mode=polymerization_mode,
            num_blocks=num_blocks,
            arms=arms,
            write_log=False
        )
        return smiles

    if blend_mode:
        print("Blend mode enabled")
        all_masses = []
        repeat_units_list = []

        for i, (s, f) in enumerate(zip(monomers, fractions)):
            atoms_per_unit = get_atom_count(s, poly_count=1)
            repeat_units = max(1, round(polymer_chain_length / atoms_per_unit))

            def adjust_chain_length(repeat_units):
                while True:
                    smiles = create_polymer_for_estimation(repeat_units)
                    atom_count = get_atom_count(smiles)
                    if atom_count > polymer_chain_length * 1.1:
                        repeat_units -= 1
                    elif atom_count < polymer_chain_length * 0.9:
                        repeat_units += 1
                    else:
                        break
                return smiles, repeat_units

            long_smiles, repeat_units = adjust_chain_length(repeat_units)

            long_smiles, _ = create_long_smiles(
                save_path=save_path,
                smiles=[s],
                fractions=[1.0],
                total_repeats=repeat_units,
                add_end_Cs=add_end_Cs,
                polymerization_mode="homopolymer",
                write_log=True,
                log_filename=f"homopolymer_{i}.txt"
            )

            mass = get_mol_mass(long_smiles)
            all_masses.append(mass)
            repeat_units_list.append(repeat_units)

        avg_molar_mass = sum(f * m for f, m in zip(fractions, all_masses))
        concentration = round(molality * avg_molar_mass * polymer_count / 1000)
        return [concentration, concentration], repeat_units_list

    atom_count_monomer = sum([get_atom_count(s, poly_count=1) * f for s, f in zip(monomers, fractions)])
    repeat_units = round(polymer_chain_length / atom_count_monomer)

    long_smiles = None
    final_repeat_units = repeat_units

    def adjust_chain_length(repeat_units):
        nonlocal long_smiles, final_repeat_units
        while True:
            if polymerization_mode == "alternating" and repeat_units % 2 != 0:
                repeat_units += 1
            smiles = create_polymer_for_estimation(repeat_units)
            atom_count = get_atom_count(smiles)
            print(f"Trying repeat_units={repeat_units} → Atom count: {atom_count}")
            if atom_count > polymer_chain_length * 1.1:
                repeat_units -= 1
            elif atom_count < polymer_chain_length * 0.9:
                repeat_units += 1
            else:
                final_repeat_units = repeat_units
                long_smiles = smiles
                break
        return long_smiles, final_repeat_units

    long_smiles, repeat_units = adjust_chain_length(repeat_units)

    if polymerization_mode == "star":
        if arms is None or arms <= 0:
            raise ValueError("For star polymerization, arms must be > 0.")
        repeat_units = max(1, repeat_units // arms)

    long_smiles, _ = create_long_smiles(
        save_path=save_path,
        smiles=monomers,
        fractions=fractions,
        total_repeats=repeat_units,
        add_end_Cs=add_end_Cs,
        polymerization_mode=polymerization_mode,
        num_blocks=num_blocks,
        arms=arms,
        write_log=True,
        log_filename="final_polymer_details.txt"
    )

    concentration = round(molality * get_mol_mass(long_smiles) * polymer_count / 1000)
    return [concentration, concentration], repeat_units

def average_dft_charges_and_update_param_dict(dft_charge_data, original_param_dict):
    """
    Averages DFT charges by atom type number and updates a deep copy of param_dict
    with the new charges. Handles atom type keys as either int or str.

    Returns:
    - new_param_dict: with updated charges
    """
    # Step 1: Average DFT charges
    charge_accumulator = defaultdict(list)
    for _, row in dft_charge_data.iterrows():
        try:
            atom_type = int(row["Type (name)"])
            charge = float(row["Charge"])
            charge_accumulator[atom_type].append(charge)
        except (ValueError, KeyError) as e:
            print(f"Skipping row due to error: {e}")
            continue

    averaged_charges = {
        atom_type: sum(charges) / len(charges)
        for atom_type, charges in charge_accumulator.items()
    }

    # Step 2: Deep copy param_dict
    new_param_dict = copy.deepcopy(original_param_dict)

    # Step 3: Update charges, coercing keys to int
    updated_atom_types = set()
    param_atom_types = set()

    for atom_type_key, props in new_param_dict["atoms"].items():
        try:
            atom_type_int = int(atom_type_key)
            param_atom_types.add(atom_type_int)
        except ValueError:
            print(f"Skipping non-integer atom type key: {atom_type_key}")
            continue

        if atom_type_int in averaged_charges:
            props["charges"] = [averaged_charges[atom_type_int]]
            updated_atom_types.add(atom_type_int)
            print(f"Updated atom type {atom_type_int} with DFT charge {averaged_charges[atom_type_int]:.4f}")
        else:
            print(f"No DFT charge found for atom type {atom_type_int}")

    # Step 4: Report unmatched DFT types
    dft_atom_types = set(charge_accumulator.keys())
    unmatched_atom_types = dft_atom_types - param_atom_types
    if unmatched_atom_types:
        print("Atom types in DFT table but missing from param_dict['atoms']:")
        print(sorted(unmatched_atom_types))
    else:
        print("All DFT atom types matched a param_dict atom.")

    return new_param_dict

def get_concentration_from_charge_neutrality(atom_names_long, param_dict, polymer_count, cation_charge=0.75, single_ion_conductor=True, anionic_atom_count_per_polymer=None):
    """
    Calculates the number of cations (e.g., Li+) needed to neutralize the system.
    Scales all DFT-derived charges so the polymer system has a total charge that
    matches an integer number of cations. Accepts explicit count of anionic sites
    (e.g., [N-]) per polymer for more reliable balancing.
    """
    total_polymer_charge = 0.0
    missing_types = set()
    atom_type_counts = {}

    # === Step 1: Build charge lookup ===
    if "atoms" not in param_dict:
        raise KeyError("Could not find 'atoms' key in param_dict.")

    charge_lookup = {}
    for k, v in param_dict["atoms"].items():
        if v.get("charges"):
            charge_lookup[k] = sum(v["charges"]) / len(v["charges"])
        else:
            charge_lookup[k] = 0.0

    # === Step 2: Accumulate charge per polymer and count atom types ===
    print("\n🔍 Starting charge accumulation and atom type counting:")
    for atom_types in atom_names_long:
        if not atom_types:
            continue
        for atom_type in atom_types:
            charge = charge_lookup.get(atom_type, 0.0)
            total_polymer_charge += charge
            atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1

    # === Step 3: Use externally provided anionic count ===
    if anionic_atom_count_per_polymer is None:
        print("No anionic count provided — defaulting to 0")
        anionic_atom_count_per_polymer = 0

    total_anionic_atoms = anionic_atom_count_per_polymer * polymer_count
    print(f"Anionic atoms counted: {anionic_atom_count_per_polymer} per polymer, {total_anionic_atoms} total")

    # === Step 4: Compute total system charge before scaling ===
    total_polymer_charge *= polymer_count
    print(f"Total unscaled polymer system charge: {total_polymer_charge:.6f}")

    # === Step 5: Determine target charge from anionic count ===
    ideal_cation_count = total_anionic_atoms
    target_polymer_charge = -1 * ideal_cation_count * cation_charge

    if total_polymer_charge == 0:
        print("Total polymer charge is 0, skipping scaling.")
        scale_factor = 1.0
    else:
        scale_factor = target_polymer_charge / total_polymer_charge

    print(f"Scaling all polymer charges by factor: {scale_factor:.6f}")

    if single_ion_conductor:
        for atom_type in param_dict["atoms"]:
            if "charges" in param_dict["atoms"][atom_type]:
                original_charges = param_dict["atoms"][atom_type]["charges"]
                scaled_charges = [q * scale_factor for q in original_charges]
                param_dict["atoms"][atom_type]["charges"] = scaled_charges
                print(f"  Scaled {atom_type}: {original_charges} → {scaled_charges}")

    # === Step 6: Recalculate total charge after scaling ===
    total_scaled_charge = 0.0
    for atom_types in atom_names_long:
        if not atom_types:
            continue
        for atom_type in atom_types:
            charge = param_dict["atoms"].get(atom_type, {}).get("charges", [0.0])[0]
            total_scaled_charge += charge

    total_scaled_charge *= polymer_count
    print(f"Total scaled polymer system charge: {total_scaled_charge:.6f}")

    return [ideal_cation_count, 0], scale_factor

def write_atom_labels_from_log(
    atom_names_file: str,
    log_file: str,
    output_file: str = "atom_labels_rdf.txt"
):
    # --- Step 1: Parse log file ---
    with open(log_file, "r") as f:
        lines = f.readlines()

    architecture = None
    monomer_counts = None
    final_smiles = None
    total_atoms = None
    block_sizes_and_monomers = None

    for line in lines:
        if line.startswith("Architecture:"):
            architecture = line.split("Architecture:")[1].strip()
        elif line.startswith("Block sizes and monomers:"):
            block_sizes_and_monomers = ast.literal_eval(line.split("Block sizes and monomers:")[1].strip())
        elif line.startswith("Monomer counts:"):
            monomer_counts = ast.literal_eval(line.split("Monomer counts:")[1].strip())
        elif line.startswith("Final SMILES:"):
            final_smiles = line.split("Final SMILES:")[1].strip()
        elif line.startswith("Total number of atoms:"):
            total_atoms = int(line.split("Total number of atoms:")[1].strip())

    if architecture not in ["alternating", "block"]:
        raise NotImplementedError("Only 'alternating' and 'block' architectures are supported.")
    if not monomer_counts or not final_smiles or not total_atoms:
        raise ValueError("Missing required fields in log file.")

    # --- Step 2: Determine heavy atom counts for unique monomers ---
    unique_monomers = list(monomer_counts.keys())
    monomer_id_map = {smi: idx for idx, smi in enumerate(unique_monomers)}
    heavy_counts = {
        smi: Chem.MolFromSmiles(smi.replace("[Cu]", "").replace("[Au]", "")).GetNumAtoms()
        for smi in unique_monomers
    }

    # --- Step 3: Build label pattern for one chain ---
    label_pattern = [2]  # initial cap
    if architecture == "alternating":
        num_repeats = sum(monomer_counts.values())
        for i in range(num_repeats):
            m_idx = i % 2
            smi = unique_monomers[m_idx]
            label_pattern.extend([m_idx] * heavy_counts[smi])
    elif architecture == "block":
        for block_size, smi in block_sizes_and_monomers:
            mon_id = monomer_id_map[smi]
            label_pattern.extend([mon_id] * (block_size * heavy_counts[smi]))
    label_pattern.append(2)  # final cap

    # --- Step 4: Read atom file ---
    with open(atom_names_file, "r") as f:
        atoms = [line.strip().split(",") for line in f.readlines()]

    # --- Step 5: Group heavy atom indices by chain ---
    chains = []
    current_chain = []
    for i, (element, resname, _) in enumerate(atoms):
        element = element.strip()
        resname = resname.strip()
        if resname == "P1" and element != "H":
            current_chain.append(i)
        elif current_chain:
            chains.append(current_chain)
            current_chain = []
    if current_chain:
        chains.append(current_chain)

    # --- Step 6: Apply label pattern to each chain ---
    labels = [3] * len(atoms)
    for chain in chains:
        if len(chain) != len(label_pattern):
            raise ValueError(f"Chain length {len(chain)} != expected pattern length {len(label_pattern)}")
        for idx, label in zip(chain, label_pattern):
            labels[idx] = label

    # --- Step 7: Label H and other residues ---
    for i, (element, resname, _) in enumerate(atoms):
        if element.strip() == "H":
            labels[i] = 3

    # --- Step 8: Write output ---
    with open(output_file, "w") as f:
        for (element, resname, _), label in zip(atoms, labels):
            f.write(f"{element.strip()}, {resname.strip()}, {label}\n")

    print(f"✅ Saved labeled RDF file to: {output_file}")


# def create_box_and_ff_files(
#     save_path,
#     long_smiles,
#     short_smiles,
#     filename,
#     polymer_count,
#     concentration,
#     packmol_path,
#     atoms_short,
#     atoms_long,
#     atom_names_short,
#     atom_names_long,
#     param_dict,
#     lit_charges_save_path,
#     charges,
#     charge_scale,
#     htvs_path,
#     htvs_details,
#     salt_smiles,
#     salt_paths,
#     salt_data_paths,
#     box_multiplier,
#     salt,
# ):
#     poly_paths = [f"{save_path}/{name}" for name in filename]
#     print(f"POLY PATHS HERE {poly_paths}")

#     print(
#         f"Creating and running packmol files with {polymer_count} chains and {concentration} LiTFSIs"
#     )
#     print(f"at {save_path} path")
#     create_packmol_input_file(
#         poly_paths,
#         salt_paths,
#         f"{save_path}/packmol.inp",
#         f"{save_path}/packed_box.pdb",
#         salt_smiles=salt_smiles,
#         polymer_count=polymer_count,
#         salt_concentrations=concentration,
#         box_multiplier=box_multiplier,
#         tolerance=5.0,
#         salt=salt,
#     )

#     run_packmol(save_path, packmol_path)
    
#     # Set counts as a list
#     polymer_count = [polymer_count] * len(long_smiles)

#     # Ensure polymer_count is correct
#     if len(polymer_count) != len(long_smiles):
#         raise ValueError("polymer_count length must match the number of long_smiles")

#     # Ensure salt_count is correct
#     if salt and len(concentration) != len(salt_smiles):
#         raise ValueError("concentration length must match the number of salt_smiles")

#     print(f"Making all the force field files for simulations")
#     train_dataset = load_hitpoly_params(
#         long_smiles,
#         salt_smiles,
#         salt_data_paths=salt_data_paths,
#         charge_scale=charge_scale,
#     )

#     train_dataset = assign_lpg_params(
#         short_smiles_list=short_smiles,
#         atoms_long_list=atoms_long,
#         atom_names_long_list=atom_names_long,
#         atoms_short=atoms_short,
#         atom_names_short=atom_names_short,
#         train_dataset=train_dataset,
#         param_dict=param_dict,
#         lit_charges_save_path=lit_charges_save_path,
#         charges=charges,
#         htvs_path=htvs_path,
#         htvs_details=htvs_details,
#     )

#     #this mol_dict should have everythin since the train_dataset has all the molecule already. no need for combined
#     mol_dict, pdb_file = load_pdb_create_mol_dict(save_path, train_dataset)


#     # Print basic information about the PDB file
#     print(f"Number of atoms: {pdb_file.topology.getNumAtoms()}")
#     print(f"Number of residues: {pdb_file.topology.getNumResidues()}")
#     print(f"Number of chains: {pdb_file.topology.getNumChains()}")
    
#     # Iterate through chains, residues, and atoms
#     for chain in pdb_file.topology.chains():
#         print(f"Chain {chain.index}")
#         for residue in chain.residues():
#             print(f"  Residue {residue.name} ({residue.index})")
#             for atom in residue.atoms():
#                 print(f"    Atom {atom.name} ({atom.element.symbol}), index {atom.index}")
                
#     # Print mol_dict before reindexing
#     print("Before reindexing:")
#     for key, mol in mol_dict.items():
#         print(f"Molecule key: {key}")
#         #for atom in mol["mol_pdb"]._residues[0]._atoms:
#             #print(f"Atom: {atom}, Name: {atom.name}, Element: {atom.element.symbol}")

#     atom_dict, atom_types_list = atom_name_reindexing(mol_dict)
    
#     # Print mol_dict after reindexing
#     print("\nAfter reindexing:")
#     for key, mol in mol_dict.items():
#         print(f"Molecule key: {key}")
#         #for atom in mol["mol_pdb"]._residues[0]._atoms:
#             #print(f"Atom: {atom}, Name: {atom.name}, Element: {atom.element.symbol}")
    
#     print(f"ATOM TYPES LIST {atom_types_list}")
#     print(f"ATOM DICT {atom_dict}")

#     # ONLY SCALE THE SALTS NOT THE POLYMERS!!!!! or code will FAIL
#     total_molecules = len(mol_dict)
#     for idx, mol_key in enumerate(mol_dict.keys()):
#         # Only scale for the last two molecules in the dictionary
#         if idx >= total_molecules - 2:
#             lammps_openmm_scaling = True
#         else:
#             lammps_openmm_scaling = False

#         # Print the molecule being processed
#         print(f"Processing molecule: {mol_key} (Scaling: {lammps_openmm_scaling})")

#         # Apply scaling based on the condition above
#         mol_dict[mol_key]["mol_hitpoly"] = param_scaling_openmm(
#             mol_dict[mol_key]["mol_hitpoly"], lammps_openmm_scaling
#     )

        
#     ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
#         mol_dict, atom_types_list
#     )
#     write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables)

#     ff_files, box_file, name_iterables, ff_file, topology_file = (
#         creating_ff_and_resid_files_gromacs(
#             mol_dict,
#             atom_types_list,
#             save_path,
#             pdb_file,
#             polymer_count,
#             concentration,
#         )
#     )
#     write_gromacs_files(
#         save_path=save_path,
#         ff_files=ff_files,
#         ff_file=ff_file,
#         box_file=box_file,
#         name_iterables=name_iterables,
#         topology_file=topology_file,
#     )
#     print(f"Box builder for - done!")

def create_box_and_ff_files_openmm(
    save_path,
    long_smiles,
    short_smiles,
    filename,
    polymer_count,
    concentration,
    packmol_path,
    atoms_short,
    atoms_long,
    atom_names_short,
    atom_names_long,
    param_dict,
    lit_charges_save_path,
    charges,
    charge_scale,
    htvs_path,
    htvs_details,
    salt_smiles,
    salt_paths,
    salt_data_paths,
    box_multiplier,
    salt,
    single_ion_conductor,
):
    poly_paths = [f"{save_path}/{name}" for name in filename]
    print(f"POLY PATHS HERE {poly_paths}")

    print(
        f"Creating and running packmol files with {polymer_count} chains and {concentration} LiTFSIs"
    )
    print(f"at {save_path} path")
    create_packmol_input_file(
        poly_paths,
        salt_paths,
        f"{save_path}/packmol.inp",
        f"{save_path}/packed_box.pdb",
        salt_smiles=salt_smiles,
        polymer_count=polymer_count,
        salt_concentrations=concentration,
        box_multiplier=box_multiplier,
        tolerance=5.0,
        salt=salt,
    )

    run_packmol(save_path, packmol_path)
    
    # Set counts as a list
    polymer_count = [polymer_count] * len(long_smiles)

    # Ensure polymer_count is correct
    if len(polymer_count) != len(long_smiles):
        raise ValueError("polymer_count length must match the number of long_smiles")

    # Ensure salt_count is correct
    if salt and len(concentration) != len(salt_smiles):
        raise ValueError("concentration length must match the number of salt_smiles")

    print(f"Making all the force field files for simulations")

    # Calculate poly_count as the number of unique SMILES in long_smiles
    print(f"total count of random polymers: {len(long_smiles)}")
    poly_count = len(set(long_smiles))
    print(f"Polymer count (number of unique polymers): {poly_count}")

    train_dataset = load_hitpoly_params(
        long_smiles,
        salt_smiles,
        salt_data_paths=salt_data_paths,
        charge_scale=charge_scale,
        poly_count=poly_count,
    )

    train_dataset = assign_lpg_params(
        short_smiles_list=short_smiles,
        atoms_long_list=atoms_long,
        atom_names_long_list=atom_names_long,
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        train_dataset=train_dataset,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        htvs_path=htvs_path,
        htvs_details=htvs_details,
    )

    #this mol_dict should have everythin since the train_dataset has all the molecule already. no need for combined
    mol_dict, pdb_file = load_pdb_create_mol_dict(save_path, train_dataset)


    # Print basic information about the PDB file
    print(f"Number of atoms: {pdb_file.topology.getNumAtoms()}")
    print(f"Number of residues: {pdb_file.topology.getNumResidues()}")
    print(f"Number of chains: {pdb_file.topology.getNumChains()}")
    
    # Iterate through chains, residues, and atoms
    for chain in pdb_file.topology.chains():
        print(f"Chain {chain.index}")
        for residue in chain.residues():
            print(f"  Residue {residue.name} ({residue.index})")
            for atom in residue.atoms():
                print(f"    Atom {atom.name} ({atom.element.symbol}), index {atom.index}")
                
    # Print mol_dict before reindexing
    print("Before reindexing:")
    for key, mol in mol_dict.items():
        print(f"Molecule key: {key}")
        #for atom in mol["mol_pdb"]._residues[0]._atoms:
            #print(f"Atom: {atom}, Name: {atom.name}, Element: {atom.element.symbol}")

    atom_dict, atom_types_list = atom_name_reindexing(mol_dict)
    
    # Print mol_dict after reindexing
    print("\nAfter reindexing:")
    for key, mol in mol_dict.items():
        print(f"Molecule key: {key}")
        #for atom in mol["mol_pdb"]._residues[0]._atoms:
            #print(f"Atom: {atom}, Name: {atom.name}, Element: {atom.element.symbol}")
    
    print(f"ATOM TYPES LIST {atom_types_list}")
    print(f"ATOM DICT {atom_dict}")

    # ONLY SCALE THE SALTS NOT THE POLYMERS!!!!! or code will FAIL
    total_molecules = len(mol_dict)

    if single_ion_conductor:
        molecules_to_scale = 1
    else:
        molecules_to_scale = 2

    for idx, mol_key in enumerate(mol_dict.keys()):
        if idx >= total_molecules - molecules_to_scale:
            lammps_openmm_scaling = True
        else:
            lammps_openmm_scaling = False

        # Print the molecule being processed
        print(f"Processing molecule: {mol_key} (Scaling: {lammps_openmm_scaling})")

        # Apply scaling based on the condition above
        mol_dict[mol_key]["mol_hitpoly"] = param_scaling_openmm(
            mol_dict[mol_key]["mol_hitpoly"], lammps_openmm_scaling
    )
        
    ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
        mol_dict, atom_types_list
    )
    write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables)

    print(f"Box builder for - done!")
    
