import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def extract_density_info_npt(save_path):
    """
    Takes an input of the save_path and creates
    a bash script that saves all the xvg files
    with the energy, temperature, volume and density
    """
    with open(os.path.join(save_path, "analysis_script.sh"), "w") as f:
        f.write("#!/bin/bash" + "\n")
        f.write("#SBATCH --job-name=opv-md" + "\n")
        f.write("#SBATCH --partition=xeon-p8" + "\n")
        f.write("#SBATCH --nodes=1" + "\n")
        f.write("#SBATCH --ntasks-per-node=1" + "\n")
        f.write("#SBATCH --cpus-per-task=4" + "\n")
        f.write("#SBATCH --time=4-00:00:00" + "\n")
        f.write("\n")
        f.write("# Load modules" + "\n")
        f.write("module purge" + "\n")
        f.write("module use --append /data1/groups/rgb_shared/jnam/opt/modules" + "\n")
        f.write("module load gromacs/2023" + "\n")
        f.write("source /etc/profile" + "\n")
        f.write("source /home/gridsan/jruza/.bashrc" + "\n")
        f.write("conda activate htvs" + "\n")
        f.write("\n")
        f.write("# Function to search for string in files" + "\n")
        f.write('search_string="npt_prod"' + "\n")
        f.write('search_string2="edr"' + "\n")
        f.write("files=()" + "\n")
        f.write("# Iterate through files in the directory" + "\n")
        f.write("for file in ./*; do" + "\n")
        f.write("    # Check if file is a regular file" + "\n")
        f.write('    if [ -f "$file" ]; then' + "\n")
        f.write("        # Check if the file contains the search string" + "\n")
        f.write('        if [[ "$file" == *"$search_string"* ]]; then' + "\n")
        f.write('            if [[ "$file" == *"$search_string2"* ]]; then' + "\n")
        f.write('                files+=("$file")' + "\n")
        f.write("            fi" + "\n")
        f.write("        fi" + "\n")
        f.write("    fi" + "\n")
        f.write("done" + "\n")
        f.write("\n")
        f.write("# Iterate through the list of files" + "\n")
        f.write("# 10 total energy" + "\n")
        f.write("# 12 temperature" + "\n")
        f.write("# 19 volume" + "\n")
        f.write("# 20 density" + "\n")
        f.write('for file in "${files[@]}"; do' + "\n")
        f.write("    # Extract energy outputs using gmx energy command" + "\n")
        f.write(
            '    gmx_mpi energy -f "$file" -o "${file%.edr}_energy.xvg" -xvg none << EOF'
            + "\n"
        )
        f.write("10" + "\n")
        f.write("12" + "\n")
        f.write("19" + "\n")
        f.write("20" + "\n")
        f.write("EOF" + "\n")
        f.write("done" + "\n")
        f.write("\n")
