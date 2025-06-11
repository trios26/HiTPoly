import os
import time
import pandas as pd
import sys

sys.setrecursionlimit(5000)

results_path = (
    f"{os.path.expanduser('~')}/HiTPoly/results/LPG_COND_T415"
)
final_path = results_path

df = pd.read_csv("data/polymer_folders_T415_LPG_250110.csv")
# simu_type = "tg"
simu_type = None

for ind, (row, val) in enumerate(df.iterrows()):
    with open("smiles.smi", "w") as f:
        f.write(val["smiles"])

    try:
        command = [
            "python run_box_builder.py",
            "--results_path",
            f"{results_path}",
            "--smiles_path",
            "smiles.smi",
            "--lpg_repeats",
            f"{val['ligpargen_repeats']}",
            "--end_carbons",
            f"{val['add_end_Cs']}",
            "--repeats",
            f"{val['repeat units']}",
            "--polymer_count",
            f"{val['chains']}",
            "--concentration",
            f"{val['salt amount']}",
            "--poly_name",
            f"{val['polymer name']}",
            "--charge_scale",
            f"{val['charge scale']}",
            "--charge_type",
            f"{val['charge']}",
            "--final_path",
            f"{final_path}",
            "--box_multiplier",
            "0.3",
        ]
        if simu_type == "tg":
            command.append("--simu_type")
            command.append("tg")
        else:
            command.append("--temperature")
            command.append(f"{val['temperature']}")
        # if val["salt_in_simu"] == "False":
        # command.append("--salt_type")
        # command.append("None")
        command = " ".join(command)
        print(f"RUNNING {command}")
        os.system(command)

    except:
        folder_date = time.strftime("%y%m%d_H%HM%M")
        with open(f"error_file_{folder_date}.txt", "w+") as f:
            f.write(
                f"polymer {val['polymer name']} at concentration {val['salt amount']} failed to create files\n"
            )

    os.remove("smiles.smi")
