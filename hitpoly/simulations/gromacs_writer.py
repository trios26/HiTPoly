import os
import numpy as np


class GromacsWriter:
    def __init__(
        self,
        save_path,
        overall_save_path=None,
        name=None,
        simu_temp=430,
        simu_pressure=1,  # bar
        timestep=2,  # fs
        xyz_output=50,  # ps
    ):
        self.save_path = save_path
        self.overall_save_path = overall_save_path
        self.name = name
        self.simu_temp = simu_temp
        self.simu_pressure = simu_pressure
        self.timestep = timestep
        self.xyz_output = xyz_output

    def relaxation_writer(
        self,
    ):
        file = []
        file.append(f"{'integrator':<23}= steep")
        file.append(f"{'emtol':<23}= 500.0")
        file.append(f"{'emstep':<23}= 0.01")
        file.append(f"{'nsteps':<23}= 100000")
        file.append(f"{'nstlist':<23}= 1")
        file.append(f"{'cutoff-scheme':<23}= Verlet")
        file.append(f"{'coulombtype':<23}= PME")
        file.append(f"{'rcoulomb':<23}= 1.2")
        file.append(f"{'rvdw':<23}= 1.2")
        file.append(f"{'pbc':<23}= xyz")
        file.append(f"{'constraints':<23}= h-bonds")
        file.append(f"{'constraint-algorithm':<23}= lincs")
        return file

    def generic_props_writer(
        self,
        simulation_time,
        timestep,
        nstlist=20,
        initial=False,
    ):
        file = []
        if self.name:
            file.append(f"{'title':<23}= {self.name}")
        file.append(f"{'integrator':<23}= md-vv")
        file.append(
            f"{'nsteps':<23}= {int(float(simulation_time/timestep)*1000000)}"
        )  # in ns
        file.append(f"{'dt':<23}= {timestep/1000}")  # fs - ps
        if not initial:
            file.append(f"{'constraints':<23}= h-bonds")
            file.append(f"{'constraint-algorithm':<23}= lincs")
            file.append(f"{'lincs_iter':<23}= 1")
            file.append(f"{'lincs_order':<23}= 4")
        file.append(f"{'cutoff-scheme':<23}= Verlet")
        file.append(f"{'nstlist':<23}= 20")
        file.append(f"{'rcoulomb':<23}= 1.2")
        file.append(f"{'rvdw':<23}= 1.2")
        file.append(f"{'DispCorr':<23}= EnerPres")
        file.append(f"{'coulombtype':<23}= PME")
        file.append(f"{'pme_order':<23}= 4")
        file.append(f"{'fourierspacing':<23}= 0.5")
        file.append(f"{'pbc':<23}= xyz")
        return file

    def npt_run(
        self,
        simulation_time,
        simu_temperature=430,  # K
        simu_pressure=1,  # bar
        energy_output=5000,
        xyz_output=0,
        velocity_output=0,
        nstlist=20,
        initial=False,
        prod=False,
    ):
        if initial:
            sim_time = 1
            file = self.generic_props_writer(1, sim_time, initial)
        else:
            file = self.generic_props_writer(simulation_time, self.timestep)
        file.append(f"{'nstlog':<23}= {energy_output}")
        if prod:
            file.append(f"{'nstxout':<23}= {int(xyz_output*1000/self.timestep)}")
            file.append(f"{'nstvout':<23}= {int(velocity_output*1000/self.timestep)}")
        # Thermostat
        if prod:
            file.append(f"{'tcoupl':<23}= nose-hoover")
        else:
            file.append(f"{'tcoupl':<23}= V-rescale")
        file.append(f"{'tc-grps':<23}= System")
        file.append(f"{'tau_t':<23}= 0.1")
        file.append(f"{'ref_t':<23}= {simu_temperature}")
        # Barostat (same for prod and equilibration)
        file.append(f"{'pcoupl':<23}= C-rescale")
        file.append(f"{'pcoupltype':<23}= isotropic")
        file.append(f"{'tau_p':<23}= 1.0")
        file.append(f"{'ref_p':<23}= {simu_pressure}")
        file.append(f"{'compressibility':<23}= 4.5e-5")  # Maybe this has to be adjusted
        file.append(f"{'refcoord_scaling':<23}= com")
        return file

    def nvt_run(
        self,
        simulation_time,
        simu_temperature=430,  # K
        energy_output=5000,
        xyz_output=0,
        velocity_output=0,
    ):
        file = self.generic_props_writer(simulation_time, self.timestep)
        file.append(f"{'nstlog':<23}= {energy_output}")
        file.append(f"{'nstxout':<23}= {int(xyz_output*1000/self.timestep)}")
        file.append(f"{'nstvout':<23}= {int(velocity_output*1000/self.timestep)}")
        # Thermostat
        file.append(f"{'tcoupl':<23}= nose-hoover")
        file.append(f"{'tc-grps':<23}= System")
        file.append(f"{'tau_t':<23}= 1")
        file.append(f"{'ref_t':<23}= {simu_temperature}")
        return file

    def equil_and_prod_balsara(
        self,
        prod_run_time,  # ns
        simu_temperature,
        analysis=False,
        image_name=None,
        platform="local",
    ):
        # Equilibration scheme from the Malonate paper
        # https://pubs.acs.org/doi/10.1021/acsmacrolett.3c00041

        if not os.path.isdir(self.save_path):
            print(f"Creating the GROMACS directory at {self.save_path}")
            os.makedirs(self.save_path)

        file_names = []
        # Energy relaxation
        relax = self.relaxation_writer()
        relax_name = "ener_relax"
        file_names.append(relax_name)
        with open(f"{self.save_path}/{relax_name}.mdp", "w") as f:
            for i in relax:
                f.write(i + "\n")

        initial = True
        initial1 = True
        t1 = simu_temperature
        for ind, (temp, pres, name) in enumerate(
            zip(
                [
                    350,
                    t1 + 70,
                    t1 + 70,
                    t1 + 20,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                ],
                [
                    10,
                    1,
                    100,
                    1,
                    100,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    "npt_pre",
                    "npt1",
                    "npt2",
                    "npt3",
                    "npt4",
                    "npt5",
                    "npt6",
                    "npt7",
                    "npt8",
                    "npt9",
                    "npt10",
                    "npt11",
                ],
            )
        ):
            if ind > 2:
                initial = False
            npt = self.npt_run(
                simulation_time=2,
                simu_temperature=temp,
                simu_pressure=pres,
                initial=initial,
            )
            file_names.append(name)
            with open(f"{self.save_path}/{name}.mdp", "w") as f:
                for i in npt:
                    f.write(i + "\n")

        # Production NVT
        nvt = self.nvt_run(
            simulation_time=prod_run_time,
            simu_temperature=simu_temperature,
            xyz_output=self.xyz_output,
        )
        nvt_name = "nvt_prod"
        file_names.append(nvt_name)
        with open(f"{self.save_path}/{nvt_name}.mdp", "w") as f:
            for i in nvt:
                f.write(i + "\n")

        # write run_equil.sh file
        if platform == "supercloud":
            with open(f"{self.save_path}/run.sh", "w") as f:
                f.write("#!/bin/bash" + "\n")
                f.write("# Load modules" + "\n")
                f.write("module purge" + "\n")
                f.write(
                    "module use --append /data1/groups/rgb_shared/jnam/opt/modules"
                    + "\n"
                )
                f.write("module load gromacs/2023" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source /home/gridsan/$USER/.bashrc" + "\n")
                f.write("source activate htvs" + "\n")
                f.write("\n")
                f.write("mkdir -p simu")
                f.write("\n")
                for ind, i in enumerate(file_names[:3]):
                    output = "packed_box" if ind == 0 else "simu/" + file_names[ind - 1]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c {output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                    f.write("\n")
                f.write("\n")
                f.write("# Reload modules" + "\n")
                f.write("module purge" + "\n")
                f.write(
                    "module use --append /data1/groups/rgb_shared/jnam/opt/modules"
                    + "\n"
                )
                f.write("module load gromacs/2023.cuda" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source $HOME/.bashrc" + "\n")
                f.write("source activate htvs" + "\n")
                for ind, i in enumerate(file_names[3:]):
                    output = file_names[ind + 2]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c simu/{output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                    f.write("\n")
                f.write("\n")

            if analysis:
                with open(f"{self.save_path}/run_analysis.sh", "w") as f:
                    f.write("#!/bin/bash" + "\n")
                    f.write("# Load modules" + "\n")
                    f.write("module purge" + "\n")
                    f.write(
                        "module use --append /data1/groups/rgb_shared/jnam/opt/modules"
                        + "\n"
                    )
                    f.write("module load gromacs/2023" + "\n")
                    f.write("source /etc/profile" + "\n")
                    f.write("source /home/gridsan/$USER/.bashrc" + "\n")
                    f.write("source activate htvs" + "\n")
                    f.write("\n")
                    f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
                    f.write(f"export DATA_PATH={self.save_path}" + "\n")
                    f.write(f"export NAME={image_name}" + "\n")
                    f.write(
                        f"python $HiTPoly/run_analysis.py -p $DATA_PATH -d {int(prod_run_time/2*3/4)}"
                    )
                    f.write(
                        f" -n $NAME -f {self.xyz_output} -temp {simu_temperature} --gromacs --platform {platform}"
                        + "\n"
                    )
                    f.write(
                        "gmx_mpi energy -f simu/nvt_prod.edr -o results/potential.xvg -xvg none << EOF"
                        + "\n"
                    )
                    f.write("8" + "\n")
                    f.write("12" + "\n")
                    f.write("14" + "\n")
                    f.write("25" + "\n")
                    f.write("26" + "\n")
                    f.write("27" + "\n")
                    f.write("28" + "\n")
                    f.write("29" + "\n")
                    f.write("30" + "\n")
                    f.write("31" + "\n")
                    f.write("32" + "\n")
                    f.write("33" + "\n")
                    f.write("EOF" + "\n")

        if platform == "engaging":
            with open(f"{self.save_path}/run.sh", "w") as f:
                f.write("#!/bin/bash" + "\n")
                f.write("# Load modules" + "\n")
                f.write("module purge" + "\n")

                f.write("module load gcc" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source /home/$USER/.bashrc" + "\n")
                f.write("source activate htvs" + "\n")
                f.write("\n")
                f.write("mkdir -p simu")
                f.write("\n")
                for ind, i in enumerate(file_names[:3]):
                    output = "packed_box" if ind == 0 else "simu/" + file_names[ind - 1]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c {output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(
                        f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i} -ntomp 1" + "\n"
                    )
                    f.write("\n")
                f.write("\n")
                f.write("# Reload modules" + "\n")
                f.write("module purge" + "\n")

                f.write("module load gcc" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source $HOME/.bashrc" + "\n")
                f.write("source activate htvs" + "\n")
                for ind, i in enumerate(file_names[3:]):
                    output = file_names[ind + 2]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c simu/{output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(
                        f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i} -ntomp 1" + "\n"
                    )
                    f.write("\n")
                f.write("\n")

            if analysis:
                with open(f"{self.save_path}/run_analysis.sh", "w") as f:
                    f.write("#!/bin/bash" + "\n")
                    f.write("# Load modules" + "\n")
                    f.write("module purge" + "\n")

                    f.write("source /etc/profile" + "\n")
                    f.write("source /home/$USER/.bashrc" + "\n")
                    f.write("source activate htvs" + "\n")
                    f.write("module load gcc" + "\n")
                    f.write("\n")
                    f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
                    f.write(f"export DATA_PATH={self.save_path}" + "\n")
                    f.write(f"export NAME={image_name}" + "\n")
                    f.write(
                        f"python $HiTPoly/run_analysis.py -p $DATA_PATH -d {int(prod_run_time/2*3/4)}"
                    )
                    f.write(
                        f" -n $NAME -f {self.xyz_output} -temp {simu_temperature} --gromacs --platform {platform}"
                        + "\n"
                    )
                    f.write(
                        "gmx_mpi energy -f simu/nvt_prod.edr -o results/potential.xvg -xvg none << EOF"
                        + "\n"
                    )
                    f.write("8" + "\n")
                    f.write("12" + "\n")
                    f.write("14" + "\n")
                    f.write("25" + "\n")
                    f.write("26" + "\n")
                    f.write("27" + "\n")
                    f.write("28" + "\n")
                    f.write("29" + "\n")
                    f.write("30" + "\n")
                    f.write("31" + "\n")
                    f.write("32" + "\n")
                    f.write("33" + "\n")
                    f.write("EOF" + "\n")

        elif platform == "local":
            with open(f"{self.save_path}/run_equil.sh", "w") as f:
                f.write("#!/bin/bash" + "\n")
                if platform == "local":
                    f.write("#SBATCH --job-name=poly-electro-md" + "\n")
                    f.write("#SBATCH --partition=xeon-p8" + "\n")
                    f.write("#SBATCH --nodes=1" + "\n")
                    f.write("#SBATCH --ntasks-per-node=1" + "\n")
                    f.write("#SBATCH --cpus-per-task=48" + "\n")
                    f.write("#SBATCH --time=4-00:00:00" + "\n")
                f.write("\n")
                f.write("# Load modules" + "\n")
                f.write("module purge" + "\n")
                f.write(
                    "module use --append /data1/groups/rgb_shared/jnam/opt/modules"
                    + "\n"
                )
                f.write("module load gromacs/2023" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source $HOME/.bashrc" + "\n")
                f.write("conda activate htvs" + "\n")
                f.write("\n")
                f.write("mkdir -p simu")
                f.write("\n")
                for ind, i in enumerate(file_names[:2]):
                    output = "packed_box" if ind == 0 else "simu/" + file_names[ind - 1]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c {output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                    f.write("\n")
                f.write("\n")
                if platform == "local":
                    f.write("sbatch run.sh" + "\n")

            # write run.sh file
            with open(f"{self.save_path}/run.sh", "w") as f:
                f.write("#!/bin/bash" + "\n")
                f.write("#SBATCH --job-name=poly-electro-md" + "\n")
                f.write("#SBATCH --partition=xeon-g6-volta" + "\n")
                f.write("#SBATCH --nodes=1" + "\n")
                f.write("#SBATCH --ntasks-per-node=1" + "\n")
                f.write("#SBATCH --cpus-per-task=40" + "\n")
                f.write("#SBATCH --gres=gpu:volta:1" + "\n")
                f.write("#SBATCH --time=4-00:00:00" + "\n")
                f.write("\n")
                f.write("# Load modules" + "\n")
                f.write("module purge" + "\n")
                f.write(
                    "module use --append /data1/groups/rgb_shared/jnam/opt/modules"
                    + "\n"
                )
                f.write("module load gromacs/2023.cuda" + "\n")
                f.write("source /etc/profile" + "\n")
                f.write("source $HOME/.bashrc" + "\n")
                f.write("conda activate htvs" + "\n")
                f.write("\n")
                for ind, i in enumerate(file_names[2:]):
                    output = file_names[ind + 1]
                    f.write(f"gmx_mpi grompp -f {i}.mdp -c simu/{output}.gro ")
                    f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                    f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                    f.write("\n")
                f.write("\n")
                if analysis:
                    f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
                    f.write(f"export NAME={image_name}" + "\n")
                    f.write(
                        f"python $HiTPoly/run_analysis.py -p $HiTPoly/{self.overall_save_path} -d {int(prod_run_time/2*3/4)}"
                    )
                    f.write(
                        f" -n $NAME -f {self.xyz_output} -temp {simu_temperature} --gromacs"
                        + "\n"
                    )

    def tg_simulations(
        self,
        prod_run_time,  # ns
        start_temperature,
        end_temperature,
        temperature_step,
        analysis=False,
        image_name=None,
    ):
        # Tg simulation scheme, a mix from the Schrodinger and Balsara malonate papers
        # https://pubs.acs.org/doi/full/10.1021/acsapm.0c00524
        # https://pubs.acs.org/doi/10.1021/acsmacrolett.3c00041

        # When scanning above 700K it is recommended to use 1fs timestep

        if not os.path.isdir(self.save_path):
            print(f"Creating the GROMACS directory at {self.save_path}")
            os.makedirs(self.save_path)

        file_names = []
        # Energy relaxation
        relax = self.relaxation_writer()
        relax_name = "ener_relax"
        file_names.append(relax_name)
        with open(f"{self.save_path}/{relax_name}.mdp", "w") as f:
            for i in relax:
                f.write(i + "\n")

        initial = True
        # Hardcoding this temperature boundary
        t1 = 400
        for ind, (temp, pres, name) in enumerate(
            zip(
                [
                    t1 + 70,
                    t1 + 70,
                    t1 + 20,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                    t1 + 20,
                    t1,
                ],
                [
                    1,
                    100,
                    1,
                    100,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    "npt1",
                    "npt2",
                    "npt3",
                    "npt4",
                    "npt5",
                    "npt6",
                    "npt7",
                    "npt8",
                    "npt9",
                    "npt10",
                    "npt11",
                ],
            )
        ):
            if ind > 1:
                initial = False
            npt = self.npt_run(
                simulation_time=2,
                simu_temperature=temp,
                simu_pressure=pres,
                initial=initial,
            )
            file_names.append(name)
            with open(f"{self.save_path}/{name}.mdp", "w") as f:
                for i in npt:
                    f.write(i + "\n")

        # Production NPTs
        for ind, t in enumerate(
            np.arange(end_temperature, start_temperature + 1, temperature_step)[::-1]
        ):
            # First iteration run a longer simlation to let the box properly
            #  equilibrate on the high temperature range
            if ind == 0:
                simu_run_time = prod_run_time + 10
            else:
                simu_run_time = prod_run_time
            npt = self.npt_run(
                simulation_time=simu_run_time,
                simu_temperature=t,
                xyz_output=self.xyz_output,
                prod=True,
            )
            npt_name = f"npt_prod_T{int(t)}"
            file_names.append(npt_name)
            with open(f"{self.save_path}/{npt_name}.mdp", "w") as f:
                for i in npt:
                    f.write(i + "\n")

        # write run_equil.sh file
        with open(f"{self.save_path}/run_equil.sh", "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write("#SBATCH --job-name=poly-electro-md" + "\n")
            f.write("#SBATCH --partition=xeon-p8" + "\n")
            f.write("#SBATCH --nodes=1" + "\n")
            f.write("#SBATCH --ntasks-per-node=1" + "\n")
            f.write("#SBATCH --cpus-per-task=48" + "\n")
            f.write("#SBATCH --time=4-00:00:00" + "\n")
            f.write("\n")
            f.write("# Load modules" + "\n")
            f.write("module purge" + "\n")
            f.write(
                "module use --append /data1/groups/rgb_shared/jnam/opt/modules" + "\n"
            )
            f.write("module load gromacs/2023" + "\n")
            f.write("source /etc/profile" + "\n")
            f.write("source /home/gridsan/$USER/.bashrc" + "\n")
            f.write("conda activate htvs" + "\n")
            f.write("\n")
            f.write("mkdir -p simu")
            f.write("\n")
            for ind, i in enumerate(file_names[:3]):
                output = "packed_box" if ind == 0 else "simu/" + file_names[ind - 1]
                f.write(f"gmx_mpi grompp -f {i}.mdp -c {output}.gro ")
                f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                f.write("\n")
            f.write("\n")
            f.write("sbatch run.sh" + "\n")

        # write run.sh file
        with open(f"{self.save_path}/run.sh", "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write("#SBATCH --job-name=poly-electro-md" + "\n")
            f.write("#SBATCH --partition=xeon-g6-volta" + "\n")
            f.write("#SBATCH --nodes=1" + "\n")
            f.write("#SBATCH --ntasks-per-node=1" + "\n")
            f.write("#SBATCH --cpus-per-task=40" + "\n")
            f.write("#SBATCH --gres=gpu:volta:1" + "\n")
            f.write("#SBATCH --time=4-00:00:00" + "\n")
            f.write("\n")
            f.write("# Load modules" + "\n")
            f.write("module purge" + "\n")
            f.write(
                "module use --append /data1/groups/rgb_shared/jnam/opt/modules" + "\n"
            )
            f.write("module load gromacs/2023.cuda" + "\n")
            f.write("source /etc/profile" + "\n")
            f.write("source /home/gridsan/$USER/.bashrc" + "\n")
            f.write("conda activate htvs" + "\n")
            f.write("\n")
            for ind, i in enumerate(file_names[3:]):
                output = file_names[ind + 2]
                f.write(f"gmx_mpi grompp -f {i}.mdp -c simu/{output}.gro ")
                f.write(f"-p packed_box.top -o simu/{i}.tpr -maxwarn 2" + "\n")
                f.write(f"mpirun -np 1 gmx_mpi mdrun -deffnm simu/{i}" + "\n")
                f.write("\n")
            f.write("\n")
