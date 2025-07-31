from sys import stdout
import os

from openmm.app import (
    PDBFile,
    Modeller,
    ForceField,
    Simulation,
    PME,
    HBonds,
    PDBReporter,
    StateDataReporter,
    CutoffNonPeriodic,
)
from openmm import (
    NoseHooverIntegrator,
    MonteCarloAnisotropicBarostat,
    MonteCarloBarostat,
    XmlSerializer,
    Platform,
)
from openmm.unit import *


def iniatilize_simulation(
    save_path,
    final_save_path,
    temperature,
    pressure=1,
    cuda_device="0",
    equilibration=False,
    barostat=False,
    timestep=0.001,
    polymer=False,
    barostat_type=None,
    fraction_froze=None,
):
    print(
        f"Initializing system at directory {save_path}, with T={temperature}, device={cuda_device}"
    )

    if polymer:
        packed_name = "polymer_conformation"
    else:
        packed_name = "packed_box"
    pdb = PDBFile(f"{save_path}/{packed_name}.pdb")

    bondDefinitionFiles = [f"{save_path}/force_field_resids.xml"]
    for ff in bondDefinitionFiles:
        pdb.topology.loadBondDefinitions(ff)
    pdb.topology.createStandardBonds()
    if equilibration:
        with open(f"{final_save_path}/equilibrated_state.xml") as input:
            equlibrated_state = XmlSerializer.deserialize(input.read())
        modeller = Modeller(pdb.topology, equlibrated_state.getPositions())
    else:
        modeller = Modeller(pdb.topology, pdb.positions)

    forceFieldFiles = [f"{save_path}/force_field.xml"]
    forcefield = ForceField(*forceFieldFiles)
    modeller.addExtraParticles(forcefield)
    if equilibration:
        if barostat_type:
            modeller.topology.setPeriodicBoxVectors(
                pdb.topology.getPeriodicBoxVectors()
            )
        else:
            modeller.topology.setPeriodicBoxVectors(
                [
                    equlibrated_state.getPeriodicBoxVectors()[0]._value,
                    equlibrated_state.getPeriodicBoxVectors()[1]._value,
                    equlibrated_state.getPeriodicBoxVectors()[2]._value,
                ]
            )
    elif polymer:
        print("Non periodic system for polymer relaxation")
    elif barostat_type:
        modeller.topology.setPeriodicBoxVectors(pdb.topology.getPeriodicBoxVectors())
    else:
        pbc_multiplier = 1.1
        max_box_dimension = pdb.positions.max().max()._value
        modeller.topology.setPeriodicBoxVectors(
            [
                [max_box_dimension * pbc_multiplier, 0, 0],
                [0, max_box_dimension * pbc_multiplier, 0],
                [0, 0, max_box_dimension * pbc_multiplier],
            ]
        )

    if polymer:
        cutoff = 0.3
    else:
        cutoff = 1.2

    if not polymer:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff * nanometer,
            constraints=HBonds,  # The lengths of all bonds that involve a hydrogen atom are constrained.
            # (Maybe similar to LINCS)
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=CutoffNonPeriodic,
            constraints=HBonds,  # The lengths of all bonds that involve a hydrogen atom are constrained.
            # (Maybe similar to LINCS)
        )

    if fraction_froze:
        print(f"Setting the masses of {fraction_froze} fraction")
        cutoff_z = (
            modeller.topology.getPeriodicBoxVectors().value_in_unit(nanometer)[-1].z
            * fraction_froze
        )

        pos_rn = pdb.getPositions()
        for i in range(system.getNumParticles()):
            if pos_rn[i].value_in_unit(nanometer).z < cutoff_z:
                system.setParticleMass(i, 10**10)

    integrator = NoseHooverIntegrator(
        temperature * kelvin,
        1 / picosecond,
        timestep * picoseconds,
    )

    if equilibration and not barostat or polymer:
        barostat = None
        barostat_id = None
    elif barostat_type == "isotropic":
        barostat = MonteCarloAnisotropicBarostat(
            (pressure, pressure, pressure) * bar, 300.0 * kelvin, False, False, True, 5
        )
        barostat_id = system.addForce(barostat)
    else:
        barostat = MonteCarloBarostat(pressure * bar, temperature * kelvin, 20)
        barostat_id = system.addForce(barostat)

    platform = Platform.getPlatformByName("CUDA")
    properties = {"CudaPrecision": "mixed"}
    print("CUDA DEVICE -", cuda_device)
    properties["DeviceIndex"] = cuda_device

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)

    if equilibration:
        simulation.loadState(f"{final_save_path}/equilibrated_state.xml")
    else:
        simulation.context.setPositions(modeller.positions)
        simulation.context.setVelocitiesToTemperature(temperature * kelvin)

    return integrator, barostat, barostat_id, simulation, system, modeller


def npt_pressure_change(
    integrator,
    simulation,
    ramp_time,
    temperature,
    barostat,
    pressure_start=1,
    pressure_end=1000,
    pressure_step=40,
):
    """
    ramp_time - fs
    """
    pressure_range = max([pressure_start, pressure_end])

    ramp_steps = int(ramp_time / (pressure_range / abs(pressure_step)))
    # NPT compression ramp
    integrator.setTemperature(temperature * kelvin)
    simulation.context.setParameter(barostat.Temperature(), temperature * kelvin)

    for i in range(round(pressure_range / abs(pressure_step))):
        # Take step, change pressure
        simulation.step(ramp_steps)
        if type(barostat) is MonteCarloAnisotropicBarostat:
            simulation.context.setParameter(
                barostat.PressureZ(),
                (pressure_start + (i * pressure_step)) * bar,
            )
        else:
            simulation.context.setParameter(
                barostat.Pressure(), (pressure_start + (i * pressure_step)) * bar
            )

    if type(barostat) is MonteCarloAnisotropicBarostat:
        simulation.context.setParameter(barostat.PressureZ(), pressure_end * bar)
    else:
        simulation.context.setParameter(barostat.Pressure(), pressure_end)


def npt_temperature_change(
    integrator,
    simulation,
    ramp_time,
    barostat,
    temp_start=350,
    temp_end=430,
    temp_step=5,
):
    high_temp = max([temp_end, temp_start])
    low_temp = min([temp_end, temp_start])

    ramp_steps = int(ramp_time / ((high_temp - low_temp) / abs(temp_step)))
    integrator.setTemperature(temp_start * kelvin)
    simulation.context.setParameter(barostat.Temperature(), temp_start * kelvin)

    for i in range(round((high_temp - low_temp) / abs(temp_step))):
        # Take step, change temperature
        simulation.step(ramp_steps)
        integrator.setTemperature((temp_start + (i * temp_step)) * kelvin)
        simulation.context.setParameter(
            barostat.Temperature(), (temp_start + (i * temp_step)) * kelvin
        )

    integrator.setTemperature(temp_end * kelvin)
    simulation.context.setParameter(barostat.Temperature(), temp_end * kelvin)


def npt_run(
    integrator,
    simulation,
    simu_time,
    temperature,
    pressure,
    barostat,
):
    integrator.setTemperature(temperature * kelvin)
    simulation.context.setParameter(barostat.Temperature(), temperature * kelvin)
    if type(barostat) is MonteCarloAnisotropicBarostat:
        simulation.context.setParameter(barostat.PressureZ(), pressure * bar)
    else:
        simulation.context.setParameter(barostat.Pressure(), pressure * bar)
    simulation.step(simu_time)


def equilibrate_polymer(
    save_path,
    simu_temp=353,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    name="polymer_conformation",
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=save_path,
            temperature=10,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=False,
            timestep=0.0005,
            polymer=True,
        )
    )
    print("Minimizing energy and saving polymer positions")
    simulation.minimizeEnergy()
    minpositions = simulation.context.getState(getPositions=True).getPositions(
        asNumpy=True
    )

    with open(f"{save_path}/{name}.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)


def equilibrate_system_1(
    save_path,
    final_save_path,
    barostat_type=None,
    simu_temp=353,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    timestep=0.001,
    fraction_froze=None,
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=10,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=False,
            timestep=timestep,
            barostat_type=barostat_type,
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{final_save_path}/equilibration.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=5000000,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{final_save_path}/equilibration.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )
    print("Minimizing energy and saving inital positions")
    for _ in range(10):
        simulation.minimizeEnergy()

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    with open(f"{save_path}/minimized_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)

    print(
        f"Small NPT (10ps) run at {int(timestep*1000)}fs stepsize with Temperature=1 and pressure=1"
    )
    npt_run(
        integrator,
        simulation,
        simu_time=20000,
        temperature=1,
        pressure=1,
        barostat=barostat,
    )

    simulation.minimizeEnergy()

    print(
        f"Small (100ps) npt ramp at {int(timestep*1000)}fs stepsize at Temperature=50 and pressure ramp from 1 to 100"
    )
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=500000,
        temperature=50,
        barostat=barostat,
        pressure_start=1,
        pressure_end=100,
        pressure_step=1,
    )

    print("Minimizing energy and saving first equilibrated output")
    simulation.minimizeEnergy()

    equil_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
    )
    equil_state_file = f"{final_save_path}/equilibrated_state.xml"
    with open(equil_state_file, "w") as f:
        f.write(XmlSerializer.serialize(equil_state))

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    modeller.topology.setPeriodicBoxVectors(equil_state.getPeriodicBoxVectors())
    with open(f"{save_path}/packed_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)


# has 2fs timestep
def equilibrate_system_2(
    save_path,
    final_save_path,
    barostat_type=None,
    simu_temp=393,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    timestep=0.002,
    fraction_froze=None,
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=simu_temp,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=True,
            barostat=True,
            timestep=timestep,
            barostat_type=barostat_type,
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{final_save_path}/equilibration_2.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=5000000,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{final_save_path}/equilibration_2.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )
    # print(f"NPT pressure ramp from 1 to 4000 bar and {simu_temp+70} K")
    # npt_pressure_change(
    #     integrator,
    #     simulation,
    #     ramp_time=500000,
    #     temperature=simu_temp + 70,
    #     pressure_start=simu_pressure,
    #     pressure_end=4000,
    #     pressure_step=40,
    # )
    print(f"Low pressure NPT hold at 1 bar, {simu_temp+70} K")
    npt_run(
        integrator,
        simulation,
        simu_time=500000,
        temperature=simu_temp + 270,
        pressure=1,
        barostat=barostat,
    )
    # print(f"NPT pressure ramp from 4000 to 1 bar, {simu_temp+70} K")
    # npt_pressure_change(
    #     integrator,
    #     simulation,
    #     ramp_time=500000,
    #     temperature=simu_temp + 70,
    #     pressure_start=4000,
    #     pressure_end=simu_pressure,
    #     pressure_step=-40,
    # )
    print(f"High pressure NPT hold at 100 bar, {simu_temp+70} K")
    npt_run(
        integrator,
        simulation,
        simu_time=500000,
        temperature=simu_temp + 270,
        pressure=100,
        barostat=barostat,
    )

    # print(f"NPT pressure ramp from 1 to 4000 bar and {simu_temp+20} K")
    # npt_pressure_change(
    #     integrator,
    #     simulation,
    #     ramp_time=500000,
    #     temperature=simu_temp + 20,
    #     pressure_start=simu_pressure,
    #     pressure_end=4000,
    #     pressure_step=40,
    # )
    print(f"Low pressure NPT hold at 1 bar, {simu_temp+20} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp + 120,
        pressure=1,
        barostat=barostat,
    )
    # print(f"NPT pressure ramp from 4000 to 1 bar, {simu_temp+20} K")
    # npt_pressure_change(
    #     integrator,
    #     simulation,
    #     ramp_time=500000,
    #     temperature=simu_temp + 20,
    #     pressure_start=4000,
    #     pressure_end=simu_pressure,
    #     pressure_step=-40,
    # )
    print(f"High pressure NPT hold at 100 bar, {simu_temp+20}K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp + 120,
        pressure=100,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp+20} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp + 120,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp+20} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp + 120,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp+20} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp + 120,
        pressure=1,
        barostat=barostat,
    )
    print(f"Low pressure NPT run at 1 bar, {simu_temp} K")
    npt_run(
        integrator,
        simulation,
        simu_time=1000000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )

    # NPT short run, don't need this
    # print(f"Adding barostat - {barostat_id}")
    # npt_run(
    #     integrator,
    #     simulation,
    #     simu_time=10000000, #10ns
    #     temperature=simu_temp,
    #     pressure=simu_pressure,
    # )

    # NVT short run
    # print(f"Removing barostat - {barostat_id}")
    # system.removeForce(barostat_id)
    # simulation.context.reinitialize(preserveState=True)
    # integrator.setTemperature(simu_temp * kelvin)
    # print("Short NVT equilibration run")
    # simulation.step(1000000)

    simulation.reporters.clear()
    print("Equilibration has finished, saving equilibartion final state!")

    equil_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
    )
    equil_state_file = f"{final_save_path}/equilibrated_state.xml"
    with open(equil_state_file, "w") as f:
        f.write(XmlSerializer.serialize(equil_state))

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    modeller.topology.setPeriodicBoxVectors(equil_state.getPeriodicBoxVectors())
    print(
        f"Post minimization box dimension along the X axis: {equil_state.getPeriodicBoxVectors()[0]._value}"
    )
    with open(f"{save_path}/packed_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)

    return integrator, simulation



def equilibrate_system_liquid1(
    save_path,
    final_save_path,
    barostat_type=None,
    simu_temp=353,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    timestep=0.001,
    fraction_froze=None,
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=10,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=False,
            timestep=timestep,
            barostat_type=barostat_type,
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{final_save_path}/equilibration.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=5000000,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{final_save_path}/equilibration.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )
    print("Minimizing energy and saving inital positions")
    for _ in range(10):
        simulation.minimizeEnergy()

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    with open(f"{save_path}/minimized_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)


    print(
        f"Small NPT (10ps) run at {int(timestep*1000)}fs stepsize with Temperature=1 and pressure=1"
    )
    npt_run(
        integrator,
        simulation,
        simu_time=20000,
        temperature=1,
        pressure=1,
        barostat=barostat,
    )

    simulation.minimizeEnergy()

    print(f"NPT temperature ramp from 1 K to {simu_temp}")
    npt_temperature_change(
        integrator,
        simulation,
        ramp_time=500000,
        temp_start=1,
        temp_end=simu_temp,
        temp_step=10,
        barostat=barostat,
    )
    """
    print(
        f"Small (100ps) npt ramp at {int(timestep*1000)}fs stepsize at Temperature=50 and pressure ramp from 1 to 100"
    )
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=500000,
        temperature=50,
        barostat=barostat,
        pressure_start=1,
        pressure_end=100,
        pressure_step=1,
    )
    """
    print("Minimizing energy and saving first equilibrated output")
    simulation.minimizeEnergy()

    equil_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
    )
    equil_state_file = f"{final_save_path}/equilibrated_state.xml"
    with open(equil_state_file, "w") as f:
        f.write(XmlSerializer.serialize(equil_state))

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    modeller.topology.setPeriodicBoxVectors(equil_state.getPeriodicBoxVectors())
    with open(f"{save_path}/packed_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)

# has 2fs timestep
def equilibrate_system_liquid2(
    save_path,
    final_save_path,
    barostat_type=None,
    simu_temp=393,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    timestep=0.002,
    fraction_froze=None,
):
    """
    For liquids, we do not need to use high pressure equilibration. So simply run NPT at 1 bar for sufficient time.
    """
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=simu_temp,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=True,
            barostat=True,
            timestep=timestep,
            barostat_type=barostat_type,
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{final_save_path}/equilibration_2.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=5000000,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{final_save_path}/equilibration_2.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )
    # print(f"NPT pressure ramp from 1 to 4000 bar and {simu_temp+70} K")
    # npt_pressure_change(
    #     integrator,
    #     simulation,
    #     ramp_time=500000,
    #     temperature=simu_temp + 70,
    #     pressure_start=simu_pressure,
    #     pressure_end=4000,
    #     pressure_step=40,
    # )


    print(f"Low pressure NPT run at 1 bar, {simu_temp} K")
    npt_run(
        integrator,
        simulation,
        simu_time=2500000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )

    print(f"temperature ramp from {simu_temp} to {simu_temp+50} K")
    npt_temperature_change(
        integrator,
        simulation,
        ramp_time=500000,
        temp_start=simu_temp,
        temp_end=simu_temp + 50,
        temp_step=10,
        barostat=barostat,
    )
    print(f"annealing at {simu_temp+50} K for 10 ns")
    npt_run(
        integrator,
        simulation,
        simu_time=2500000,
        temperature=simu_temp + 50,
        pressure=1,
        barostat=barostat,
    )

    print(f"temperature quenching from {simu_temp+50} to {simu_temp} K")
    npt_temperature_change(
        integrator,
        simulation,
        ramp_time=500000,
        temp_start=simu_temp + 50,
        temp_end=simu_temp,
        temp_step=-10,
        barostat=barostat,
    )
    print(f"annealing at {simu_temp} K for 10 ns")
    npt_run(
        integrator,
        simulation,
        simu_time=2500000,
        temperature=simu_temp,
        pressure=1,
        barostat=barostat,
    )
    simulation.reporters.clear()
    print("Equilibration has finished, saving equilibartion final state!")

    equil_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
    )
    equil_state_file = f"{final_save_path}/equilibrated_state.xml"
    with open(equil_state_file, "w") as f:
        f.write(XmlSerializer.serialize(equil_state))

    minpositions = simulation.context.getState(getPositions=True).getPositions()
    modeller.topology.setPeriodicBoxVectors(equil_state.getPeriodicBoxVectors())
    print(
        f"Post minimization box dimension along the X axis: {equil_state.getPeriodicBoxVectors()[0]._value}"
    )
    with open(f"{save_path}/packed_box.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, minpositions, f)

    return integrator, simulation


def equilibrate_system_IR(
    save_path,
    final_save_path,
    simu_temp=353,
    simu_pressure=1,
    logperiod=5000,
    mdOutputTime=5000,
    cuda_device="0",
    timestep=0.001,
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=simu_temp,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=True,
            barostat=True,
            timestep=timestep,
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{final_save_path}/equilibration_2.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=65000000,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{final_save_path}/equilibration_2.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )

    first_r_t = 200000
    pressure_step = 5
    print(f"NPT ramp for {first_r_t} fs with a {pressure_step}step")
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=first_r_t,
        temperature=simu_temp,
        pressure_start=0,
        pressure_end=1000,
        pressure_step=pressure_step,
    )
    simulation.minimizeEnergy()

    print("NPT pressure ramp from 1 to 4000 bar")
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=500000,
        temperature=simu_temp,
        pressure_start=simu_pressure,
        pressure_end=4000,
        pressure_step=40,
    )
    print("High pressure NPT run at 4000 bar")
    npt_run(
        integrator, simulation, simu_time=400000, temperature=simu_temp, pressure=4000
    )
    print("NPT pressure ramp from 4000 to 1 bar")
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=600000,
        temperature=simu_temp,
        pressure_start=4000,
        pressure_end=simu_pressure,
        pressure_step=-40,
    )
    print(f"NPT temperature ramp from {simu_temp} to {simu_temp+200} bar")
    npt_temperature_change(
        integrator,
        simulation,
        ramp_time=400000,
        temp_start=simu_temp,
        temp_end=simu_temp + 200,
        temp_step=10,
    )
    print(f"NPT temperature ramp from {simu_temp+200} to {simu_temp} bar")
    npt_temperature_change(
        integrator,
        simulation,
        ramp_time=400000,
        temp_start=simu_temp,
        temp_end=simu_temp + 200,
        temp_step=-10,
    )
    print("NPT pressure ramp from 1 to 4000 bar")
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=300000,
        temperature=simu_temp,
        pressure_start=simu_pressure,
        pressure_end=4000,
        pressure_step=40,
    )
    print("NPT pressure ramp from 4000 to 1 bar")
    npt_pressure_change(
        integrator,
        simulation,
        ramp_time=300000,
        temperature=simu_temp,
        pressure_start=4000,
        pressure_end=simu_pressure,
        pressure_step=-40,
    )

    # NPT short run
    print(f"Adding barostat - {barostat_id}")
    npt_run(
        integrator,
        simulation,
        simu_time=10000000,
        temperature=simu_temp,
        pressure=simu_pressure,
    )

    # NVT short run
    print(f"Removing barostat - {barostat_id}")
    system.removeForce(barostat_id)
    simulation.context.reinitialize(preserveState=True)
    integrator.setTemperature(simu_temp * kelvin)
    print("Short NVT equilibration run")
    simulation.step(1000000)

    simulation.reporters.clear()
    print("Equilibration has finished, saving equilibartion final state!")

    equil_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
    )
    equil_state_file = f"{final_save_path}/equilibrated_state.xml"
    with open(equil_state_file, "w") as f:
        f.write(XmlSerializer.serialize(equil_state))

    return integrator, simulation


def prod_run_nvt(
    save_path,
    final_save_path,
    # integrator,
    # simulation,
    simu_time,
    simu_temp=353,
    logperiod=5000,
    mdOutputTime=12500,
    timestep=0.002,
    extra_name=None,
    simu_pressure=1,
    cuda_device="0",
    fraction_froze=None,
):
    integrator, barostat, barostat_id, simulation, system, modeller = (
        iniatilize_simulation(
            save_path=save_path,
            final_save_path=final_save_path,
            temperature=simu_temp,
            pressure=simu_pressure,
            cuda_device=cuda_device,
            equilibration=True,
            barostat=False,
            timestep=timestep,
            fraction_froze=fraction_froze,
        )
    )

    # ns to ps
    simu_steps = (simu_time * 1000) / timestep

    simulation.reporters.append(
        StateDataReporter(
            f"{save_path}/simulation.log",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=simu_steps,
            separator="\t",
        )
    )

    simulation.reporters.append(
        PDBReporter(f"{save_path}/simu_output.pdb", mdOutputTime)
    )
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            mdOutputTime,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            density=True,
        )
    )

    integrator.setTemperature(simu_temp * kelvin)
    print(f"Starting production run for {simu_steps} timesteps")

    for i in range(4):
        simulation.step(simu_steps // 4)

        final_state = simulation.context.getState(
            getPositions=True, getVelocities=True, getParameters=True
        )

        final_state_file = (
            f"{save_path}/final_state_{int((i+1)*(simu_steps/1000000)//4)}.xml"
        )

        with open(final_state_file, "w") as f:
            f.write(XmlSerializer.serialize(final_state))

        if i > 0:
            prev_state_file = (
                f"{save_path}/final_state_{int(i*(simu_steps/1000000)//4)}.xml"
            )
            os.remove(prev_state_file)

        minpositions = simulation.context.getState(getPositions=True).getPositions()
        modeller.topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())
        print(
            f"Post minimization box dimension along the X axis: {final_state.getPeriodicBoxVectors()[0]._value}"
        )
        with open(f"{save_path}/final_state_box.pdb", "w") as f:
            PDBFile.writeFile(modeller.topology, minpositions, f)

#changed results path to save_path forn now
def write_analysis_script(
    save_path,
    results_path,
    repeat_units,
    cation,
    anion,
    platform,
    simu_temperature,
    prod_run_time,
    xyz_output=25,  # ps
    ani_name_rdf=None,
):
    if platform == "supercloud":
        with open(f"{save_path}/run_analysis.sh", "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write("# Load modules" + "\n")
            f.write("source /etc/profile" + "\n")
            f.write("source /home/gridsan/$USER/.bashrc" + "\n")
            f.write("source activate htvs" + "\n")
            f.write("\n")
            f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
            f.write(f"export DATA_PATH={results_path}" + "\n")
            f.write(f"export NAME=T{simu_temperature}" + "\n")
            f.write(
                f"python $HiTPoly/run_analysis_openmm.py -p $DATA_PATH -d {int(prod_run_time/2*3/4)}"
            )
            f.write(
                f" --repeat_units {repeat_units} -n $NAME -f {xyz_output} -temp {simu_temperature} --platform {platform}"
                + f" --cat {cation} --ani {anion} --ani_rdf {ani_name_rdf} \n"
            )

    elif platform == "engaging":
        with open(f"{results_path}/run_analysis.sh", "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write("# Load modules" + "\n")
            f.write("source /etc/profile" + "\n")
            f.write("source /home/$USER/.bashrc" + "\n")
            f.write("source activate htvs" + "\n")
            f.write("\n")
            f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
            f.write(f"export DATA_PATH={results_path}" + "\n")
            f.write(f"export NAME=T{simu_temperature}" + "\n")
            f.write(
                f"python $HiTPoly/run_analysis_openmm.py -p $DATA_PATH -d {int(prod_run_time/2*3/4)}"
            )
            f.write(
                f" -n $NAME -f {xyz_output} -temp {simu_temperature} --platform {platform}"
                + f" --cat {cation} --ani {anion} --ani_rdf {ani_name_rdf} \n"
            )
    elif platform == "perlmutter":
        with open(f"{save_path}/run_analysis.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=analyze_ffnet_cpu\n")
            f.write("#SBATCH -C cpu\n")
            f.write("#SBATCH --qos=debug\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks=1\n")
            f.write("#SBATCH --cpus-per-task=8\n")
            f.write("#SBATCH --time=00:30:00\n")
            f.write("#SBATCH --account=m5068\n")
            f.write("#SBATCH --output=ffnet_debug_cpu_%j.out\n")
            f.write("#SBATCH --error=ffnet_debug_cpu_%j.err\n")
            f.write("\n")
            f.write("#!/bin/bash" + "\n")
            f.write("# Load modules" + "\n")
            f.write("source /etc/profile" + "\n")
            f.write("source /home/gridsan/$USER/.bashrc" + "\n")
            f.write("source activate htvs" + "\n")
            f.write("\n")
            f.write("export HiTPoly=$HOME/HiTPoly" + "\n")
            f.write(f"export DATA_PATH={results_path}" + "\n")
            f.write(f"export NAME=T{simu_temperature}" + "\n")
            f.write(
                f"python $HiTPoly/run_analysis_openmm.py -p $DATA_PATH -d {int(prod_run_time/2*3/4)}"
            )
            f.write(
                f" --repeat_units {repeat_units} -n $NAME -f {xyz_output} -temp {simu_temperature} --platform {platform}"
                + f" --cat {cation} --ani {anion} --ani_rdf {ani_name_rdf} \n"
            )

##New
def tg_simulation(
    prod_run_time,   # ns
    start_temperature,
    end_temperature,
    temperature_step,
    save_path,
    initial_pdb,
    forcefield_files,
    platform,
    xyz_output=False,
):
    """
    Tg scan protocol using OpenMM, including a two-step equilibration
    followed by the original 11-stage NPT relaxations and temperature scan.

    Parameters:
    - prod_run_time: production run time in ns
    - start_temperature, end_temperature, temperature_step: K values for scan
    - save_path: directory for outputs
    - initial_pdb: path to packed_box.pdb
    - forcefield_files: list of ForceField XML filenames
    - platform: OpenMM Platform object
    - xyz_output: whether to write .xyz files
    """
    # 1) create directories
    final_save_path = os.path.join(save_path, "openmm_saver")
    if not os.path.isdir(final_save_path):
        os.makedirs(final_save_path)
        print(f"Created directory: {final_save_path}")

    # 2) standard equilibration scheme (pre-existing functions)
    equilibrate_system_1(
        save_path=save_path,
        final_save_path=final_save_path,
        simu_temp=start_temperature,
        simu_pressure=1,
    )
    equilibrate_system_2(
        save_path=save_path,
        final_save_path=final_save_path,
        simu_temp=start_temperature,
        simu_pressure=1,
    )

    # Load minimized & equilibrated box
    pdb = PDBFile(initial_pdb)
    ff = ForceField(*forcefield_files)
    modeller = Modeller(pdb.topology, pdb.positions)
    topology = modeller.topology
    last_positions = modeller.positions
    file_names = []

    # 3) Original NPT relaxation stages (11 runs × 2 ns)
    t1 = 400
    temps = [t1+70, t1+70, t1+20, t1+20, t1, t1+20, t1, t1+20, t1, t1+20, t1]
    pressures = [1, 100, 1, 100, 1, 1, 1, 1, 1, 1, 1]
    names = [f"npt{i+1}" for i in range(len(temps))]
    for ind, (temp, pres, name) in enumerate(zip(temps, pressures, names)):
        dt = 1.0 if temp > 700 else 2.0
        # build system + barostat
        system = ff.createSystem(
            topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds,
        )
        barostat = MonteCarloBarostat(pres*bar, temp*kelvin, MonteCarloAnisotropicBarostat)
        system.addForce(barostat)
        integrator = NoseHooverIntegrator(
            temp*kelvin,
            1.0/picosecond,
            dt*femtoseconds,
        )
        sim = Simulation(topology, system, integrator, platform)
        sim.context.setPositions(last_positions)
        interval = int((1.0*picosecond)/(dt*femtoseconds))
        sim.reporters.append(DCDReporter(f"{save_path}/{name}.dcd", interval))
        sim.reporters.append(
            StateDataReporter(
                f"{save_path}/{name}.log", interval,
                step=True, time=True, potentialEnergy=True,
                temperature=True, volume=True
            )
        )
        sim.step(int(2e6 / dt))  # 2 ns
        state = sim.context.getState(getPositions=True)
        with open(f"{save_path}/{name}.pdb", 'w') as f:
            PDBFile.writeFile(sim.topology, state.getPositions(), f)
        last_positions = state.getPositions()
        file_names.append(name)

    # 4) Production NPT temperature scan (extra 10 ns on first)
    scan_temps = np.arange(end_temperature, start_temperature+1, temperature_step)[::-1]
    for idx, T in enumerate(scan_temps):
        run_time = prod_run_time + 10 if idx == 0 else prod_run_time
        name = f"npt_prod_T{int(T)}"
        dt = 1.0 if T > 700 else 2.0
        system = ff.createSystem(
            topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds,
        )
        barostat = MonteCarloBarostat(1*bar, T*kelvin, MonteCarloAnisotropicBarostat)
        system.addForce(barostat)
        integrator = NoseHooverIntegrator(
            T*kelvin,
            1.0/picosecond,
            dt*femtoseconds,
        )
        sim = Simulation(topology, system, integrator, platform)
        sim.context.setPositions(last_positions)
        interval = int((1.0*picosecond)/(dt*femtoseconds))
        sim.reporters.append(DCDReporter(f"{save_path}/{name}.dcd", interval))
        sim.reporters.append(
            StateDataReporter(
                f"{save_path}/{name}.log", interval,
                step=True, time=True, potentialEnergy=True,
                temperature=True, volume=True
            )
        )
        if xyz_output:
            sim.reporters.append(PDBReporter(f"{save_path}/{name}.xyz", interval))
        sim.step(int(run_time * 1e6 / dt))
        state = sim.context.getState(getPositions=True)
        with open(f"{save_path}/{name}.pdb", 'w') as f:
            PDBFile.writeFile(sim.topology, state.getPositions(), f)
        last_positions = state.getPositions()
        file_names.append(name)

            print("Tg simulation complete — stages:", file_names)

    # 5) Run Tg analysis on the production data
    try:
        Tg_value = analyze_tg(
            save_path=save_path,
            temperatures=scan_temps,
        )
        print(f"Glass transition temperature (Tg) estimate: {Tg_value:.1f} K")
    except Exception as e:
        print(f"Tg analysis failed: {e}")


def analyze_tg(
    save_path,
    temperatures,
    discard_fraction=0.5,
    fit_ranges=None,
    output_plot="tg_analysis.png",
):
    """
    Analyze Tg from NPT production runs.

    Parameters:
    - save_path: directory where log files are stored
    - temperatures: list of temperatures (K) corresponding to npt_prod_T{T}.log files
    - discard_fraction: fraction of initial data to discard for equilibration
    - fit_ranges: tuple of two tuples defining low-T and high-T fit ranges, e.g. ((300,350),(700,800))
    - output_plot: filename for saving the density vs temperature plot

    Returns:
    - Tg_estimate: estimated Tg in K
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # Collect mean densities
    data = []
    for T in temperatures:
        logfile = os.path.join(save_path, f"npt_prod_T{int(T)}.log")
        df = pd.read_csv(logfile, sep=" ", comment='#', skipinitialspace=True)
        if 'density' not in df.columns:
            raise ValueError(f"Density column not found in {logfile}")
        # discard initial frames
        n = len(df)
        df2 = df.iloc[int(n*discard_fraction):]
        mean_rho = df2['density'].mean()
        data.append((T, mean_rho))
    df_data = pd.DataFrame(data, columns=['T','density'])
    df_data['spec_vol'] = 1.0/df_data['density']

    # Fit linear regimes
    if fit_ranges is None:
        low_range = (df_data['T'].min(), df_data['T'].min() + (df_data['T'].max()-df_data['T'].min())/3)
        high_range = (df_data['T'].max() - (df_data['T'].max()-df_data['T'].min())/3, df_data['T'].max())
    else:
        low_range, high_range = fit_ranges
    low_df = df_data[(df_data['T'] >= low_range[0]) & (df_data['T'] <= low_range[1])]
    high_df = df_data[(df_data['T'] >= high_range[0]) & (df_data['T'] <= high_range[1])]
    m1, b1 = np.polyfit(high_df['T'], high_df['spec_vol'], 1)
    m2, b2 = np.polyfit(low_df['T'],  low_df['spec_vol'], 1)
    # Intersection
    Tg = (b2 - b1) / (m1 - m2)

    # Plot
    plt.figure()
    plt.scatter(df_data['T'], df_data['spec_vol'], label='Data')
    xs = np.linspace(df_data['T'].min(), df_data['T'].max(), 100)
    plt.plot(xs, m1*xs + b1, '--', label='High-T fit')
    plt.plot(xs, m2*xs + b2, '--', label='Low-T fit')
    plt.axvline(Tg, color='k', linestyle=':', label=f'Tg = {Tg:.1f} K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Specific Volume (1/density)')
    plt.legend()
    plt.savefig(os.path.join(save_path, output_plot))
    plt.close()

    print(f"Estimated Tg = {Tg:.1f} K")
    return Tg

