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


def initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
        initialize_simulation(
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
def tg_simulations(
    save_path,
    final_save_path,
    prod_run_time=5.0, #nanoseconds
    start_temperature=450.0,
    end_temperature=200.0,
    temperature_step=25.0
):
    """
    Perform Tg simulation temperature/pressure sweeps in OpenMM,
    mirroring the GROMACS scheme from ACS Appl. Polym. Mater. 2021 and ACS Macro Lett. 2023.

    Steps:
    1. Energy relaxation
    2. Short NPT conditioning cycles
    3. Production NPT runs scanning temperatures
    4. Analysis of density vs temperature
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(final_save_path, exist_ok=True)

    # 1. Energy relaxation
    print("Starting energy relaxation...")
    equilibrate_system_1(save_path=save_path, final_save_path=final_save_path)
    equilibrate_system_2(save_path=save_path, final_save_path=final_save_path)

    # 2. Short NPT conditioning cycles around t1
    t1 = start_temperature
    temps = [t1 + 70, t1 + 70, t1 + 20, t1 + 20, t1,
             t1 + 20, t1, t1 + 20, t1, t1 + 20, t1]
    pressures = [1, 100, 1, 100, 1, 1, 1, 1, 1, 1, 1]
    print("Running short NPT conditioning cycles...")
    for i, (temp, pres) in enumerate(zip(temps, pressures)):
        print(f" Cycle {i+1}: T={temp} K, P={pres} bar")
        integrator, barostat, barostat_id, simulation, system, modeller = \
            initialize_simulation(
                save_path=save_path,
                final_save_path=final_save_path,
                temperature=temp,
                pressure=pres,
                equilibration=True,
                barostat=True,
                timestep=0.001
            )
        # 2 ps at 1 fs timestep => 2000 steps
        npt_run(
            integrator, simulation,
            simu_time=2000,
            temperature=temp * unit.kelvin,
            pressure=pres * unit.bar,
            barostat=barostat
        )
        # save state & box
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getParameters=True
        )
        xml_out = os.path.join(final_save_path, f"state_cycle_{i+1}.xml")
        with open(xml_out, 'w') as f:
            f.write(XmlSerializer.serialize(state))
        pos = state.getPositions()
        PDBFile.writeFile(
            modeller.topology, pos,
            open(os.path.join(save_path, f"box_cycle_{i+1}.pdb"), 'w')
        )

    # 3. Production NPT runs scanning temperature
    print("Running production NPT runs...")
    scan_temps = np.arange(end_temperature, start_temperature + 1, temperature_step)[::-1]
    for idx, T in enumerate(scan_temps):
        print(f" Prod cycle {idx+1}: T={T} K")
        integrator, barostat, barostat_id, simulation, system, modeller = \
            initialize_simulation(
                save_path=save_path,
                final_save_path=final_save_path,
                temperature=T,
                pressure=1,
                equilibration=True,
                barostat=True,
                timestep=0.002
            )
        extra_ps = 10000 if idx == 0 else 0
        total_ps = prod_run_time * 1000 + extra_ps
        npt_run(
            integrator, simulation,
            simu_time=total_ps / 0.002,
            temperature=T * unit.kelvin,
            pressure=1 * unit.bar,
            barostat=barostat
        )
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getParameters=True
        )
        xml_out = os.path.join(save_path, f"state_prod_T{int(T)}.xml")
        with open(xml_out, 'w') as f:
            f.write(XmlSerializer.serialize(state))
        pos = state.getPositions()
        PDBFile.writeFile(
            modeller.topology, pos,
            open(os.path.join(save_path, f"box_prod_T{int(T)}.pdb"), 'w')
        )

    print("Tg simulation completed. All states and PDBs saved.")

    # 4. Analyze results automatically
    analyze_tg_results(final_save_path)


def analyze_tg_results(
    final_save_path: str,
    output_plot: str = "density_vs_temp.png"
):
    """
    Analyze Tg simulation results by extracting densities from production boxes
    and plotting density vs temperature to identify Tg.
    """
    temps = []
    densities = []
    for fname in os.listdir(final_save_path):
        if fname.startswith("box_prod_T") and fname.endswith(".pdb"):
            T = int(fname.replace("box_prod_T", "").replace(".pdb", ""))
            u = mda.Universe(os.path.join(final_save_path, fname))
            dims = u.atoms.dimensions
            vol_ang3 = dims[0] * dims[1] * dims[2]
            mass_amu = sum(atom.mass for atom in u.atoms)
            density = (mass_amu * 1.66054e-24) / (vol_ang3 * 1e-24)
            temps.append(T)
            densities.append(density)
    temps, densities = zip(*sorted(zip(temps, densities)))
    plt.figure()
    plt.plot(temps, densities, marker='o')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Density (g/cm³)")
    plt.title("Density vs Temperature")
    plt.tight_layout()
    output_path = os.path.join(final_save_path, output_plot)
    plt.savefig(output_path)
    print(f"Density vs Temperature plot saved to {output_path}")


