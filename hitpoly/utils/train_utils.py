import torch
from hitpoly.writers.file_writers import (
    lammps_writer,
    training_details_writer,
    loss_writer,
)
import signal, sys


def calc_ff(
    molecule,
    force_field,
    train_args,
    calc,
    batch,
    lammps=False,
    name=None,
    full_path=None,
    debug=False,
    test_geoms=False,
    predict=False,
    epoch=None,
):

    true_batch, pred_batch, true_charges = force_field.calculate(
        train_args.ff_type,
        calc,
        molecule,
        batch,
        debug=debug,
        test_geoms=test_geoms,
    )

    if lammps:
        if train_args.discrete_flag:
            pair_params = torch.vstack(
                [
                    molecule.electrostatic_params.view(-1, 1).transpose(1, 0).detach(),
                    molecule.dispersion_params[:, 0].detach(),
                    molecule.dispersion_params[:, 1].detach(),
                ]
            ).transpose(1, 0)
        else:
            pair_params = molecule.pair_params.detach()
        lammps_writer(
            f"{full_path}/mol_{name}_{molecule.species_id}.lmp",
            molecule.bond_params.detach(),
            molecule.angle_params.detach(),
            molecule.dihedral_params.detach(),
            molecule.improper_params.detach(),
            pair_params,
            molecule,
        )
    return true_batch, pred_batch, true_charges.to(train_args.device)


def extra_loss_calc(params, param_type: str, extra_loss, train_args):
    extra_loss = 0
    if param_type == "bond":
        spread = train_args.bond_scaling_param_1
        mean = train_args.bond_scaling_param_2
        penalty = train_args.bond_extra_penalty_term
    elif param_type == "angle":
        spread = train_args.angle_scaling_param_1
        mean = train_args.angle_scaling_param_2
        penalty = train_args.angle_extra_penalty_term
    if params.min() < (mean - spread) / 1.5:  # Soft boundary for the spread
        extra_loss += ((mean - spread) - params.min()).pow(2) * penalty
    if params.max() > (mean + spread) * 1.5:
        extra_loss += (params.min() - (mean + spread)).pow(2) * penalty
    return extra_loss


def train_test_split_df(df, frac=0.2, state=1):
    # get random sample
    test = df.sample(frac=frac, axis=0, random_state=state)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test
