from hitpoly.utils.geometry_calc import get_bond_geometry, get_angle_geometry
from hitpoly.utils.args import hitpolyArgs
from hitpoly.utils.constants import (
    SIGMA_ATOMS,
    EPSILON_ATOMS,
)
import random
import numpy as np
import torch


def initialize_discrete_params(dataset, train_args=hitpolyArgs):
    # Bond
    unique_types = dataset.bond_type.unique().tolist()
    num_types = len(unique_types)
    top_target, types = np.array([]), np.array([])
    ks, tops = [], []
    for molecule in dataset:
        top_target = np.concatenate(
            (
                top_target,
                get_bond_geometry(molecule.geometries, molecule.bonds)
                .flatten()
                .tolist(),
            )
        )
        # each molecule represents ONE species, but contains multiple geometries. Thus, the molecule.bond_type will only have
        # the types for one replica, so we need to copy that for every geometry so that there is abond type for every bond
        # (in every geometry for each species)
        types = np.concatenate(
            (types, molecule.bond_type.tolist() * molecule.geometries.shape[0])
        )
    for cur_type in unique_types:
        inx = np.where(types == cur_type)[0]
        ks.append(
            300
            * (
                1
                - (
                    random.random() * 2 * train_args.top_perturb_hyp
                    - train_args.top_perturb_hyp
                )
            )
        )
        tops.append(
            (
                (
                    1
                    - (
                        random.random() * 2 * train_args.top_perturb_hyp
                        - train_args.top_perturb_hyp
                    )
                )
                * top_target[inx].mean(0)
            )
        )
    dataset.bond_params = torch.vstack(
        [torch.tensor(ks), torch.tensor(tops)]
    ).transpose(1, 0)
    dataset.bond_params.type(torch.float)
    dataset.bond_params.requires_grad = True

    # Angle
    unique_types = dataset.angle_type.unique().tolist()
    num_types = len(unique_types)
    top_target, types = np.array([]), np.array([])
    ks, tops = [], []
    for molecule in dataset:
        top_target = np.concatenate(
            (
                top_target,
                get_angle_geometry(molecule.geometries, molecule.angles)[1]
                .flatten()
                .tolist(),
            )
        )
        types = np.concatenate(
            (types, molecule.angle_type.tolist() * molecule.geometries.shape[0])
        )

    for cur_type in unique_types:
        inx = np.where(types == cur_type)[0]
        ks.append(
            50
            * (
                1
                - (
                    random.random() * 2 * train_args.top_perturb_hyp
                    - train_args.top_perturb_hyp
                )
            )
        )
        tops.append(
            (
                (
                    1
                    - (
                        random.random() * 2 * train_args.top_perturb_hyp
                        - train_args.top_perturb_hyp
                    )
                )
                * top_target[inx].mean(0)
            )
        )

    dataset.angle_params = torch.vstack(
        [torch.tensor(ks), torch.tensor(tops)]
    ).transpose(1, 0)
    dataset.angle_params.type(torch.float)
    dataset.angle_params.requires_grad = True

    # Dihedral
    num_types = len(dataset.dihedral_type.unique())
    dataset.dihedral_params = torch.zeros(num_types, 4) + 0.01
    dataset.dihedral_params.type(torch.float)
    dataset.dihedral_params.requires_grad = True

    # Pair
    # params[0] - atomic charge (q)
    # params[1] - dielectric constant (LJ epsilon) (From tabulated data)
    # params[2] - LJ sigma (From tabulated data)
    unique_types = dataset.atom_type.unique().tolist()
    num_types = len(unique_types)
    top_target, types = np.array([]), np.array([])
    qs, Q_target = (
        [],
        [],
    )  # Q_target is the actual average, while qs is the initial guess (Q_target +- random noise)
    for molecule in dataset:
        top_target = np.concatenate((top_target, molecule.charges.flatten().tolist()))
        types = np.concatenate(
            (types, molecule.atom_type.tolist() * molecule.geometries.shape[0])
        )
    for cur_type in unique_types:
        inx = np.where(types == cur_type)[0]
        Q_target.append(top_target[inx].mean(0))

        qs.append(
            (
                1
                - (
                    random.random() * 2 * train_args.top_perturb_hyp
                    - train_args.top_perturb_hyp
                )
            )
            * top_target[inx].mean(0)
        )
    # get LJ params from lookup table
    sigmas, eps = [], []
    for cur_type in unique_types:
        atomic_num_of_type = np.array(dataset.atomic_nums)[
            np.where(dataset.atom_type == unique_types[cur_type])[0][0]
        ]
        sigmas.append(SIGMA_ATOMS[atomic_num_of_type])
        eps.append(EPSILON_ATOMS[atomic_num_of_type])

    # Improper
    num_types = len(dataset.improper_type.unique())
    dataset.improper_params = torch.zeros(num_types, 2) + 0.01
    dataset.improper_params.type(torch.float)
    dataset.improper_params.requires_grad = True
    dataset.electrostatic_params = torch.tensor(
        qs, dtype=torch.float, requires_grad=True
    )
    dataset.true_charge = torch.tensor(Q_target, dtype=torch.float, requires_grad=False)
    dataset.dispersion_params = torch.vstack(
        [torch.tensor(eps), torch.tensor(sigmas)]
    ).transpose(1, 0)
    dataset.dispersion_params.type(torch.float)
    dataset.dispersion_params.requires_grad = False
