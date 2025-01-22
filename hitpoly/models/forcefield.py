import time
import torch
import numpy as np
from hitpoly.utils.args import hitpolyArgs
from hitpoly.utils.geometry_calc import *
from hitpoly.utils.constants import *
from hitpoly.utils.useful_math import mse


class ForceFieldCalc:
    def __init__(self, args: hitpolyArgs):
        self.args = args

    def calculate(
        self,
        ff_type,
        prop,
        molecule,
        geom_count,
        debug=False,
        test_geoms=False,
        geometries=None,
        true_property=None,
    ):
        if not geometries.any():
            if test_geoms:
                geometries, true_property = molecule.sample_geometries(
                    geom_count, prop, test_geoms
                )
            else:
                geometries, true_property = molecule.sample_geometries(geom_count, prop)
        true_property = true_property.to(self.args.device)
        geom_size = geometries.shape[0]
        if prop == "energy":
            pred_property = torch.zeros(geom_size).to(self.args.device)
        elif prop == "force":
            pred_property = torch.zeros(geom_size, molecule.atom_num, 3).to(
                self.args.device
            )  # Forces are 3N

        pred_property += self.bonds(
            geometries, molecule.bonds, molecule.bond_params, prop, ff_type
        )
        pred_property += self.angles(
            geometries, molecule.angles, molecule.angle_params, prop, ff_type
        )
        pred_property += self.dihedrals(
            geometries, molecule.dihedrals, molecule.dihedral_params, prop, ff_type
        )
        coloumb, lj = self.pairs(
            geometries, molecule.pairs, molecule.pair_params, prop, ff_type
        )
        pred_property += coloumb
        pred_property += lj

        return true_property, pred_property  # , molecule.charges.mean(axis=0)

    def bonds(self, geometry, bonds, params, prop, ff_type):
        r0 = get_bond_geometry(geometry, bonds)
        r0_force_vec = get_bond_derivatives(geometry, bonds)
        if ff_type == "opls":
            # https://docs.lammps.org/bond_harmonic.html
            # From model.py len(params) = 2
            # https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.23897
            # Linear basis functions
            # params[0] - prediceted energy param K1, energy/distance^2
            # params[1] - prediceted energy param K2, energy/distance^2
            # params[2] - distance param min, r1
            # params[3] - distance param max, r2

            # returns separate energy for each bond
            if prop == "energy":
                return (params[:, 0] * (params[:, 1] - r0).pow(2)).sum(-1)
                # return ((params[:,0]*(params[:,2]-r0).pow(2))
                #         + (params[:,1]*(params[:,3]-r0).pow(2))).sum()
            elif prop == "force":
                f = 0.0
                total = torch.zeros(geometry.shape[0], geometry.shape[1], 3).to(
                    self.args.device
                )
                f = 2 * params[:, 0] * (params[:, 1] - r0).pow(1)
                # f += (2*params[:,0]*(params[:,2]-r0).pow(1)+2*params[:,1]*(params[:,3]-r0).pow(1)).to(self.args.device)
                bonds = bonds.view(-1, 1).expand(geometry.shape[0], 2 * len(bonds), 3)
                if len(bonds) != 0:
                    f = (f.view(geometry.shape[0], -1, 1, 1) * r0_force_vec).view(
                        geometry.shape[0], -1, 3
                    )
                    total = torch.zeros_like(total).scatter_add(1, bonds, f)
                return total
        else:
            raise NameError("Only OPLS available")

    def angles(self, geometry, angles, params, prop, ff_type):
        cos_theta0, theta0, r0 = get_angle_geometry(geometry, angles)
        theta0_force_vec, r0_force_vec = get_angle_derivatives(geometry, angles)

        if ff_type == "opls":
            # https://docs.lammps.org/angle_harmonic.html
            # From model.py len(params) = 2
            # params[0] - prediceted energy param K1, energy
            # params[1] - prediceted energy param K2, energy
            # params[2] - angle min, theta1 (calculation has to be done in radians)
            # params[3] - angle max, theta2 (calculation has to be done in radians)

            # returns separate energy for each angle
            if prop == "energy":
                return (params[:, 0] * (params[:, 1] - theta0).pow(2)).sum(-1)
                # return ((params[:,0]*(params[:,2]-theta0).pow(2))
                #         + (params[:,1]*(params[:,3]-theta0).pow(2))).sum()
            elif prop == "force":
                f = 0.0
                total = torch.zeros(geometry.shape[0], geometry.shape[1], 3).to(
                    self.args.device
                )
                f += 2 * params[:, 0] * (params[:, 1] - theta0).pow(1)
                # f += (2*params[:,0]*(params[:,2]-theta0).pow(1)+2*params[:,1]*(params[:,3]-theta0).pow(1)).to(self.args.device)
                angles = angles.view(-1, 1).expand(
                    geometry.shape[0], 3 * len(angles), 3
                )
                if len(angles) != 0:
                    f = (f.view(geometry.shape[0], -1, 1, 1) * theta0_force_vec).view(
                        geometry.shape[0], -1, 3
                    )
                    total = torch.zeros_like(total).scatter_add(1, angles, f)
                return total
        else:
            raise NameError("Only OPLS available")

    def dihedrals(self, geometry, dihedrals, params, prop, ff_type):
        cos_phi0, phi0, theta0, r0 = get_dihedral_geometry(
            geometry, dihedrals
        )  # output in radians
        phi0_force_vec, theta0_force_vec, r0_force_vec = get_dihedral_derivatives(
            geometry, dihedrals
        )

        if ff_type == "opls":
            # https://docs.lammps.org/dihedral_opls.html
            # From model.py len(params) = 4
            # params[0] - K1, energy
            # params[1] - K2, energy
            # params[2] - K3, energy
            # params[3] - K4, energy

            if prop == "energy":
                energy = (
                    0.5 * params[:, 0] * (1 + torch.cos(phi0))
                    + 0.5 * params[:, 1] * (1 - torch.cos(2 * phi0))
                    + 0.5 * params[:, 2] * (1 + torch.cos(3 * phi0))
                    + 0.5 * params[:, 3] * (1 - torch.cos(4 * phi0))
                ).to(self.args.device)

                # returns separate energy for each dihedral
                return energy.sum(-1)
            elif prop == "force":
                f = 0.0
                total = torch.zeros(geometry.shape[0], geometry.shape[1], 3).to(
                    self.args.device
                )
                f += (
                    0.5 * params[:, 0] * (-1) * 1 * torch.sin(phi0)
                    + 0.5 * params[:, 1] * (+1) * 2 * torch.sin(2 * phi0)
                    + 0.5 * params[:, 2] * (-1) * 3 * torch.sin(3 * phi0)
                    + 0.5 * params[:, 3] * (+1) * 4 * torch.sin(4 * phi0)
                )
                dihedrals = dihedrals.view(-1, 1).expand(
                    geometry.shape[0], 4 * len(dihedrals), 3
                )
                if len(dihedrals) != 0:
                    f = (f.view(geometry.shape[0], -1, 1, 1) * phi0_force_vec).view(
                        geometry.shape[0], -1, 3
                    )
                    total = torch.zeros_like(total).scatter_add(1, dihedrals, f)
                return total
        else:
            raise NameError("Only OPLS available")

    def pairs(self, geometry, pairs, params, prop, ff_type):
        r0 = get_bond_geometry(geometry, pairs)
        r0_force_vec = get_bond_derivatives(geometry, pairs)
        if ff_type == "opls":
            # ~ https://docs.lammps.org/pair_lj_cut_coul.html
            # From model.py len(params) = 1
            # params[0] - atomic charge (q)
            # params[1] - dielectric constant (LJ epsilon) (From tabulated data)
            # params[2] - LJ sigma (From tabulated data)

            charge_dict = dict(zip(np.arange(len(params[:, 0])), params[:, 0]))
            epsilon_dict = dict(zip(np.arange(len(params[:, 1])), params[:, 1]))
            sigma_dict = dict(zip(np.arange(len(params[:, 2])), params[:, 2]))

            sigmas_1 = torch.tensor(list(map(sigma_dict.get, pairs[:, 0].tolist()))).to(
                self.args.device
            )
            sigmas_2 = torch.tensor(list(map(sigma_dict.get, pairs[:, 1].tolist()))).to(
                self.args.device
            )

            epsilons_1 = torch.tensor(
                list(map(epsilon_dict.get, pairs[:, 0].tolist()))
            ).to(self.args.device)
            epsilons_2 = torch.tensor(
                list(map(epsilon_dict.get, pairs[:, 1].tolist()))
            ).to(self.args.device)

            charges_1 = torch.tensor(
                list(map(charge_dict.get, pairs[:, 0].tolist()))
            ).to(self.args.device)
            charges_2 = torch.tensor(
                list(map(charge_dict.get, pairs[:, 1].tolist()))
            ).to(self.args.device)

            sigmas = 1 / 2 * (sigmas_1 + sigmas_2)
            epsilons = (epsilons_1 * epsilons_2).sqrt()

            inv_r0 = r0.pow(-1)

            lj_12 = (sigmas * inv_r0).pow(12)
            lj_6 = (sigmas * inv_r0).pow(6)

            if prop == "energy":
                coloumb = (COUL_CONST * charges_1 * charges_2 * inv_r0 / EPSILON_R).sum(
                    -1
                )
                lj = (4 * epsilons * (lj_12 - lj_6)).sum(-1)

            elif prop == "force":
                f_coul = 0.0
                f_lj = 0.0
                coloumb = torch.zeros(geometry.shape[0], geometry.shape[1], 3).to(
                    self.args.device
                )
                f_coul += COUL_CONST * charges_1 * charges_2 * inv_r0.pow(2) / EPSILON_R

                lj = torch.zeros(geometry.shape[0], geometry.shape[1], 3).to(
                    self.args.device
                )
                f_lj += 48 * epsilons * inv_r0 * (lj_12 - 0.5 * lj_6)

                pairs = pairs.view(-1, 1).expand(geometry.shape[0], 2 * len(pairs), 3)

                if len(pairs) != 0:
                    f_coul = (
                        f_coul.view(geometry.shape[0], -1, 1, 1) * r0_force_vec
                    ).view(geometry.shape[0], -1, 3)
                    coloumb = torch.zeros_like(coloumb).scatter_add(1, pairs, f_coul)

                    f_lj = (f_lj.view(geometry.shape[0], -1, 1, 1) * r0_force_vec).view(
                        geometry.shape[0], -1, 3
                    )
                    lj = torch.zeros_like(lj).scatter_add(1, pairs, f_lj)

            # returns separate energy for each pair, and separate for coloumb and lj interactions
            return coloumb, lj
