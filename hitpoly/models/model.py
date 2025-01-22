import torch
import torch.nn as nn
from hitpoly.data.builder import TopologyDataset
from hitpoly.utils.args import hitpolyArgs
from hitpoly.utils.useful_math import scaler, charge_scaler
import copy


class HiTPoly(nn.Module):
    def __init__(
        self,
        args: hitpolyArgs,
    ):
        nn.Module.__init__(self)
        self.train_args = args
        self.make_encoder(self.train_args)
        self.make_paramnets(self.train_args)
        # initialize_weights(self)

    def make_encoder(self, train_args):
        self.encoder = MPN(train_args)

    def make_paramnets(self, train_args):
        if train_args.sep_bond_pred:
            self.bondnet = [
                TermNet(
                    train_args,
                    input_mult=1,
                    pred_output=1,
                    final_layer=train_args.bondnet_final_act,
                    weight_init=train_args.bond_weights,
                    param_1=train_args.bond_weight_param_1,
                    param_2=train_args.bond_weight_param_2,
                ).to(self.train_args.device)
                for _ in range(train_args.bond_output)
            ]
        else:
            self.bondnet = TermNet(
                train_args,
                input_mult=1,
                pred_output=train_args.bond_output,
                final_layer=train_args.bondnet_final_act,
                weight_init=train_args.bond_weights,
                param_1=train_args.bond_weight_param_1,
                param_2=train_args.bond_weight_param_2,
            ).to(self.train_args.device)
        if train_args.sep_angle_pred:
            self.anglenet = [
                TermNet(
                    train_args,
                    input_mult=2,
                    pred_output=1,
                    final_layer=train_args.anglenet_final_act,
                    weight_init=train_args.angle_weights,
                    param_1=train_args.angle_weight_param_1,
                    param_2=train_args.angle_weight_param_2,
                ).to(self.train_args.device)
                for _ in range(train_args.angle_output)
            ]
        else:
            self.anglenet = TermNet(
                train_args,
                input_mult=2,
                pred_output=train_args.angle_output,
                final_layer=train_args.anglenet_final_act,
                weight_init=train_args.angle_weights,
                param_1=train_args.angle_weight_param_1,
                param_2=train_args.angle_weight_param_2,
            ).to(self.train_args.device)
        if train_args.sep_dihedral_pred:
            self.dihedralnet = [
                TermNet(
                    train_args,
                    input_mult=4,
                    pred_output=1,
                    final_layer=train_args.dihedralnet_final_act,
                ).to(self.train_args.device)
                for _ in range(train_args.dihedral_output)
            ]
        else:
            self.dihedralnet = TermNet(
                train_args,
                input_mult=4,
                pred_output=train_args.dihedral_output,
                final_layer=train_args.dihedralnet_final_act,
            ).to(self.train_args.device)
        self.pairnet = TermNet(
            train_args, input_mult=1, pred_output=train_args.pair_output
        ).to(self.train_args.device)

    def forward(self, batch: TopologyDataset):
        mol_batch = batch.batch_graph()
        encoding, mol_indeces = self.encoder(mol_batch)

        for molecule, (ind, (a_start, a_size)) in zip(batch, enumerate(mol_indeces)):
            encode = encoding.narrow(0, a_start, a_size)
            molecule.reorder_encoding(encode)

            lj_params = torch.tensor([])
            charge = torch.tensor([])
            # if self.train_args.pred_lj_params: # Disabling this for now! (Sep2022)
            charge = molecule.lj_params[:, 0].view(-1, 1).to(self.train_args.device)
            lj_params = molecule.lj_params[:, 1:].to(self.train_args.device)

            # true_charges = torch.cat((true_charges, charge), axis=0).to(self.train_args.device)
            assert molecule.true_charge.shape == charge.shape
            molecule.true_charge = charge

            # Bond prediction
            if molecule.bond_params.any():
                encoded_bonds = molecule.encode[molecule.bonds].sum(
                    1
                )  # Creating the input vector for the NN
                if self.train_args.sep_bond_pred:  # If bond params are separated then the energy param k and distance param r are predicted separately
                    temp_bond = torch.cat(
                        [i(encoded_bonds) for i in self.bondnet], axis=1
                    )
                    if (
                        self.train_args.bond_scaling
                    ):  # Scaling output angle values to some mean and spread
                        temp_bond[:, 0] = (
                            temp_bond[:, 0] * self.train_args.bond_scaling_param_1
                            + self.train_args.bond_scaling_param_2
                        )
                        # This transformations should be: |(k*mju+sigma)|/2 (note added 020223 after talking with Pablo)
                else:
                    temp_bond = self.bondnet(encoded_bonds)
                    if (
                        self.train_args.bond_scaling
                    ):  # Scaling output bond values to some mean and spread
                        temp_bond = (
                            temp_bond * self.train_args.bond_scaling_param_1
                            + self.train_args.bond_scaling_param_2
                        ) / 2

                    if self.train_args.bond_term_linearization:
                        temp_bond_1 = temp_bond[:, 1].clone().detach()
                        temp_bond[:, 1] = (
                            temp_bond[:, 0] * self.train_args.bond_min_OPLS
                            + temp_bond[:, 1] * self.train_args.bond_max_OPLS
                        ) / (temp_bond[:, 0] + temp_bond[:, 1])
                        temp_bond[:, 0] = temp_bond[:, 0] + temp_bond_1

                assert molecule.bond_params.shape == temp_bond.shape
                molecule.bond_params = temp_bond

            # Angle prediction
            if molecule.angle_params.any():
                encoded_angles = torch.cat(
                    [
                        molecule.encode[molecule.angles[:, [0, 2]]].sum(1),
                        molecule.encode[molecule.angles[:, 1]],
                    ],
                    dim=1,
                )  # Creating the input vector for the NN

                if self.train_args.sep_angle_pred:  # If angle params are separated then the energy param k and angle param theta are predicted separately
                    temp_angle = torch.cat(
                        [i(encoded_angles) for i in self.anglenet], axis=1
                    )
                    if (
                        self.train_args.angle_scaling
                    ):  # Scaling output angle values to some mean and spread
                        temp_angle[:, 0] = (
                            temp_angle[:, 0] * self.train_args.angle_scaling_param_1
                            + self.train_args.angle_scaling_param_2
                        )
                else:
                    temp_angle = self.anglenet(encoded_angles)
                    if (
                        self.train_args.angle_scaling
                    ):  # Scaling output angle values to some mean and spread
                        temp_angle = (
                            temp_angle * self.train_args.angle_scaling_param_1
                            + self.train_args.angle_scaling_param_2
                        ) / 2

                    if self.train_args.angle_term_linearization:
                        temp_angle_1 = temp_angle[:, 1].clone().detach()
                        temp_angle[:, 1] = (
                            temp_angle[:, 0] * self.train_args.angle_min_OPLS
                            + temp_angle[:, 1] * self.train_args.angle_max_OPLS
                        ) / (temp_angle[:, 0] + temp_angle[:, 1])
                        temp_angle[:, 0] = temp_angle[:, 0] + temp_angle_1

                assert molecule.angle_params.shape == temp_angle.shape
                molecule.angle_params = temp_angle

            # dihedral prediction

            # encoded_dihedrals = (
            #     (encode[molecule.dihedrals[:]] +
            #     encode[torch.flip(molecule.dihedrals[:], [0])])/2
            #     ).view(molecule.dihedrals.shape[0], -1) # Creating the input vector for the NN
            if molecule.dihedral_params.any():
                dih_1 = (
                    molecule.encode[molecule.dihedrals[:]]
                    .view(molecule.dihedrals.shape[0], -1)
                    .to(self.train_args.device)
                )
                dih_2 = (
                    molecule.encode[torch.flip(molecule.dihedrals[:], [1])]
                    .view(molecule.dihedrals.shape[0], -1)
                    .to(self.train_args.device)
                )
                if self.train_args.sep_dihedral_pred:
                    temp_dihedral_1 = torch.cat(
                        [i(dih_1) for i in self.dihedralnet], axis=1
                    ).to(self.train_args.device)
                    temp_dihedral_2 = torch.cat(
                        [i(dih_2) for i in self.dihedralnet], axis=1
                    ).to(self.train_args.device)
                else:
                    temp_dihedral_1 = self.dihedralnet(dih_1).to(self.train_args.device)
                    temp_dihedral_2 = self.dihedralnet(dih_2).to(self.train_args.device)

                temp_dihedral = (temp_dihedral_1 + temp_dihedral_2) / 2
                assert molecule.dihedral_params.shape == temp_dihedral.shape
                if self.train_args.dihedral_cutoff:
                    temp_dihedral = torch.sign(temp_dihedral) * torch.relu(
                        torch.relu(
                            torch.abs(temp_dihedral) - self.train_args.dihedral_cutoff
                        )
                        + torch.minimum(
                            torch.abs(temp_dihedral) - self.train_args.dihedral_cutoff,
                            self.train_args.dihedral_cutoff
                            * torch.ones_like(temp_dihedral),
                        )
                    )
                molecule.dihedral_params = temp_dihedral

            if self.train_args.ff_type == "compass":
                raise NameError("Compass not yet implemnented, use OPLS")

            # pair prediction
            temp_pair = self.pairnet(molecule.encode).to(
                self.train_args.device
            )  # input needs to just be atomwise vectors
            # temp_pair = charge_scaler(temp_pair)
            temp_pair = torch.cat((temp_pair, lj_params), axis=1).to(
                self.train_args.device
            )
            assert molecule.pair_params.shape == temp_pair.shape
            molecule.pair_params = temp_pair


class DiscreteModel(nn.Module):
    def __init__(
        self,
        args: hitpolyArgs,
    ):
        nn.Module.__init__(self)
        self.train_args = args
        self.counter = 0
        self.sub_e = 0

    def forward(self, batch: TopologyDataset, dataset: TopologyDataset):
        """
        in the discrete model, the "weights" we are trying to learn are the parameters assigned across the entire
        dataset (so the same parameters for the same environments even if they show up in different molecules).
        We use those "weights" to "predict" the parameters of the individual molecules (in this case, by just assigning htem
        to the parameters for the same atomic environment from the dataset) in the forward pass. We then use these predictions to calculate
        the Forces and get the loss (relative to the DFT forces). This is then backpropagated to update the dataset parameters, which we would
        then use to update the molecule parameters [etc...]
        """
        # import pdb
        # import sys

        # def get_size(obj, seen=None):
        #     """Recursively finds size of objects"""
        #     size = sys.getsizeof(obj)
        #     if seen is None:
        #         seen = set()
        #     obj_id = id(obj)
        #     if obj_id in seen:
        #         return 0
        #     # Important mark as seen *before* entering recursion to gracefully handle
        #     # self-referential objects
        #     seen.add(obj_id)
        #     if isinstance(obj, dict):
        #         size += sum([get_size(v, seen) for v in obj.values()])
        #         size += sum([get_size(k, seen) for k in obj.keys()])
        #     elif hasattr(obj, "__dict__"):
        #         size += get_size(obj.__dict__, seen)
        #     elif hasattr(obj, "__iter__") and not isinstance(
        #         obj, (str, bytes, bytearray)
        #     ):
        #         size += sum([get_size(i, seen) for i in obj])
        #     return size

        # pdb.set_trace()
        # if hasattr(batch[0], "bond_params"):
        #     dataset.charge_enforcer()
        # if self.train_args.drude_flag:
        #     if hasattr(batch[0], "drude_bond_params") and self.counter > 10*len(batch):
        #         # import pdb

        #         # pdb.set_trace()
        #         change_bool = dataset.remove_drudes()
        #         if change_bool:
        #             self.counter = 0
        for molecule in batch:
            self.counter += 1
            if dataset.bond_params.numel():
                # transform_tensor = torch.zeros(
                #     len(molecule.bond_type),
                #     len(dataset.bond_params),
                # )
                # transform_tensor = transform_tensor.to(self.train_args.device)
                # transform_tensor = transform_tensor.type(torch.double)
                # # transform_tensor.requires_grad = True

                # for ind, i in enumerate(molecule.bond_type):
                #     transform_tensor[ind, i] = 1.0
                # molecule.bond_params = torch.matmul(
                #     transform_tensor, dataset.bond_params
                # )

                # molecule.bond_params = dataset.bond_params[molecule.bond_type.cpu()].to(
                #     self.train_args.device
                # )
                molecule.bond_params = dataset.bond_params[molecule.bond_type]
            else:
                molecule.bond_params = None

            if dataset.angle_params.numel():
                molecule.angle_params = dataset.angle_params[
                    molecule.angle_type.cpu()
                ].to(self.train_args.device)
            else:
                molecule.angle_params = None
            if dataset.dihedral_params.numel():
                molecule.dihedral_params = dataset.dihedral_params[
                    molecule.dihedral_type.cpu()
                ].to(self.train_args.device)
            else:
                molecule.dihedral_params = None
            if dataset.dispersion_params.numel():
                molecule.dispersion_params = dataset.dispersion_params[
                    molecule.atom_type.cpu()
                ].to(self.train_args.device)
            else:
                molecule.dispersion_params = None
            if dataset.improper_params.numel():
                molecule.improper_params = dataset.improper_params[
                    molecule.improper_type.cpu()
                ].to(self.train_args.device)
            else:
                molecule.improper_params = None
            if self.train_args.drude_flag:
                # if torch.isnan(dataset.drude_bond_params).any():
                #     import pdb
                #     pdb.set_trace()
                # if hasattr(molecule, "drude_bond_params"):
                #     # import pdb

                #     # pdb.set_trace()
                #     dataset.remove_drudes()
                # pdb.set_trace()
                idx = copy.deepcopy(molecule.drude_bond_type)
                for i in range(len(molecule.drude_bond_type)):
                    # idx[i] = molecule.drude_type_to_idx_dict[
                    #     molecule.drude_bond_type[i].item()
                    # ]
                    idx[i] = dataset.master_drude_type_to_idx_dict[
                        molecule.drude_bond_type[i].item()
                    ]
                # import pdb

                # pdb.set_trace()
                molecule.drude_bond_params = dataset.drude_bond_params[idx].to(
                    self.train_args.device
                )
            if dataset.electrostatic_params.numel():
                molecule.electrostatic_params = dataset.electrostatic_params[
                    molecule.atom_type.cpu()
                ].to(self.train_args.device)
            else:
                molecule.electrostatic_params = None
            molecule.true_charge = dataset.true_charge[molecule.atom_type.cpu()].to(
                self.train_args.device
            )
            # if molecule.smiles == [
            #     "COCCOCCOCCOCCOC.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]"
            # ]:
            #     import pdb

            #     pdb.set_trace()
            # print(5)

            # print(molecule.atomic_nums)
            # print(molecule.drude_type_to_idx_dict)
            # print(dataset.drude_bond_params)
            # print(molecule.drude_bond_params)
            # print(molecule.drude_bond_type)
            # print(dataset.master_drude_type_to_idx_dict)

    # def charge_enforcer(self, dataset):
    #     #regressor insists on violating charge conservation so prior to each force calc we normalize it
    #     # with torch.no_grad():
    #     # print(dataset.drude_bond_params)
    #     for molecule in dataset:
    #         if "." in molecule.smiles[0]:
    #             for ind, sep_idxs in enumerate(molecule.atom_idxs_separate):
    #                 if molecule.formal_charges_separate[ind]:
    #                     if molecule.num_drudes:
    #                         for_charge_transform = torch.zeros(
    #                             (len(molecule.electrostatic_params), molecule.num_drudes),
    #                             dtype=torch.float,
    #                         ).to(
    #                             self.train_args.device,
    #                         )
    #                         for i in range(molecule.num_drudes):
    #                                 for_charge_transform[molecule.drude_center_idxs[i], molecule.drude_particle_idxs[i]-molecule.num_atoms] = 1

    #                         combined_charges = (
    #                             torch.matmul(for_charge_transform, molecule.drude_bond_params[:, 0])
    #                             + molecule.electrostatic_params
    #                         )[sep_idxs]
    #                     else:
    #                         combined_charges = molecule.electrostatic_params[sep_idxs]
    #                     norm_factor = molecule.formal_charges_separate[ind]/combined_charges.sum()
    #                     # import pdb
    #                     # pdb.set_trace()
    #                     dataset.electrostatic_params[molecule.atom_type.cpu()][sep_idxs] = norm_factor*molecule.electrostatic_params[sep_idxs]
    #                     # import pdb
    #                     # pdb.set_trace()
    #                     if molecule.num_drudes:
    #                         idx = copy.deepcopy(molecule.drude_bond_type)
    #                         for j in range(len(molecule.drude_bond_type)):
    #                             # if dataset.master_drude_type_to_idx_dict[molecule.drude_bond_type[j].item()] in sep_idxs:
    #                             # if molecule.drude_bond_type[j].item() in molecule.atom_type[sep_idxs]:
    #                             idx[j] = dataset.master_drude_type_to_idx_dict[
    #                                 molecule.drude_bond_type[j].item()
    #                             ]
    #                         # dataset.drude_bond_params[idx][sep_idxs-molecule.num_atoms][:, 0] = norm_factor*molecule.drude_bond_params[:, 0][sep_idxs-molecule.num_atoms]
    #                         # dataset.drude_bond_params[idx][sep_idxs][:, 0] = norm_factor*molecule.drude_bond_params[:, 0][sep_idxs]
    #                         sep_d_idxs = []
    #                         d_to_part = {v: k for k, v in molecule.drude_dict.items()}
    #                         # import pdb
    #                         # pdb.set_trace()
    #                         for ac in sep_idxs:
    #                             if ac in molecule.drude_dict.values():
    #                                 sep_d_idxs += [d_to_part[ac]-molecule.num_atoms]
    #                         # if len(sep_idxs) ==1:
    #                         #     import pdb
    #                         #     pdb.set_trace()
    #                         dataset.drude_bond_params[idx][:, 0][sep_d_idxs] = (norm_factor*molecule.drude_bond_params[:, 0][sep_d_idxs]).cpu()
    #                     # print(molecule.smiles)
    #                     # print(sep_idxs)
    #                     # print()
    #                     # print((molecule.electrostatic_params[sep_idxs]+molecule.drude_bond_params[:, 0][sep_d_idxs]).sum())
    #                     # print(norm_factor)
    #                     # print(norm_factor*(molecule.electrostatic_params[sep_idxs]+molecule.drude_bond_params[:, 0][sep_d_idxs]).sum())
    #         else:
    #             if molecule.formal_charge:
    #                 if molecule.num_drudes:
    #                     for_charge_transform = torch.zeros(
    #                         (len(molecule.electrostatic_params), molecule.num_drudes),
    #                         dtype=torch.float,
    #                     ).to(
    #                         self.train_args.device,
    #                     )
    #                     for i in range(molecule.num_drudes):
    #                             for_charge_transform[molecule.drude_center_idxs[i], molecule.drude_particle_idxs[i]-molecule.num_atoms] = 1

    #                     combined_charges = (
    #                         torch.matmul(for_charge_transform, molecule.drude_bond_params[:, 0])
    #                         + molecule.electrostatic_params
    #                     )
    #                 else:
    #                     combined_charges = molecule.electrostatic_params
    #                 norm_factor = molecule.formal_charge/combined_charges.sum()
    #                 dataset.electrostatic_params[molecule.atom_type.cpu()] = norm_factor*molecule.electrostatic_params
    #                 if molecule.num_drudes:
    #                     idx = copy.deepcopy(molecule.drude_bond_type)
    #                     for j in range(len(molecule.drude_bond_type)):
    #                         idx[j] = dataset.master_drude_type_to_idx_dict[
    #                             molecule.drude_bond_type[j].item()
    #                         ]
    #                     dataset.drude_bond_params[idx][:, 0] = (norm_factor*molecule.drude_bond_params[:, 0]).cpu()
    #         # print(dataset.drude_bond_params)
    #         # if torch.isnan(dataset.drude_bond_params).any():
    #         #     import pdb
    #         #     pdb.set_trace()
