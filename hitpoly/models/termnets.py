import torch
import torch.nn as nn
import numpy as np
from hitpoly.utils.args import hitpolyArgs
from hitpoly.data.builder import TopologyBuilder
from hitpoly.utils.nn_utils import weight_init_func



class TermNet(nn.Module):
    """
    input_mult denotes the input size of the encoded atoms in the system
    For bonds it is 1, angles - 2, dihedrals - 4.
    This comes from attempting to obtain permutation invarience for the system.
    """

    def __init__(
        self,
        train_args: hitpolyArgs,
        input_mult: int,
        pred_output: int,
        final_layer: bool = False,
        weight_init: bool = False,
        param_1: int = None,
        param_2: int = None,
    ):
        nn.Module.__init__(self)
        input_dim = train_args.hidden_size * input_mult  # From chemprop
        depth = train_args.hitpoly_depth
        hid_nodes = train_args.hitpoly_hidden_nodes
        activ = get_activation_function(train_args.hitpoly_activation)
        final_activ = (
            None
            if not final_layer
            else get_activation_function(train_args.hitpoly_final_layer)
        )
        dropout = nn.Dropout(train_args.hitpoly_dropout)
        output = pred_output
        bias = train_args.hitpoly_bias

        if depth == 1:
            self.model = nn.Sequential(dropout, nn.Linear(input_dim, output, bias))
        else:
            layers = [dropout, nn.Linear(input_dim, hid_nodes, bias)]
            for _ in range(depth - 2):
                layers.extend([activ, dropout, nn.Linear(hid_nodes, hid_nodes, bias)])
            layers.extend(
                [
                    activ,
                    dropout,
                    nn.Linear(hid_nodes, output, bias),
                ]
            )
            if train_args.hitpoly_weight_initialization and weight_init:
                for l in layers:
                    if isinstance(l, nn.Linear):
                        weight_init_func(
                            train_args.hitpoly_weight_initialization,
                            l.weight,
                            param_1,
                            param_2,
                            train_args.hitpoly_activation,
                        )

            if final_activ:
                layers.extend([final_activ])

            self.model = nn.Sequential(*layers)

    def forward(self, encoded_elements):
        return self.model(encoded_elements).pow(1)
