import torch
import torch.nn as nn


def weight_init_func(
    weight_init: str, tensor: torch.Tensor, param_1: int, param_2: int, activ: str
):
    """
    Gets an weight initialization function module given the name.

    Supports:

    * :code:`uniform_`
    * :code:`normal_`
    * :code:`constant_`
    * :code:`xavier_uniform_`
    * :code:`xavier_normal_`
    * :code:`kaiming_uniform_`
    * :code:`kaiming_normal_`

    :param weight_init: The name of the weight_init function.
    :return: The weight_init function module.
    """
    if weight_init == "uniform_":
        nn.init.uniform_(tensor, param_1, param_2)
    elif weight_init == "normal_":
        nn.init.normal_(tensor, param_1, param_2)
    elif weight_init == "constant_":
        nn.init.constant_(tensor, param_1)
    elif weight_init == "xavier_uniform_":
        nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain(activ))
    elif weight_init == "xavier_normal_":
        nn.init.xavier_normal_(tensor, gain=nn.init.calculate_gain(activ))
    elif weight_init == "kaiming_uniform_":
        nn.init.kaiming_uniform_(tensor, gain=nn.init.calculate_gain(activ))
    elif weight_init == "kaiming_normal_":
        nn.init.kaiming_normal_(tensor, gain=nn.init.calculate_gain(activ))
    else:
        raise ValueError(f'Weight initialization "{weight_init}" not supported.')
