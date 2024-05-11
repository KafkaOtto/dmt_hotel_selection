from typing import Any

import numpy as np
import torch
import torch.nn as nn


from rank_ndcg_batch.utils.logging_r import get_logger

logger = get_logger()


def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_num_params(model: nn.Module) -> int:
    """
    Calculation of the number of nn.Module parameters.
    :param model: nn.Module
    :return: number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params  # type: ignore


def log_num_params(num_params: int) -> None:
    """
    Logging num_params to the global logger.
    :param num_params: number of parameters to log
    """
    logger.info("Model has {} trainable parameters".format(num_params))

def load_state_dict_from_file(path: str, device: Any):
    return torch.load(path, map_location=device)