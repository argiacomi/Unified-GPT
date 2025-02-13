import os
import random

import numpy as np
import tensorflow as tf
import torch
import yaml


def set_global_random_seed(seed=123, framework="pytorch"):
    """
    Sets the random seed for Python, NumPy, TensorFlow, and PyTorch (if installed).
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    if framework in ("tensorflow", "tf"):
        tf.random.set_seed(seed)
    if framework in ("pytorch", "torch"):
        torch.manual_seed(seed)


def load_yaml_config(config_path):
    """
    Loads a YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_device(
    device,
    framework="pytorch",
):
    """Set the device based on availability and user preference for PyTorch or TensorFlow."""
    device = device.lower()
    framework = framework.lower()

    if framework == "pytorch":
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            print(f'{device} is not available for PyTorch. Defaulting to "CPU"')
            return torch.device("cpu")

    elif framework == "tensorflow":
        if device in ["cuda", "mps"]:
            return "GPU"  # TensorFlow abstracts device handling and automatically assigns GPU if available
        elif device == "cpu":
            return "CPU"
        else:
            print(f'{device} is not available for TensorFlow. Defaulting to "CPU"')
            return "CPU"

    else:
        raise ValueError("Unsupported framework. Choose 'pytorch' or 'tensorflow'.")
