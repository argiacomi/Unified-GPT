#!/usr/bin/env python3
import argparse
import os
import sys

from utils.utils import load_yaml_config, set_device, set_global_random_seed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = argparse.ArgumentParser(description="GPT Training Entry Point")
    parser.add_argument(
        "--config", type=str, default="./config/config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["tensorflow", "pytorch"],
        help="Override framework specified in the config",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.framework:
        config["model"]["framework"] = args.framework

    # Initialize tokenizer (using tiktoken)
    try:
        import tiktoken
    except ImportError:
        print("tiktoken package not found. Install it via pip install tiktoken")
        sys.exit(1)

    tokenizer = tiktoken.get_encoding("gpt2")
    framework = config["model"]["framework"].lower()
    set_global_random_seed(config["seed"], framework)

    if framework in ["tensorflow", "tf"]:
        print("Starting TensorFlow training...")
        from training.train_tf import train_tf_custom

        device = set_device(config["device"], framework)

        train_tf_custom(config, tokenizer, device)

    elif framework in ["pytorch", "torch"]:
        print(f"Starting PyTorch training on device: {device}...")
        from training.train_torch import train_torch

        device = set_device(config["device"], framework)

        train_torch(config, tokenizer, device)

    else:
        raise ValueError(f"Unsupported framework: {framework}")


if __name__ == "__main__":
    main()
