def create_model(config):
    """
    Factory function to create a GPT model based on configuration.
    """
    framework = config.get("model", {}).get("framework", "pytorch").lower()
    variant = config.get("model", {}).get("variant", "custom").lower()
    size = config.get("model", {}).get("size", "gpt_small").lower()

    # Select model implementation based on framework and variant.
    if framework in ("tensorflow", "tf"):
        if variant == "builtin":
            from models.tensorflow.gpt_builtin import GPTModelBuiltin as Model
        elif variant == "custom":
            from models.tensorflow.gpt_custom import GPTModelCustom as Model
        else:
            raise ValueError(f"Unsupported TensorFlow variant: {variant}")
    elif framework in ("pytorch", "torch"):
        if variant == "builtin":
            from models.pytorch.gpt_builtin import GPTModelBuiltin as Model
        elif variant == "custom":
            from models.pytorch.gpt_custom import GPTModelCustom as Model
        else:
            raise ValueError(f"Unsupported PyTorch variant: {variant}")
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Extract model configuration for the selected size.
    model_configs = config.get("model", {}).get("configs", {})
    if size not in model_configs:
        raise ValueError(f"Configuration for model size '{size}' not found.")
    model_config = model_configs[size]

    return Model(model_config)
