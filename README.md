# Unified-GPT

Modular repository for training example GPT-based language models. Models built using both TensorFlow and PyTorch and using both library pre-built and custom-built components.

## Features

- **Multi-Framework:** Easily switch between TensorFlow and PyTorch.
- **Configurable Variants:** Choose between built-in and custom model implementations.
- **Modular Design:** Separate modules for data processing, model components, training loops, and utilities.
- **Unified Configuration:** All hyperparameters and model settings are defined in a single `config.yaml` file.
- **Unified Generation Interface:** Consistent text generation across frameworks.

## Repository Structure

Unified-GPT/
├── README.md # This file
├── unifiedGPT.py # Entry point for training
├── config/
│ └── config.yaml # Global configuration file
├── data/
│ └── dataset.py # Data loading & tokenization (TF and PyTorch)
├── models/
│ ├── tensorflow/
│ │ ├── gpt_builtin.py # GPT using TensorFlow's built-in layers
│ │ └── gpt_custom.py # GPT using custom TensorFlow components
│ └── pytorch/
│ ├── gpt_builtin.py # GPT using PyTorch's built-in layers
│ └── gpt_custom.py # GPT using custom PyTorch components
├── training/
│ ├── train_tf.py # TensorFlow training loop
│ ├── train_torch.py # PyTorch training loop
│ └── callbacks.py # Logging and text generation callbacks
│── utils/
│ ├── lr_schedule.py # Learning rate schedulers for TF and PyTorch
├── plots.py # Utility functions for plotting metrics
├── utils.py # Miscellaneous utilities (e.g., config loader, set_device)
└── generate.py # Unified text generation interface

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/argiacomi/Unified-GPT.git
cd Unified-GPT
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit config.yaml to adjust hyperparameters, select the framework (tensorflow or pytorch), and choose the model variant (builtin or custom).

### Training

Run the training script using the entry point:

```bash
python main.py --config config.yaml --framework pytorch
```

### Text Generation

After training, use the unified generation interface in generate.py to produce text from your trained model.
