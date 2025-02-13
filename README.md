# Unified-GPT

Modular repository for training example GPT-based language models. Models built using both TensorFlow and PyTorch and using both library pre-built and custom-built components.

## Features

- **Multi-Framework:** Easily switch between TensorFlow and PyTorch.
- **Configurable Variants:** Choose between built-in and custom model implementations.
- **Modular Design:** Separate modules for data processing, model components, training loops, and utilities.
- **Unified Configuration:** All hyperparameters and model settings are defined in a single `config.yaml` file.
- **Unified Generation Interface:** Consistent text generation across frameworks.

## Repository Structure

<pre>
Unified-GPT/  
├── README.md
├── unifiedGPT.py
├── config/  
│   └── config.yaml
├── data/  
│   └── dataset.py
├── models/  
│   ├── tensorflow/  
│   │   ├── gpt_builtin.py 
│   │   └── gpt_custom.py 
│   └── pytorch/  
│     ├── gpt_builtin.py
│     └── gpt_custom.py
├── training/  
│   ├── train_tf.py
│   ├── train_torch.py
│   └── callbacks.py
└── utils/  
    ├── lr_schedule.py
    ├── plots.py
    ├── utils.py
    └── generate.py
</pre>
  
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
