# Sentiment Classification with RNN Architectures

A comprehensive comparative analysis system for evaluating different RNN architectures on sentiment classification using the IMDb Movie Review Dataset.

## Overview

This project implements and compares multiple RNN variants (RNN, LSTM, Bidirectional LSTM) for binary sentiment classification. The system systematically evaluates different configurations including:

- **Architectures**: Basic RNN, LSTM, Bidirectional LSTM
- **Activation Functions**: Sigmoid, ReLU, Tanh
- **Optimizers**: Adam, SGD, RMSProp
- **Sequence Lengths**: 25, 50, 100 words
- **Stability Strategies**: With/without gradient clipping

## Project Structure

```
├── data/                    # Dataset storage (IMDb reviews)
│   └── IMDB_Dataset.csv
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── config.py           # Configuration classes and constants
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── models.py           # RNN model architectures
│   ├── train.py            # Training infrastructure
│   ├── evaluate.py         # Evaluation and metrics
│   └── utils.py            # Utility functions
├── results/                # Experimental results and plots
│   ├── experiments_summary.csv
│   └── plots/
├── models/                 # Trained model checkpoints (.pth files)
├── runner.py               # Main training script
├── plot_results.py         # Results visualization script
├── requirements.txt        # Python dependencies
├── report.md               # Detailed project report
└── README.md
```

## Setup Instructions

### Prerequisites

- **Python Version**: Python 3.8 or higher (tested with Python 3.8-3.12)
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Hardware**: 
  - CPU: 2+ cores recommended
  - RAM: 4-8 GB available memory
  - Storage: 2+ GB free space

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd msml_602
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate   # On Windows
   
   # Or using conda
   conda create -n rnn_sentiment python=3.10
   conda activate rnn_sentiment
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Training Experiments

The main training script is `runner.py`, which runs a comprehensive suite of experiments across different model configurations.

#### Run All Experiments
```bash
python runner.py
```

This will:
- Test all combinations of architectures, activations, optimizers, sequence lengths, and gradient clipping settings
- Train each configuration for 10 epochs (default)
- Save trained models to `models/` directory
- Log results to `results/experiments_summary.csv`

#### Configuration Settings

Default configurations (defined in `src/config.py`):
- **Architectures**: `['bidirectional_lstm']` (can include 'rnn', 'lstm', 'bidirectional_lstm')
- **Activation Functions**: `['relu', 'tanh']` (can include 'sigmoid')
- **Optimizers**: `['adam', 'sgd', 'rmsprop']`
- **Sequence Lengths**: `[25, 50, 100]`
- **Gradient Clipping**: `[False, True]`
- **Epochs**: `10`
- **Batch Size**: `32`
- **Learning Rate**: `0.001`
- **Hidden Size**: `64`
- **Embedding Dimension**: `100`

#### Modify Experiment Configuration

To customize experiments, edit `src/config.py`:
```python
# Example: Test only LSTM with different optimizers
ARCHITECTURES = ['lstm']
ACTIVATION_FUNCTIONS = ['relu']
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
SEQUENCE_LENGTHS = [50]
EPOCHS = 15
```

### 2. Visualization and Analysis

After training, generate plots to analyze the results:

```bash
python plot_results.py
```

This script generates:
- **Accuracy and F1 vs Sequence Length**: Shows performance trends across different sequence lengths
- **Training Loss Curves**: Compares best and worst performing models

Plots are saved to `results/plots/`:
- `accuracy_f1_vs_seq_length.png`
- `training_loss_best_worst.png`

### 3. Evaluating Individual Models

To evaluate a specific trained model:

```python
import torch
from src.evaluate import ModelEvaluator

# Load a saved model
checkpoint = torch.load('models/exp_20251024_000446_model.pth')
model_state = checkpoint['model_state_dict']
results = checkpoint['result']

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")
```

## Expected Runtime and Output Files

### Runtime Estimates (CPU)

| Configuration | Per Epoch | 10 Epochs | Full Suite* |
|---------------|-----------|-----------|-------------|
| RNN, seq=25 | 30-45s | 5-8 min | - |
| LSTM, seq=50 | 45-60s | 8-10 min | - |
| BiLSTM, seq=100 | 60-90s | 10-15 min | - |
| All combinations | - | - | 4-8 hours |

*Full suite varies based on number of parameter combinations (default: 36 combinations with current config)

### Memory Requirements
- **RAM**: 4-8 GB recommended (minimum 2 GB)
- **Storage**: 
  - Dataset: ~130 MB (IMDB_Dataset.csv)
  - Models: ~50-100 MB per saved model
  - Results: ~10-50 MB
  - Total: 2+ GB recommended

### Output Files

#### 1. Training Results (`results/`)
- **`experiments_summary.csv`**: Complete log of all experiments with columns:
  - Model, Activation, Optimizer, Seq Length, Grad Clipping
  - Accuracy, F1 Score
  - Epoch Time (s), Final Loss
  - Loss History (per-epoch training losses)

Example row:
```csv
bidirectional_lstm,relu,adam,50,Yes,0.8542,0.8523,45.23,0.3124,"[0.693, 0.512, 0.421, ...]"
```

#### 2. Model Checkpoints (`models/`)
- **Naming Convention**: `exp_YYYYMMDD_HHMMSS_model.pth`
- **Contents**: Each `.pth` file contains:
  - `model_state_dict`: Trained model weights
  - `result`: Experiment configuration and metrics

Example:
```
models/exp_20251024_000446_model.pth
models/exp_20251024_001249_model.pth
...
```

#### 3. Visualization Plots (`results/plots/`)
- **`accuracy_f1_vs_seq_length.png`**: Performance metrics across sequence lengths
- **`training_loss_best_worst.png`**: Training curves comparing best/worst models

#### 4. Dataset (`data/`)
- **`IMDB_Dataset.csv`**: IMDb movie reviews dataset (should be placed here before training)