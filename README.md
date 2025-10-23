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
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── config.py           # Configuration classes and constants
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── models.py           # RNN model architectures
│   ├── train.py            # Training infrastructure
│   ├── evaluate.py         # Evaluation and metrics
│   ├── experiment_runner.py # Experiment orchestration
│   ├── results_aggregator.py # Results analysis
│   └── utils.py            # Utility functions
├── results/                # Experimental results and plots
├── models/                 # Trained model checkpoints
├── train_main.py           # Main training script
├── evaluate_main.py        # Model evaluation script
├── analyze_results.py      # Results analysis script
├── requirements.txt        # Python dependencies
├── example_config.json     # Example configuration file
└── README.md
```

## Requirements

### Python Version
- **Python 3.8 or higher** (tested with Python 3.8, 3.9, 3.10)

### Dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=1.12.0` - PyTorch for neural network implementation
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation and analysis
- `matplotlib>=3.5.0` - Plotting and visualization
- `nltk>=3.7` - Natural language processing toolkit
- `scikit-learn>=1.1.0` - Machine learning metrics and utilities

### Additional Setup
Download required NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

### 1. Training Models

#### Single Experiment
Run a single experiment with specific parameters:
```bash
# Basic usage
python train_main.py --single --architecture lstm --activation relu --optimizer adam

# With custom parameters
python train_main.py --single --architecture bidirectional_lstm --activation tanh \
    --optimizer sgd --sequence-length 100 --learning-rate 0.01 --epochs 15
```

#### Systematic Experiments
Run systematic experiments varying specific parameters:
```bash
# Vary architectures (keeping other parameters fixed)
python train_main.py --systematic --vary architecture

# Vary multiple parameters
python train_main.py --systematic --vary architecture activation optimizer

# With custom base configuration
python train_main.py --systematic --vary sequence_length --epochs 20 --verbose
```

#### Batch Experiments from Configuration File
Run multiple experiments from a JSON configuration file:
```bash
# Use example configuration
python train_main.py --config example_config.json

# With custom settings
python train_main.py --config my_experiments.json --save-models --verbose
```

**Configuration File Format:**
```json
{
  "experiments": [
    {
      "architecture": "lstm",
      "activation": "relu",
      "optimizer": "adam",
      "sequence_length": 50,
      "gradient_clipping": false,
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 10,
      "dropout": 0.4
    }
  ]
}
```

### 2. Evaluating Models

#### Evaluate Saved Model
```bash
# Evaluate specific model file
python evaluate_main.py --model models/best_model.pth

# With detailed metrics and timing analysis
python evaluate_main.py --model models/lstm_relu_adam.pth --detailed-metrics --timing-analysis
```

#### Evaluate All Models in Directory
```bash
# Evaluate all models from experiment results
python evaluate_main.py --results-dir results/batch_20240101_120000

# Export results in different formats
python evaluate_main.py --results-dir results/ --export-format excel --save-predictions
```

#### Compare Multiple Models
```bash
# Compare specific models
python evaluate_main.py --compare --models model1.pth model2.pth model3.pth

# With verbose output
python evaluate_main.py --compare --models models/*.pth --verbose
```

### 3. Analyzing Results

#### Basic Analysis
```bash
# Analyze all results in directory
python analyze_results.py --results-dir results/

# Analyze specific results file
python analyze_results.py --results-file results/all_results.json
```

#### Advanced Analysis
```bash
# Compare specific parameters with statistical tests
python analyze_results.py --results-dir results/ --compare-parameters architecture activation \
    --statistical-tests --generate-plots

# Find optimal configurations with constraints
python analyze_results.py --results-dir results/ --find-optimal \
    --constraints max_epoch_time=10.0 min_accuracy=0.85

# Generate comprehensive report
python analyze_results.py --results-dir results/ --generate-plots --report-format pdf \
    --export-data --top-k 10
```

## Expected Runtime and Output

### Training Performance
- **Single Experiment**: 5-15 minutes (10 epochs, CPU)
- **Systematic Experiments**: 1-3 hours (depends on parameter variations)
- **Full Experimental Suite**: 4-8 hours (all combinations)

### Memory Requirements
- **RAM**: 4-8 GB recommended
- **Storage**: 1-2 GB for datasets and results
- **GPU**: Optional (CPU-optimized implementation)

### Output Files

#### Training Outputs
- `results/experiment_YYYYMMDD_HHMMSS/`
  - `experiment_results.json` - Detailed results with metrics
  - `training_logs.txt` - Training progress logs
  - `config.json` - Experiment configuration
  - `plots/` - Training curves and visualizations

#### Model Checkpoints
- `models/` - Saved model weights (.pth files)
- Naming convention: `{architecture}_{activation}_{optimizer}_{timestamp}.pth`

#### Analysis Outputs
- `analysis_results/`
  - `analysis_report_YYYYMMDD_HHMMSS.txt` - Comprehensive analysis report
  - `processed_results_YYYYMMDD_HHMMSS.csv` - Processed experimental data
  - `performance_distributions.png` - Performance distribution plots
  - `{parameter}_comparison.png` - Parameter comparison plots

## Hardware Requirements and Performance Notes

### Minimum Requirements
- **CPU**: 2+ cores, 2.0+ GHz
- **RAM**: 4 GB available memory
- **Storage**: 2 GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Recommended Configuration
- **CPU**: 4+ cores, 3.0+ GHz (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: 8+ GB available memory
- **Storage**: 5+ GB free space (SSD preferred)

### Performance Optimization
- **CPU Training**: Optimized for multi-core CPU execution
- **Batch Size**: Default 32 (adjust based on available memory)
- **Sequence Length**: Shorter sequences (25-50) train faster than longer ones (100)
- **Model Complexity**: LSTM > Bidirectional LSTM > RNN (in terms of training time)

### Expected Training Times (CPU)
| Configuration | Single Epoch | 10 Epochs | Full Suite |
|---------------|--------------|-----------|------------|
| RNN, seq=25   | 30-45s      | 5-8 min   | -          |
| LSTM, seq=50  | 45-60s      | 8-10 min  | -          |
| BiLSTM, seq=100| 60-90s     | 10-15 min | -          |
| All combinations| -          | -         | 4-8 hours  |

## Reproducibility

All experiments use fixed random seeds for consistent results:

```python
# Fixed seeds used throughout
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Hardware Specifications Logging
The system automatically logs hardware specifications in results:
- CPU information and core count
- Available memory
- Python and package versions
- Operating system details

### Reproducing Results
1. Use identical Python environment (requirements.txt)
2. Use same random seed (42)
3. Use same dataset split (fixed train/test split)
4. Use same hyperparameters as documented in results

## Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size
python train_main.py --single --batch-size 16

# Use shorter sequences
python train_main.py --single --sequence-length 25
```

#### Slow Training
```bash
# Reduce epochs for testing
python train_main.py --single --epochs 5

# Use simpler architecture
python train_main.py --single --architecture rnn
```

#### NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')  # Optional
```

#### Dataset Download Issues
The IMDb dataset is automatically downloaded on first use. If download fails:
1. Check internet connection
2. Manually download from: https://ai.stanford.edu/~amaas/data/sentiment/
3. Extract to `data/aclImdb/` directory

### Performance Tips
1. **Use SSD storage** for faster data loading
2. **Close other applications** to free memory
3. **Use shorter sequences** for faster experimentation
4. **Start with single experiments** before running full suites
5. **Monitor system resources** during long experiments
