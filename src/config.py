"""
Configuration settings for the sentiment classification system.
Contains all hyperparameters and system settings.
"""

from dataclasses import dataclass
from typing import List


# Reproducibility settings
RANDOM_SEED = 42

# Dataset settings
VOCAB_SIZE = 10000
SEQUENCE_LENGTHS = [25, 50, 100]
TRAIN_TEST_SPLIT = 0.5  # IMDb dataset comes pre-split

# Model architecture settings
EMBEDDING_DIM = 100
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT_RANGE = (0.3, 0.5)
DEFAULT_DROPOUT = 0.4

# Training settings
BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10

# Supported configurations
ARCHITECTURES = ['rnn', 'lstm', 'bidirectional_lstm']
ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'tanh']
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']

# File paths
DATA_DIR = 'data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
SRC_DIR = 'src'

experiments = [
    # {
    #   "architecture": "rnn",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "rnn",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "sigmoid",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "relu",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 25,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "bidirectional_lstm",
    #   "activation": "tanh",
    #   "optimizer": "adam",
    #   "sequence_length": 100,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "relu",
    #   "optimizer": "sgd",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.01,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    # {
    #   "architecture": "lstm",
    #   "activation": "relu",
    #   "optimizer": "rmsprop",
    #   "sequence_length": 50,
    #   "gradient_clipping": False,
    #   "learning_rate": 0.001,
    #   "batch_size": 32,
    #   "epochs": 10,
    #   "dropout": 0.4
    # },
    {
      "architecture": "lstm",
      "activation": "relu",
      "optimizer": "adam",
      "sequence_length": 50,
      "gradient_clipping": True,
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 10,
      "dropout": 0.4
    }
]

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    architecture: str
    activation: str
    optimizer: str
    sequence_length: int
    gradient_clipping: bool
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    dropout: float = DEFAULT_DROPOUT

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    accuracy: float
    f1_score: float
    avg_epoch_time: float
    total_training_time: float
    final_loss: float
    loss_history: List[float]