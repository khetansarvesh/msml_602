import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Iterator
import traceback
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import numpy as np
import itertools
from dataclasses import dataclass, asdict
from copy import deepcopy
import os

from src.config import *
from src.preprocess import IMDbDataLoader, TextPreprocessor, SequenceProcessor
from src.models import SimpleRNN, LSTMModel, BidirectionalLSTMModel
from src.train import Trainer
from src.evaluate import ModelEvaluator


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed} for reproducible results")

def get_device():
    return torch.device('cpu')

def prepare_data(train_texts_clean, test_texts_clean, train_labels, test_labels, sequence_length, batch_size) -> tuple:

    sequence_processor = SequenceProcessor()
    vocabulary = sequence_processor.build_vocabulary(train_texts_clean, vocab_size=10000)
    
    train_sequences = sequence_processor.texts_to_sequences(train_texts_clean)
    test_sequences = sequence_processor.texts_to_sequences(test_texts_clean)

    
    train_sequences_padded = sequence_processor.pad_sequences(train_sequences, sequence_length)
    test_sequences_padded = sequence_processor.pad_sequences(test_sequences, sequence_length)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.LongTensor(train_sequences_padded),torch.LongTensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.LongTensor(test_sequences_padded),torch.LongTensor(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(vocabulary)


def main():

    device = get_device()

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Device: {device}")
    
    # Set random seed
    set_random_seeds(RANDOM_SEED)

    # Load dataset
    data_loader = IMDbDataLoader(data_dir=DATA_DIR)
    train_texts, train_labels, test_texts, test_labels = data_loader.load_data()
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    train_texts_clean = [preprocessor.clean_text(text) for text in train_texts]
    test_texts_clean = [preprocessor.clean_text(text) for text in test_texts]

    
    for exp_config in experiments:
        # Prepare data
        train_loader, test_loader, vocab_size = prepare_data(train_texts_clean, test_texts_clean, train_labels, test_labels, exp_config["sequence_length"], BATCH_SIZE)

        # Create model
        model_kwargs = {
            'vocab_size': vocab_size,
            'embedding_dim': 100,  # Fixed as per requirements
            'hidden_size': 64,     # Fixed as per requirements
            'num_layers': 2,       # Fixed as per requirements
            'dropout': config.dropout,
            'activation': config.activation
        }
        
        if config.architecture == 'rnn':
            model = SimpleRNN(**model_kwargs)
        elif config.architecture == 'lstm':
            model = LSTMModel(**model_kwargs)
        elif config.architecture == 'bidirectional_lstm':
            model = BidirectionalLSTMModel(**model_kwargs)


        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate,
            gradient_clipping=config.gradient_clipping,
            device=device
        )



        # Run experiment
        start_time = time.time()
        epoch_metrics = trainer.train_multiple_epochs(
                                                        train_loader=train_loader,
                                                        num_epochs=config.epochs,
                                                        val_loader=val_loader,
                                                        verbose=False
                                                    )

        # result = runner.run_single_experiment(
        #                                         config=ExperimentConfig(**exp_config),
        #                                         train_loader=train_loader,
        #                                         val_loader=None,
        #                                         test_loader=test_loader,
        #                                         vocab_size=vocab_size
        #                                     )
        end_time = time.time()
    

        total_training_time = end_time - start_time
        
        # Calculate average epoch time
        avg_epoch_time = total_training_time / config.epochs

        # Get training summary
        training_summary = trainer.get_training_summary()
        



        # Evaluation phase
        evaluator = ModelEvaluator(device=device)
        evaluation_metrics = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            verbose=False
        )
        result = ExperimentResult(
            config=config,
            accuracy=evaluation_metrics.accuracy,
            f1_score=evaluation_metrics.f1_score,
            avg_epoch_time=avg_epoch_time,
            total_training_time=total_training_time,
            final_loss=training_summary.get('final_train_loss', 0.0),
            loss_history=training_summary.get('loss_history', [])
        )


        


        # Save model
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        models_dir = Path(models_dir)
        model_path = models_dir / f'{experiment_id}_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'result': asdict(result)
        }, model_path)


        print(f"\nExperiment completed successfully!")
        print(f"Results:")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  F1-Score: {result.f1_score:.4f}")
        print(f"  Avg Epoch Time: {result.avg_epoch_time:.2f}s")
        print(f"  Total Time: {end_time - start_time:.2f}s")

if __name__ == '__main__':
    main()