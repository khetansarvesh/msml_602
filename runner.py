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
import csv

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

def main():

    device = torch.device('cpu')

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

    # Build vocabulary
    sequence_processor = SequenceProcessor()
    vocabulary = sequence_processor.build_vocabulary(train_texts_clean, vocab_size=10000)
    vocab_size = len(vocabulary)

    # text to tokens [I am sarvesh] -> [12, 45, 678]
    train_sequences = sequence_processor.texts_to_sequences(train_texts_clean)
    test_sequences = sequence_processor.texts_to_sequences(test_texts_clean)

    for architecture in ARCHITECTURES:
        for activation in ACTIVATION_FUNCTIONS:
            for optimizer in OPTIMIZERS:
                for sequence_length in SEQUENCE_LENGTHS:
                    for gradient_clipping in [False, True]:

                        # perform padding for train and test
                        train_sequences_padded = sequence_processor.pad_sequences(train_sequences, sequence_length)
                        test_sequences_padded = sequence_processor.pad_sequences(test_sequences, sequence_length)

                        # Create data loaders
                        train_dataset = torch.utils.data.TensorDataset(torch.LongTensor(train_sequences_padded), torch.LongTensor(train_labels))
                        test_dataset = torch.utils.data.TensorDataset(torch.LongTensor(test_sequences_padded), torch.LongTensor(test_labels))
                        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                        # Create model
                        model_kwargs = {
                            'vocab_size': vocab_size,
                            'embedding_dim': EMBEDDING_DIM,
                            'hidden_size': HIDDEN_SIZE,
                            'num_layers': NUM_LAYERS,
                            'dropout': DROPOUT,
                            'activation': activation,
                        }

                        if architecture == 'rnn':
                            model = SimpleRNN(**model_kwargs)
                        elif architecture == 'lstm':
                            model = LSTMModel(**model_kwargs)
                        elif architecture == 'bidirectional_lstm':
                            model = BidirectionalLSTMModel(**model_kwargs)

                        # Create trainer
                        trainer = Trainer(
                            model=model,
                            optimizer_type=optimizer,
                            learning_rate=LEARNING_RATE,
                            gradient_clipping=gradient_clipping,
                            device=device
                        )

                        # Run experiment
                        start_time = time.time()
                        epoch_metrics = trainer.train_multiple_epochs(
                            train_loader=train_loader,
                            num_epochs=EPOCHS,
                            val_loader=None,
                            verbose=False
                        )
                        end_time = time.time()
                        total_training_time = end_time - start_time
                        avg_epoch_time = total_training_time / EPOCHS

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
                            config={
                                        'architecture': architecture,
                                        'activation': activation,
                                        'optimizer': optimizer,
                                        'sequence_length': sequence_length,
                                        'gradient_clipping': gradient_clipping,
                                    },
                            accuracy=evaluation_metrics.accuracy,
                            f1_score=evaluation_metrics.f1_score,
                            avg_epoch_time=avg_epoch_time,
                            total_training_time=total_training_time,
                            final_loss=training_summary.get('final_train_loss', 0.0),
                            loss_history=training_summary.get('loss_history', [])
                        )

                        # Save model
                        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        models_dir = Path(MODELS_DIR)
                        model_path = models_dir / f'{experiment_id}_model.pth'
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'result': asdict(result)
                        }, model_path)

                        # Save experiment summary to CSV in results folder
                        results_dir = Path(RESULTS_DIR)
                        csv_path = results_dir / 'experiments_summary.csv'
                        write_header = not csv_path.exists()

                        with csv_path.open('a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            if write_header:
                                writer.writerow(["Model", "Activation", "Optimizer", "Seq Length", "Grad Clipping", "Accuracy", "F1", "Epoch Time (s)", "Final Loss", "Loss History"])

                            writer.writerow([
                                architecture,
                                activation,
                                optimizer,
                                sequence_length,
                                'Yes' if gradient_clipping else 'No',
                                f"{result.accuracy:.4f}",
                                f"{result.f1_score:.4f}",
                                f"{result.avg_epoch_time:.2f}",
                                f"{result.final_loss:.4f}",
                                json.dumps(result.loss_history)
                            ])


if __name__ == '__main__':
    main()