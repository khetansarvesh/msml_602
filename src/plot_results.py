"""Utility to generate summary plots from experiment results.

Creates two plots and saves them into `results/plots/`:

- Accuracy and F1 vs Sequence Length (reads `results/experiments_summary.csv`)
- Training Loss vs Epochs for the best and worst models (reads `models/*.pth` for saved `result.loss_history`)

Usage:
    python -m src.plot_results
    or
    python src/plot_results.py

This script is defensive: if required files are missing it will print a message and exit.
"""
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = Path("models")
EXPERIMENTS_CSV = RESULTS_DIR / "experiments_summary.csv"


def plot_accuracy_f1_vs_seq_len(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        print(f"Experiment CSV not found at {csv_path}. Skipping Accuracy/F1 plot.")
        return

    df = pd.read_csv(csv_path)

    # Ensure numeric seq length
    if 'Seq Length' in df.columns:
        df['Seq Length'] = pd.to_numeric(df['Seq Length'], errors='coerce')
    elif 'Seq_Length' in df.columns:
        df['Seq Length'] = pd.to_numeric(df['Seq_Length'], errors='coerce')
    else:
        print("No 'Seq Length' column found in experiments CSV. Skipping.")
        return

    # Convert accuracy and F1 to numeric
    for col in ['Accuracy', 'F1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Group by sequence length and compute mean (in case multiple runs)
    grouped = df.groupby('Seq Length').agg({'Accuracy': 'mean', 'F1': 'mean'}).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped['Seq Length'], grouped['Accuracy'], marker='o', label='Accuracy')
    plt.plot(grouped['Seq Length'], grouped['F1'], marker='s', label='F1')
    plt.xlabel('Sequence Length')
    plt.ylabel('Score')
    plt.title('Accuracy and F1 vs Sequence Length')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Accuracy/F1 vs Seq Length plot to {out_path}")


def load_results_from_model_file(pth_path: Path) -> Dict[str, Any]:
    try:
        data = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {pth_path}: {e}")
        return {}

    # The saver stored 'result' in the checkpoint as a dict
    result = data.get('result') if isinstance(data, dict) else None
    if isinstance(result, str):
        # sometimes saved as json string
        try:
            result = json.loads(result)
        except Exception:
            result = None

    return result or {}


def plot_training_loss_best_worst(models_dir: Path, out_path: Path) -> None:
    pth_files = sorted(models_dir.glob('*.pth'))
    if not pth_files:
        print(f"No model .pth files found in {models_dir}. Skipping training loss plot.")
        return

    records = []
    for p in pth_files:
        res = load_results_from_model_file(p)
        if not res:
            continue
        # Try to extract accuracy and loss_history
        acc = None
        for key in ['accuracy', 'Accuracy']:
            if key in res:
                try:
                    acc = float(res[key])
                    break
                except Exception:
                    pass

        loss_history = res.get('loss_history') or res.get('losses') or []
        # Normalize to list of floats
        try:
            loss_history = [float(x) for x in loss_history]
        except Exception:
            loss_history = []

        records.append({'path': p, 'accuracy': acc, 'loss_history': loss_history})

    if not records:
        print("No valid result records found inside model files. Skipping training loss plot.")
        return

    # Filter to those with non-empty loss_history
    records = [r for r in records if r['loss_history']]
    if not records:
        print("No models contained loss history. Skipping training loss plot.")
        return

    # Determine best and worst by accuracy (fallback to final loss if accuracy missing)
    records_with_acc = [r for r in records if r['accuracy'] is not None]
    if records_with_acc:
        best = max(records_with_acc, key=lambda x: x['accuracy'])
        worst = min(records_with_acc, key=lambda x: x['accuracy'])
    else:
        # Fallback: use final loss (lower is better)
        for r in records:
            r['final_loss'] = r['loss_history'][-1]
        best = min(records, key=lambda x: x['final_loss'])
        worst = max(records, key=lambda x: x['final_loss'])

    plt.figure(figsize=(8, 5))
    epochs_best = list(range(1, len(best['loss_history']) + 1))
    epochs_worst = list(range(1, len(worst['loss_history']) + 1))
    plt.plot(epochs_best, best['loss_history'], marker='o', label=f"Best ({best.get('accuracy')})")
    plt.plot(epochs_worst, worst['loss_history'], marker='s', label=f"Worst ({worst.get('accuracy')})")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs (Best and Worst Models)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Training Loss plot to {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accuracy and F1 vs Sequence Length
    plot_accuracy_f1_vs_seq_len(EXPERIMENTS_CSV, PLOTS_DIR / 'accuracy_f1_vs_seq_length.png')

    # Plot 2: Training Loss vs Epochs (best and worst models)
    plot_training_loss_best_worst(MODELS_DIR, PLOTS_DIR / 'training_loss_best_worst.png')


if __name__ == '__main__':
    main()
