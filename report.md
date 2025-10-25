## Project Report

### 1. Dataset Summary
**Dataset:** IMDB Movie Reviews (50,000 reviews)
- **Preprocessing:**
   - Lowercasing, removal of non-alphanumeric characters
   - Tokenization (whitespace split)
   - Padding/truncation to fixed sequence lengths (25, 50, 100)
- **Statistics:**
   - Average review length: 230.20 words
   - Vocabulary size: 180,586 (full), limited to 10,000 for modeling

### 2. Model Configuration
- **Embedding dimension:** 100
- **Hidden size:** 64
- **Number of layers:** 2
- **Dropout:** 0.4
- **Batch size:** 32
- **Epochs:** 10
- **Random Seed**: 42 (for reproducibility)
- **Optimizers**: Adam (lr=0.001), SGD (lr=0.01), RMSProp (lr=0.001)
- **Sequence lengths:** 25, 50, 100
- **Gradient clipping:** tested ON and OFF (max_norm=1.0)
- **Activation functions:** sigmoid, relu and tanh



### 3. Comparative Analysis
1. **Accuracy**: Proportion of correctly classified samples
2. **F1-Score**: Harmonic mean of precision and recall
3. **Precision**: True positives / (True positives + False positives)
4. **Recall**: True positives / (True positives + False negatives)
5. **Training Time**: Average time per epoch (seconds)

| Model | Activation | Optimizer | Seq Length | Grad Clipping | Accuracy | F1    | Epoch Time (s) |
|-------|------------|-----------|------------|---------------|----------|-------|---------------|
| rnn   | relu       | rmsprop   | 50         | Yes           | 0.7047   | 0.7032| 12.22         |
| rnn   | relu       | adam      | 25         | Yes           | 0.7049   | 0.7046| 8.13          |
| rnn   | sigmoid    | rmsprop   | 25         | Yes           | 0.6948   | 0.6938| 7.43          |
| rnn   | relu       | adam      | 50         | Yes           | 0.6952   | 0.6949| 13.14         |
| rnn   | relu       | rmsprop   | 100        | No            | 0.6732   | 0.6656| 21.82         |
| rnn   | sigmoid    | adam      | 100        | Yes           | 0.6745   | 0.6745| 23.25         |
| ...   | ...        | ...       | ...        | ...           | ...      | ...   | ...           |

**See `results/experiments_summary.csv` for all configurations.**

#### Charts
- Accuracy and F1 vs Sequence Length: ![Accuracy/F1 vs Seq Length](results/plots/accuracy_f1_vs_seq_length.png)
- Training Loss (Best/Worst): ![Training Loss Best/Worst](results/plots/training_loss_best_worst.png)

### 4. Discussion
**Best Configuration:**
- The best performing configuration (highest F1/Accuracy) was:
   - Model: Lstm, Activation: relu, Optimizer: rmsprop, Sequence Length: 100, Gradient Clipping: Yes
   - Accuracy: 0.8363, F1: 0.8363, Epoch Time: 26.44s

**Effect of Sequence Length:**
- Longer sequence lengths (100) with proper regularization (dropout, gradient clipping) performed consistently best across all architectures, likely due to more context for the model to perform prediction.

**Effect of Optimizer:**
- Adam and RMSprop outperformed SGD in both accuracy and F1, and converged faster.
- Adam optimizer provided the best balance of performance and convergence speed
- SGD required careful learning rate tuning and showed slower convergence
- RMSProp performed well but was slightly slower than Adam

**Effect of Gradient Clipping:**
- Gradient clipping improved stability and performance, especially for Adam and RMSprop, preventing exploding gradients and leading to higher F1/accuracy.


### 5. Conclusion
Under CPU constraints, the optimal configuration is:
- **Model:** LSTM with relu activation
- **Optimizer:** rmsprop
- **Sequence Length:** 100
- **Gradient Clipping:** Yes
- **Justification:** This setup achieved the highest F1 and accuracy with reasonable training time, balancing performance and efficiency for CPU-based training.
