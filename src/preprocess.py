import os
import re
import pickle
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize

from .config import DATA_DIR, VOCAB_SIZE, RANDOM_SEED
from .utils import set_random_seeds

from typing import Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Lowercase all text.
# Remove punctuation and special characters.
# Tokenize sentences (use Keras Tokenizer or nltk.word_tokenize).
# Keep only the top 10,000 most frequent words.
# Convert each review to a sequence of token IDs.
# Pad or truncate sequences to fixed lengths of 25, 50, and 100 words (you will test these variations).


class TextPreprocessor:
    """
    Handles text cleaning and normalization operations.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        # Create translation table for punctuation removal
        self.punct_translator = str.maketrans('', '', string.punctuation)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text by converting to lowercase and removing punctuation.
        
        Requirements addressed:
        - 1.1: Convert all text to lowercase
        - 1.2: Remove all punctuation and special characters
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text with lowercase and no punctuation
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase (Requirement 1.1)
        text = text.lower()
        
        # Remove punctuation and special characters (Requirement 1.2)
        text = text.translate(self.punct_translator)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words using NLTK tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens (words)
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Tokenize using NLTK
        tokens = word_tokenize(cleaned_text)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens


class SequenceProcessor:
    """
    Manages tokenization, vocabulary building, and sequence padding operations.
    Implements requirements 1.3, 1.4, 1.5 for sequence processing.
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        """
        Initialize the sequence processor.
        
        Args:
            vocab_size (int): Maximum vocabulary size (default from config)
        """
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.preprocessor = TextPreprocessor()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_IDX = 0
        self.UNK_IDX = 1
    
    def build_vocabulary(self, texts: List[str], vocab_size: int = None) -> Dict[str, int]:
        """
        Build vocabulary from texts using top most frequent words.
        
        Requirements addressed:
        - 1.3: Use the top 10,000 most frequent words only
        
        Args:
            texts (List[str]): List of texts to build vocabulary from
            vocab_size (int, optional): Maximum vocabulary size
            
        Returns:
            Dict[str, int]: Word to index mapping
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        print(f"Building vocabulary with top {vocab_size} words...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.preprocessor.tokenize(text)
            word_counts.update(tokens)
        
        # Get top vocab_size - 2 words (reserve space for PAD and UNK tokens)
        most_common_words = word_counts.most_common(vocab_size - 2)
        
        # Build word to index mapping
        word_to_idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        
        # Add most frequent words
        for idx, (word, count) in enumerate(most_common_words, start=2):
            word_to_idx[word] = idx
        
        # Build reverse mapping
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Store vocabulary information
        self.word_counts = word_counts
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocabulary = word_to_idx  # For backward compatibility
        
        print(f"Vocabulary built: {len(word_to_idx)} words")
        print(f"Most common words: {[word for word, _ in most_common_words[:10]]}")
        
        return word_to_idx
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of token IDs using the built vocabulary.
        
        Requirements addressed:
        - 1.4: Convert each review to token ID sequences
        
        Args:
            texts (List[str]): List of texts to convert
            
        Returns:
            List[List[int]]: List of token ID sequences
        """
        if not self.word_to_idx:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        sequences = []
        unknown_word_count = 0
        total_tokens = 0
        
        for text in texts:
            tokens = self.preprocessor.tokenize(text)
            total_tokens += len(tokens)
            
            # Convert tokens to IDs, using UNK_IDX for unknown words
            sequence = []
            for token in tokens:
                if token in self.word_to_idx:
                    sequence.append(self.word_to_idx[token])
                else:
                    sequence.append(self.UNK_IDX)
                    unknown_word_count += 1
            
            sequences.append(sequence)
        
        # Print conversion statistics
        if total_tokens > 0:
            unknown_rate = (unknown_word_count / total_tokens) * 100
            print(f"Text-to-sequence conversion completed:")
            print(f"  - Total tokens: {total_tokens:,}")
            print(f"  - Unknown tokens: {unknown_word_count:,} ({unknown_rate:.2f}%)")
            print(f"  - Sequences created: {len(sequences):,}")
        
        return sequences
    
    def process_sequences_with_validation(self, texts: List[str], sequence_length: int, 
                                        show_stats: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Complete sequence processing pipeline with validation and statistics.
        
        This method combines text-to-sequence conversion, padding/truncation, and statistics
        computation in a single call with validation.
        
        Args:
            texts (List[str]): List of texts to process
            sequence_length (int): Target sequence length (must be 25, 50, or 100)
            show_stats (bool): Whether to print statistics
            
        Returns:
            Tuple[np.ndarray, Dict]: Processed sequences and statistics
        """
        # Validate sequence length
        self._validate_sequence_length(sequence_length)
        
        # Convert texts to sequences
        sequences = self.texts_to_sequences(texts)
        
        # Get statistics before padding
        stats = self.get_sequence_statistics(sequences)
        
        # Pad sequences to target length
        padded_sequences = self.pad_sequences(sequences, sequence_length)
        
        # Add padding statistics
        stats["target_length"] = sequence_length
        stats["final_shape"] = padded_sequences.shape
        
        if show_stats:
            self._print_sequence_statistics(stats)
        
        return padded_sequences, stats
    
    def _print_sequence_statistics(self, stats: Dict):
        """
        Print formatted sequence statistics.
        
        Args:
            stats (Dict): Statistics dictionary from get_sequence_statistics
        """
        print(f"\n=== Sequence Statistics ===")
        print(f"Total sequences: {stats['total_sequences']:,}")
        print(f"Length range: {stats['min_length']} - {stats['max_length']} tokens")
        print(f"Mean length: {stats['mean_length']:.1f} tokens")
        print(f"Median length: {stats['median_length']:.1f} tokens")
        print(f"Standard deviation: {stats['std_length']:.1f} tokens")
        
        print(f"\nLength percentiles:")
        for pct, value in stats['length_percentiles'].items():
            print(f"  {pct}: {value:.1f} tokens")
        
        print(f"\nLength distribution:")
        for bin_range, count in stats['length_distribution'].items():
            pct = (count / stats['total_sequences']) * 100
            print(f"  {bin_range} tokens: {count:,} sequences ({pct:.1f}%)")
        
        if 'target_length' in stats:
            target_length = stats['target_length']
            truncation_info = stats['truncation_stats'][f'length_{target_length}']
            print(f"\nPadding/Truncation for length {target_length}:")
            print(f"  Truncated: {truncation_info['truncated']:,} ({truncation_info['truncated_pct']:.1f}%)")
            print(f"  Padded: {truncation_info['padded']:,} ({truncation_info['padded_pct']:.1f}%)")
            print(f"  Exact length: {truncation_info['exact']:,} ({truncation_info['exact_pct']:.1f}%)")
            print(f"  Final shape: {stats['final_shape']}")
        
        print("=" * 30)
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int) -> np.ndarray:
        """
        Pad or truncate sequences to a fixed length.
        
        Requirements addressed:
        - 1.5: Pad or truncate all sequences to specified length (25, 50, or 100)
        
        Args:
            sequences (List[List[int]]): List of token ID sequences
            max_length (int): Target sequence length
            
        Returns:
            np.ndarray: Padded sequences array of shape (num_sequences, max_length)
        """
        # Validate sequence length parameter
        self._validate_sequence_length(max_length)
        
        padded_sequences = np.full((len(sequences), max_length), self.PAD_IDX, dtype=np.int32)
        
        for i, sequence in enumerate(sequences):
            if len(sequence) > max_length:
                # Truncate if too long
                padded_sequences[i] = sequence[:max_length]
            else:
                # Pad if too short
                padded_sequences[i, :len(sequence)] = sequence
        
        return padded_sequences
    
    def _validate_sequence_length(self, sequence_length: int):
        """
        Validate that sequence length is one of the supported values.
        
        Args:
            sequence_length (int): The sequence length to validate
            
        Raises:
            ValueError: If sequence length is not 25, 50, or 100
        """
        valid_lengths = [25, 50, 100]
        if sequence_length not in valid_lengths:
            raise ValueError(f"Sequence length must be one of {valid_lengths}, got {sequence_length}")
    
    def get_sequence_statistics(self, sequences: List[List[int]]) -> Dict:
        """
        Compute comprehensive statistics about sequence lengths.
        
        Args:
            sequences (List[List[int]]): List of token ID sequences
            
        Returns:
            Dict: Dictionary containing sequence length statistics
        """
        if not sequences:
            return {"error": "No sequences provided"}
        
        lengths = [len(seq) for seq in sequences]
        
        stats = {
            "total_sequences": len(sequences),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "std_length": np.std(lengths),
            "length_percentiles": {
                "25th": np.percentile(lengths, 25),
                "75th": np.percentile(lengths, 75),
                "90th": np.percentile(lengths, 90),
                "95th": np.percentile(lengths, 95)
            },
            "length_distribution": self._get_length_distribution(lengths),
            "truncation_stats": self._get_truncation_stats(lengths)
        }
        
        return stats
    
    def _get_length_distribution(self, lengths: List[int]) -> Dict:
        """
        Get distribution of sequence lengths in bins.
        
        Args:
            lengths (List[int]): List of sequence lengths
            
        Returns:
            Dict: Length distribution statistics
        """
        length_counts = Counter(lengths)
        
        # Create bins for common sequence lengths
        bins = {
            "0-10": 0,
            "11-25": 0,
            "26-50": 0,
            "51-100": 0,
            "101-200": 0,
            "201-500": 0,
            "500+": 0
        }
        
        for length in lengths:
            if length <= 10:
                bins["0-10"] += 1
            elif length <= 25:
                bins["11-25"] += 1
            elif length <= 50:
                bins["26-50"] += 1
            elif length <= 100:
                bins["51-100"] += 1
            elif length <= 200:
                bins["101-200"] += 1
            elif length <= 500:
                bins["201-500"] += 1
            else:
                bins["500+"] += 1
        
        return bins
    
    def _get_truncation_stats(self, lengths: List[int]) -> Dict:
        """
        Get statistics about how many sequences would be truncated at different lengths.
        
        Args:
            lengths (List[int]): List of sequence lengths
            
        Returns:
            Dict: Truncation statistics for different sequence lengths
        """
        total_sequences = len(lengths)
        
        truncation_stats = {}
        for target_length in [25, 50, 100]:
            truncated_count = sum(1 for length in lengths if length > target_length)
            padded_count = sum(1 for length in lengths if length < target_length)
            exact_count = sum(1 for length in lengths if length == target_length)
            
            truncation_stats[f"length_{target_length}"] = {
                "truncated": truncated_count,
                "padded": padded_count,
                "exact": exact_count,
                "truncated_pct": (truncated_count / total_sequences) * 100,
                "padded_pct": (padded_count / total_sequences) * 100,
                "exact_pct": (exact_count / total_sequences) * 100
            }
        
        return truncation_stats
    
    def get_vocabulary_stats(self) -> Dict:
        """
        Get statistics about the built vocabulary.
        
        Returns:
            Dict: Vocabulary statistics
        """
        if not self.word_to_idx:
            return {"error": "Vocabulary not built"}
        
        return {
            "vocab_size": len(self.word_to_idx),
            "total_words_seen": len(self.word_counts),
            "most_common_words": self.word_counts.most_common(10),
            "coverage": len(self.word_to_idx) / len(self.word_counts) if self.word_counts else 0
        }
    
    def save_vocabulary(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath (str): Path to save vocabulary
        """
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_counts': dict(self.word_counts),
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """
        Load vocabulary from file.
        
        Args:
            filepath (str): Path to load vocabulary from
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.word_counts = Counter(vocab_data['word_counts'])
        self.vocab_size = vocab_data['vocab_size']
        self.vocabulary = self.word_to_idx  # For backward compatibility
        
        print(f"Vocabulary loaded from {filepath}: {len(self.word_to_idx)} words")


class SentimentDataset(Dataset):
    """
    PyTorch Dataset class for sentiment classification.
    """
    
    def __init__(self, texts: List[str], labels: List[int]):
        """
        Initialize the dataset.
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of sentiment labels (0 or 1)
        """
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


class IMDbDataLoader:

    def __init__(self, data_dir: str = DATA_DIR, filename: str = 'IMDB_Dataset.csv'):
        self.data_dir = data_dir
        self.filename = filename
        self.filepath = os.path.join(self.data_dir, self.filename)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize expected columns to 'text' and 'sentiment'"""
        # Common column name mappings
        col_map = {}
        lower_cols = {c.lower(): c for c in df.columns}

        # Map text column
        for candidate in ['review', 'text', 'content']:
            if candidate in lower_cols:
                col_map[lower_cols[candidate]] = 'text'
                break

        # Map sentiment/label column
        for candidate in ['sentiment', 'label', 'sentiment_label', 'polarity']:
            if candidate in lower_cols:
                col_map[lower_cols[candidate]] = 'sentiment'
                break

        df = df.rename(columns=col_map)

        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain review text and sentiment label columns")

        # Convert sentiment to binary 0/1 if needed
        if df['sentiment'].dtype == object:
            df['sentiment'] = df['sentiment'].str.strip().str.lower()
            df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1}).astype(int)
        else:
            # Ensure integers
            df['sentiment'] = df['sentiment'].astype(int)

        return df[['text', 'sentiment']]

    def load_data(self) -> Tuple[list, list, list, list]:

        df = pd.read_csv(self.filepath)

        set_random_seeds(RANDOM_SEED)
        df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        mid = len(df_shuffled) // 2
        train_df = df_shuffled.iloc[:mid].reset_index(drop=True)
        test_df = df_shuffled.iloc[mid:].reset_index(drop=True)

        # Standardize and return lists
        train_df = self._standardize_dataframe(train_df)
        test_df = self._standardize_dataframe(test_df)

        return train_df['text'].astype(str).tolist(), train_df['sentiment'].astype(int).tolist(), test_df['text'].astype(str).tolist(), test_df['sentiment'].astype(int).tolist()
