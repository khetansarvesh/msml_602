import pandas as pd
import re
from collections import Counter

# Path to the IMDB dataset
DATA_PATH = 'data/IMDB_Dataset.csv'

# Read in chunks to avoid memory issues
chunk_size = 10000
review_lengths = []
vocab_counter = Counter()
total_reviews = 0

# Simple text cleaning and tokenization
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
    reviews = chunk['review'].astype(str)
    for review in reviews:
        tokens = clean_and_tokenize(review)
        review_lengths.append(len(tokens))
        vocab_counter.update(tokens)
        total_reviews += 1

avg_review_length = sum(review_lengths) / len(review_lengths)
vocab_size = len(vocab_counter)

print(f"Total reviews: {total_reviews}")
print(f"Average review length: {avg_review_length:.2f} words")
print(f"Vocabulary size: {vocab_size}")
