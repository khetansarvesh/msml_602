2. Dataset Selection
Dataset: IMDb Movie Review Dataset (50,000 reviews)
Dataset Preparation Guidelines
Use the predefined 50/50 split (25k for training, 25k for testing).
Preprocess the text as follows:


Lowercase all text.
Remove punctuation and special characters.
Tokenize sentences (use Keras Tokenizer or nltk.word_tokenize).
Keep only the top 10,000 most frequent words.
Convert each review to a sequence of token IDs.
Pad or truncate sequences to fixed lengths of 25, 50, and 100 words (you will test these variations).
