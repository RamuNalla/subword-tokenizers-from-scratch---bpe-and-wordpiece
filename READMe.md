# Subword Tokenizers From Scratch

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational implementation of subword tokenization algorithms - Byte-Pair Encoding (BPE) and WordPiece tokenizers, built from scratch for learning and research purposes.

## Overview

This project provides clean and well-documented implementations of two fundamental subword tokenization algorithms:

- **Byte-Pair Encoding (BPE)**: The algorithm behind GPT and many transformer models
- **WordPiece**: Google's tokenization method used in BERT and related models

## Algorithms Explained

### Byte-Pair Encoding (BPE)

1. **Initialize**: Start with character-level vocabulary
2. **Count Pairs**: Find most frequent adjacent character pairs
3. **Merge**: Replace most frequent pair with new token
4. **Repeat**: Continue until desired vocabulary size
5. **Apply**: Use learned merge rules for tokenization

**Key Characteristics**:
- Frequency-driven merging
- Handles out-of-vocabulary words well

### WordPiece

WordPiece uses likelihood maximization for merging:

1. **Initialize**: Start with character vocabulary + prefix notation
2. **Score Pairs**: Calculate likelihood score for each pair
3. **Select Best**: Choose pair with highest likelihood improvement
4. **Merge**: Create new subword token
5. **Iterate**: Continue until vocabulary target reached

**Key Characteristics**:
- Likelihood-based optimization
- Prefix notation (##) for subwords
- Used in BERT and modern transformers

### Algorithm Comparison

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge Strategy | Most frequent pair | Highest likelihood score |
| Optimization | Compression ratio | Data likelihood |
| Complexity | Simpler | More sophisticated |
| Use Cases | GPT family, Translation | BERT family, Classification |

## Usage Examples

### Compare Different Tokenizers

```python
from tokenizers.utils import compare_tokenizations, visualize_tokenization_comparison

bpe = BPETokenizer()
bpe.load('models/bpe_tokenizer.json')

wordpiece = WordPieceTokenizer()
wordpiece.load('models/wordpiece_tokenizer.json')

text = "Tokenization is preprocessing text for machine learning."
tokenizers = {'BPE': bpe, 'WordPiece': wordpiece}

results = compare_tokenizations(text, tokenizers)
visualize_tokenization_comparison(text, tokenizers)
```

### Analyze Tokenizer Quality

```python
from tokenizers.utils import evaluate_tokenizer_quality

test_texts = ["Your test sentences here..."]
metrics = evaluate_tokenizer_quality(bpe, test_texts)

print("Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Custom Tokenizer Training

```python
custom_bpe = BPETokenizer(
    vocab_size=2000,
    min_frequency=10,
    word_end_token="<|endofword|>"
)

domain_corpus = load_domain_specific_data()
custom_bpe.train(domain_corpus)

merge_rules = custom_bpe.get_merge_rules()
print(f"Learned {len(merge_rules)} merge rules")
```

## Project Structure

```
subword-tokenizers/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                    # Training data
│   ├── sample_text.txt     # Educational dataset
├── tokenizers/             # Core implementation
│   ├── __init__.py
│   ├── base_tokenizer.py   # Abstract base class
│   ├── bpe_tokenizer.py    # BPE implementation
│   ├── wordpiece_tokenizer.py  # WordPiece implementation
│   └── utils.py            # Analysis and visualization
├── examples/               # Usage examples
│   ├── train_bpe.py       # BPE training script
│   ├── train_wordpiece.py   # WordPiece training script
│   └── interactive_test.py  # Comparison utilities
└── models/                  # Saved tokenizers (created)
```

