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