import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Use relative import for module
from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.utils import visualize_token_frequencies, evaluate_tokenizer_quality

def load_data(data_path: str) -> list:              # load training data from file
    
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    return lines

def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--data', type=str, default='data/sample_text.txt', help='Path to training data')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Target vocabulary size')
    parser.add_argument('--min-freq', type=int, default=2, help='Minimum frequency for tokens')
    parser.add_argument('--output', type=str, default='models/bpe_tokenizer.json', help='Output path for trained tokenizer')
    parser.add_argument('--visualize', action='store_true', help='Show token frequency visualization')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data from {args.data}...")
    try:                                                    # load training data from file
        corpus = load_data(args.data)
        print(f"Loaded {len(corpus)} text segments")
    except FileNotFoundError:
        print(f"Error: Could not find data file {args.data}")
        return
    
    print(f"\nInitializing BPE tokenizer with vocab_size={args.vocab_size}")
    tokenizer = BPETokenizer(                                                   # initialize BPE tokenizer
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )

    tokenizer.train(corpus)                                   # train tokenizer

    print(f"\nSaving tokenizer to {args.output}")
    tokenizer.save(args.output)

    stats = tokenizer.get_stats()
    print(f"\nTokenizer Statistics:")
    print(f"  Final vocabulary size: {stats['vocab_size']}")
    print(f"  Training completed: {stats['trained']}")
    print(f"  Tokenizer type: {stats['tokenizer_type']}")
    
    print(f"\nMost frequent tokens:")
    for token, freq in stats['most_frequent_tokens']:
        print(f"  '{token}': {freq}")

    merge_rules = tokenizer.get_merge_rules()                       # show merge rules
    print(f"\nLearned {len(merge_rules)} merge rules")
    print("First 10 merge rules:")
    for i, (left, right) in enumerate(merge_rules[:10]):
        merged = left + right
        print(f"  {i+1}. '{left}' + '{right}' -> '{merged}'")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "This is a completely new sentence not in training data."
    ]

    print(f"\nTesting tokenization:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        token_strings = [tokenizer.id_to_token[t] for t in tokens]
        decoded = tokenizer.decode(tokens)
        
        print(f"\nInput: '{text}'")
        print(f"Tokens: {token_strings}")
        print(f"Token IDs: {tokens}")
        print(f"Decoded: '{decoded}'")
        print(f"Compression: {len(text.split())} words -> {len(tokens)} tokens")

    print(f"\nEvaluating on training corpus...")
    quality_metrics = evaluate_tokenizer_quality(tokenizer, corpus)
    print(f"Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.4f}")

    if args.visualize:
        print(f"\nGenerating visualization...")
        try:
            visualize_token_frequencies(tokenizer, top_k=20, 
                                      title=f"BPE Token Frequencies (vocab_size={args.vocab_size})")
        except ImportError:
            print("Matplotlib not available for visualization")
    
    print(f"\nTraining complete! Tokenizer saved to {args.output}")

if __name__ == "__main__":
    main()

