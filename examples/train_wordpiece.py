import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tokenizers.wordpiece_tokenizer import WordPieceTokenizer
from tokenizers.utils import visualize_token_frequencies, evaluate_tokenizer_quality

def load_data(data_path: str) -> list:              # load training data from file.
    
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]      # split into sentences/paragraphs for training
    return lines

def main():
    parser = argparse.ArgumentParser(description='Train WordPiece tokenizer')
    parser.add_argument('--data', type=str, default='data/sample_text.txt', help='Path to training data')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Target vocabulary size')
    parser.add_argument('--min-freq', type=int, default=2, help='Minimum frequency for tokens')
    parser.add_argument('--output', type=str, default='models/wordpiece_tokenizer.json', help='Output path for trained tokenizer')
    parser.add_argument('--visualize', action='store_true', help='Show token frequency visualization')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data from {args.data}...")
    try:
        corpus = load_data(args.data)
        print(f"Loaded {len(corpus)} text segments")
    except FileNotFoundError:
        print(f"Error: Could not find data file {args.data}")
        return

    print(f"\nInitializing WordPiece tokenizer with vocab_size={args.vocab_size}")
    tokenizer = WordPieceTokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )
    
    tokenizer.train(corpus)
    
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

    subword_vocab = tokenizer.get_subword_vocab()
    print(f"\nSubword vocabulary size: {len(subword_vocab)}")
    
    print(f"\nToken type examples:")
    char_tokens = [t for t in tokenizer.vocab.keys() if len(t) == 1 and t.isalpha()]
    prefixed_tokens = [t for t in tokenizer.vocab.keys() if t.startswith(tokenizer.word_prefix)]
    regular_tokens = [t for t in tokenizer.vocab.keys() 
                     if not t.startswith(tokenizer.word_prefix) 
                     and len(t) > 1 
                     and t not in {tokenizer.PAD, tokenizer.UNK, tokenizer.BOS, tokenizer.EOS}]
    
    print(f"  Character tokens: {char_tokens[:10]}")
    print(f"  Prefixed tokens: {prefixed_tokens[:10]}")
    print(f"  Regular tokens: {regular_tokens[:10]}")

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "This is a completely new sentence not in training data.",
        "Preprocessing and tokenization are fundamental steps."
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
    
    print(f"\nWord analysis examples:")
    test_words = ["tokenization", "preprocessing", "artificial", "intelligence", "understanding"]
    for word in test_words:
        subwords = tokenizer.analyze_word(word)
        print(f"  '{word}' -> {subwords}")
    
    print(f"\nEvaluating on training corpus...")
    quality_metrics = evaluate_tokenizer_quality(tokenizer, corpus)
    print(f"Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    if args.visualize:
        print(f"\nGenerating visualization...")
        try:
            visualize_token_frequencies(tokenizer, top_k=20, 
                                      title=f"WordPiece Token Frequencies (vocab_size={args.vocab_size})")
        except ImportError:
            print("Matplotlib not available for visualization") 

    print(f"\nTraining complete! Tokenizer saved to {args.output}")

if __name__ == "__main__":
    main()