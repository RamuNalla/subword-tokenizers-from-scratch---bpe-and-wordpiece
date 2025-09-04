import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.wordpiece_tokenizer import WordPieceTokenizer

def load_tokenizers():
    tokenizers = {}
    
    bpe_path = "models/bpe_tokenizer.json"              # load BPE tokenizer
    if os.path.exists(bpe_path):
        bpe = BPETokenizer()
        bpe.load(bpe_path)
        tokenizers['BPE'] = bpe
        print("✓ BPE tokenizer loaded")
    else:
        print("✗ BPE tokenizer not found - train it first with: python examples/train_bpe.py")
    
    wp_path = "models/wordpiece_tokenizer.json"     # load WordPiece tokenizer
    if os.path.exists(wp_path):
        wp = WordPieceTokenizer()
        wp.load(wp_path)
        tokenizers['WordPiece'] = wp
        print("✓ WordPiece tokenizer loaded")
    else:
        print("✗ WordPiece tokenizer not found - train it first with: python examples/train_wordpiece.py")
    
    return tokenizers

def main():
    print("=== Interactive Tokenizer Testing ===\n")
    
    tokenizers = load_tokenizers()              # Load available tokenizers
    
    if not tokenizers:
        print("No tokenizers available. Please train them first!")
        return
    
    print(f"\nAvailable tokenizers: {list(tokenizers.keys())}")
    print("\nEnter text to tokenize (or 'quit' to exit):\n")
    
    while True:
        try:
            text = input(">>> ").strip()            # Read user input
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            print(f"\nTokenizing: '{text}'\n")
            
            for name, tokenizer in tokenizers.items():          # Tokenize with each available tokenizer
                print(f"--- {name} ---")
                
                try:
                    token_ids = tokenizer.encode(text)
                    tokens = [tokenizer.id_to_token[tid] for tid in token_ids]
                    decoded = tokenizer.decode(token_ids)
                    
                    print(f"Tokens: {tokens}")
                    print(f"Token IDs: {token_ids}")
                    print(f"Token count: {len(tokens)}")
                    print(f"Decoded: '{decoded}'")
                    
                    original_words = len(text.split())              # Calculate compression ratio
                    compression = original_words / len(tokens) if tokens else 0
                    print(f"Compression: {original_words} words → {len(tokens)} tokens ({compression:.2f}x)")
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                print()
            
            print("-" * 50)
            print()
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            print("\n\nExiting...")
            break
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
