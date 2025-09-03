import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

# visualize token frequencies from a trained tokenizer
def visualize_token_frequencies(tokenizer, top_k: int = 20, title: str = "Token Frequencies"):
    
    if not tokenizer.trained:
        raise ValueError("Tokenizer must be trained first")
    
    most_common = tokenizer.token_freqs.most_common(top_k)      # get top k most frequent tokens
    tokens, frequencies = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tokens)), frequencies)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def compare_tokenizations(text: str, tokenizers: Dict[str, object]) -> Dict:        # compare tokenizations across different tokenizers
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        if not tokenizer.trained:
            results[name] = {"error": "Tokenizer not trained"}
            continue
            
        try:
            token_ids = tokenizer.encode(text)
            tokens = [tokenizer.id_to_token[id] for id in token_ids]
            decoded = tokenizer.decode(token_ids)
            
            results[name] = {
                "token_ids": token_ids,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "decoded": decoded,
                "vocab_size": tokenizer.get_vocab_size()
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results