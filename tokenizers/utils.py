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

def calculate_compression_ratio(text: str, tokenizer) -> float:     # calculate compression ratio by tokenizer.
    
    if not tokenizer.trained:
        raise ValueError("Tokenizer must be trained first")
    
    original_length = len(text.split())
    tokens = tokenizer.encode(text)
    tokenized_length = len(tokens)
    
    if tokenized_length == 0:
        return 0.0
    
    return original_length / tokenized_length


def analyze_subword_patterns(tokenizer, sample_words: List[str]) -> Dict:       # analyze how tokenizer breaks down specific words.
    
    if not tokenizer.trained:
        raise ValueError("Tokenizer must be trained first")
    
    analysis = {}
    
    for word in sample_words:
        tokens = tokenizer.encode(word)
        token_strings = [tokenizer.id_to_token.get(t, '[UNK]') for t in tokens]
        
        analysis[word] = {
            'tokens': token_strings,
            'num_tokens': len(tokens),
            'token_ids': tokens
        }
    
    return analysis

def calculate_vocabulary_overlap(tokenizer1, tokenizer2) -> Dict:           # calculate vocabulary overlap between two tokenizers.
    
    vocab1 = set(tokenizer1.vocab.keys())
    vocab2 = set(tokenizer2.vocab.keys())
    
    intersection = vocab1.intersection(vocab2)
    union = vocab1.union(vocab2)
    
    return {
        'vocab1_size': len(vocab1),
        'vocab2_size': len(vocab2),
        'intersection_size': len(intersection),
        'union_size': len(union),
        'jaccard_similarity': len(intersection) / len(union) if union else 0.0,
        'overlap_ratio_1': len(intersection) / len(vocab1) if vocab1 else 0.0,
        'overlap_ratio_2': len(intersection) / len(vocab2) if vocab2 else 0.0
    }


def visualize_tokenization_comparison(text: str, tokenizers: Dict[str, object]):            # visualize tokenization results from multiple tokenizers.
    
    results = compare_tokenizations(text, tokenizers)
    
    names = []
    num_tokens = []
    vocab_sizes = []
    
    for name, result in results.items():
        if 'error' not in result:
            names.append(name)
            num_tokens.append(result['num_tokens'])
            vocab_sizes.append(result['vocab_size'])
    
    if not names:
        print("No valid tokenization results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(names, num_tokens)
    ax1.set_title('Number of Tokens per Tokenizer')
    ax1.set_ylabel('Number of Tokens')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(names, vocab_sizes)
    ax2.set_title('Vocabulary Size per Tokenizer')
    ax2.set_ylabel('Vocabulary Size')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTokenization of: '{text}'\n")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ERROR - {result['error']}")
        else:
            print(f"{name}:")
            print(f"  Tokens: {result['tokens']}")
            print(f"  Count: {result['num_tokens']}")
            print(f"  Decoded: '{result['decoded']}'")
            print()


def evaluate_tokenizer_quality(tokenizer, test_corpus: List[str]) -> Dict:              # evaluate tokenizer quality on a test corpus.
    
    if not tokenizer.trained:
        raise ValueError("Tokenizer must be trained first")
    
    total_chars = 0
    total_tokens = 0
    total_words = 0
    unk_count = 0
    
    for text in test_corpus:
        chars = len(text)
        words = len(text.split())
        tokens = tokenizer.encode(text)
        unks = sum(1 for t in tokens if tokenizer.id_to_token[t] == tokenizer.UNK)
        
        total_chars += chars
        total_words += words
        total_tokens += len(tokens)
        unk_count += unks
    
    return {
        'avg_tokens_per_word': total_tokens / total_words if total_words > 0 else 0,
        'compression_ratio': total_words / total_tokens if total_tokens > 0 else 0,
        'unk_rate': unk_count / total_tokens if total_tokens > 0 else 0,
        'chars_per_token': total_chars / total_tokens if total_tokens > 0 else 0,
        'total_texts': len(test_corpus),
        'total_tokens': total_tokens,
        'total_words': total_words,
        'vocab_utilization': len([t for t in tokenizer.token_freqs if tokenizer.token_freqs[t] > 0]) / len(tokenizer.vocab)
    }