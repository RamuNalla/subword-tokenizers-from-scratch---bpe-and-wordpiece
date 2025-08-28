import json
import pickle
import regex as re
from typing import List, Dict, Tuple, Self
from collections import Counter, defaultdict
from .base_tokenizer import BaseTokenizer

class BPETokenizer(BaseTokenizer):

    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2, word_end_token: str = "</w>"):

        super().__init__(vocab_size, min_frequency)
        self.word_end_token = word_end_token
        self.merges = {}                            # (pair_tuple) --> merge_token mapping
        self.merge_order = []                       # list of merges in the order they were learned

    
    def _get_character_level_vocab(self, word_freqs: Dict[str, int]) -> Dict[str, int]:     # vocabulary with character level tokens

        char_freqs = Counter()

        for word, freq in word_freqs.items():
            chars = list(word) + [self.word_end_token]          # split word into characters and add word-end token
            for char in chars:
                char_freqs[char] += freq
        
        return dict(char_freqs)


    def _get_pairs(self, word_splits: Dict[str, List[str]]) -> Counter:             # Gets all adjacent pairs from word splits

        pairs = Counter()

        for word, splits in word_splits.items():
            for i in range(len(splits)-1):
                pairs[(splits[i], splits[i+1])] += self.word_freqs[word]
        
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], word_splits: Dict[str, List[str]]) -> Dict[str, List[str]]:        # Merge a specific pair in all word splits

        new_word_splits = {}
        new_token = pair[0] + pair[1]

        for word, splits in word_splits.items():
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits)-1 and splits[i] == pair[0] and splits[i+1] == pair[1]:
                    new_splits.append(new_token)
                    i += 2
                else:
                    new_splits.append(splits[i])
            new_splits[word] = new_splits
        
        return new_word_splits


    def train(self, corpus: List[str]) -> None:         # Train BPE on corpus

        word_freqs = self._get_word_frequencies(corpus)
        self.word_freqs = word_freqs

        char_freqs = self._get_character_level_vocab(word_freqs)        # Initialize with character level vocabulary

        next_id = len(self.vocab)

        for char, freq in char_freqs.items():
            if freq >= self.min_frequency:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                self.token_freqs[char] = freq
                next_id += 1
        
        word_splits = {}                            # Initialize word splits (each word split into characters + word end tokens)
        for word in word_freqs:
            word_splits[word] = list(word) + [self.word_end_token]
        
        num_merges = self.vocab_size - len(self.vocab)          # BPE merges

        for merge_num in range(num_merges):

            pairs = self._get_pairs(word_splits)

            if not pairs:
                print(f"No more pairs to merge. Stopping ar {len(self.vocab)} tokens")
            
            best_pair = pairs.most_common(1)[0][0]
            best_freq = pairs[best_pair]

            if best_freq < self.min_frequency:
                print(f"Best pair frequency ({best_freq}) below theshold. Stopping")
                break
            
            new_token = best_pair[0] + best_pair[1]     # create new token by merging pait

            self.vocab[new_token] = next_id
            self.id_to_token[next_id] = new_token
            self.token_freqs[new_token] = best_freq
            next_id += 1

            self.merges[best_pair] = new_token
            self.merge_order.append(best_pair)

            word_splits = self._merge_pair(best_pair, word_splits)

            if (merge_num + 1) % 100 == 0:
                print(f"Completed {merge_num + 1} merges. Vocab size: {len(self.vocab)}")
        
        self.trained = True
        print(f"Training complete! Final vocabulary size: {len(self.vocab)}")

    
    def _apply_bpe(self, word: str) -> List[str]:       # Apply learned BPE to a word

        tokens = list(word) + [self.word_end_token]

        for pair in self.merge_order:
            i=0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    merged_token = self.merges[pair]
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                else:
                    i += 1
        
        return tokens

    









    
