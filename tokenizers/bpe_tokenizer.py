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

        def _merge_pair(self, pair: Tuple[str, str], word_splits: Dict[str, List[str]]) -> Dict[str, List[str]]:

            new_word_splits = {}
            new_token = pair[0] + pair[1]
        
            for word, splits in word_splits.items():
                new_splits = []
                i = 0
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == pair[0] and splits[i + 1] == pair[1]:
                    # Merge the pair
                        new_splits.append(new_token)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                new_word_splits[word] = new_splits
        
            return new_word_splits

    
