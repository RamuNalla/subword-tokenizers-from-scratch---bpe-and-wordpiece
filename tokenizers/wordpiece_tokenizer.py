import json
import pickle
import math
import regex as re
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
from .base_tokenizer import BaseTokenizer

class WordPieceTokenizer(BaseTokenizer):            # Implementation with likelihood based merging
    
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2, word_prefix: str = "##", unk_token: str = "[UNK]"):
        super.__init__(vocab_size, min_frequency)

        self.word_prefix = word_prefix
        self.UNK = unk_token
        self._update_special_tokens()

        self.subword_counts = Counter()
        self.pair_counts = Counter()


    def _update_special_tokens(self):
        self.vocab.clear()
        self.id_to_token.clear()
        
        special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
    

    def _get_initial_subwords(self, word_freqs: Dict[str, int]) -> Dict[str, int]:      # get initial subword vocabulary from character level

        subword_freqs = Counter()
        
        for word, freq in word_freqs.items():
            if word:
                subword_freqs[word[0]] += freq
                
                for char in word[1:]:
                    prefixed_char = self.word_prefix + char
                    subword_freqs[prefixed_char] += freq
        
        return dict(subword_freqs)
    
    
    def _split_word_into_subwords(self, word: str, vocab_set: Set[str]) -> List[str]:       # split a word into subwords

        if not word:
            return []
        
        subwords = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = False
            
            while start < end:                   # find the longest subword starting at 'start'
                subword = word[start:end]
                
                if start > 0:                   # Add prefix if not at word beginning
                    subword = self.word_prefix + subword
                
                if subword in vocab_set:
                    subwords.append(subword)
                    start = end
                    found = True
                    break
                
                end -= 1
            
            if not found:
                return [self.UNK]               # Cannot split further, return UNK
        
        return subwords