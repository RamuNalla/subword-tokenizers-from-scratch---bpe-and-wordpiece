import json
import pickle
import math
import regex as re
from typing import List, Tuple, Dict, set
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
