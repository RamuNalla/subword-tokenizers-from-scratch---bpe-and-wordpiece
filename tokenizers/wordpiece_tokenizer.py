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
        