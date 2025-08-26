import json
import pickle
import regex as re
from typing import List, Dict, Tuple, Self
from collections import Counter, defaultdict
from .base_tokenizer import BaseTokenizer

class BPETokenizer(BaseTokenizer):

    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2, word_end_token: str = "</w>"):

        super().__init__(vocab_size, min_frequency)