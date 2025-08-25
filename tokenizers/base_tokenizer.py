import json
import pickle
import regex as re
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import Counter, defaultdict

class BaseTokenizer(ABC):               # Abstract base class for subword tokenizers

    def __init__(self, vocab_size: int=1000, min_frequency: int = 2):           # Input: Target vocabulary size, Min frequency for tokens to be considered
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = {}                 # token to id mapping
        self.id_to_token = {}           # id to token mapping
        self.token_freqs = Counter()    # token frequencies
        self.trained = False

        self.UNK = "<UNK>"              # special tokens
        self.PAD = "<PAD>"
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"

        self._init_special_tokens()     # Initialize with special tokens
    


    def _init_special_tokens(self):             # Initialize special tokens

        special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
        

    def _preprocess_text(self, text: str) -> str:       # Raw input text to preprocessed text

        text = re.sub(r'\s+', " ", text.strip())        # Normalize white space
        text = re.sub(r'([.!?;,:()])', r" \1 ", text)   # add spaces around punctuation for better tokenization
        text = re.sub(r'\s+', " ", text)                # clean up extra spaces

        return text.lower()


    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:       # gets word frequencies from corpus

        word_freqs = Counter()
        for text in corpus:
            preprocessed = self._preprocess_text(text)
            words = preprocessed.split()
            word_freqs.update(words)
        
        return dict(word_freqs)


    @abstractmethod
    def train(self, corpus: List[str]) -> None:     # train the tokenizer on a corpus
        pass

    
    @abstractmethod
    def encode(self, text: str) -> List[int]:       # encode text into token_ids
        pass
    

    def decode(self, token_ids: List[int]) -> str:  # decode token ids back to text

        if not self.trained:
            raise ValueError("Tokenizer must be trained before decoding")

        tokens = []

        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.UNK)
        
        return self._postprocess_tokens(tokens)


    def _postprocess_tokens(self, tokens: List[str]) -> str:        # Postprocess tokens back to readable text

        cleaned_tokens = [t for t in tokens if t not in {self.PAD, self.BOS, self.EOS}]     # Remove special tokens for output

        return " ".join(cleaned_tokens)


    def get_vocab_size(self) -> int:
        return len(self.vocab)


    def get_token_frequency(self, token: str) -> int:
        return self.token_freqs.get(token, 0)


    def save(self, filepath: str) -> None:          # save tokenizer to a file

        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'token_freqs': dict(self.token_freqs),
            'trained': self.trained,
            'tokenizer_type': self.__class__.__name__
        }

        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    

    def load(self, filepath: str) -> None:          # load tokenizer from file

        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.min_frequency = data['min_frequency']
        self.vocab = data['vocab']

        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.token_freqs = Counter(data['token_freqs'])
        self.trained = data['trained']
    

    def get_stats(self) -> Dict:            # Returns tokenizer statistics

        return {
            'vocab_size': self.get_vocab_size(),
            'trained': self.trained,
            'most_frequent_tokens': self.token_freqs.most_common(10),
            'tokenizer_type': self.__class__.__name__
        }




    
