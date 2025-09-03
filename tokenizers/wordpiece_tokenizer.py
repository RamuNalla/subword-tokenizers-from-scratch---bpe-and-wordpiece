import json
import pickle
import math
import regex as re
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
from .base_tokenizer import BaseTokenizer

class WordPieceTokenizer(BaseTokenizer):            # Implementation with likelihood based merging
    
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2, word_prefix: str = "##", unk_token: str = "[UNK]"):
        super().__init__(vocab_size, min_frequency)
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
    

    def _get_all_subword_pairs(self, word_freqs: Dict[str, int], vocab_set: Set[str]) -> Counter:       # get all adjacent subword pairs and their frequencies.

        pair_counts = Counter()
        
        for word, freq in word_freqs.items():
            subwords = self._split_word_into_subwords(word, vocab_set)
            
            for i in range(len(subwords) - 1):
                pair = (subwords[i], subwords[i + 1])
                pair_counts[pair] += freq
        
        return pair_counts
    

    def _calculate_pair_score(self, pair: Tuple[str, str], pair_count: int, left_count: int, right_count: int) -> float:

        if left_count == 0 or right_count == 0:
            return float('-inf')
        
        return math.log(pair_count / (left_count * right_count))        # WordPiece score: log(pair_count / (left_count * right_count)), higher score means better merges
    

    def train(self, corpus: List[str]) -> None:             # train word piece tokenizer on the corpus
       
        word_freqs = self._get_word_frequencies(corpus)     # get word frequencies
        self.word_freqs = word_freqs
        
        subword_freqs = self._get_initial_subwords(word_freqs)      # initialize with character-level subwords
        
        next_id = len(self.vocab)                                   # add initial subwords to vocabulary
        for subword, freq in subword_freqs.items():
            if freq >= self.min_frequency:
                self.vocab[subword] = next_id
                self.id_to_token[next_id] = subword
                self.token_freqs[subword] = freq
                self.subword_counts[subword] = freq
                next_id += 1
        
        vocab_set = set(self.vocab.keys())
        
        num_merges = self.vocab_size - len(self.vocab)              # Perform WordPiece merges
        
        for merge_num in range(num_merges):
            pair_counts = self._get_all_subword_pairs(word_freqs, vocab_set)            # Get all pairs and their counts
            
            if not pair_counts:
                print(f"No more pairs found. Stopping at {len(self.vocab)} tokens.")
                break
            
            best_pair = None                
            best_score = float('-inf')                          # Calculate scores for all pairs
            
            for pair, pair_count in pair_counts.items():
                if pair_count < self.min_frequency:
                    continue
                
                left_token, right_token = pair
                left_count = self.subword_counts.get(left_token, 0)
                right_count = self.subword_counts.get(right_token, 0)
                
                score = self._calculate_pair_score(pair, pair_count, left_count, right_count)
                
                if score > best_score:
                    best_score = score
                    best_pair = pair
            
            if best_pair is None:
                print(f"No valid pairs found. Stopping at {len(self.vocab)} tokens.")
                break
            
            left_token, right_token = best_pair                 # Create new merged token
            if right_token.startswith(self.word_prefix):
                new_token = left_token + right_token[len(self.word_prefix):]        # Remove prefix when merging
            else:
                new_token = left_token + right_token
            
            self.vocab[new_token] = next_id                     # Add new token to vocabulary
            self.id_to_token[next_id] = new_token
            
            merged_count = pair_counts[best_pair]               # Update counts
            self.token_freqs[new_token] = merged_count
            self.subword_counts[new_token] = merged_count
            
            vocab_set.add(new_token)                            # Update vocabulary set
            next_id += 1
            
            if (merge_num + 1) % 100 == 0:
                print(f"Completed {merge_num + 1} merges. Vocab size: {len(self.vocab)}")
                print(f"Best pair: {best_pair} -> {new_token} (score: {best_score:.4f})")
        
        self.trained = True
        print(f"Training complete! Final vocabulary size: {len(self.vocab)}")


    def encode(self, text: str) -> List[int]:               # encode text into token IDs with wordpiece

        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        preprocessed = self._preprocess_text(text)
        words = preprocessed.split()
        
        token_ids = []
        vocab_set = set(self.vocab.keys())
        
        for word in words:
            
            subwords = self._split_word_into_subwords(word, vocab_set)      # split word into subwords
            
            for subword in subwords:                                        # convert to IDs
                if subword in self.vocab:
                    token_ids.append(self.vocab[subword])
                else:
                    token_ids.append(self.vocab[self.UNK])
        
        return token_ids
    

    def _postprocess_tokens(self, tokens: List[str]) -> str:        # postprocess wordpiece tokens back to readable text
        
        cleaned_tokens = [t for t in tokens if t not in {self.PAD, self.BOS, self.EOS, self.UNK}]   # remove special tokens
        
        words = []
        current_word = ""
        
        for token in cleaned_tokens:
            if token.startswith(self.word_prefix):
                current_word += token[len(self.word_prefix):]           # continuation of current word
            else:
                if current_word:                                        # start of new word
                    words.append(current_word)
                current_word = token
        
        if current_word:
            words.append(current_word)                                  # add the last word
        
        return ' '.join(words)


    def get_subword_vocab(self) -> Dict[str, int]:                      # get the subword vocab frequencies
        return dict(self.subword_counts)
    
    
    def analyze_word(self, word: str) -> List[str]:                     # analyze how a word split into subwords

        if not self.trained:
            raise ValueError("Tokenizer must be trained before analysis")
        
        vocab_set = set(self.vocab.keys())
        return self._split_word_into_subwords(word.lower(), vocab_set)

    def save(self, filepath: str) -> None:                              # save the wordpiece tokenizer to a file
        
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'word_prefix': self.word_prefix,
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'token_freqs': dict(self.token_freqs),
            'subword_counts': dict(self.subword_counts),
            'word_freqs': dict(self.word_freqs) if hasattr(self, 'word_freqs') else {},
            'trained': self.trained,
            'tokenizer_type': 'WordPieceTokenizer'
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    
    def load(self, filepath: str) -> None:                              # load the wordpiece tokenizer from a file
        
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.min_frequency = data['min_frequency']
        self.word_prefix = data['word_prefix']
        self.vocab = data['vocab']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.token_freqs = Counter(data['token_freqs'])
        self.subword_counts = Counter(data['subword_counts'])
        self.word_freqs = data.get('word_freqs', {})
        self.trained = data['trained']