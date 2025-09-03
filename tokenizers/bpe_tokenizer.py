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
                    i += 1
            new_word_splits[word] = new_splits
        
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

    
    def encode(self, text: str) -> List[int]:       # Encodes text into tokenids using BPE

        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding")

        preprocessed = self._preprocess_text(text)
        words = preprocessed.split()

        token_ids = []

        for word in words:
            subword_tokens = self._apply_bpe(word)

            for token in subword_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.vocab[self.UNK])
        
        return token_ids
    

    def _postprocess_tokens(self, tokens: List[str]) -> str:        # Postprocess BPE tokens back to readable text

        cleaned_tokens = [t for t in tokens if t not in {self.PAD, self.BOS, self.EOS, self.UNK}]

        text_parts = []
        current_word = ""

        for token in cleaned_tokens:
            if token.endswith(self.word_end_token):
                current_word += token[:-len(self.word_end_token)]

                text_parts.append(current_word)
                current_word = ""
            else:
                current_word += token
        
        if current_word:
            text_parts.append(current_word)
        
        return ' '.join(text_parts)


    def get_merge_rules(self) -> List[Tuple[str, str]]:
        return self.merge_order.copy()
    
    def save(self, filepath: str) -> None:                  # Save BPE tokenizer to a file

        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'word_end_token': self.word_end_token,
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'token_freqs': dict(self.token_freqs),
            'word_freqs': dict(self.token_freqs) if hasattr(self, 'word_freqs') else {},
            'merges': {f"{k[0]}___{k[1]}": v for k, v in self.merges.items()},              # Serialize tuple keys
            'merge_order': [[pair[0], pair[1]] for pair in self.merge_order],               # Serialize tuples
            'trained': self.trained,
            'tokenizer_type': 'BPETokenizer'
        }

        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)


    def load(self, filepath: str) -> None:          # load BPE tokenizer from file

        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f) 

        self.vocab_size = data['vocab_size']
        self.min_frequency = data['min_frequency']
        self.word_end_token = data['word_end_token']
        self.vocab = data['vocab']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.token_freqs = Counter(data['token_freqs'])
        self.word_freqs = data.get('word_freqs', {})

        self.merges = {}                            # Deserialize merge rules
        for key, value in data['merges'].items():
            parts = key.split('___')
            self.merges[(parts[0], parts[1])] = value
        
        self.merge_order = [(pair[0], pair[1]) for pair in data['merge_order']]
        self.trained = data['trained']











