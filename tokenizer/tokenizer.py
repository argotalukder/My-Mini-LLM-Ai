# ============================================================
#   tokenizer/tokenizer.py
#   BPE Tokenizer (Updated with AI Thinking tags)
#   Team Claude AI | Made for Argo
# ============================================================

import json
import re
import os
from collections import Counter, defaultdict


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer with Chain of Thought (Thinking) support
    """

    # নতুন Thinking টোকেন যোগ করা হয়েছে
    SPECIAL_TOKENS = {
        "<PAD>": 0,   
        "<UNK>": 1,   
        "<BOS>": 2,   
        "<EOS>": 3,   
        "<SEP>": 4,   
        "<USR>": 5,   
        "<AST>": 6,   
        "<THINK>": 7,     # AI এর চিন্তা শুরু
        "</THINK>": 8,    # AI এর চিন্তা শেষ
    }

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}          
        self.reverse_vocab = {}  
        self.merges = {}         
        self.trained = False

    def train(self, texts, verbose=True):
        if verbose:
            print("🔧 Tokenizer training শুরু...")

        char_freq = Counter()
        word_freq = Counter()

        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                # Special token গুলো char level এ ভাঙবে না
                if word not in self.SPECIAL_TOKENS:
                    word_freq[word] += 1
                    for char in word:
                        char_freq[char] += 1

        self.vocab = dict(self.SPECIAL_TOKENS)
        next_id = len(self.SPECIAL_TOKENS)

        for char, freq in char_freq.most_common():
            if freq >= 2:  
                if char not in self.vocab:
                    self.vocab[char] = next_id
                    next_id += 1

        word_splits = {}
        for word, freq in word_freq.items():
            chars = list(word) + ['</w>']
            word_splits[tuple(chars)] = freq

        target_vocab_size = self.vocab_size
        merge_count = 0

        while len(self.vocab) < target_vocab_size:
            pair_freq = self._get_pair_frequencies(word_splits)
            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < 2:
                break

            merged = ''.join(best_pair)
            self.merges[best_pair] = merged

            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1

            word_splits = self._merge_pair(best_pair, word_splits)
            merge_count += 1

            if verbose and merge_count % 100 == 0:
                print(f"   Merges: {merge_count}, Vocab size: {len(self.vocab)}")

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True

        if verbose:
            print(f"✅ Training শেষ! Vocab size: {len(self.vocab)}")

    def _pre_tokenize(self, text):
        # <THINK> ট্যাগগুলোকে আলাদাভাবে চেনার জন্য Regex আপডেট করা হয়েছে
        words = re.findall(r'<THINK>|</THINK>|[\u0980-\u09FF]+|[a-zA-Z]+|[0-9]+|[^\s\w]', text)
        processed_words = []
        for w in words:
            if w in ['<THINK>', '</THINK>']:
                processed_words.append(w)
            elif w.strip():
                processed_words.append(w.lower())
        return processed_words

    def _get_pair_frequencies(self, word_splits):
        pairs = defaultdict(int)
        for word, freq in word_splits.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, word_splits):
        new_splits = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in word_splits:
            new_word = ' '.join(word)
            new_word = new_word.replace(bigram, replacement)
            new_splits[tuple(new_word.split())] = word_splits[word]

        return new_splits

    def encode(self, text, add_special=True):
        if not self.trained:
            raise RuntimeError("আগে train() করো!")

        tokens = []
        if add_special:
            tokens.append(self.SPECIAL_TOKENS["<BOS>"])

        words = self._pre_tokenize(text)

        for word in words:
            # Special Token হলে সরাসরি ID বসাও
            if word in self.SPECIAL_TOKENS:
                tokens.append(self.SPECIAL_TOKENS[word])
            else:
                word_tokens = self._tokenize_word(word + '</w>')
                for token in word_tokens:
                    tokens.append(self.vocab.get(token, self.SPECIAL_TOKENS["<UNK>"]))

        if add_special:
            tokens.append(self.SPECIAL_TOKENS["<EOS>"])

        return tokens

    def _tokenize_word(self, word):
        if word in self.vocab:
            return [word]
        chars = list(word)
        for pair, merged in self.merges.items():
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars

    def decode(self, token_ids):
        tokens = []
        for tid in token_ids:
            if tid in [self.SPECIAL_TOKENS["<BOS>"], self.SPECIAL_TOKENS["<EOS>"], self.SPECIAL_TOKENS["<PAD>"]]:
                continue
            token = self.reverse_vocab.get(tid, '<UNK>')
            tokens.append(token)

        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def encode_conversation(self, user_text, ai_text=None):
        tokens = [self.SPECIAL_TOKENS["<USR>"]]
        tokens.extend(self.encode(user_text, add_special=False))
        tokens.append(self.SPECIAL_TOKENS["<SEP>"])

        if ai_text:
            tokens.append(self.SPECIAL_TOKENS["<AST>"])
            tokens.extend(self.encode(ai_text, add_special=False))
            tokens.append(self.SPECIAL_TOKENS["<EOS>"])

        return tokens

    def save(self, filepath="tokenizer/vocab.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath="tokenizer/vocab.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = {tuple(k.split("|||")): v for k, v in data["merges"].items()}
        self.vocab_size = data["vocab_size"]
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.trained = True

    def __len__(self):
        return len(self.vocab)
