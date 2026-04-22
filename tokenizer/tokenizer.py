# ============================================================
#   tokenizer/tokenizer.py
#   নিজের BPE Tokenizer — 100% from scratch
#   Bengali + English + Banglish support
# ============================================================

import json
import re
import os
from collections import Counter, defaultdict


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer
    GPT এর মতো same approach — নিজে বানানো
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,   # padding
        "<UNK>": 1,   # unknown word
        "<BOS>": 2,   # beginning of sequence
        "<EOS>": 3,   # end of sequence
        "<SEP>": 4,   # separator (user/ai এর মধ্যে)
        "<USR>": 5,   # user turn
        "<AST>": 6,   # assistant turn
    }

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}          # token → id
        self.reverse_vocab = {}  # id → token
        self.merges = {}         # BPE merge rules
        self.trained = False

    # ── Training ──────────────────────────────────────────

    def train(self, texts, verbose=True):
        """
        text list দিয়ে tokenizer train করো
        texts: ["আমি ভালো আছি", "কি খবর", ...]
        """
        if verbose:
            print("🔧 Tokenizer training শুরু...")

        # Step 1: Character vocabulary বানাও
        char_freq = Counter()
        word_freq = Counter()

        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_freq[word] += 1
                for char in word:
                    char_freq[char] += 1

        # Step 2: Initial vocab = special tokens + characters
        self.vocab = dict(self.SPECIAL_TOKENS)
        next_id = len(self.SPECIAL_TOKENS)

        # Common characters যোগ করো
        for char, freq in char_freq.most_common():
            if freq >= 2:  # কমপক্ষে ২বার দেখা গেছে
                if char not in self.vocab:
                    self.vocab[char] = next_id
                    next_id += 1

        # Step 3: BPE merges
        # Word frequency দিয়ে শুরু করো
        word_splits = {}
        for word, freq in word_freq.items():
            chars = list(word) + ['</w>']
            word_splits[tuple(chars)] = freq

        target_vocab_size = self.vocab_size
        merge_count = 0

        while len(self.vocab) < target_vocab_size:
            # সবচেয়ে frequent pair খোঁজো
            pair_freq = self._get_pair_frequencies(word_splits)
            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < 2:
                break

            # Merge করো
            merged = ''.join(best_pair)
            self.merges[best_pair] = merged

            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1

            # Word splits update করো
            word_splits = self._merge_pair(best_pair, word_splits)
            merge_count += 1

            if verbose and merge_count % 100 == 0:
                print(f"   Merges: {merge_count}, Vocab size: {len(self.vocab)}")

        # Reverse vocab বানাও
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True

        if verbose:
            print(f"✅ Training শেষ! Vocab size: {len(self.vocab)}")

    def _pre_tokenize(self, text):
        """Text কে words এ ভাগ করো"""
        # Bengali + English words
        words = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+|[0-9]+|[^\s\w]', text)
        return [w.lower() for w in words if w.strip()]

    def _get_pair_frequencies(self, word_splits):
        """সব adjacent pair এর frequency count করো"""
        pairs = defaultdict(int)
        for word, freq in word_splits.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, word_splits):
        """Best pair কে সব জায়গায় merge করো"""
        new_splits = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in word_splits:
            new_word = ' '.join(word)
            new_word = new_word.replace(bigram, replacement)
            new_splits[tuple(new_word.split())] = word_splits[word]

        return new_splits

    # ── Encode / Decode ────────────────────────────────────

    def encode(self, text, add_special=True):
        """
        Text → Token IDs
        "আমি ভালো" → [123, 456, ...]
        """
        if not self.trained:
            raise RuntimeError("আগে train() করো!")

        tokens = []

        if add_special:
            tokens.append(self.SPECIAL_TOKENS["<BOS>"])

        words = self._pre_tokenize(text)

        for word in words:
            word_tokens = self._tokenize_word(word + '</w>')
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.SPECIAL_TOKENS["<UNK>"]))

        if add_special:
            tokens.append(self.SPECIAL_TOKENS["<EOS>"])

        return tokens

    def _tokenize_word(self, word):
        """একটা word কে BPE rules দিয়ে tokenize করো"""
        if word in self.vocab:
            return [word]

        chars = list(word)

        # BPE merges apply করো
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
        """
        Token IDs → Text
        [123, 456, ...] → "আমি ভালো"
        """
        tokens = []
        for tid in token_ids:
            if tid in [self.SPECIAL_TOKENS["<BOS>"], self.SPECIAL_TOKENS["<EOS>"],
                       self.SPECIAL_TOKENS["<PAD>"]]:
                continue
            token = self.reverse_vocab.get(tid, '<UNK>')
            tokens.append(token)

        # Join করো এবং word boundary ঠিক করো
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def encode_conversation(self, user_text, ai_text=None):
        """
        Conversation format এ encode করো
        """
        tokens = [self.SPECIAL_TOKENS["<USR>"]]
        tokens.extend(self.encode(user_text, add_special=False))
        tokens.append(self.SPECIAL_TOKENS["<SEP>"])

        if ai_text:
            tokens.append(self.SPECIAL_TOKENS["<AST>"])
            tokens.extend(self.encode(ai_text, add_special=False))
            tokens.append(self.SPECIAL_TOKENS["<EOS>"])

        return tokens

    # ── Save / Load ────────────────────────────────────────

    def save(self, filepath="tokenizer/vocab.json"):
        """Tokenizer save করো"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Tokenizer saved: {filepath}")

    def load(self, filepath="tokenizer/vocab.json"):
        """Tokenizer load করো"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.merges = {
            tuple(k.split("|||")): v
            for k, v in data["merges"].items()
        }
        self.vocab_size = data["vocab_size"]
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.trained = True
        print(f"✅ Tokenizer loaded! Vocab size: {len(self.vocab)}")

    def __len__(self):
        return len(self.vocab)
