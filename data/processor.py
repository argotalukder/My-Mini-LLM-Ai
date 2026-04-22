# ============================================================
#    data/processor.py
#    Training data load + process + batch বানানো
#    Team Claude AI | Made for Argo
# ============================================================

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

def load_conversations(filepath, hf_filepath=None, limit=None):
    """JSONL file থেকে conversations load করো + HF data merge"""
    conversations = []

    # ── তোমার নিজের data ─────────────────────────────────────
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'user' in item and 'ai' in item:
                            conversations.append(item)
                    except json.JSONDecodeError:
                        continue
        print(f"✅ নিজের data: {len(conversations)} conversations")

    # ── HuggingFace data (যদি থাকে) ──────────────────────────
    if hf_filepath and os.path.exists(hf_filepath):
        hf_count = 0
        with open(hf_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'user' in item and 'ai' in item:
                            conversations.append(item)
                            hf_count += 1
                    except json.JSONDecodeError:
                        continue
        print(f"✅ HuggingFace data: {hf_count} conversations")
    else:
        print("⚠️  HF data পাওয়া যায়নি, শুধু নিজের data দিয়ে train হবে।")

    # দুটো data ভালোভাবে mix করো
    random.shuffle(conversations)

    # ── এখানে Data Limit সেট করা হচ্ছে ──
    if limit and len(conversations) > limit:
        conversations = conversations[:limit]
        print(f"✂️  Data limited to: {limit} conversations for training.")

    print(f"✅ মোট ট্রেইনিং ডেটা: {len(conversations)}")
    return conversations


class ConversationDataset(Dataset):
    """
    PyTorch Dataset for conversation training
    """
    def __init__(self, conversations, tokenizer, max_seq_len=128):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples     = []

        print("🔧 Dataset তৈরি করছি...")

        for conv in conversations:
            user_text = conv['user']
            ai_text   = conv['ai']

            # Conversation encode করো
            tokens = tokenizer.encode_conversation(user_text, ai_text)

            # Max length এ trim করো
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]

            # Padding করো
            padding = [tokenizer.SPECIAL_TOKENS["<PAD>"]] * (max_seq_len - len(tokens))
            tokens = tokens + padding

            self.samples.append(tokens)

        print(f"✅ {len(self.samples)} training samples ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels    = torch.tensor(tokens[1:],  dtype=torch.long)
        return input_ids, labels


def get_dataloader(conversations, tokenizer, config, shuffle=True):
    """DataLoader বানাও"""
    dataset = ConversationDataset(conversations, tokenizer, config.max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader
