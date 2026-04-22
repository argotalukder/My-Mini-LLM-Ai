# ============================================================
#   config.py — সব settings এক জায়গায়
#   তোমার AI এর brain এর blueprint
#   Team Claude AI | Made for Argo
# ============================================================

class ModelConfig:
    # ── Model Architecture ──
    vocab_size     = 5000       # কতটা unique token জানবে
    embed_dim      = 256        # প্রতিটা token এর vector size
    num_heads      = 8          # attention heads
    num_layers     = 4          # transformer layers
    ffn_dim        = 1024       # feed forward hidden size
    max_seq_len    = 256        # একবারে কতটা text দেখবে
    dropout        = 0.1        # overfitting এড়ানোর জন্য

    # ── Training ──
    batch_size     = 32         # একসাথে কতটা sample
    learning_rate  = 3e-4       # শেখার speed
    epochs         = 50         # কতবার পুরো data দেখবে
    warmup_steps   = 100        # শুরুতে ধীরে শেখা
    grad_clip      = 1.0        # gradient explosion রোধ

    # ── Generation ──
    temperature    = 0.8        # creativity (0.1=boring, 1.5=wild)
    top_k          = 50         # top k tokens থেকে select
    top_p          = 0.9        # nucleus sampling
    max_new_tokens = 150        # সর্বোচ্চ কতটা generate করবে

    # ── Files ──
    data_file      = "data/conversations.jsonl"
    vocab_file     = "tokenizer/vocab.json"
    model_save     = "checkpoints/model.pt"
    log_file       = "training.log"


class PersonalityConfig:
    # ── তোমার AI এর identity ──
    name           = "ARIA"     # তোমার AI এর নাম (যা ইচ্ছা রাখো)
    owner          = "Argo"     # তোমার নাম

    # যা বলবে না
    never_say = [
        "আমি একটি AI assistant",
        "আমি জানি না কীভাবে",
        "আমি সাহায্য করতে পারব না",
        "আমার কোনো অনুভূতি নেই",
    ]

    # কথা বলার style
    style = "casual Banglish — Bengali base, English terms mixed"
    humor = "dry and sarcastic"
    emotion_aware = True
