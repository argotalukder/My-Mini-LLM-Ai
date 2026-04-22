class ModelConfig:
    # ── Model Architecture (Brain Size) ──
    vocab_size     = 5000       
    embed_dim      = 512        # 256 থেকে 512 করা হয়েছে (বেশি পাওয়ারফুল)
    num_heads      = 8          
    num_layers     = 6          # 4 থেকে 6 করা হয়েছে (গভীর চিন্তা করার জন্য)
    ffn_dim        = 2048       # Feed forward network দ্বিগুণ করা হয়েছে
    max_seq_len    = 256        
    dropout        = 0.1        

    # ── Training ──
    batch_size     = 32         
    learning_rate  = 3e-4       
    epochs         = 100        # 100 বার ট্রেইন হবে (Colab এ জলদি হয়ে যাবে)
    warmup_steps   = 100        
    grad_clip      = 1.0        

    # ── Generation ──
    temperature    = 0.7        # 0.7 দিলে লজিক্যাল অ্যানসার দেবে
    top_k          = 50         
    top_p          = 0.9        
    max_new_tokens = 200        # থিংকিং+অ্যানসার এর জন্য বেশি টোকেন লাগবে

    # ── Files ──
    data_file      = "data/conversations.jsonl"
    vocab_file     = "tokenizer/vocab.json"
    model_save     = "checkpoints/model.pt"
    log_file       = "training.log"


class PersonalityConfig:
    name           = "ARIA"     
    owner          = "Argo"     

    never_say = [
        "আমি একটি AI assistant",
        "আমি জানি না কীভাবে",
    ]

    style = "casual Banglish, honest, helpful and smart"
    emotion_aware = True
