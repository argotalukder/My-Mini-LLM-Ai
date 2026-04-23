# config.py
import os

class ArgoConfig:
    # Model Specifications (PRD অনুযায়ী)
    vocab_size = 16000      # Bangla + English vocab
    n_embd = 1024           # Embedding dimension
    n_head = 16             # Attention heads
    n_layer = 24            # Transformer layers
    block_size = 1024       # Context window
    n_experts = 8           # MoE experts
    top_k = 2               # Active experts per token
    hidden_dim = 4096       # FFN hidden dimension
    dropout = 0.1
    
    # Default Paths (Google Drive Setup)
    base_dir = '/content/drive/MyDrive/ArgoLM_Training'
    data_pretrain_dir = os.path.join(base_dir, 'data', 'pretrain')
    data_sft_dir = os.path.join(base_dir, 'data', 'sft')
    data_grpo_dir = os.path.join(base_dir, 'data', 'grpo')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    
    # Training Parameters
    learning_rate = 3e-4
    batch_size = 8
    epochs = 2
    save_steps = 500
