# ============================================================
#   train.py — Model Training
#   Google Colab এ run করো (free GPU!)
#   Team Claude AI | Made for Argo
# ============================================================

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn

# Project files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig
from tokenizer.tokenizer import BPETokenizer
from model.transformer import ConversationalAI
from data.processor import load_conversations, get_dataloader


# ── Setup ─────────────────────────────────────────────────────

def setup():
    """Training environment setup"""
    config = ModelConfig()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("tokenizer",   exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    return config, device


# ── Tokenizer Training ────────────────────────────────────────

def train_tokenizer(conversations, config):
    """Tokenizer train করো conversation data দিয়ে"""
    print("\n" + "="*50)
    print("📝 TOKENIZER TRAINING")
    print("="*50)

    # Vocab file already আছে কিনা check
    if os.path.exists(config.vocab_file):
        print("✅ Tokenizer already trained! Loading...")
        tokenizer = BPETokenizer(config.vocab_size)
        tokenizer.load(config.vocab_file)
        return tokenizer

    # সব text collect করো
    all_texts = []
    for conv in conversations:
        all_texts.append(conv['user'])
        all_texts.append(conv['ai'])

    # Train
    tokenizer = BPETokenizer(config.vocab_size)
    tokenizer.train(all_texts, verbose=True)
    tokenizer.save(config.vocab_file)

    return tokenizer


# ── Learning Rate Scheduler ───────────────────────────────────

def get_lr(step, config):
    """Cosine learning rate with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    # Cosine decay
    progress = (step - config.warmup_steps) / max(1, 1000 - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Training Loop ─────────────────────────────────────────────

def train(config, device, tokenizer, dataloader):
    """Main training loop"""
    print("\n" + "="*50)
    print("🏋️  MODEL TRAINING শুরু")
    print("="*50)

    # Model
    config.vocab_size = len(tokenizer)
    model = ConversationalAI(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Log file
    log_data = []
    best_loss = float('inf')
    global_step = 0

    print(f"📊 Total parameters: {model.count_parameters():,}")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"🔄 Epochs: {config.epochs}")
    print()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            # Learning rate update
            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Update
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1
            global_step += 1

            # Progress print
            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch+1}/{config.epochs}] "
                      f"Step [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {lr:.6f}")

        # Epoch শেষ
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed  = time.time() - start_time

        print(f"\n✅ Epoch {epoch+1} শেষ | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Log save
        log_data.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": lr,
            "time": elapsed
        })

        with open("logs/training_log.json", 'w') as f:
            json.dump(log_data, f, indent=2)

        # Best model save
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'vocab_size': config.vocab_size,
                    'embed_dim': config.embed_dim,
                    'num_heads': config.num_heads,
                    'num_layers': config.num_layers,
                    'ffn_dim': config.ffn_dim,
                    'max_seq_len': config.max_seq_len,
                    'dropout': config.dropout,
                }
            }, config.model_save)
            print(f"💾 Best model saved! Loss: {best_loss:.4f}")

        # Sample generation (প্রতি ১০ epoch)
        if (epoch + 1) % 10 == 0:
            print("\n🤖 Sample generation:")
            generate_sample(model, tokenizer, device, config)
            model.train()

        print()

    print(f"\n🎉 Training শেষ! Best loss: {best_loss:.4f}")
    return model


# ── Sample Generation ─────────────────────────────────────────

def generate_sample(model, tokenizer, device, config):
    """Training এর মধ্যে দেখো model কি বলছে"""
    model.eval()
    test_inputs = ["হ্যালো", "কেমন আছো?", "পড়াশোনা করি"]

    for test_input in test_inputs:
        tokens = tokenizer.encode_conversation(test_input)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        output = model.generate(
            input_ids, tokenizer,
            max_new_tokens=50,
            temperature=config.temperature,
            top_k=config.top_k
        )

        # Decode করো
        generated = output[0].tolist()
        sep_id = tokenizer.SPECIAL_TOKENS.get("<AST>", 6)
        if sep_id in generated:
            response_tokens = generated[generated.index(sep_id) + 1:]
        else:
            response_tokens = generated[len(tokens):]

        response = tokenizer.decode(response_tokens)
        print(f"  User: {test_input}")
        print(f"  AI  : {response}")
        print()


# ── Main ──────────────────────────────────────────────────────

def main():
    print("🚀 Bengali Conversational AI — Training")
    print("Team Claude AI | Made for Argo")
    print("=" * 50)

    # Setup
    config, device = setup()

    # Data load
    print("\n📂 Data loading...")
    conversations = load_conversations(config.data_file)

    if len(conversations) == 0:
        print("❌ কোনো data নেই! data/conversations.jsonl check করো।")
        return

    # Tokenizer
    tokenizer = train_tokenizer(conversations, config)

    # DataLoader
    dataloader = get_dataloader(conversations, tokenizer, config)

    # Train
    model = train(config, device, tokenizer, dataloader)

    print("\n" + "="*50)
    print("✅ সব শেষ!")
    print("📁 Model: checkpoints/model.pt")
    print("📁 Tokenizer: tokenizer/vocab.json")
    print("🎯 এখন চালাও: python chat.py")
    print("="*50)


if __name__ == "__main__":
    main()
