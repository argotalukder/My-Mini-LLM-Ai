# ============================================================
#   chat.py — তোমার AI এর সাথে কথা বলো
#   Train হয়ে গেলে এটা run করো
#   Team Claude AI | Made for Argo
# ============================================================

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig, PersonalityConfig
from tokenizer.tokenizer import BPETokenizer
from model.transformer import ConversationalAI


def load_model(config, device):
    """Trained model load করো"""
    if not os.path.exists(config.model_save):
        print("❌ Model পাওয়া যায়নি!")
        print("   আগে run করো: python train.py")
        sys.exit(1)

    checkpoint = torch.load(config.model_save, map_location=device)

    # Config update করো checkpoint থেকে
    saved_config = checkpoint.get('config', {})
    for key, val in saved_config.items():
        setattr(config, key, val)

    model = ConversationalAI(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    loss  = checkpoint.get('loss', '?')
    print(f"✅ Model loaded! (Epoch: {epoch}, Loss: {loss:.4f})")

    return model


def load_tokenizer(config):
    """Tokenizer load করো"""
    if not os.path.exists(config.vocab_file):
        print("❌ Tokenizer পাওয়া যায়নি!")
        print("   আগে run করো: python train.py")
        sys.exit(1)

    tokenizer = BPETokenizer()
    tokenizer.load(config.vocab_file)
    return tokenizer


def generate_response(model, tokenizer, user_input, config, device):
    """User input দিলে AI response generate করো"""
    model.eval()

    with torch.no_grad():
        # Encode
        tokens = tokenizer.encode_conversation(user_input)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        # Generate
        output = model.generate(
            input_ids,
            tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )

        # Decode response only
        generated = output[0].tolist()
        ast_id = tokenizer.SPECIAL_TOKENS.get("<AST>", 6)

        if ast_id in generated:
            response_tokens = generated[generated.index(ast_id) + 1:]
        else:
            response_tokens = generated[len(tokens):]

        response = tokenizer.decode(response_tokens)
        return response.strip() if response.strip() else "..."


def chat_loop(model, tokenizer, config, device):
    """Main conversation loop"""
    personality = PersonalityConfig()

    print("\n" + "="*50)
    print(f"🤖 {personality.name} — তোমার Personal AI")
    print(f"👤 Owner: {personality.owner}")
    print("="*50)
    print("'quit' বা 'exit' লিখলে বের হবে")
    print("'clear' লিখলে context reset হবে")
    print("="*50 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("তুমি: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'বিদায়']:
                print(f"\n{personality.name}: bye।\n")
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                print("(Context cleared)\n")
                continue

            # Response generate করো
            response = generate_response(
                model, tokenizer, user_input, config, device
            )

            # Empty response handle
            if not response or response == "...":
                response = "হুম।"

            print(f"{personality.name}: {response}\n")

            # History তে রাখো
            conversation_history.append({
                "user": user_input,
                "ai": response
            })

        except KeyboardInterrupt:
            print(f"\n\n{personality.name}: আসি।\n")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load
    tokenizer = load_tokenizer(config)
    model     = load_model(config, device)

    # Chat
    chat_loop(model, tokenizer, config, device)


if __name__ == "__main__":
    main()
