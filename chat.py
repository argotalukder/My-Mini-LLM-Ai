# ============================================================
#   chat.py — তোমার AI এর সাথে কথা বলো (Advanced CoT Update)
#   Team Claude AI | Made for Argo
# ============================================================

import os
import sys
import torch
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig, PersonalityConfig
from tokenizer.tokenizer import BPETokenizer
from model.transformer import ConversationalAI

# Color Codes for Terminal
COLOR_USER = "\033[92m"    # Green
COLOR_THINK = "\033[90m"   # Dark Grey
COLOR_AI = "\033[96m"      # Cyan
COLOR_RESET = "\033[0m"    # Reset

def load_model(config, device):
    if not os.path.exists(config.model_save):
        print("❌ Model পাওয়া যায়নি! আগে run করো: python train.py")
        sys.exit(1)

    checkpoint = torch.load(config.model_save, map_location=device)
    
    saved_config = checkpoint.get('config', {})
    for key, val in saved_config.items():
        setattr(config, key, val)

    model = ConversationalAI(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✅ Advanced AI loaded! (Loss: {checkpoint.get('loss', '?'):.4f})")
    return model

def load_tokenizer(config):
    if not os.path.exists(config.vocab_file):
        print("❌ Tokenizer পাওয়া যায়নি! আগে run করো: python train.py")
        sys.exit(1)

    tokenizer = BPETokenizer()
    tokenizer.load(config.vocab_file)
    return tokenizer

def generate_response(model, tokenizer, user_input, config, device):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode_conversation(user_input)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        output = model.generate(
            input_ids,
            tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )

        generated = output[0].tolist()
        ast_id = tokenizer.SPECIAL_TOKENS.get("<AST>", 6)

        if ast_id in generated:
            response_tokens = generated[generated.index(ast_id) + 1:]
        else:
            response_tokens = generated[len(tokens):]

        response = tokenizer.decode(response_tokens)
        return response.strip() if response.strip() else "..."

def print_ai_response(personality_name, response):
    """AI এর রেসপন্স থেকে 'Thinking' আলাদা করে কালারফুল আউটপুট দেবে"""
    
    # Check if there is <THINK> tag in the response
    if '<THINK>' in response:
        if '</THINK>' in response:
            parts = response.split('</THINK>')
            think_part = parts[0].replace('<THINK>', '').strip()
            actual_part = parts[1].strip()
            
            # Print Thinking Process in Grey
            print(f"{COLOR_THINK}[🧠 {personality_name} ভাবছে: {think_part}]{COLOR_RESET}")
            # Print Actual response in Cyan
            print(f"{COLOR_AI}{personality_name}: {actual_part}{COLOR_RESET}\n")
        else:
            # Tag opened but generation stopped before closing
            think_part = response.replace('<THINK>', '').strip()
            print(f"{COLOR_THINK}[🧠 {personality_name} ভাবছে: {think_part}...]{COLOR_RESET}")
            print(f"{COLOR_AI}{personality_name}: (আমি আরও ভাবছিলাম, কিন্তু লিমিট শেষ!){COLOR_RESET}\n")
    else:
        # No thinking tag, direct response
        print(f"{COLOR_AI}{personality_name}: {response}{COLOR_RESET}\n")


def chat_loop(model, tokenizer, config, device):
    personality = PersonalityConfig()

    print("\n" + "="*50)
    print(f"🤖 {personality.name} — Advanced Personal AI")
    print(f"👤 Owner: {personality.owner}")
    print("="*50)
    print("💡 AI এখন নিজে নিজে চিন্তা করতে পারে! (Chain of Thought)")
    print("'quit' বা 'exit' লিখলে বের হবে")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input(f"{COLOR_USER}তুমি: {COLOR_RESET}").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"\n{COLOR_AI}{personality.name}: bye Argo।{COLOR_RESET}\n")
                break

            response = generate_response(model, tokenizer, user_input, config, device)
            
            # Empty response handle
            if not response or response == "...":
                response = "হুম।"

            # Parse and print with colors
            print_ai_response(personality.name, response)

        except KeyboardInterrupt:
            print(f"\n\n{COLOR_AI}{personality.name}: আসি বস।{COLOR_RESET}\n")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    tokenizer = load_tokenizer(config)
    model     = load_model(config, device)

    chat_loop(model, tokenizer, config, device)

if __name__ == "__main__":
    main()
