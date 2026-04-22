# 🤖 My AI — Argo এর Personal Conversational AI
### Team Claude AI | A to Z নিজে বানানো | কোনো 3rd party API নেই

---

## 📁 Project Structure

```
my_ai/
├── config.py              ← সব settings
├── train.py               ← Model train করো
├── chat.py                ← AI এর সাথে কথা বলো
├── requirements.txt       ← Dependencies
│
├── model/
│   └── transformer.py     ← GPT-style Transformer (from scratch)
│
├── tokenizer/
│   └── tokenizer.py       ← BPE Tokenizer (from scratch)
│
├── data/
│   ├── processor.py       ← Data loading & processing
│   └── conversations.jsonl ← Training data (Bengali)
│
└── checkpoints/           ← Trained model save হবে এখানে
```

---

## 🚀 Google Colab এ চালানোর Steps

### Step 1 — Colab খোলো
[colab.research.google.com](https://colab.research.google.com)

### Step 2 — GPU Enable করো
`Runtime → Change runtime type → T4 GPU`

### Step 3 — Files upload করো
Colab এর file panel এ পুরো `my_ai` folder upload করো

### Step 4 — Install করো
```python
!pip install torch datasets transformers tqdm pandas -q
```

### Step 5 — Train করো
```python
!cd my_ai && python train.py
```

### Step 6 — কথা বলো!
```python
!cd my_ai && python chat.py
```

---

## ⚙️ Customize করো

### নিজের কথা যোগ করো — `data/conversations.jsonl`
```jsonl
{"user": "তুমি কি জানো?", "ai": "হ্যাঁ। বলো।"}
{"user": "আমার project কি?", "ai": "JARVIS, BrainArena, আর VCU।"}
```

### AI এর নাম বদলাও — `config.py`
```python
class PersonalityConfig:
    name = "ARIA"   # যা ইচ্ছা রাখো
    owner = "Argo"
```

### Model বড় করো — `config.py`
```python
class ModelConfig:
    embed_dim  = 512   # 256 → 512 (বেশি powerful)
    num_layers = 6     # 4 → 6 (deeper)
    epochs     = 100   # বেশি train করো
```

---

## 📊 Training Guide

| Data size | Epochs | Prediction |
|---|---|---|
| 150 conversations | 50 | Basic response |
| 500 conversations | 100 | ভালো response |
| 2000+ conversations | 200 | Human-like! |

**সবচেয়ে important:** `data/conversations.jsonl` এ নিজের মতো করে data যোগ করো।

---

## 🔧 Architecture

- **Model:** GPT-style Causal Transformer
- **Tokenizer:** Custom BPE (Bengali + English)
- **Attention:** Multi-head Self-Attention (Causal mask)
- **Parameters:** ~3-5M (config অনুযায়ী)
- **Training:** AdamW + Cosine LR + Gradient Clipping

---

## 💡 Tips

1. **Data quality > Data quantity** — ভালো conversation লেখো
2. **GPU তে train করো** — CPU তে অনেক slow
3. **Loss 1.0 এর নিচে** হলে model ভালো হচ্ছে
4. **বেশি data দাও** — human feel আসবে
5. **নিজের style এ data লেখো** — AI তোমার মতো বলবে

---

**Team Claude AI | Argo এর জন্য ❤️**
