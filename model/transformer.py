# ============================================================
#   model/transformer.py
#   Transformer Model — A to Z নিজে বানানো
#   GPT-style Causal Language Model
#   Team Claude AI | Made for Argo
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Attention Mechanism ───────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Self-Attention: AI কীভাবে context বোঝে
    "আমি ভালো আছি" → "আমি" কার সাথে related সেটা বোঝে
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # Batch, Time, Channels

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Causal mask — ভবিষ্যৎ দেখতে পাবে না
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out


# ── Feed Forward Network ──────────────────────────────────────

class FeedForward(nn.Module):
    """
    AI এর 'চিন্তা করার' layer
    Attention এর পরে deeper processing
    """

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),                          # smooth activation
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    একটা complete transformer layer
    Attention + FFN + Residual + Norm
    """

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn       = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Residual connection with attention
        x = x + self.attention(self.norm1(x), mask)
        # Residual connection with FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ── Positional Encoding ───────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Word এর position মনে রাখা
    "আমি তোমাকে ভালোবাসি" ≠ "তোমাকে আমি ভালোবাসি"
    """

    def __init__(self, embed_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Main Model ────────────────────────────────────────────────

class ConversationalAI(nn.Module):
    """
    তোমার নিজের Conversational AI Model
    GPT-style Causal Language Model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.embed_dim,
            config.max_seq_len,
            config.dropout
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ffn_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(config.embed_dim)

        # Output projection → vocabulary
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (token embedding = output weights)
        self.output.weight = self.token_embedding.weight

        # Weights initialize করো
        self.apply(self._init_weights)

        print(f"🤖 Model ready! Parameters: {self.count_parameters():,}")

    def _init_weights(self, module):
        """Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _make_causal_mask(self, seq_len, device):
        """Causal mask — ভবিষ্যৎ দেখতে পারবে না"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, labels=None):
        """
        Forward pass
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (training এর জন্য)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token + Positional embedding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # Causal attention mask
        mask = self._make_causal_mask(T, device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Normalize
        x = self.norm(x)

        # Logits
        logits = self.output(x)

        # Loss calculate করো (training এর জন্য)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=0  # PAD token ignore
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=100,
                 temperature=0.8, top_k=50, top_p=0.9):
        """
        Text generate করো
        Conversation response তৈরি করার main function
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Context window এ fit করো
            context = generated[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self.forward(context)

            # Last token এর logits নাও
            next_logits = logits[:, -1, :] / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_vals = torch.topk(next_logits, top_k, dim=-1)
                min_val = top_k_vals.values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float('-inf'))

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # Sample করো
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # EOS হলে থামো
            if next_token.item() == tokenizer.SPECIAL_TOKENS.get("<EOS>", 3):
                break

            generated = torch.cat([generated, next_token], dim=-1)

        return generated
