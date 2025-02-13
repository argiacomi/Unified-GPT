import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class CustomGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class CustomFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            CustomGELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", self._generate_causal_mask(context_length))

    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.causal_mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class CustomTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = CustomLayerNorm(cfg["emb_dim"])
        self.att = CustomMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.dropout1 = CustomLayerNorm(cfg["emb_dim"])

        self.norm2 = CustomLayerNorm(cfg["emb_dim"])
        self.ff = CustomFeedForward(cfg)
        self.dropout2 = CustomLayerNorm(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout1(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        x = x + shortcut
        return x


class GPTModelCustom(nn.Module):
    """
    GPT built using custom PyTorch components.
    """

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.positional_embedding = nn.Embedding(
            config["context_length"], config["emb_dim"]
        )
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.ModuleList(
            [CustomTransformerBlock(config) for _ in range(config["n_layers"])]
        )
        # Optionally, you can use CustomLayerNorm here.
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=config["qkv_bias"]
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeds = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeds = self.positional_embedding(positions)
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits
