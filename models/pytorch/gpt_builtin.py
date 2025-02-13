import torch
import torch.nn as nn


class GPTModelBuiltin(nn.Module):
    """
    GPT built using PyTorch’s built-in Transformer encoder layers.
    """
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.positional_embedding = nn.Embedding(
            config["context_length"], config["emb_dim"]
        )
        self.dropout = nn.Dropout(config["drop_rate"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["emb_dim"],
            nhead=config["n_heads"],
            dim_feedforward=4 * config["emb_dim"],
            dropout=config["drop_rate"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config["n_layers"], enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=config["qkv_bias"]
        )

        self.register_buffer(
            "causal_mask",
            self._generate_causal_mask(config["context_length"])
        )

    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeds = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeds = self.positional_embedding(positions)
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        mask_bool = self.causal_mask.bool()[:seq_len, :seq_len]
        x = self.transformer(x, mask=mask_bool, is_causal=True)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits
