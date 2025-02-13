# gpt_builtin.py: GPT built with TensorFlow’s built-in layers.
import tensorflow as tf


class TransformerBlockBuiltin(tf.keras.layers.Layer):
    """
    Transformer block using TensorFlow’s built-in layers.
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg["n_heads"],
            key_dim=cfg["emb_dim"] // cfg["n_heads"],
            dropout=cfg["drop_rate"],
            use_bias=cfg["qkv_bias"],
            kernel_initializer="he_normal",
            name="multi_head_attention",
        )
        self.dropout1 = tf.keras.layers.Dropout(cfg["drop_rate"], name="dropout1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    4 * cfg["emb_dim"],
                    activation=tf.nn.gelu,
                    kernel_initializer="he_normal",
                    name="ffn_dense1",
                ),
                tf.keras.layers.Dense(
                    cfg["emb_dim"],
                    kernel_initializer="he_normal",
                    name="ffn_dense2",
                ),
            ],
            name="feed_forward_network",
        )
        self.dropout2 = tf.keras.layers.Dropout(cfg["drop_rate"], name="dropout2")

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, x, use_causal_mask=True)
        x = self.dropout1(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        return x + shortcut

class GPTModelBuiltin(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=config["vocab_size"],
            output_dim=config["emb_dim"],
            name="token_embedding",
        )
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=config["context_length"],
            output_dim=config["emb_dim"],
            name="positional_embedding",
        )
        self.dropout = tf.keras.layers.Dropout(config["drop_rate"])
        self.transformer_blocks = [
            TransformerBlockBuiltin(config) for _ in range(config["n_layers"])
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.out_head = tf.keras.layers.Dense(
            config["vocab_size"],
            use_bias=config["qkv_bias"],
            kernel_initializer="he_normal",
            name="output_projection",
        )

    def call(self, x):
        seq_length = tf.shape(x)[1]
        token_embeds = self.token_embedding(x)
        positions = tf.range(seq_length)
        pos_embeds = self.positional_embedding(positions)
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits
