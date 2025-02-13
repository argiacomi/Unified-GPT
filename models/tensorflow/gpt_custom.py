import tensorflow as tf


class CustomLayerNorm(tf.keras.Layer):
    def __init__(self, emb_dim, eps=1e-5, name="custom_layer_norm"):
        super().__init__(name=name)
        self.eps = eps
        self.scale = self.add_weight(
            shape=(emb_dim,), initializer="ones", trainable=True, name="scale"
        )
        self.shift = self.add_weight(
            shape=(emb_dim,), initializer="zeros", trainable=True, name="shift"
        )

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / tf.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class CustomGELU(tf.keras.Layer):
    def __init__(self, name="custom_gelu"):
        super().__init__(name=name)

    def call(self, x):
        return (
            0.5
            * x
            * (
                1
                + tf.tanh(
                    tf.sqrt(2.0 / tf.constant(tf.experimental.numpy.pi))
                    * (x + 0.044715 * tf.pow(x, 3))
                )
            )
        )


class CustomFeedForward(tf.keras.Layer):
    def __init__(self, cfg, name="custom_feed_forward"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(
            4 * cfg["emb_dim"], kernel_initializer="he_normal", name="ffn_dense1"
        )
        self.gelu = CustomGELU(name="gelu_activation")
        self.dense2 = tf.keras.layers.Dense(
            cfg["emb_dim"], kernel_initializer="he_normal", name="ffn_dense2"
        )

    def call(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x


class CustomMultiHeadAttention(tf.keras.Layer):
    def __init__(
        self,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        name="custom_mha",
    ):
        super().__init__(name=name)
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = tf.keras.layers.Dense(
            d_out, use_bias=qkv_bias, kernel_initializer="he_normal", name="W_query"
        )
        self.W_key = tf.keras.layers.Dense(
            d_out, use_bias=qkv_bias, kernel_initializer="he_normal", name="W_key"
        )
        self.W_value = tf.keras.layers.Dense(
            d_out, use_bias=qkv_bias, kernel_initializer="he_normal", name="W_value"
        )
        self.out_proj = tf.keras.layers.Dense(
            d_out, kernel_initializer="he_normal", name="out_proj"
        )
        self.dropout = tf.keras.layers.Dropout(dropout, name="attention_dropout")
        mask = tf.linalg.band_part(tf.ones((context_length, context_length)), -1, 0)
        self.causal_mask = tf.where(mask == 0, -1e10, tf.zeros_like(mask))

    def call(self, x):
        b = tf.shape(x)[0]
        num_tokens = tf.shape(x)[1]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = tf.reshape(keys, (b, num_tokens, self.num_heads, self.head_dim))
        queries = tf.reshape(queries, (b, num_tokens, self.num_heads, self.head_dim))
        values = tf.reshape(values, (b, num_tokens, self.num_heads, self.head_dim))
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        attn_scores = tf.matmul(queries, keys, transpose_b=True)
        attn_scores = attn_scores + self.causal_mask[:num_tokens, :num_tokens]
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn_scores = attn_scores / scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        context = tf.matmul(attn_weights, values)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (b, num_tokens, self.d_out))
        return self.out_proj(context)


class CustomTransformerBlock(tf.keras.Layer):
    """
    Transformer block using custom components.
    """

    def __init__(self, cfg, name="custom_transformer_block"):
        super().__init__(name=name)
        self.norm1 = CustomLayerNorm(cfg["emb_dim"], name="norm1")
        self.att = CustomMultiHeadAttention(
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            name="multi_head_attention",
        )
        self.dropout1 = tf.keras.layers.Dropout(cfg["drop_rate"], name="dropout1")
        self.norm2 = CustomLayerNorm(cfg["emb_dim"], name="norm2")
        self.ff = CustomFeedForward(cfg, name="feed_forward_network")
        self.dropout2 = tf.keras.layers.Dropout(cfg["drop_rate"], name="dropout2")

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout1(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        return x + shortcut


class GPTModelCustom(tf.keras.Model):
    def __init__(self, config, name="gpt_custom"):
        super().__init__(name=name)
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
        self.dropout = tf.keras.layers.Dropout(
            config["drop_rate"], name="embedding_dropout"
        )
        self.transformer_blocks = [
            CustomTransformerBlock(config, name=f"transformer_block_{i}")
            for i in range(config["n_layers"])
        ]
        self.norm = CustomLayerNorm(config["emb_dim"], name="final_norm")
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
