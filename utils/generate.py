import tensorflow as tf
import torch


# ---------- PyTorch Generation ----------
def generate_torch(
    model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(1)
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (next_token == eos_id).all():
            break

        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ---------- TensorFlow Generation ----------
def generate_tf(
    model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None
):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond, training=False)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = tf.math.top_k(logits, k=top_k)
            min_val = top_logits[:, -1, tf.newaxis]
            logits = tf.where(logits < min_val, -1e10 * tf.ones_like(logits), logits)

        if temperature > 0.0:
            logits = logits / temperature
            next_token = tf.random.categorical(logits, num_samples=1)
        else:
            next_token = tf.keras.ops.argmax(logits, axis=-1, keepdims=True)

        if eos_id is not None and tf.reduce_all(tf.equal(next_token, eos_id)):
            break

        idx = tf.concat([idx, tf.cast(next_token, dtype=idx.dtype)], axis=1)
    return idx


# ---------- Unified Generation Interface ----------
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=25,
    eos_id=None,
    framework="pytorch",
):
    """
    Autoregressively generate tokens using a GPT model.
    """
    fw = framework.lower()
    if fw in ["pytorch", "torch"]:
        return generate_torch(
            model, idx, max_new_tokens, context_size, temperature, top_k, eos_id
        )
    elif fw in ["tensorflow", "tf"]:
        return generate_tf(
            model, idx, max_new_tokens, context_size, temperature, top_k, eos_id
        )
    else:
        raise ValueError("Unknown framework specified. Use 'pytorch' or 'tensorflow'.")


# ---------- Helper Functions ----------
def text_to_token_ids(text, tokenizer, framework="pytorch"):
    """
    Convert input text to a tensor of token IDs.
    """
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    fw = framework.lower()
    if fw in ["pytorch", "torch"]:
        return torch.tensor(token_ids).unsqueeze(0)
    elif fw in ["tensorflow", "tf"]:
        return tf.expand_dims(tf.convert_to_tensor(token_ids), axis=0)
    else:
        raise ValueError("Unknown framework specified. Use 'pytorch' or 'tensorflow'.")


def token_ids_to_text(token_ids, tokenizer, framework="pytorch"):
    """
    Convert a tensor of token IDs back to text.
    """
    fw = framework.lower()
    if fw in ["pytorch", "torch"]:
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return tokenizer.decode(token_ids.tolist()).replace("\n"," ")
    elif fw in ["tensorflow", "tf"]:
        if len(token_ids.shape) > 1:
            token_ids = tf.squeeze(token_ids, axis=0)
        return tokenizer.decode(token_ids.numpy().tolist()).replace("\n", " ")
    else:
        raise ValueError("Unknown framework specified. Use 'pytorch' or 'tensorflow'.")
