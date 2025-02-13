import tensorflow as tf
import torch

from utils.generate import generate, text_to_token_ids, token_ids_to_text


# TensorFlow callback for logging and text generation at epoch end.
class TrainingCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset_size, batch_size, start_context, tokenizer, context_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.start_context = text_to_token_ids(
            start_context, self.tokenizer, framework="tensorflow"
        )
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.lrs = []
        self.tokens_seen = 0
        self.num_tokens = []

    def on_train_batch_begin(self, batch, logs=None):
        """Update learning rate at the start of each batch."""
        lr = self.model.optimizer.learning_rate.numpy()
        self.lrs.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        tokens_in_epoch = self.batch_size * (self.dataset_size // self.batch_size)
        remainder = self.dataset_size % self.batch_size
        if remainder > 0:
            tokens_in_epoch += remainder
        self.tokens_seen += tokens_in_epoch
        self.num_tokens.append(self.tokens_seen)

        generated = generate(
            self.model,
            self.start_context,
            max_new_tokens=50,
            context_size=self.context_size,
            temperature=0.0,
            top_k=None,
            framework="tensorflow",
        )

        text = token_ids_to_text(generated, self.tokenizer, framework="tensorflow")
        print(f"\nGenerated text:\n{text}\n")


# PyTorch helper function for logging text generation.
def log_text_generation(
    model, tokenizer, device, start_context, generate_func, context_size
):

    model.eval()
    input_ids = text_to_token_ids(start_context, tokenizer, framework="pytorch").to(
        device
    )
    with torch.no_grad():
        generated = generate_func(
            model,
            input_ids,
            max_new_tokens=50,
            context_size=context_size,
            framework="pytorch",
        )
    decoded_output = token_ids_to_text(generated, tokenizer, framework="pytorch")
    print(f"\nGenerated text:\n{decoded_output}\n")
    model.train()
