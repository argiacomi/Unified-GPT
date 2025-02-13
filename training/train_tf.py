import tensorflow as tf
import tiktoken

from data.dataset import create_tf_dataloader, download_and_load_text
from models import init as tf_init
from training.callbacks import TrainingCallback
from utils import utils
from utils.generate import generate, text_to_token_ids, token_ids_to_text
from utils.lr_scheduler import WarmupCosineDecay
from utils.plots import plot_losses, plot_lrs


def train_tf(config, tokenizer, device):
    with tf.device(f"/{device}:0"):
        # Load and split text
        data_cfg = config["data"]
        text = download_and_load_text(data_cfg["url"], data_cfg["file_path"])
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        # Create datasets
        train_loader = create_tf_dataloader(
            train_text,
            tokenizer,
            max_length=data_cfg["max_length"],
            stride=data_cfg["stride"],
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = create_tf_dataloader(
            val_text,
            tokenizer,
            max_length=data_cfg["max_length"],
            stride=data_cfg["stride"],
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        # Create model via factory (uses framework & variant settings)
        model = tf_init.create_model(config)

        # Setup learning rate schedule & optimizer
        total_steps = train_loader.cardinality().numpy() * config["training"]["num_epochs"]
        warmup_steps = int(total_steps * config["training"]["optimizer"]["warmup_ratio"])
        lr_schedule = WarmupCosineDecay(
            initial_lr=float(config["training"]["optimizer"]["initial_lr"]),
            peak_lr=float(config["training"]["optimizer"]["peak_lr"]),
            min_lr=float(config["training"]["optimizer"]["min_lr"]),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config["training"]["optimizer"]["weight_decay"],
            clipnorm=config["training"]["optimizer"]["clipnorm"],
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss_fn)

        # Setup callbacks (text generation at end of each epoch)
        context_size = config["model"]["configs"][config["model"]["size"]]["context_length"]
        train_cb = TrainingCallback(
            dataset_size=train_loader.cardinality().numpy(),
            batch_size=config["training"]["batch_size"],
            start_context="Every effort moves you",
            tokenizer=tokenizer,
            context_size=context_size,
        )

        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=config["training"]["num_epochs"],
            callbacks=[train_cb],
        )

        plot_losses(
            range(len(history.history["loss"])),
            train_cb.num_tokens,
            history.history["loss"],
            history.history["val_loss"],
            save_path=None,
        )
        plot_lrs(range(len(train_cb.lrs)), train_cb.lrs)

        return model


def train_tf_custom(config, tokenizer, device):
    with tf.device(f"/{device}:0"):
        # Load and split text
        data_cfg = config["data"]
        text = download_and_load_text(data_cfg["url"], data_cfg["file_path"])
        split_idx = int(len(text) * 0.9)
        train_text, val_text = text[:split_idx], text[split_idx:]

        # Create datasets
        train_loader = create_tf_dataloader(
            train_text,
            tokenizer,
            max_length=data_cfg["max_length"],
            stride=data_cfg["stride"],
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = create_tf_dataloader(
            val_text,
            tokenizer,
            max_length=data_cfg["max_length"],
            stride=data_cfg["stride"],
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        # Create model
        model = tf_init.create_model(config)

        # Setup LR schedule & optimizer
        total_steps = (
            train_loader.cardinality().numpy() * config["training"]["num_epochs"]
        )
        warmup_steps = int(
            total_steps * config["training"]["optimizer"]["warmup_ratio"]
        )
        lr_schedule = WarmupCosineDecay(
            initial_lr=float(config["training"]["optimizer"]["initial_lr"]),
            peak_lr=float(config["training"]["optimizer"]["peak_lr"]),
            min_lr=float(config["training"]["optimizer"]["min_lr"]),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config["training"]["optimizer"]["weight_decay"],
            clipnorm=config["training"]["optimizer"]["clipnorm"],
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Logging containers
        train_losses, val_losses, num_tokens, lrs = [], [], [], []
        tokens_seen = 0
        context_size = config["model"]["configs"][config["model"]["size"]][
            "context_length"
        ]
        start_context = "Every effort moves you"
        num_epochs = config["training"]["num_epochs"]

        global_step = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss, steps = 0.0, 0

            # Training loop
            for inputs, targets in train_loader:
                with tf.GradientTape() as tape:
                    logits = model(inputs, training=True)
                    loss = loss_fn(targets, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Log LR and accumulate loss/tokens
                current_lr = (
                    optimizer.learning_rate.numpy()
                    if hasattr(optimizer.learning_rate, "numpy")
                    else optimizer.learning_rate
                )
                lrs.append(current_lr)
                epoch_loss += loss.numpy()
                steps += 1
                global_step += 1
                tokens_seen += tf.size(inputs).numpy()

            avg_train_loss = epoch_loss / steps
            train_losses.append(avg_train_loss)
            num_tokens.append(tokens_seen)

            # Validation loop
            val_epoch_loss, val_steps = 0.0, 0
            for inputs, targets in val_loader:
                logits = model(inputs, training=False)
                loss = loss_fn(targets, logits)
                val_epoch_loss += loss.numpy()
                val_steps += 1
            avg_val_loss = val_epoch_loss / val_steps if val_steps > 0 else float("nan")
            val_losses.append(avg_val_loss)

            print(f"Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

            # Generate sample text at epoch end (ensure inference mode)
            input_ids = text_to_token_ids(
                start_context, tokenizer, framework="tensorflow"
            )
            generated = generate(
                model,
                input_ids,
                max_new_tokens=50,
                context_size=context_size,
                temperature=0.0,
                top_k=None,
                framework="tensorflow"
            )

            generated_text = token_ids_to_text(
                generated, tokenizer, framework="tensorflow"
            )
            print(f"Epoch {epoch+1} generated text:\n{generated_text}\n")

        plot_losses(
            range(num_epochs), num_tokens, train_losses, val_losses, save_path=None
        )
        plot_lrs(range(len(lrs)), lrs)
        return model
