import random

import numpy as np
import tiktoken
import torch
import torch.nn as nn

from data.dataset import create_torch_dataloader, download_and_load_text
from models import init as pt_init
from training.callbacks import log_text_generation
from utils import utils
from utils.generate import generate
from utils.lr_scheduler import get_torch_lr_scheduler
from utils.plots import plot_losses, plot_lrs


def calc_loss_batch(
    input_batch, target_batch, model, device, criterion
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = criterion(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, criterion, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device, criterion)
        total_loss += loss.item()

    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, criterion):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, criterion
        )
        val_loss = calc_loss_loader(val_loader, model, device, criterion)
    model.train()
    return train_loss, val_loss


def train_torch(config, tokenizer, device):
    data_cfg = config["data"]
    text = download_and_load_text(data_cfg["url"], data_cfg["file_path"])

    train_ratio = config["training"]["train_ratio"]
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_loader = create_torch_dataloader(
        train_text,
        tokenizer,
        max_length=data_cfg["max_length"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_torch_dataloader(
        val_text,
        tokenizer,
        max_length=data_cfg["max_length"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    model = pt_init.create_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["optimizer"]["peak_lr"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    warmup_steps = int(total_steps * config["training"]["optimizer"]["warmup_ratio"])
    lr_scheduler = get_torch_lr_scheduler(
        optimizer,
        float(config["training"]["optimizer"]["initial_lr"]),
        float(config["training"]["optimizer"]["peak_lr"]),
        float(config["training"]["optimizer"]["min_lr"]),
        warmup_steps,
        total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    num_tokens = []
    lrs = []

    tokens_seen = 0
    global_step = 0
    num_epochs = range(config["training"]["num_epochs"])

    for epoch in num_epochs:
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            global_step += 1
            lrs.append(lr_scheduler.get_last_lr())

            loss = calc_loss_batch(inputs, targets, model, device, criterion)
            loss.backward()

            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()

            tokens_seen += inputs.numel()

        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, criterion
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        num_tokens.append(tokens_seen)
        print(
            f"Ep {epoch+1} (Iter {(global_step):06d}): "
            f"Train loss {train_loss:.3f}, "
            f"Val loss {val_loss:.3f}"
        )

        context_size = model.positional_embedding.weight.shape[0]

        log_text_generation(
            model, tokenizer, device, "Every effort moves you", generate, context_size
        )

    plot_losses(num_epochs, num_tokens, train_losses, val_losses, save_path=None)
    plot_lrs(range(len(lrs)), lrs)

    return model
