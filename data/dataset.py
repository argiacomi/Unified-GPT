import os
import urllib.request

import tensorflow as tf
import torch
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader


# ---------------------- Utility: Data Downloading ----------------------
def download_and_load_text(url, file_path, exclude_last=6):
    """
    Downloads text from a URL if the file does not exist.
    Extracts text by concatenating paragraphs from the HTML (excluding the last few).
    Returns the text content.
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            html = response.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.body.find_all("p")
        text = "".join([p.get_text() + "\n" for p in paragraphs[:-exclude_last]])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------- TensorFlow ----------------------
class GPTDatasetTF(tf.data.Dataset):
    """
    A TensorFlow Dataset that creates input-target pairs for language modeling.
    Splits tokenized text into sliding windows.
    """

    def __new__(cls, text, tokenizer, max_length, stride):
        token_ids = tf.convert_to_tensor(tokenizer.encode(text))
        inputs = tf.signal.frame(
            token_ids, frame_length=max_length, frame_step=stride, pad_end=False
        )
        targets = tf.signal.frame(
            token_ids[1:], frame_length=max_length, frame_step=stride, pad_end=False
        )
        n = tf.minimum(tf.shape(inputs)[0], tf.shape(targets)[0])
        inputs, targets = inputs[:n], targets[:n]
        return tf.data.Dataset.from_tensor_slices((inputs, targets))


def create_tf_dataloader(
    text, tokenizer, max_length, stride, batch_size=4, shuffle=True, drop_remainder=True
):
    """
    Creates a TensorFlow DataLoader (tf.data.Dataset) from the given text.
    """
    ds = GPTDatasetTF(text, tokenizer, max_length, stride)
    if shuffle:
        # Use dataset cardinality if available; fallback to a large constant.
        try:
            buf_size = int(ds.cardinality().numpy())
        except:
            buf_size = 10000
        ds = ds.shuffle(buffer_size=buf_size)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------- PyTorch ----------------------
class GPTDatasetTorch:
    """
    A PyTorch Dataset that creates input-target pairs for language modeling.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_torch_dataloader(
    text,
    tokenizer,
    max_length,
    stride,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Creates a PyTorch DataLoader from the given text.
    """

    dataset = GPTDatasetTorch(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
