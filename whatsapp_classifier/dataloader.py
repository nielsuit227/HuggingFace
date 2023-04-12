import os

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(os.getenv("llm", "distilbert-base-uncased"))


def tokenize(batch):
    tokenized_batch = tokenizer(
        batch["text"], padding=True, truncation=True, max_length=128
    )
    return tokenized_batch


def load_data(path: str) -> Dataset:
    """
    Given a path, loads a CSV, and creates PyTorch Dataloaders for training and testing.

    Parameters
    ----------
    path : str
        Location of the CSV

    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.read_csv(path)

    data = Dataset.from_pandas(df)
    data = data.train_test_split(test_size=0.2)
    data = data.class_encode_column("label")

    data = data.map(tokenize, batched=True)

    data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Transform to Torch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        data["train"],
        shuffle=True,
        batch_size=int(os.getenv("batch_size", 8)),
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        data["test"],
        shuffle=True,
        batch_size=int(os.getenv("batch_size", 8)),
        collate_fn=data_collator,
    )

    return train_loader, test_loader
