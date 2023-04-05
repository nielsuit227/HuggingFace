import os

from datasets import Dataset, load_dataset


def load_data() -> Dataset:
    path = "data/train/"
    path = path if os.path.exists(path) else "HuggingFace/finetune/" + path

    data = load_dataset("csv", data_dir=path)

    data = data["train"].train_test_split(test_size=0.2)
    data = data.shuffle(seed=22)

    return data
