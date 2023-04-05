import os

import pandas as pd


def load_data() -> list[str]:
    """Loads data from data/training folder"""
    path = "data/raw/_chat.txt"
    path = path if os.path.exists(path) else "HuggingFace/finetune/" + path
    with open(path, "r") as f:
        data = f.read()
    return data.split("\n")


def merge_rows(data: list[str]) -> list[str]:
    """
    Loops backwards, if it starts with a non-date, it appends the message to the
    message above.
    """
    filtered: list[str] = []
    i = len(data) - 1

    while i > 0:
        row = data[i]

        # Merge enters
        if i > 1 and row[0] != "[":
            data[i - 1] += "\n" + data[i]
            i -= 1
            continue

        # Merge sender
        sender = row[21 : row.find(":", 20)]
        while i > 1 and sender == data[i - 1][21 : data[i - 1].find(":", 20)]:
            next_row = data[i - 1]

            message = row[row.find(":", 20) + 2 :]
            row = add_sentence(next_row, message)
            i -= 1

        filtered.insert(0, row)
        i -= 1

    return filtered


def add_sentence(row: str, add: str) -> str:
    """
    Makes sure the right punctuation is used.
    """
    row = row.strip()
    if row[-1] not in ".?!":
        return row + ". " + add
    else:
        return row + " " + add


def extract_questions(data: list[str]) -> pd.DataFrame:
    """
    First, sender is extracted.
    Then, all continuous questions of the same sender are merged.
    Then, all rea's messages are parsed as questions, mine as answers.

    TODO: add time filter
    """
    if "Rea" not in data[0]:
        data.pop(0)

    df = pd.DataFrame({"question": data[::2], "answer": data[1::2]})
    for key in df:
        df[key] = df[key].apply(lambda x: x[x.find(":", 21) + 2 :])

    return df


# Filter photos, videos, calls
rows = load_data()[:100]
rows = [r for r in rows if "[U+200E]" not in r]
rows = merge_rows(rows)
data = extract_questions(rows)
data.to_csv("data/train/first.csv", index=False)
