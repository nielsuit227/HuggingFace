import pandas as pd


def load_data() -> list[str]:
    """Loads data from data/training folder"""
    path = "data/_chat.txt"
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

    for i in range(len(data) - 1, -1, -1):
        row = data[i]

        if len(row) == 0:
            continue

        # Merge enters
        if i > 1 and row[0] != "[":
            data[i - 1] += "\n" + data[i]
            continue

        filtered.insert(0, row)

    return filtered


def label_rows(data: list[str]) -> pd.DataFrame:
    """
    This splits the rows and labels them, using the structure of alternating messages
    """
    labelled: list[tuple[str, str]] = []
    for row in data:
        labelled.append((row[21 : row.find(":", 21)], row[row.find(":", 21) + 2 :]))
    return pd.DataFrame(labelled, columns=["label", "text"])


# Filter photos, videos, calls
rows = load_data()
rows = [r for r in rows if "\u200e" not in r]
rows = merge_rows(rows)
label_rows(rows).to_csv("data/cleaned.csv", index=False)
