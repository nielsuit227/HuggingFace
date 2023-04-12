import torch
from dataloader import tokenizer
from transformers import AutoModelForSequenceClassification

# Load Model
model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_v1")

input_ = "Going for a run"
tokens = tokenizer(input_, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    predictions = torch.argmax(model(**tokens).logits, dim=-1)

print(f"Likely printed by: {'Rea' if predictions.item() else 'Niels'}")
