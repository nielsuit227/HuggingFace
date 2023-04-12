import evaluate
import torch
from dataloader import load_data
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_v1")


# Load metric
_, test_loader = load_data("data/cleaned.csv")
metric = evaluate.load("accuracy")

# Iteratively evaluate the model and compute metrics
model.eval()
for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Get model accuracy and F1 score
result = metric.compute()
print(result)
