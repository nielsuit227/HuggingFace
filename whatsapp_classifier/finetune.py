import os

from dataloader import load_data
from dotenv import load_dotenv
from optimizer import get_optimizer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

load_dotenv()

train_loader, test_loader = load_data("data/first.csv")

# Load model from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    os.getenv("llm", "distilbert-base-uncased"), num_labels=2
)

# Get optimizer
optimizer, lr_scheduler = get_optimizer(model, len(train_loader))

# Get params
num_epochs = int(os.getenv("num_epochs", 5))
num_training_steps = num_epochs * len(train_loader)

# Setup progress bar
progress_bar = tqdm(range(num_training_steps))

# Train with torch
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Save model to disk
model.save_pretrained("HuggingFace/whatsapp_classifier/models/distilbert_v1")
