import os

from dotenv import load_dotenv
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, AutoModel, get_scheduler

load_dotenv()


def get_optimizer(model: AutoModel, num_batches: int) -> tuple[AdamW, LambdaLR]:
    # Model parameters
    learning_rate = float(os.getenv("learning_rate", 5e-5))
    num_epochs = int(os.getenv("num_epochs", 5))
    num_warmup_steps = int(os.getenv("num_warmup_steps", 10))

    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Further define learning rate scheduler
    num_training_steps = num_epochs * num_batches
    scheduler = get_scheduler(
        "linear",  # linear decay
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler
