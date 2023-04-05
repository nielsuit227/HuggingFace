from typing import Any

import evaluate
import numpy as np
from prep_dataset import load_data
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments


def tokenize_function(example: dict[str, Any]) -> list[int]:
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


ckpt = "BelleGroup/BELLE-LLAMA-7B-2M"
model = LlamaForCausalLM.from_pretrained(
    ckpt, device_map="auto", low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
metric = evaluate.load("accuracy")
training_args = TrainingArguments()
data = load_data()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
