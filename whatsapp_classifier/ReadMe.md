# Whatsapp Sender Classifier
Based on a whatsapp conversation, the scripts here finetune DistilBert for the conversation and based on new input, predicts who is more likely to have written the text.

## Quickstart
1. Download conversation and store as `/data/_chat.txt`
2. Run `clean.py`
3. Run `finetune.py`
4. Run `eval.py`

> One can also import the `predict_on_input` function from `eval.py`, to see the prediction on any arbitrary text input. 

## Data
In WhatsApp, you can export a conversation. Given the single modularity, please do so without attachments. You can store this in `/data/_chat.txt`. No further edits have to be made :) 

### Cleaning 
Few steps are involved to clean the data. 

1. Removing references (\[U+200E])
2. Merge rows that are part of the same message (line starts with an enter instead of \[)
3. Label the sender
4. Store as CSV

These steps are executed in `clean.py`. 

### Loading
The script `dataloader.py` contains a function `load_data`, that takes the path that takes a csv, and 

## Finetuning
Loads the data, model, and optimizer, executes the training loops and stores the resulting model.

### Optimizer
A basic AdamW is used, with weight decay.