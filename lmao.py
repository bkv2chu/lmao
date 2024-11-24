import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with the exact model name/path
TRAIN_FILE = "haha.txt"  # Your training data file
OUTPUT_DIR = "./llama-3.1-8b-finetuned"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 500
LOGGING_STEPS = 50

# Load Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

tokenizer.pad_token = tokenizer.eos_token

# Load Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Prepare Dataset
def preprocess_data(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": TRAIN_FILE})

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["text"])

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    push_to_hub=False,
    remove_unused_columns=False,
    report_to="none"
)

# Trainer Setup
print("Setting up Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

# Fine-tune the model
print("Starting training...")
trainer.train()

# Save the final model
print("Saving the final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")