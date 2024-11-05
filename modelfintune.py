from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd
import sentencepiece as spm


# Load your data
data = pd.read_csv('qa_dataset.csv')
data = data.dropna()  # Drop any rows with missing values
data['input_text'] = "Question: " + data['Question'] + " </s>"
data['target_text'] = data['Answer'] + " </s>"

# Convert to Hugging Face dataset format
dataset = Dataset.from_pandas(data[['input_text', 'target_text']])

# Tokenize the data
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess(example):
    inputs = tokenizer(example['input_text'], max_length=512, padding='max_length', truncation=True)
    targets = tokenizer(example['target_text'], max_length=512, padding='max_length', truncation=True)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Load model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

model.save_pretrained('./t5-qa-model')
tokenizer.save_pretrained('./t5-qa-tokenizer')


