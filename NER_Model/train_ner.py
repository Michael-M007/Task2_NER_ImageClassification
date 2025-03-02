import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import load_metric
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset (replace with your Animals-10 NER dataset if needed)
# If you are using your own dataset, format it properly.
dataset = load_dataset('conll2003')  # Replace with your NER dataset or preprocessed dataset

# Initialize the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)  # Adjust num_labels as per your dataset

# Define the label names (for the Animals-10 dataset, adjust according to your NER labels)
label_names = ['O', 'B-animal', 'I-animal']  # Adjust this based on your dataset

# Function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['tokens'], truncation=True, padding=True)

# Tokenize the dataset
train_dataset = dataset["train"].map(tokenize_function, batched=True)
valid_dataset = dataset["validation"].map(tokenize_function, batched=True)

# Ensure that the dataset has the appropriate format for token classification
def align_labels_with_tokens(examples):
    word_ids = examples["word_ids"]
    labels = examples["labels"]
    previous_word_idx = None
    aligned_labels = []

    for word_idx in word_ids:
        if word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(-100)  # We will ignore this label during training
        previous_word_idx = word_idx
    examples["labels"] = aligned_labels
    return examples

# Apply label alignment
train_dataset = train_dataset.map(align_labels_with_tokens, batched=True)
valid_dataset = valid_dataset.map(align_labels_with_tokens, batched=True)

# Load metric for evaluation
metric = load_metric("seqeval")

# Define the compute_metrics function to evaluate the model
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label for label, idx in zip(example, label_example) if idx != -100]
        for example, label_example in zip(labels, predictions)
    ]
    pred_labels = [
        [label for label, idx in zip(example, label_example) if idx != -100]
        for example, label_example in zip(predictions, predictions)
    ]
    
    return metric.compute(predictions=true_labels, references=pred_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=500,
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("animal_ner_model")

