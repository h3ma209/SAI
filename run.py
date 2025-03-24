#!/usr/bin/env python
# coding: utf-8

import torch
from torchtext import data
import numpy as np
import torch.optim as optim
import os
import random
import torch.nn as nn
from Model import *
from transformers import BertTokenizer
from libs import sql_tokenizer  # Import sql_tokenizer from libs.py
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

# Constants
SEED = 1000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2
N_EPOCHS = 50
DATA_PATH = 'csv_files/safe_xss_sql.csv'  # Removed ../
WEIGHTS_PATH = 'saved_weights.pt'
MODELS_DIR = "saved_models"  # Removed ../
WEIGHTS_DIR = "saved_weights"  # Removed ../
BERT_MODEL_NAME = 'bert-base-uncased'
VOCAB_PATH = "pickles/vocab.pt"
LABEL_PATH = "pickles/label.pt"

# Set random seeds for reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Tokenization function using BERT tokenizer
def bert_tokenizer(text,debug=False):
    sql = sql_tokenizer(text)
    if debug:
        print(f"tok: {sql}")

    return tokenizer.tokenize(sql)

# Define fields
# TEXT = data.Field(tokenize=bert_tokenizer, batch_first=True, include_lengths=True)
TEXT = data.Field(tokenize=sql_tokenizer, batch_first=True, include_lengths=True)

LABEL = data.LabelField(dtype=torch.float, batch_first=True)
fields = [(None, None), ('text', TEXT), ('label', LABEL)]

# Load and preprocess data
def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    training_data = data.TabularDataset(path=DATA_PATH, format='csv', fields=fields, skip_header=True)
    print(vars(training_data.examples[0]))

    filtered_examples = []
    for example in training_data.examples:
        if example.text and all(word.strip() for word in example.text):
            filtered_examples.append(example)
        else:
            example.text = ["[PAD]"]
            filtered_examples.append(example)

    training_data.examples = filtered_examples
    print("Data loaded and preprocessed.")
    return training_data

# Build vocabulary and save
def build_and_save_vocab(train_data):
    print("Building vocabulary...")
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    print(f"Size of TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Size of LABEL vocabulary: {len(LABEL.vocab)}")

    print("Saving vocabulary and label...")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(TEXT, f)
    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(LABEL, f)
    print("Vocabulary and label saved.")

# Create iterators
def create_iterators(train_data, valid_data):
    print("Creating iterators...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device
    )
    print("Iterators created.")
    return train_iterator, valid_iterator, device

# Initialize model
def initialize_model(device):
    print("Initializing model...")
    model = classifier(len(TEXT.vocab), EMBEDDING_DIM, HIDDEN_DIM, len(LABEL.vocab.stoi), NUM_LAYERS, bidirectional=True, dropout=DROPOUT).to(device)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("Model initialized.")
    return model

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define loss and optimizer
def define_loss_optimizer(model, device):
    print("Defining loss and optimizer...")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    print("Loss and optimizer defined.")
    return optimizer, criterion

# Accuracy function
def binary_accuracy(preds, y):
    rounded_pred = torch.round(preds)
    _, pred_label = torch.max(rounded_pred, dim=1)
    correct = (pred_label == y).float()
    return correct.sum() / len(correct)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text

        # Explicitly move text_lengths to CPU and ensure it's int64
        text_lengths_cpu = text_lengths.cpu().long()

        # Print the device and dtype for debugging
        # print(f"Device of text_lengths: {text_lengths.device}")
        # print(f"Device of text_lengths_cpu: {text_lengths_cpu.device}")
        # print(f"Type of text_lengths_cpu: {text_lengths_cpu.dtype}")

        predictions = model(text, text_lengths_cpu).squeeze()
        loss = criterion(predictions, batch.label.long())
        acc = binary_accuracy(predictions, batch.label.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
# Evaluation loop

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            # Explicitly move text_lengths to CPU and ensure it's int64
            text_lengths_cpu = text_lengths.cpu().long()

            # Print the device and dtype for debugging (optional for final code)
            # print(f"Evaluate - Device of text_lengths: {text_lengths.device}")
            # print(f"Evaluate - Device of text_lengths_cpu: {text_lengths_cpu.device}")
            # print(f"Evaluate - Type of text_lengths_cpu: {text_lengths_cpu.dtype}")

            predictions = model(text, text_lengths_cpu).squeeze()
            loss = criterion(predictions, batch.label.long())
            acc = binary_accuracy(predictions, batch.label.long())
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
# Save model function
def save_model(model, valid_acc):
    md_val_acc = "%.2f" % (valid_acc * 100)
    model_name = f"Acc {md_val_acc}.pt"
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, model_name))
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, model_name))
    print(f"SAVED\n{model_name}")

# Prediction function
def predict(model, sentence):
    pred_2_lbl = {1: 'xss', 0: 'sql', 2: "safe"}
    # tokenized = bert_tokenizer(sentence, debug=True)
    tokenized = sql_tokenizer(sentence )
    print(f"tokenized: {tokenized}")
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device).unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    print(f"tensor: {tensor}")
    print(f"length_tensor: {length_tensor}")
    prediction = model(tensor, length_tensor)
    pred_lbl = np.argmax(prediction.detach().cpu().numpy())
    print('\npredicted threat type:', pred_2_lbl[pred_lbl])
    return prediction

if __name__ == "__main__":
    training_data = load_and_preprocess_data()
    train_data, valid_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))
    build_and_save_vocab(train_data)
    train_iterator, valid_iterator, device = create_iterators(train_data, valid_data)
    model = initialize_model(device)
    optimizer, criterion = define_loss_optimizer(model, device)

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    save_model(model, valid_acc)
    print("\nExample prediction:")
    predict(model, 'group_concat(namapemohon,0x3a,email),3,4,5,6 from pendaftaran_user ')