import torch
import os
import numpy as np
import torch.nn.functional as F
import pickle
from Model import classifier # Ensure this is the correct import for your model

# --- Assume these dependencies are available in the calling environment ---
# from libs import sql_tokenizer # You need your custom tokenizer
# from Model import classifier # You need your model class definition

# --- Assume these constants are defined in the calling environment ---
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 32
# NUM_LAYERS = 1 # Must match the saved model's training config
# DROPOUT = 0.2 # Should match the saved model's training config
# MAX_SEQ_LEN = 128 # Must match the max_seq_len used in TextDataset during training

# Ensure paths are correct relative to where the calling script is run
# VOCAB_STOI_PATH = "pickles/vocab_stoi.pkl"
# VOCAB_ITOS_PATH = "pickles/vocab_itos.pkl" # Not strictly needed for prediction, but good practice
# LABEL_TO_INT_PATH = "pickles/label_to_int.pkl"
# SAVED_WEIGHTS_DIR = 'saved_weights/' # Directory where model weights are saved


def initialize_device():
    """
    Initializes and returns the appropriate device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The device to use for computations.
    """
    # torch.backends.cudnn.deterministic = True # Optional, not strictly needed for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_custom_mappings(stoi_path, label_map_path):
    """
    Loads custom vocabulary (token to index) and label mappings from pickle files.

    Args:
        stoi_path (str): Path to the pickled token to index mapping (stoi).
        label_map_path (str): Path to the pickled label mappings (label_to_int, int_to_label).

    Returns:
        tuple: (stoi, int_to_label), or (None, None) if loading fails.
    """
    try:
        with open(stoi_path, 'rb') as f:
            stoi = pickle.load(f)
        with open(label_map_path, 'rb') as f:
            label_maps = pickle.load(f)
            int_to_label = label_maps['int_to_label'] # We need int_to_label for prediction output

        print(f"Loaded vocabulary from {stoi_path}")
        print(f"Loaded label mapping from {label_map_path}")
        return stoi, int_to_label
    except FileNotFoundError as e:
        print(f"Error loading mapping files: {e}")
        print("Please ensure the training script has been run to generate these files.")
        return None, None
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return None, None


def create_model(vocab_size, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
    """
    Creates an instance of the classifier model with the specified architecture.

    Args:
        vocab_size (int): The size of the vocabulary.
        output_dim (int): The number of output classes.
        embedding_dim (int): The dimension of the token embeddings.
        hidden_dim (int): The dimension of the RNN hidden states.
        num_layers (int): The number of RNN layers.
        dropout (float): The dropout probability.

    Returns:
        Model.classifier: An instance of the model, or None if the classifier class is not available.
    """
    # Ensure the classifier class is available in the calling environment
    try:
        # Access the classifier class from the calling environment
        # This assumes 'classifier' is imported where this function is called
        model = classifier(
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            num_layers,
            bidirectional=True,
            dropout=dropout
        )
        return model
    except NameError:
        print("Error: 'classifier' class not found. Please ensure Model.py is imported.")
        return None
    except Exception as e:
        print(f"Error creating model instance: {e}")
        return None


def load_model_weights(model, device, saved_weights_dir):
    """
    Loads the latest model state_dict from the specified directory into the model.

    Args:
        model: The model instance to load weights into.
        device (torch.device): The device to map the weights to.
        saved_weights_dir (str): Directory containing the saved .pt model weights.

    Returns:
        Model.classifier: The model instance with loaded weights, or None if loading fails.
    """
    try:
        # Find the latest saved weight file (e.g., based on timestamp)
        list_of_files = [os.path.join(saved_weights_dir, f) for f in os.listdir(saved_weights_dir) if f.endswith('.pt')]
        if not list_of_files:
            print(f"Error: No .pt files found in {saved_weights_dir}")
            return None

        # You might want a more robust way to select the 'best' model if you have many
        # For now, loading the latest by modification time
        latest_file = max(list_of_files, key=os.path.getmtime)

        print(f"Attempting to load weights from: {latest_file}")
        model.load_state_dict(torch.load(latest_file, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Loaded model weights successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Saved weights directory not found at {saved_weights_dir}")
        return None
    except Exception as e:
        print(f"Error loading model weights from {latest_file}: {e}")
        return None


def predict_sentence(model, sentence, stoi, int_to_label, tokenizer, device, max_seq_len):
    """
    Makes a prediction for a single input sentence using the loaded model.

    Args:
        model: The loaded PyTorch model in evaluation mode.
        sentence (str): The input sentence to predict.
        stoi (dict): Token to index mapping (string to integer).
        int_to_label (dict): Index to label string mapping (integer to string).
        tokenizer (function): The tokenizer function (e.g., sql_tokenizer from libs).
        device (torch.device): The device the model is on.
        max_seq_len (int): Maximum sequence length used for padding/truncation during training.

    Returns:
        tuple: (predicted_label, confidence_percentage, probabilities_dict),
               or (None, None, None) if tokenization fails or mappings are invalid.
    """
    model.eval() # Ensure model is in evaluation mode

    # Process the sentence using the custom tokenizer
    tokenized = tokenizer(sentence)

    # Handle empty tokenized list
    if not tokenized:
        print(f"Warning: Tokenization resulted in an empty list for sentence: '{sentence}'")
        return None, None, None

    # Convert tokens to indices using the stoi mapping
    # Use .get() with a default for <unk>
    unk_idx = stoi.get('<unk>', 0) # Default <unk> index is typically 0
    indexed = [stoi.get(token, unk_idx) for token in tokenized]

    # Handle padding and truncation for prediction
    if len(indexed) > max_seq_len:
        indexed = indexed[:max_seq_len]
        length = max_seq_len
    else:
        length = len(indexed)
        # Pad the sequence
        pad_idx = stoi.get('<pad>', 1) # Default <pad> index is typically 1
        padding_needed = max_seq_len - length
        indexed = indexed + [pad_idx] * padding_needed

    # Convert to tensors and move to device
    # Add batch dimension (batch size = 1 for a single sentence)
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device) # Shape becomes [1, max_seq_len]

    # Lengths tensor stays on CPU, shape [1]
    length_tensor = torch.LongTensor([length])

    with torch.no_grad():
        # Pass the length tensor (on CPU) to the model
        prediction_logits = model(tensor, length_tensor)

    # Apply softmax to get probabilities
    prediction_probs = F.softmax(prediction_logits, dim=1) # dim=1 because it's a batch of 1

    # Get the predicted class index (highest probability)
    predicted_class_index = prediction_probs.argmax(dim=1).item()

    # Get the predicted label string
    pred_lbl = "Unknown" # Default in case of index out of bounds or mapping issue
    if int_to_label and 0 <= predicted_class_index < len(int_to_label):
        pred_lbl = int_to_label[predicted_class_index]
    else:
        print(f"Warning: Predicted class index {predicted_class_index} out of bounds for label mapping.")


    # Get the confidence (probability of the predicted label)
    confidence = prediction_probs[0, predicted_class_index].item() * 100

    # Get probabilities for all labels
    probabilities_dict = {}
    # Ensure int_to_label is valid before iterating
    if int_to_label:
        # Sort labels by their index for consistent output order
        sorted_labels = sorted(int_to_label.items(), key=lambda item: item[0]) # Sort by index
        for idx, label in sorted_labels:
            if 0 <= idx < prediction_probs.shape[1]:
                 prob = prediction_probs[0, idx].item()
                 probabilities_dict[label] = prob * 100 # Store as percentage
            else:
                 probabilities_dict[label] = 0.0 # Assign 0 if index out of bounds


    # Return predicted label, confidence, and probabilities dictionary
    return pred_lbl, confidence, probabilities_dict


# Example of how to use these functions in another script:
"""
# --- In your separate script (e.g., another_script.py) ---

import torch
from Model import classifier # Import your model class
from libs import sql_tokenizer # Import your tokenizer
from predictor import (
    initialize_device,
    load_custom_mappings,
    create_model,
    load_model_weights,
    predict_sentence
)

# Define constants used for training and prediction (must match training)
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 1 # <<< Make sure this matches your SAVED model
DROPOUT = 0.2
MAX_SEQ_LEN = 128 # <<< Make sure this matches your training dataset padding/truncation

# Define paths to your saved files
VOCAB_STOI_PATH = "pickles/vocab_stoi.pkl"
LABEL_TO_INT_PATH = "pickles/label_to_int.pkl"
SAVED_WEIGHTS_DIR = 'saved_weights/'

# 1. Initialize Device
device = initialize_device()

# 2. Load Custom Vocabulary and Label Mappings
stoi, int_to_label = load_custom_mappings(VOCAB_STOI_PATH, LABEL_TO_INT_PATH)

if stoi is None or int_to_label is None:
    print("Failed to load mappings. Cannot proceed with prediction.")
    exit()

# Get vocabulary size and output dimension from loaded mappings
vocab_size = len(stoi)
output_dim = len(int_to_label)

# 3. Create Model Instance (architecture must match training)
model = create_model(vocab_size, output_dim, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)

if model is None:
    print("Failed to create model instance. Cannot proceed with prediction.")
    exit()

# 4. Load Trained Model Weights
model = load_model_weights(model, device, SAVED_WEIGHTS_DIR)

if model is None:
    print("Failed to load model weights. Cannot proceed with prediction.")
    exit()

# 5. Use the predict_sentence function
sentence_to_test = "SELECT * FROM users WHERE id='1'"
predicted_label, confidence, probabilities = predict_sentence(
    model,
    sentence_to_test,
    stoi,
    int_to_label,
    sql_tokenizer, # Pass your tokenizer function
    device,
    MAX_SEQ_LEN
)

if predicted_label is not None:
    print(f"\nPrediction for '{sentence_to_test}':")
    print(f"  Predicted Label: {predicted_label}")
    print(f"  Confidence: {confidence:.2f}%")
    print("  Probabilities:", probabilities)

# You can now use the 'predict_sentence' function multiple times
# or integrate it into an application.
"""