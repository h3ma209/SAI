import torch
import os
import numpy as np
import torch.nn.functional as F
import pickle # Import pickle for loading mappings

# Assuming libs is a package in the same directory
try:
    from libs import * # Import all public names from libs.py
    # Ensure necessary functions are available after import *
    # If you didn't use __all__ in libs.py, you might need specific imports:
    # from libs.libs import sql_tokenizer, load_vocab # Example specific imports
except ImportError:
    print("Error: Could not import from libs package.")
    print("Please ensure you have a 'libs' directory with an '__init__.py' and 'libs.py'.")
    exit()

# Assuming Model.py is in the same directory
try:
    from Model import classifier
except ImportError:
    print("Error: Could not import 'classifier' from Model.py.")
    print("Please ensure Model.py exists in the same directory or update the import path.")
    exit()

# --- Constants (should match the SAVED model's training) ---
# Ensure these match the values used during training for the model file you are loading!
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 1  # <<< CORRECTED: Set to 1 to match the saved model based on error keys
DROPOUT = 0.2   # Dropout might be different for a 1-layer model, check your training config
MAX_SEQ_LEN = 128 # Should match the max_seq_len used in TextDataset

# Ensure paths are correct relative to where you run predict.py
VOCAB_STOI_PATH = "pickles/vocab_stoi.pkl"
VOCAB_ITOS_PATH = "pickles/vocab_itos.pkl"
LABEL_TO_INT_PATH = "pickles/label_to_int.pkl"
SAVED_WEIGHTS_DIR = 'saved_weights/' # Directory where model weights are saved

# --- Helper Functions ---

def initialize_device():
    """Initializes and returns the device (GPU if available, otherwise CPU)."""
    # torch.backends.cudnn.deterministic = True # Not strictly needed for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_custom_mappings(stoi_path, itos_path, label_map_path):
    """Loads custom vocabulary and label mappings from pickle files."""
    try:
        with open(stoi_path, 'rb') as f:
            stoi = pickle.load(f)
        with open(itos_path, 'rb') as f:
            itos = pickle.load(f)
        with open(label_map_path, 'rb') as f:
            label_maps = pickle.load(f)
            label_to_int = label_maps['label_to_int']
            int_to_label = label_maps['int_to_label']

        print(f"Loaded vocabulary from {stoi_path} and {itos_path}")
        print(f"Loaded label mapping from {label_map_path}")
        return stoi, itos, label_to_int, int_to_label
    except FileNotFoundError as e:
        print(f"Error loading mapping files: {e}")
        print("Please ensure the training script has been run to generate these files.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return None, None, None, None

# Corrected create_model to use the global NUM_LAYERS constant
def create_model(vocab_size, output_dim, num_layers=NUM_LAYERS):
    """Creates an instance of the classifier model with specified architecture."""
    # Ensure architecture parameters match the trained model
    return classifier(
        vocab_size,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        output_dim,
        num_layers, # <<< Use the num_layers parameter
        bidirectional=True,
        dropout=DROPOUT
    )

def load_model_weights(model, device, saved_weights_dir):
    """Loads the latest model state_dict from the specified directory."""
    try:
        # Find the latest saved weight file (e.g., based on timestamp)
        list_of_files = [os.path.join(saved_weights_dir, f) for f in os.listdir(saved_weights_dir) if f.endswith('.pt')]
        if not list_of_files:
            print(f"Error: No .pt files found in {saved_weights_dir}")
            return None

        # You might want a more robust way to select the 'best' model if you have many
        # For now, loading the latest by modification time as in your original snippet
        latest_file = max(list_of_files, key=os.path.getmtime)

        print(f"Attempting to load weights from: {latest_file}") # Print the file being loaded
        model.load_state_dict(torch.load(latest_file, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Loaded model weights from {latest_file}")
        return model
    except FileNotFoundError:
        print(f"Error: Saved weights directory not found at {saved_weights_dir}")
        return None
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Print the error details for debugging loading issues
        print(f"Loading error details: {e}")
        return None

# --- Prediction Function ---

def predict_sentence(model, sentence, stoi, int_to_label, tokenizer, device, max_seq_len=MAX_SEQ_LEN, ground_truth_label=None):
    """
    Makes a prediction for a single input sentence.

    Args:
        model: The loaded PyTorch model.
        sentence (str): The input sentence to predict.
        stoi (dict): Token to index mapping.
        int_to_label (dict): Index to label string mapping.
        tokenizer (function): The tokenizer function (e.g., sql_tokenizer).
        device (torch.device): The device to use for inference.
        max_seq_len (int): Maximum sequence length for padding/truncation.
        ground_truth_label (str, optional): The true label for the sentence, if known, to check correctness.

    Returns:
        tuple: (predicted_label, confidence, probabilities), or None if tokenization fails.
    """
    model.eval() # Ensure model is in evaluation mode

    # Process the sentence using the custom tokenizer
    tokenized = tokenizer(sentence)

    # Handle empty tokenized list
    if not tokenized:
        print(f"Warning: Tokenization resulted in an empty list for sentence: '{sentence}'")
        print("--------------------------\n")
        return None, None, None # Return None for all results

    print(f"Sentence: '{sentence}'") # Print the original sentence
    print(f"Tokenized: {tokenized}")

    # Convert tokens to indices using the stoi mapping
    # Use .get() with a default for <unk>
    unk_idx = stoi.get('<unk>', 0) # Default <unk> index is 0
    indexed = [stoi.get(token, unk_idx) for token in tokenized]

    # print(f"Indexed: {indexed}") # Optional: print indexed list

    # Handle padding and truncation for prediction
    if len(indexed) > max_seq_len:
        indexed = indexed[:max_seq_len]
        length = max_seq_len
    else:
        length = len(indexed)
        # Pad the sequence
        pad_idx = stoi.get('<pad>', 1) # Default <pad> index is 1
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
    pred_lbl = "Unknown" # Default in case of index out of bounds
    if 0 <= predicted_class_index < len(int_to_label):
        pred_lbl = int_to_label[predicted_class_index]

    # Get the confidence (probability of the predicted label)
    confidence = prediction_probs[0, predicted_class_index].item() * 100

    print('Predicted threat type:', pred_lbl)
    print(f'Confidence: {confidence:.2f}%')

    # Display probabilities for all labels
    probabilities = {}
    print("Probabilities:")
    # Sort labels by their index for consistent output order
    sorted_labels = sorted(int_to_label.items(), key=lambda item: item[0]) # Sort by index
    for idx, label in sorted_labels:
        prob = prediction_probs[0, idx].item() # Get probability for this label (from batch item 0)
        probabilities[label] = prob * 100 # Store as percentage
        print(f"  {label}: {prob:.2%}") # Format and print as percentage


    # Check correctness if ground_truth_label is provided
    if ground_truth_label is not None:
        if pred_lbl == ground_truth_label:
            print(f'Prediction: Correct (Ground Truth: {ground_truth_label})')
        else:
            print(f'Prediction: Incorrect (Ground Truth: {ground_truth_label})')


    print("--------------------------\n")

    # Return predicted label, confidence, and probabilities
    return pred_lbl, confidence, probabilities


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Initialize Device
    device = initialize_device()

    # 2. Load Custom Vocabulary and Label Mappings
    stoi, itos, label_to_int, int_to_label = load_custom_mappings(
        VOCAB_STOI_PATH, VOCAB_ITOS_PATH, LABEL_TO_INT_PATH
    )

    if stoi is None or itos is None or label_to_int is None or int_to_label is None:
        print("Failed to load necessary mappings. Exiting.")
        exit()

    # Get vocabulary size and output dimension from loaded mappings
    vocab_size = len(stoi)
    output_dim = len(label_to_int)

    # 3. Create Model Instance (architecture must match training)
    # Pass the NUM_LAYERS constant here to ensure it matches the saved model's architecture
    model = create_model(vocab_size, output_dim, num_layers=NUM_LAYERS)


    # 4. Load Trained Model Weights
    model = load_model_weights(model, device, SAVED_WEIGHTS_DIR)

    if model is None:
        print("Failed to load model weights. Exiting.")
        exit()

    # 5. Make Predictions on Example Sentences
    example_sentences = [
        'this is hema', # Assuming safe
        'this is sql', # Assuming safe (unless 'sql' is a keyword)
        'this is <script>', # Assuming xss
        'this is <script> alert() </script>', # Assuming xss
        'this is <script> alert() </script> and this is sql', # Assuming xss (contains script)
        'this is <script> alert() </script> and this is sql and this is hema', # Assuming xss
        "SELECT * FROM users WHERE name='hema'", # Assuming sql
        "SELECT * FROM users WHERE name='hema' and password='password'", # Assuming sql
        "UNION SELECT username, password FROM users WHERE '1'='1' --" # Assuming sql
    ]

    # Add optional ground truth labels for correctness check
    # Make sure the order matches example_sentences
    ground_truth_labels = [
        'safe',
        'safe', # Adjust if 'sql' is tokenized as a keyword and means sql injection
        'xss',
        'xss',
        'xss',
        'xss',
        'sql',
        'sql',
        'sql'
    ]

    print("--- Example Predictions ---")
    for i, sentence in enumerate(example_sentences):
        # Pass ground truth label if available
        gt_label = ground_truth_labels[i] if i < len(ground_truth_labels) else None
        # Pass the correct tokenizer (sql_tokenizer from libs)
        predict_sentence(model, sentence, stoi, int_to_label, sql_tokenizer, device, ground_truth_label=gt_label)

    # Example of predicting a single sentence interactively (optional)
    # while True:
    #     user_input = input("Enter a sentence to predict (or 'quit' to exit): ")
    #     if user_input.lower() == 'quit':
    #         break
    #     predict_sentence(model, user_input, stoi, int_to_label, sql_tokenizer, device)