# libs/libs.py

import os
import pandas
import pickle

# Define what gets imported with 'from libs import *'
__all__ = [
    'sql_tokenizer',
    'load_data_from_csv',
    'build_vocabulary',
    'replace_symbol',
    'open_file',
    'numericalize_text',
    'load_vocab', # Include if you plan to load previously saved vocab
    # Add other functions/variables you want to be accessible via '*'
]


# Define paths relative to the location where the script using libs is run
# It's generally better to pass these paths or make them configurable
# For this example, assuming keys/ is in the parent directory of libs/
try:
    # Use utf-8 encoding
    with open('keys/keywords.txt', 'r', encoding='utf-8') as f:
        keys = {line.strip().upper() for line in f if line.strip()}
except FileNotFoundError:
    print("Error: keys/keywords.txt not found. Please ensure it exists.")
    keys = set() # Initialize as empty set to avoid errors
except Exception as e:
    print(f"Error reading keys/keywords.txt: {e}")
    keys = set()


# Load replacements
replacements = {}
try:
    # Use utf-8 encoding
    with open('keys/replace.txt', 'r', encoding='utf-8') as f:
        # CORRECTED LINE: Iterate over 'f' instead of 'line'
        for line in f:
            line = line.strip()
            if line and "==>" in line:
                parts = line.split("==>")
                if len(parts) == 2:
                    key, value = parts
                    replacements[key.strip()] = value.strip()
except FileNotFoundError:
    print("Error: keys/replace.txt not found. Please ensure it exists.")
    replacements = {} # Initialize as empty dict
except Exception as e:
    print(f"Error reading keys/replace.txt: {e}")
    replacements = {}


def replace_symbol(word):
    """
    Replaces integers with 'INT' and specific symbols with their token representations.
    """
    try:
        # This part replaces integers with 'INT'
        int(word)
        return "INT"
    except ValueError:
        # This part replaces individual symbols based on the replacements dictionary
        word_list = []
        for char in word:
            # Check if the character is in replacements, prioritize exact character match
            replacement = replacements.get(char)
            if replacement:
                 # Add spaces around the replacement token
                 word_list.append(f" {replacement} ")
            else:
                word_list.append(char) # Keep character if no specific replacement

        # Join characters, clean up extra spaces, and return
        return "".join(word_list).strip().replace("  ", " ")


def sql_tokenizer(query):
    """
    Custom tokenizer for SQL/XSS queries without spaCy.
    Processes the query, replaces symbols/integers, identifies keywords,
    and defaults other words to 'EN_WORD'.
    Returns a list of tokens.
    """
    if not isinstance(query, str):
        # Handle non-string input gracefully
        return []

    query = query.lower()
    

    # Apply symbol and integer replacement first
    processed_query_with_symbols = ' '.join(map(replace_symbol, query.split()))

    # Simple split by whitespace after symbol replacement
    split_words = [word.strip() for word in processed_query_with_symbols.split() if word.strip()]

    processed_tokens = []
    for word in split_words:
        upper = word.upper()
        if upper in keys:
            processed_tokens.append(upper)
        elif upper in replacements.values():
            processed_tokens.append(upper)  # Keep symbolic tokens like LT, GT, STAR
        # Add a check for 'INT' which is introduced by replace_symbol
        elif upper == "INT":
            processed_tokens.append("INT")
        elif upper == "EN_WORD":
             processed_tokens.append("EN_WORD") # Keep 'EN_WORD' if it somehow appears
        else:
            processed_tokens.append("EN_WORD") # Default to EN_WORD for anything else

    # Return the list of processed tokens
    return [token for token in processed_tokens if token] # Final filter for any empty tokens


def load_data_from_csv(filename, text_column, label_column):
    """
    Loads text and label data from a CSV file using pandas.
    Assumes the CSV has a header row.
    """
    try:
        df = pandas.read_csv(filename, encoding='utf-8')
        # Ensure the required columns exist
        if text_column not in df.columns or label_column not in df.columns:
            print(f"Error: CSV file '{filename}' must contain '{text_column}' and '{label_column}' columns.")
            return None, None
        # Return lists of text and labels
        return df[text_column].tolist(), df[label_column].tolist()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None


def build_vocabulary(texts, tokenizer, min_freq=1):
    """
    Builds a vocabulary (token to index and index to token) from a list of text data.
    Includes special tokens <unk> and <pad>.
    """
    token_counts = {}
    for text in texts:
        tokens = tokenizer(text)
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

    # Add special tokens
    # Ensure <unk> is index 0 and <pad> is index 1 or vice-versa
    # Common practice is <unk>: 0, <pad>: 1
    stoi = {'<unk>': 0, '<pad>': 1}
    itos = ['<unk>', '<pad>']

    # Add tokens with frequency >= min_freq
    for token, count in sorted(token_counts.items(), key=lambda item: item[1], reverse=True):
        if count >= min_freq and token not in stoi:
            stoi[token] = len(stoi)
            itos.append(token)

    print(f"Built vocabulary of size: {len(stoi)}")
    return stoi, itos


def numericalize_text(tokens, stoi):
    """
    Converts a list of tokens into a list of indices using the stoi mapping.
    Unknown tokens are mapped to the <unk> index.
    """
    # Get the index for the unknown token, default to 0 if <unk> somehow not in stoi
    unk_idx = stoi.get('<unk>', 0)
    return [stoi.get(token, unk_idx) for token in tokens]


def load_vocab(path):
    """
    Loads a pickled object (e.g., vocabulary mapping) from a file.
    """
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object loaded from {path}")
        return obj
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None # Return None if file is not found
    except Exception as e:
        print(f"Error loading object from {path}: {e}")
        return None
def open_file(filename):
    """
    Opens a file and returns its content.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None