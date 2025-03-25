import torch
import os
import numpy as np
import torch.nn.functional as F
from libs import *  
from Model import classifier

def initialize_device():
    torch.backends.cudnn.deterministic = True
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vocabularies(vocab_path='pickles/vocab.pkl', label_path='pickles/label.pkl'):
    text_vocab = load_vocab(vocab_path)
    label_vocab = load_vocab(label_path)
    return text_vocab, label_vocab

def load_latest_model(model, device,saved_weights_dir='saved_weights/' ):
    latest_file = max(
        (os.path.join(saved_weights_dir, f) for f in os.listdir(saved_weights_dir)),
        key=os.path.getmtime
    )
    model.load_state_dict(torch.load(latest_file, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {latest_file}")
    return model

def create_model(size_of_vocab, embedding_dim=100, num_hidden_nodes=32, num_output_nodes=4, num_layers=2, dropout=0.2):
    return classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, bidirectional=True, dropout=dropout)

def predict(model, sentence, text_vocab, device):
    pred_2_lbl = {2: 'xss', 1: 'sql', 0: "safe", 3: "label"}
    
    tokenized = [tok.text for tok in nlp.tokenizer(sql_tokenizer(sentence))]
    indexed = [text_vocab.vocab.stoi.get(t, text_vocab.vocab.stoi["<unk>"]) for t in tokenized]
    length = [len(indexed)]
    print(f"Tokenized: {tokenized}")
    print(f"Indexed: {indexed}")
    
    tensor = torch.LongTensor(indexed).to(device).unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    
    prediction = model(tensor, length_tensor)
    probs = F.softmax(prediction, dim=1).detach().cpu().numpy()[0]
    pred_lbl = np.argmax(probs)
    confidence = probs[pred_lbl] * 100
    probabilities = {pred_2_lbl[i]: f"{probs[i]*100:.2f}%" for i in range(len(probs))}
    
    print(f"Predicted Threat Type: {pred_2_lbl[pred_lbl]} ({confidence:.2f}%)")
    print("Probabilities:", 
          {pred_2_lbl[i]: f"{probs[i]*100:.2f}%" for i in range(len(probs))}
          )
    print("--------------------------\n")
    
    return prediction, pred_2_lbl[pred_lbl], confidence, probabilities

# Initialize setup
# device = initialize_device()
# TEXT, LABEL = load_vocabularies()
# model = create_model(len(TEXT.vocab))
# model = load_latest_model(model)

# Example predictions
# example_sentences = [
#     'this is hema',
#     'this is sql',
#     'this is <script>',
#     'this is <script> alert() </script>',
#     'this is <script> alert() </script> and this is sql',
#     'this is <script> alert() </script> and this is sql and this is hema',
#     "SELECT * FROM users WHERE name='hema'",
#     "SELECT * FROM users WHERE name='hema' and password='password'",
#     "UNION SELECT username, password FROM users WHERE '1'='1' --"
# ]

# for sentence in example_sentences:
#     predict(model, sentence, TEXT, device)
