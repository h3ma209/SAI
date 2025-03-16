import time
import torch
from libs import * # Import all from libs.py
import numpy as np
from Model import classifier

proc_start = time.time()
print("imported libs", time.time() - proc_start)

torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = load_vocab('pickles/vocab.pt')
print("loaded vocab", time.time() - proc_start)

size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 4
num_layers = 2
bidirection = True
dropout = 0.2

model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                   bidirectional=True, dropout=dropout)
model.load_state_dict(torch.load('saved_weights/Acc 89.06.pt', map_location=device))
model.eval()
print("loaded model", time.time() - proc_start)

def predict(model, sentence, all=False):
    pred_2_lbl = {1: 'xss', 0: 'sql', 2: "safe", 3: "other"}
    tokenized = bert_tokenizer(sentence)  # Use bert_tokenizer from libs.py
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device).unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    prediction = model(tensor, length_tensor)
    pred_lbl = np.argmax(prediction.detach().cpu().numpy())
    if all:
        return prediction, pred_2_lbl[pred_lbl], tokenized, indexed
    else:
        return prediction, pred_2_lbl[pred_lbl]

print(predict(model, 'this is hema'), time.time() - proc_start)