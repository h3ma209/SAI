from flask import Flask
from flask import request
app = Flask(__name__)
import time
proc_start = time.time()
import spacy
import torch
from libs import *
import torch
import numpy as np
from Model import classifier

print("imported libs",time.time()-proc_start)


nlp = spacy.load('en')
torch.backends.cudnn.deterministic = True  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
TEXT = load_vocab('pickles\\vocab.pt')
print("loaded vocab",time.time()-proc_start)

size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 4
num_layers = 2
bidirection = True
dropout = 0.2


model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)
model.load_state_dict(torch.load('..\\saved_weights\\Acc 97.32.pt'))
#print(model)
#print(load)
#model.load_state_dict(load)
model.eval()
print("loaded model",time.time()-proc_start)
def predict(model,sentence,all = False):
    pred_2_lbl = {1:'xss',0:'sql',2:"safe"}
    tokenized = [tok.text for tok in nlp.tokenizer(sql_tokenizer(sentence))] 
    #print(tokenized)

    indexed = [TEXT.vocab.stoi[t] for t in tokenized] 
    #print(indexed)
    length = [len(indexed)] 
    tensor = torch.LongTensor(indexed).to(device) 
    tensor = tensor.unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    prediction = model(tensor,length_tensor)
    pred_lbl = np.argmax(prediction.detach().numpy())
    #print('\n')
    #print('predicted threat type:',pred_2_lbl[pred_lbl])
    if all:
        return prediction, pred_2_lbl[pred_lbl],tokenized, indexed
    else:
        return prediction,pred_2_lbl[pred_lbl]

@app.route('/flask', methods=['GET'])
def index():
    sql=request.args.get('username')
    return str(predict(model, sql,all=True))

if __name__ == "__main__":
    app.run(port=5000, debug=True)