#!/usr/bin/env python
# coding: utf-8

import spacy
import torch
from torchtext import data    
from libs import *
import numpy as np
import pandas as pd
import torch.optim as optim
import time
import os
import pickle
import random
import torch.nn as nn
from Model import *

nlp = spacy.load('en')


SEED = 2019


torch.manual_seed(SEED)


torch.backends.cudnn.deterministic = True  


TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)


fields = [(None, None), ('text',TEXT),('label', LABEL)]


training_data=data.TabularDataset(path = '../csv_files/xss_safe_sql.csv',format = 'csv',fields = fields,skip_header = True)

print(vars(training_data.examples[0]))
for i,dt in enumerate(training_data):
    if(len(dt.text) <= 0):
        training_data[i].text = "<blank>"
        #print(i,training_data[i].text,training_data[i].label)


train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
#initialize glove embeddings
# uncomment to use pretrained glove and comment the other one
#TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  

TEXT.build_vocab(train_data,min_freq=3)
TEXT.build_vocab(train_data,min_freq=3)  

LABEL.build_vocab(train_data)
print("Size of TEXT vocabulary:",len(TEXT.vocab))
print("Size of LABEL vocabulary:",len(LABEL.vocab))


save_vocab(TEXT,"pickles/vocab.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


BATCH_SIZE = 64

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)



size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = len(LABEL.vocab.stoi)
num_layers = 2
bidirection = True
dropout = 0.2


model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)



print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')


# if you want to use glove and copy weights
#pretrained_embeddings = TEXT.vocab.vectors
#model.embedding.weight.data.copy_(pretrained_embeddings)

#print(pretrained_embeddings.shape)



optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


def binary_accuracy(preds, y):

    rounded_pred = torch.round(preds)
    _,pred_label = torch.max(rounded_pred, dim = 1)
    correct = (pred_label == y).float()
    acc = correct.sum() / len(correct)
    return acc

model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    

    epoch_loss = 0
    epoch_acc = 0
    

    model.train()  
    
    for batch in iterator:

        optimizer.zero_grad()   
        

        text, text_lengths = batch.text   
        

        predictions = model(text, text_lengths).squeeze()  
        

        y_tensor = torch.tensor(batch.label, dtype=torch.long, device=device)
        loss = criterion(predictions, y_tensor)        
        

        acc = binary_accuracy(predictions, batch.label)   
        

        loss.backward()       
        

        optimizer.step()      
        

        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    

    epoch_loss = 0
    epoch_acc = 0


    model.eval()
    

    with torch.no_grad():
    
        for batch in iterator:
        

            text, text_lengths = batch.text
            

            predictions = model(text, text_lengths).squeeze()
            

            y_tensor = torch.tensor(batch.label, dtype=torch.long, device=device)
            loss = criterion(predictions, y_tensor)      
            acc = binary_accuracy(predictions, batch.label)
            

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 7
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
     

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')





def save_model():
    models_path = "../saved_weights"
    md_val_acc = "%.2f"%(valid_acc*100)
    model_name = "Acc "+md_val_acc+".pt"
    torch.save(model.state_dict(), os.path.join("../saved_models",model_name))
    full_path = os.path.join(models_path, model_name)
    torch.save(model.state_dict(),full_path)
    print("SAVED\n",model_name)


save_model()




def predict(model,sentence):
    pred_2_lbl = {1:'xss',0:'sql',2:"safe"}
    tokenized = [tok.text for tok in nlp.tokenizer(sql_tokenizer(sentence))] 
    print(tokenized)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized] 
    print(indexed)
    length = [len(indexed)] 
    tensor = torch.LongTensor(indexed).to(device) 
    tensor = tensor.unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    prediction = model(tensor,length_tensor)
    pred_lbl = np.argmax(prediction.detach().numpy())
    print('\n')
    print('predicted threat type:',pred_2_lbl[pred_lbl])
    return prediction

# Examples


#pred = predict(model,""" SELECT * FROM items
#WHERE owner = 'wiley'
#AND itemname = 'name' OR 'a'='a'; """)


#predict(model,'im good')
predict(model,'group_concat(namapemohon,0x3a,email),3,4,5,6 from pendaftaran_user ')
