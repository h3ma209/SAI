import os
import pandas
import pickle
import torch
from torchtext import vocab
from transformers import BertTokenizer

# Load keywords
with open('keys/keywords.txt', 'r') as f:
    keys = [line.strip() for line in f]

# Load replacements
replacements = {}
with open('keys/replace.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split("==")
        replacements[key.strip()] = value.strip()

all_vocabs = []

def replace_symbol(word):
    try:
        int(word)
        return "INT"
    except ValueError:
        word_list = []
        for char in word:
            replacement = replacements.get(char)
            word_list.append(f" {replacement} " if replacement else char)
        return "".join(word_list).strip().replace("  ", " ")

def sql_tokenizer(query):
    query = query.lower()
    processed_query = ' '.join(map(replace_symbol, query.split()))
    tokenized = ' '.join(map(replace_symbol, processed_query.split()))
    split = tokenized.split()
    for i, word in enumerate(split):
        if word.upper() in keys:
            split[i] = word.upper()
    return ' '.join(split)

def open_file(filename, label, safe=False, limit=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if limit > 0:
            lines = lines[:limit]

    text = []
    for line in lines:
        line = line.strip()
        if line:
            words = [word for word in line.split() if word]
            if safe:
                text.append(' '.join(words))
            else:
                text.append(sql_tokenizer(' '.join(words)))

    data = pandas.DataFrame({'text': text, 'label': [label] * len(text)})
    all_vocabs.extend(data['text'].tolist())
    return data

def custom_tokenizer(sentence):
    with open('tokenizer.p', 'rb') as fp:
        data = pickle.load(fp)
    
    tokenized = []
    for word in sentence.strip().split():
        if word:
            try:
                index = data[word]
                tokenized.append(index)
            except KeyError:
                pass
    return tokenized

def load_vocab(path):
    with open(path, 'rb') as f:
        vocab_obj = pickle.load(f)
    return vocab_obj

# Add bert_tokenizer to libs.py
BERT_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

def bert_tokenizer(text,debug=False):
    sql = sql_tokenizer(text)
    if debug:
        print(f"tok: {sql}")

    return tokenizer.tokenize(sql)