import os
import pandas
import glob
import pickle

f = open('keywords.txt','r')
keys = list(map(lambda x:x.replace("\n",''),f.readlines()))
f.close()
f = open('replace.txt','r')

dicts = {}
replace = list(map(lambda x:x.replace("\n",''),f.readlines()))

for line in replace:
    x,y = line.split("==")[0].replace(" ",''),line.split("==")[1].replace(" ",'')
    dicts[x] = y
f.close()

all_vocabs = list()


def replace_symbol(word):
    wordlist = []
    
    try:
        type(int(word))
        wordlist.append("INT")
    except:
        word = list(word)
        for i,l in enumerate(word):
            try:
                replacement = dicts[l]
            except:
                replacement = None

            if replacement != None:
                l = " "+replacement+' '
            wordlist.append(l)
        
    return " ".join("".join(wordlist).strip().split("  "))


def sql_tokenizer(query):
    query = query.lower()
    bb = list(map(replace_symbol,query.split(" ")))
    bb = ' '.join(bb)
    tokenized = " ".join(list(map(replace_symbol,bb.split(" "))))
    split = tokenized.split(" ")
    for i,word in enumerate(tokenized.split(" ")):
        if word.upper() in keys:
            split[i] = word.upper()
    
    return " ".join(split)

def open_file(filename,label,safe=False,limit=0):

	


    f = open(filename,'r')
    if limit == 0:
    	txt = f.readlines()
    else:
    	txt = f.readlines()[:limit]
    f.close()
#    txt = list(map(lambda x:x.replace("\n",' ') if( len(x) > 0) else None,txt))
    text = []
    for sent in txt:
        if len(sent) > 0:
            sent_list = []
            sent = sent.replace("\n",'').split(" ")
            for word in sent:
                if len(word)> 0:
                    sent_list.append(word)
            if safe:
            	text.append(" ".join(sent_list))
            else:

            	text.append(sql_tokenizer(" ".join(sent_list)))
            

    dict = {'text':text,'label':[label for i in range(len(text))]}
    data = pandas.DataFrame(dict)
    all_vocabs.append([sent for sent in data['text']])
    return data

def custom_tokenizer(sentence):
    with open('tokenizer.p', 'rb') as fp:
        data = pickle.load(fp)
    sent = sentence.strip().split(" ")
    tokenized = []
    for word in sent:
        if len(word) > 0:
            try:
                index = data[word]
                tokenized.append(index)

            except:
                pass
    return tokenized