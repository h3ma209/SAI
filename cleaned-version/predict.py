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