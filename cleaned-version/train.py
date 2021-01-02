import torch
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
