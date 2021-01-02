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