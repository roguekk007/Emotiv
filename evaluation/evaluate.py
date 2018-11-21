import torch 
import numpy as np 

#Returns the model's accuracy on dataloader
def acc(model, loader):
    #Set in evaluation mode to alter behavior of certain modules
    model.eval()
    instance = 0
    correct = 0
    for (batch_x, batch_y) in loader:
        batch_y_pred = model(batch_x).detach().cpu().numpy()
        for i, y_pred in enumerate(batch_y_pred):
            y = batch_y[i]
            if np.argmax(y_pred) == y:
                correct += 1
            instance += 1
    model.train()
    return correct / instance 