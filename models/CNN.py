#Temporal CNN
import torch 
import torch.nn as nn 
from torch.optim import Adam
import torch.nn.functional as F 
from evaluation.evaluate import acc
import numpy as np 
import apex

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        bs = len(x)
        return x.view(bs, -1)

class CNN(nn.Module):
    def __init__(self, dev, output_dim, fp16):
        super(CNN, self).__init__()
        hidden = 1024
        dropout = 0.5
        hidden_conv = 256

        self.add_module('conv0', nn.Conv2d(1, hidden_conv, 3, padding=1))
        self.add_module('bn0', nn.BatchNorm2d(hidden_conv))
        self.add_module('act0', nn.ReLU())
        self.add_module('d0', nn.Dropout2d(dropout))
        for _ in range(3):
            self.add_module('conv'+str(_+1), nn.Conv2d(hidden_conv, hidden_conv, 3, padding=1))
            self.add_module('bn'+str(_+1), nn.BatchNorm2d(hidden_conv))
            self.add_module('act'+str(_+1), nn.ReLU())
            self.add_module('d'+str(_+1), nn.Dropout2d(dropout))
        self.add_module('conv5', nn.Conv2d(hidden_conv, hidden_conv // 2, 3, padding=1))
        self.add_module('bn5', nn.BatchNorm2d(hidden_conv // 2))
        self.add_module('act5', nn.ReLU())
        self.add_module('d5', nn.Dropout2d(dropout))
        self.add_module('p5', nn.MaxPool2d(2))

        self.add_module('Flatten', Flatten())

        self.add_module('fc6', nn.Linear(7168, hidden))
        self.add_module('bn6', nn.BatchNorm1d(hidden))
        self.add_module('act6', nn.ReLU())
        self.add_module('d7', nn.Dropout(dropout))
        self.add_module('fc8', nn.Linear(hidden, output_dim))
        self.add_module('act6', nn.Softmax(dim=1))
        self.train_acc = []
        self.val_acc = []
        self.epochs = [0]
        self.cri = F.cross_entropy
        self.op = Adam(self.parameters(), lr=0.002)
        self = self.to(dev)
        self.fp16 = fp16
        if self.fp16:
            self = self.half()
            self.op = apex.fp16_utils.FP16_Optimizer(self.op, static_loss_scale=128.0, verbose=False)
        
    def forward(self, x):
        #Add a channel dimension
        x = x.unsqueeze(1)
        #print(x.size())
        for layer in self.children():
            x = layer(x)
        return x

    def num_layers(self):
        length = 0
        for children in self.children():
            length += 1
        return length

    def layer_output(self, x, n):
        #Do make it replicable
        #call num_layers-1 for final output
        #Call num_layers-1-.. for features
        self.eval()
        for i, layer in enumerate(self.children()):
            x = layer(x)
            if i == n:
                break 
        self.train()
        return x

    def train_on(self, trainloader, validloader, epochs):
        for epoch in range(self.epochs[-1] + 1, self.epochs[-1] + epochs):
            for (x, y) in trainloader:
                y_pred = self.forward(x)
                loss = self.cri(y_pred, y)
                if self.fp16:
                    self.op.backward(loss)
                else:
                    loss.backward()
                self.op.step()
                self.op.zero_grad()

            if epoch % 3 == 0:
                train_acc = acc(self, trainloader)
                val_acc = acc(self, validloader)
                self.train_acc.append(train_acc)
                self.val_acc.append(val_acc)
                print('Epoch', epoch, 'acc:', train_acc, 'val_acc:', val_acc)
                self.epochs.append(epoch)
            if epoch % 10 == 0:
                self.save()
        self.save()
    
    def save(self):
        path = './models/CNN'
        if self.fp16:
            path = path + '16.pth'
        else:
            path = path + '32.pth'
        torch.save((self.state_dict(), self.epochs, self.train_acc, self.val_acc), path)
