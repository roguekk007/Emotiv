#Implements fully connected, untemporal
#Benchmark
import torch 
import torch.nn as nn 
from torch.optim import Adam
import torch.nn.functional as F 
from evaluation.evaluate import acc
import numpy as np 
import apex

class NN(nn.Module):
    def __init__(self, input_dim, output_dim, fp16, dev):
        super(NN, self).__init__()
        hidden = 512
        dropout = 0.5
        self.add_module('fc0', nn.Linear(input_dim, hidden))
        self.add_module('bn0', nn.BatchNorm1d(hidden))
        self.add_module('ac0', nn.ReLU())
        self.add_module('d0', nn.Dropout(dropout))
        for _ in range(5):
            self.add_module('fc'+str(_+1), nn.Linear(hidden, hidden))
            self.add_module('bn'+str(_+1), nn.BatchNorm1d(hidden))
            self.add_module('ac'+str(_+1), nn.ReLU())
            self.add_module('d'+str(_+1), nn.Dropout(dropout))
        self.add_module('fc'+str(6), nn.Linear(hidden, output_dim))
        self.add_module('ac'+str(6), nn.Softmax(dim=1))
        self.train_acc = []
        self.val_acc = []
        self.epochs = [0]
        self.cri = F.cross_entropy
        self.op = Adam(self.parameters(), lr=0.001)
        self.fp16 = fp16
        self = self.to(dev)
        if self.fp16:
            self = self.half()
            self.op = apex.fp16_utils.FP16_Optimizer(self.op, static_loss_scale=512.0, verbose=False)
    
    def forward(self, x):
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
                self.epochs.append(epoch)
                print('Epoch', epoch, 'acc:', train_acc, 'val_acc:', val_acc)
            if epoch % 10 == 0:
                self.save()
        self.save()
    
    def save(self):
        path = './models/NN'
        if self.fp16:
            path = path + '16.pth'
        else:
            path = path + '32.pth'
        torch.save((self.state_dict(), self.epochs, self.train_acc, self.val_acc), path)