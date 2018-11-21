#Implements temporal 
#Benchmark
import torch 
import torch.nn as nn 
from torch.optim import Adam, RMSprop
import torch.nn.functional as F 
from evaluation.evaluate import acc
import numpy as np 
import apex

#Temporal data as [batch, time, input_dim]
class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, dev, fp16):
        super(RNN, self).__init__()
        GRU_hidden = 512
        dropout = 0.5
        self.add_module('R1', nn.GRU(input_size=14, hidden_size=GRU_hidden))
        self.add_module('R2~4', nn.GRU(input_size=GRU_hidden, hidden_size=GRU_hidden, num_layers=3, dropout=dropout))
        self.add_module('R5', nn.GRU(input_size=GRU_hidden, hidden_size=output_dim))
        self.add_module('ac1', nn.Softmax(dim=1))
        self.train_acc = []
        self.val_acc = []
        self.epochs = [0]
        self.cri = F.cross_entropy
        self.op = RMSprop(self.parameters(), lr=0.0005)

        self.fp16 = fp16
        self = self.to(dev)
        if self.fp16:
            self = self.half()
            self.op = apex.fp16_utils.FP16_Optimizer(self.op, static_loss_scale=512.0, verbose=False)

    def forward(self, x):
        x = x.permute([1, 0, 2])
        for layer in self.children():
            if isinstance(layer, nn.GRU):
                x, hidden = layer(x)
            if isinstance(layer, nn.Softmax):
                x = layer(x[-1])
        return x

    def num_layers(self):
        length = 0
        for children in self.children():
            length += 1
        return length

    def layer_output(self, x, n):
        #Make replicable
        #call num_layers-1 for final output
        #Call num_layers-1-.. for features
        self.eval()
        for i, layer in enumerate(self.children()):
            if isinstance(layer, nn.GRU):
                x, hidden = layer(x)
            else:
                x = layer(x[-1])
            if i == n:
                break 
        self.train()
        return x

    def train_on(self, trainloader, validloader, epochs = 30):
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
        path = './models/RNN'
        if self.fp16:
            path = path + '16.pth'
        else:
            path = path + '32.pth'
        torch.save((self.state_dict(), self.epochs, self.train_acc, self.val_acc), path)
