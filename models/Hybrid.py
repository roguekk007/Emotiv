#Implements fully connected, untemporal
#Benchmark
import torch 
import torch.nn as nn 
from torch.optim import Adam, RMSprop
import torch.nn.functional as F 
from evaluation.evaluate import acc
import numpy as np 
import apex

#Defining a flatten module to convert conv output to fc input
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        bs = len(x)
        return x.view(bs, -1)

class Hybrid(nn.Module):
    def __init__(self, input_dim, output_dim, dev, fp16):
        super(Hybrid, self).__init__()
        #Hyperparameter definition
        GRU_hidden = 256
        conv_hidden = 256
        hidden = 512
        dropout = 0.5
        #RNN component definition
        self.r1 = nn.GRU(input_dim, GRU_hidden)
        self.r2 = nn.GRU(GRU_hidden, GRU_hidden, num_layers=3, dropout=dropout)

        #CNN component definition
        self.conv1 = nn.Conv2d(1, conv_hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_hidden)
        self.act1 = nn.ReLU()
        self.d1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(conv_hidden, conv_hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_hidden)
        self.act2 = nn.ReLU()
        self.d2 = nn.Dropout2d(dropout)
        self.conv3 = nn.Conv2d(conv_hidden, conv_hidden, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_hidden)
        self.act3 = nn.ReLU()
        self.d3 = nn.Dropout2d(dropout)
        self.conv4 = nn.Conv2d(conv_hidden, conv_hidden, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_hidden)
        self.act4 = nn.ReLU()
        self.d4 = nn.Dropout2d(dropout)
        self.conv5 = nn.Conv2d(conv_hidden, conv_hidden // 2, 3, padding=1)
        self.p5 = nn.MaxPool2d(2)
        self.bn5 = nn.BatchNorm2d(conv_hidden // 2)
        self.act5 = nn.ReLU()
        self.d5 = nn.Dropout2d(dropout)

        self.fc1 = nn.Linear(conv_hidden * 28 + GRU_hidden, hidden)
        self.bn6 = nn.BatchNorm1d(hidden)
        self.d6 = nn.Dropout(dropout)
        self.act6 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 5)
        self.bn7 = nn.BatchNorm1d(5)
        self.act = nn.Softmax(dim=1)

        self.train_acc = []
        self.val_acc = []
        self.epochs = [0]
        self.cri = F.cross_entropy
        self.op = RMSprop(self.parameters(), lr=0.0002)
        self = self.to(dev)
        self.fp16 = fp16
        if self.fp16:
            self = self.half()
            self.op = apex.fp16_utils.FP16_Optimizer(self.op, static_loss_scale=128.0, verbose=False)

    def forward(self, x):
        bs = len(x)
        #X input of bs, seq_len, input_dim
        rnn_x = x.permute(1, 0, 2)
        rnn_x, hidden = self.r1(rnn_x)
        rnn_x, hidden = self.r2(rnn_x)
        rnn_x = rnn_x[-1]
        rnn_x = rnn_x.view(bs, -1)

        cnn_x = x.unsqueeze(1)
        cnn_x = self.d1(self.act1(self.bn1(self.conv1(cnn_x))))
        cnn_x = self.d2(self.act2(self.bn2(self.conv2(cnn_x))))
        cnn_x = self.d3(self.act3(self.bn3(self.conv3(cnn_x))))
        cnn_x = self.d4(self.act4(self.bn4(self.conv4(cnn_x))))
        cnn_x = self.d5(self.act5(self.bn5(self.p5(self.conv5(cnn_x)))))
        cnn_x = cnn_x.view(bs, -1)

        x = torch.cat((rnn_x, cnn_x), dim=1)
        x = self.act6(self.d6(self.bn6(self.fc1(x))))
        x = self.act(self.bn7(self.fc2(x)))
        return x

    def num_layers(self):
        length = 0
        for _ in self.children():
            length += 1
        return length

    def layer_output(self, x, n):
        #Output feature before final linear layer irrespective of n
        self.eval()
        bs = len(x)
        rnn_x = x.permute(1, 0, 2)
        rnn_x, _ = self.r1(rnn_x)
        rnn_x, _ = self.r2(rnn_x)
        rnn_x = rnn_x[-1]
        rnn_x = rnn_x.view(bs, -1)

        cnn_x = x.unsqueeze(0)
        cnn_x = self.d1(self.act1(self.bn1(self.conv1(cnn_x))))
        cnn_x = self.d2(self.act2(self.bn2(self.conv2(cnn_x))))
        cnn_x = self.d3(self.act3(self.bn3(self.conv3(cnn_x))))
        cnn_x = self.d4(self.act4(self.bn4(self.conv4(cnn_x))))
        cnn_x = self.d5(self.act5(self.bn5(self.conv5(cnn_x))))
        cnn_x = cnn_x.view(bs, -1)

        x = torch.cat((rnn_x, cnn_x), dim=1)
        x = self.d6(self.bn6(self.fc1(x)))
        self.train()
        return x

    def train_on(self, trainloader, validloader, epochs = 30):
        self.train()
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
        path = './models/Hybrid'
        if self.fp16:
            path = path + '16.pth'
        else:
            path = path + '32.pth'
        torch.save((self.state_dict(), self.epochs, self.train_acc, self.val_acc), path)