import torch
import numpy as np 
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale

#Returns X, Y
def raw_data(shuffle):
    data = np.array(loadmat('./data/emotiv.mat')['emotiv'])
    #Shuffle if data is not going to be temporally correlated
    if shuffle:
        np.random.shuffle(data)
    TX = scale(data[:, :14])
    TY = data[:, 14:].reshape(-1)
    X = []
    Y = []
    #print(data.shape)
    #[1, 2, 3, 4, 6] -> [0, 1, 2, 3, 4]
    for i, x in enumerate(TX):
        TY[i] = int(TY[i])
        if (TY[i] >= 1 and TY[i] <= 4) or TY[i] == 6:
            X.append(x)
            if TY[i] == 6:
                Y.append(4)
            else:
                Y.append(TY[i] - 1)
    X = np.array(X)
    Y = np.array(Y)
    print('Shape of data:', X.shape, Y.shape)
    return X, Y

#Returns (x*t, y): time distributed
def time_data(stride=32):
    X, Y = raw_data(shuffle=False)
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    train_wrapped = []
    val_wrapped = []
    trainsition_p = 0.1

    left = 0
    while left < len(X) - stride:
        right = left + stride
        if np.random.random() < trainsition_p:
            val_wrapped.append((X[left : right], Y[right]))
            left = right
        else:
            train_wrapped.append((X[left : right], Y[right]))
            left += 1
    #Implement shuffling for time data
    np.random.shuffle(val_wrapped)
    np.random.shuffle(train_wrapped)

    for (x, y) in train_wrapped:
        train_X.append(x)
        train_Y.append(y)
    for (x, y) in val_wrapped:
        val_X.append(x)
        val_Y.append(y)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)

    print('Shape of time data:', train_X.shape, train_Y.shape, val_X.shape, val_Y.shape)
    return train_X, train_Y, val_X, val_Y

class EmotivData(Dataset):
    def __init__(self, X, Y, dev, fp16):
        super(EmotivData, self).__init__()
        self.X = X 
        self.Y = Y
        self.X = torch.tensor(self.X, dtype=torch.float).to(dev)
        self.Y = torch.tensor(self.Y, dtype=torch.long).to(dev)
        if fp16:
            self.X = self.X.half()
        self.length = len(self.X)
    
    def __len__(self):
        return self.length 
    
    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

#Returns dataloaders
def data(is_temporal, stride=16, bs=1024, dev='cuda', fp16=True):
    if is_temporal:
        X_train, Y_train, X_val, Y_val = time_data(stride=16)
        trainloader = DataLoader(EmotivData(X_train, Y_train, dev=dev, fp16=fp16), batch_size=bs, shuffle=True)
        valloader = DataLoader(EmotivData(X_val, Y_val, dev=dev, fp16=fp16), batch_size=bs, shuffle=True)
    else:
        X, Y = raw_data(True)
        pivot = int(len(X) * 0.2)
        trainloader = DataLoader(EmotivData(X[pivot:], Y[pivot:], dev=dev, fp16=fp16), batch_size=bs, shuffle=True)
        valloader = DataLoader(EmotivData(X[:pivot], Y[:pivot], dev=dev, fp16=fp16), batch_size=bs, shuffle=True)
    return trainloader, valloader