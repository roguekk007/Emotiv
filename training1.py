import torch
import numpy as np

from data.data import data
from models.NN import NN
from models.CNN import CNN
from models.RNN import RNN
from models.Hybrid import Hybrid
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Magic lines!
def replicable():
    np.random.seed(1)
    torch.manual_seed(1)

def runexperiment(fp16, epochs):
    dev = 'cuda'
    replicable()
    spatial_trainloader, spatial_validloader = data(is_temporal=False, bs=512, fp16=fp16)

    replicable()
    trainloader, validloader = data(is_temporal=True, bs=512, fp16=fp16)

    print('Training CNN')
    replicable()
    model = CNN(dev=dev, output_dim=5, fp16=fp16)
    model.train_on(trainloader, validloader, epochs=epochs)
    del model

    print('Training NN')
    replicable()
    model = NN(14, 5, fp16=fp16, dev=dev)
    model.train_on(spatial_trainloader, spatial_validloader, epochs=epochs)
    del model

runexperiment(fp16=True, epochs=150)
runexperiment(fp16=False, epochs=150)