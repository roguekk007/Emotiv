import torch 
import matplotlib.pyplot as plt 
import numpy as np 

#Visualizes performance based on saved models
#Visualize juxtaposed validation acc
def juxtapose_performance(model_list):
    fig, ax = plt.subplots(figsize=(7, 7/3*2))
    for name, path in model_list:
        state_dict, epochs, train_acc, val_acc = torch.load('./models/' + path) 
        epochs = epochs[1:]
        val_acc = val_acc = ax.plot(epochs, val_acc, label = name)
    plt.title("Models' performance")
    plt.legend()
    plt.show()

def visualize_performance(model_list):
    for name, path in model_list:
        fig, ax = plt.subplots(figsize=(7, 7/3*2))
        path = './models/' + path
        state_dict, epochs, train_acc, val_acc = torch.load(path)
        epochs = epochs[1:]
        train_acc_curve = ax.plot(epochs, train_acc, label='train acc')
        val_acc_curve = ax.plot(epochs, val_acc, label='val acc')
        point = np.argmax(np.array(val_acc))
        max_val_acc = np.max(np.array(val_acc))
        ax.annotate('Max val acc:'+str(max_val_acc)[:6], xy=(point, max_val_acc), xytext=(point - 5, max_val_acc - 0.1),
                    arrowprops=dict(facecolor='black', shrink=0),
                    horizontalalignment='right', verticalalignment='top', fontsize=14)
        plt.title(name + ' performance')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()

model_list = [
    ('NN', 'NN32.pth'),
    ('CNN', 'CNN32.pth'),
    ('RNN', 'RNN32.pth'),
    ('Hybrid', 'Hybrid32.pth')
]
juxtapose_performance(model_list)
visualize_performance(model_list)