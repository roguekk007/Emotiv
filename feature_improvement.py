from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from data.data import data
import torch 
import numpy as np

np.random.seed(7)
torch.manual_seed(7)

#Choosing mode='paper_exact' to use originally saved CNN as feature extractor
#mode = 'paper_exact'
mode = 'replicate'

#Generate & save features from the selected model
if mode == 'paper_exact':
    model = torch.load('./models/cnn_temporal_saved_models_in_paper.pth').to('cpu')
else:
    model = torch.load('./models/cnn_temporal.pth').to('cpu')
trainloader, validloader = data(is_temporal = True, bs=100, dev='cpu')
N = model.num_layers()
for i, (x, y) in enumerate(trainloader):
    print(i, '/', len(trainloader) - 1)
    features = model.layer_output(x.unsqueeze(1), N-4)
    torch.save((features.detach().numpy(), y.detach().numpy().reshape(-1, 1)), './saved_features/train'+str(i)+'.pth')
for i, (x, y) in enumerate(validloader):
    print(i, '/', len(validloader) - 1)
    features = model.layer_output(x.unsqueeze(1), N-4)
    torch.save((features.detach().numpy(), y.detach().numpy().reshape(-1, 1)), './saved_features/val'+str(i)+'.pth')

#Load saved feature data, uses different local data arrays because
#np.concatenate is slow when concatenated array size is big
def read_featured_data():
    #Utilizing features
    train_x = np.zeros([0, 512])
    train_x1 = np.zeros([0, 512])
    train_x2 = np.zeros([0, 512])
    train_x3 = np.zeros([0, 512])
    train_x4 = np.zeros([0, 512])
    train_x5 = np.zeros([0, 512])
    train_x6 = np.zeros([0, 512])
    train_y = np.zeros([0, 1])
    val_x = np.zeros([0, 512])
    val_y = np.zeros([0, 1])
    len_train = 719
    p1 = len_train // 6
    p2 = len_train * 2 // 6
    p3 = len_train * 3 // 6
    p4 = len_train * 4 // 6
    p5 = len_train * 5 // 6
    len_val = 80
    for i in range(len_train):
        #print(i, '/', len_train)
        (features, y) = torch.load('./saved_features/train'+str(i)+'.pth')
        if i >= p5:
            train_x6 = np.concatenate((train_x6, features), axis=0)
        elif i >= p4:
            train_x5 = np.concatenate((train_x5, features), axis=0)
        elif i >= p3:
            train_x4 = np.concatenate((train_x4, features), axis=0)
        elif i >= p2:
            train_x3 = np.concatenate((train_x3, features), axis=0)
        elif i >= p1:
            train_x2 = np.concatenate((train_x2, features), axis=0)
        else:
            train_x1 = np.concatenate((train_x1, features), axis=0)
        train_y = np.concatenate((train_y, y), axis=0)
    train_x = np.concatenate((train_x1, train_x2, train_x3, train_x4, train_x5, train_x6), axis=0)

    for i in range(len_val):
        #print(i, '/', len_val)
        (features, y) = torch.load('./saved_features/val'+str(i)+'.pth')
        val_x = np.concatenate((val_x, features), axis=0)
        val_y = np.concatenate((val_y, y), axis=0)
    train_x = scale(train_x)
    val_x = scale(val_x)
    return train_x, train_y, val_x, val_y

#Load original data
def original_data():
    train_x = np.zeros([0, 14 * 16])
    train_y = np.zeros([0, 1])
    val_x = np.zeros([0, 14 * 16])
    val_y = np.zeros([0, 1])
    trainloader, validloader = data(is_temporal=True, bs=7000, dev='cpu')
    for (x, y) in trainloader:
        train_x = np.concatenate((train_x, x.numpy().reshape(-1, 14 * 16)), axis=0)
        train_y = np.concatenate((train_y, y.numpy().reshape(-1, 1)), axis=0)
    for (x, y) in validloader:
        val_x = np.concatenate((val_x, x.numpy().reshape(-1, 14 * 16)), axis=0)
        val_y = np.concatenate((val_y, y.numpy().reshape(-1, 1)), axis=0)
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
    return train_x, train_y, val_x, val_y

#Evaluate and output performance in model list
def improvements(model_list, orig_x, orig_val_x, feature_x, feature_val_x, orig_y, orig_val_y, train_y, val_y):
    for (name, model) in model_list:     
        clf = model 
        clf.fit(orig_x, orig_y.ravel())
        orig_acc = accuracy_score(orig_y, clf.predict(orig_x))
        orig_val_acc = accuracy_score(orig_val_y, clf.predict(orig_val_x))
        
        clf = model 
        clf.fit(feature_x, train_y.ravel())
        feature_acc = accuracy_score(train_y, clf.predict(feature_x))
        feature_val_acc = accuracy_score(val_y, clf.predict(feature_val_x))
        print(name, ':', orig_acc, orig_val_acc, feature_acc, feature_val_acc)

orig_x, orig_y, orig_val_x, orig_val_y = original_data()
feature_x, train_y, feature_val_x, val_y = read_featured_data()
print('Data loaded')

#Candidate models
models = [
    ('GaussianNB' , GaussianNB()),
    ('LR', LogisticRegression()),
    ('Adaboost', AdaBoostClassifier(n_estimators=200)),
    ('GBT', GradientBoostingClassifier(n_estimators=200)),
    ('RF', RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ('KNN', KNeighborsClassifier(n_neighbors=5, n_jobs=-1, algorithm='kd_tree')),
    ('XGB', XGBClassifier(learning_rate=0.3, n_estimators=1000, n_jobs=-1)),
]

#Output improvement
improvements(models, orig_x, orig_val_x, feature_x, feature_val_x, orig_y, orig_val_y, train_y, val_y)
