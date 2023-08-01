import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
from .cluster import HDBSCAN, GMM

class MLP(nn.Module):
    '''
        Multilayer Perceptron.
    '''
    def __init__(self, input_size=2, layer_sizes=[100, 50, 10]):
        super().__init__()
        if len(layer_sizes) == 3:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, layer_sizes[0]),
                nn.ReLU(),
                nn.Linear(layer_sizes[0], layer_sizes[1]),
                nn.ReLU(),
                nn.Linear(layer_sizes[1], layer_sizes[2])
            )
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, layer_sizes[0]),
                nn.ReLU(),
                nn.Linear(layer_sizes[0], layer_sizes[1])
            )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
def train(data, labels, layers=[1000, 50, 10], epochs=25):
    mlp = MLP(input_size=data.shape[-1], layer_sizes=layers)
    train_data = list(zip(data, labels))
    trainloader = DataLoader(train_data, batch_size=256, shuffle=True)
  
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(epochs): # 5 epochs at maximum

        # Print epoch
        # if epoch % (epochs//10):
        #     print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs.float())

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()

            # if i % 500 == 499:
            #     print('Loss after mini-batch %5d: %.3f' %
            #         (i + 1, current_loss / 500))
            #     current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
    return mlp

def class_scores(train_reps, test_reps, num_classes, layers=[1000, 100, 10], epochs=100, mod_type='mlp', wf_ids=None, model_params={}):
    if wf_ids is None:
        wf_ids = list(range(num_classes))
    layers = [1000, 100, num_classes]
    labels_train = np.array([[i for j in range(1200)] for i in range(num_classes)]).reshape(-1)
    labels_test = np.array([[i for j in range(300)] for i in range(num_classes)]).reshape(-1)
    
    # knn = KNeighborsClassifier(n_neighbors = 10)
    # knn.fit(train_reps, labels_train)
    if mod_type == 'mlp':
        mlp = train(train_reps, labels_train, layers, epochs=epochs)
    elif mod_type == 'gmm':
        pred_labels = GMM(train_reps, test_reps)
    elif mod_type == 'hdbscan':
        pred_labels = HDBSCAN(test_reps)
    per_class_acc = {}
    for i in range(num_classes):
        # class_score = knn.score(test_reps[300*i:300*(i+1)], labels_test[300*i:300*(i+1)])*100
        if mod_type == 'mlp':
            with torch.no_grad():
                curr_labels = np.argmax(mlp(torch.from_numpy(test_reps[300*i:300*(i+1)]).float()).numpy(), axis=1)
            class_score = accuracy_score(labels_test[300*i:300*(i+1)], curr_labels) * 100
        elif mod_type == 'gmm' or mod_type == 'hdbscan':
            curr_labels = pred_labels[300*i:300*(i+1)] 
            class_score = adjusted_rand_score(labels_test[300*i:300*(i+1)], curr_labels) * 100
        per_class_acc['wf {}'.format(str(wf_ids[i]))] = class_score
    return per_class_acc
        
def avg_score(train_reps, test_reps, num_classes, labels_train, labels_test, layers=[1000, 100, 10], epochs=100, mod_type='mlp', ret_pred_labels=False, model_params={}):
    layers = [1000, 100, num_classes]
    # labels_train = np.array([[i for j in range(1200)] for i in range(num_classes)]).reshape(-1)
    # labels_test = np.array([[i for j in range(300)] for i in range(num_classes)]).reshape(-1)
    
    # knn = KNeighborsClassifier(n_neighbors = 10)
    # knn.fit(train_reps, labels_train)
    # acc['score'] = knn.score(test_reps, labels_test)*100
    
    if mod_type == 'mlp':
        mlp = train(train_reps, labels_train, layers, epochs=epochs)
        with torch.no_grad():
            pred_labels = np.argmax(mlp(torch.from_numpy(test_reps).float()).numpy(), axis=1)
        acc = {}
        acc['score'] = accuracy_score(pred_labels, labels_test)*100
    elif mod_type == 'gmm':
        # pred_labels = GMM(train_reps, test_reps, num_classes)
        if model_params['n_clusters'] is None:
            model_params['n_clusters'] = num_classes
        pred_labels, bic_scores_test, bic_scores_train = GMM(train_reps, test_reps, n_clusters=model_params['n_clusters'])
        acc = {}
        acc['score'] = adjusted_rand_score(labels_test, pred_labels)*100
        acc['bic_test'] = bic_scores_test
        acc['bic_train'] = bic_scores_train
    elif mod_type == 'hdbscan':
        pred_labels = HDBSCAN(test_reps)
        acc = {}
        acc['score'] = adjusted_rand_score(labels_test, pred_labels)*100
    if ret_pred_labels:
        return acc, pred_labels
    return acc

def per_class_accs(train_reps, test_reps, models, num_classes, labels_train, labels_test, mod_type='mlp', wf_ids=None, model_params={}):
    class_res = {}

    for i in range(len(train_reps)):
        class_res[models[i]] = class_scores(train_reps[i], test_reps[i], num_classes, labels_train, labels_test, mod_type=mod_type, wf_ids=wf_ids, model_params=model_params)
        
    return class_res

def avg_class_accs(train_reps, test_reps, models, num_classes, labels_train, labels_test, mod_type='mlp', model_params={}):
    class_res = {}

    for i in range(len(train_reps)):
        class_res[models[i]] = avg_score(train_reps[i], test_reps[i], num_classes, labels_train, labels_test,mod_type=mod_type, model_params=model_params)
        
    return class_res