import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train= pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())


X_train=np.array(train_data, dtype=np.float32) / 255.0 
y_train = np.array(train_labels) 

X_test = np.array(test_data, dtype=np.float32) / 255.0 


m = X_train.shape[0]          
indices = np.arange(m)         
np.random.seed(42)              
np.random.shuffle(indices)     


val_size = int(0.1 * m)
train_indices = indices[:-val_size]
val_indices = indices[-val_size:]

X_val = X_train[val_indices]
y_val = y_train[val_indices]

X_train = X_train[train_indices]
y_train = y_train[train_indices]

print(X_train)
print(y_train)


X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test)


input_dim = 784
hidden_dim = 100
output_dim = 10

def he_init(in_dim, out_dim):
    return torch.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)


torch.manual_seed(42)

W1 = he_init(input_dim, hidden_dim)
W2 = he_init(hidden_dim, output_dim)

b1 = torch.full((hidden_dim,), 0.01)
b2 = torch.zeros(output_dim)

epochs = 35
batch_size = 128
lr = 0.3
lambda_l2 = 5e-6 


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

def relu_grad(x):
    return (x > 0).float()

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()


num_batches = int(np.ceil(X_train.shape[0] / batch_size))

mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

for epoch in range(epochs):
    if epoch == 15:
     lr *= 0.5
    perm = torch.randperm(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    train_loss = 0.0
    train_acc = 0.0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        m = X_batch.shape[0]

        Z1 = X_batch @ W1 + b1         
        A1 = relu(Z1)    

        dropout_rate = 0.2
        mask = (torch.rand_like(A1) > dropout_rate).float()
        A1 = A1 * mask / (1 - dropout_rate)             
        Z2 = A1 @ W2 + b2            
        loss = F.cross_entropy(Z2, y_batch) + (lambda_l2/2)*(W1.norm()**2 + W2.norm()**2)
        train_loss += loss.item() * m
        train_acc += accuracy(Z2, y_batch) * m

      
        Y_hat = F.softmax(Z2, dim=1)
        Y_onehot = torch.zeros_like(Y_hat)
        Y_onehot[range(m), y_batch] = 1

        dZ2 = (Y_hat - Y_onehot) / m           
        dW2 = A1.T @ dZ2 + lambda_l2 * W2     
        db2 = dZ2.sum(dim=0)

    
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = X_batch.T @ dZ1 + lambda_l2 * W1  
        db1 = dZ1.sum(dim=0)

      
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

 
    train_loss /= X_train.shape[0]
    train_acc /= X_train.shape[0]

 
    Z1_val = X_val @ W1 + b1
    A1_val = relu(Z1_val)
    Z2_val = A1_val @ W2 + b2
    val_loss = F.cross_entropy(Z2_val, y_val).item() + (lambda_l2/2)*(W1.norm()**2 + W2.norm()**2)
    val_acc = accuracy(Z2_val, y_val)




Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)
Z2_test = A1_test @ W2 + b2
pred_class = Z2_test.argmax(dim=1).numpy()


df = pd.DataFrame({"ID": range(len(pred_class)), "target": pred_class})
df.to_csv("submission.csv", index=False)
