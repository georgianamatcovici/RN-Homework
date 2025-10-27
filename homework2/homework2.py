import pickle
import os
import pandas as pd
import numpy as np
train_file = "extended_mnist_train.pkl"
test_file = "test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

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

X=np.array(train_data, dtype=np.float32) / 255.0 
labels=np.array(train_labels) 
m=labels.size 
Target=np.zeros((m, 10)) 
Target[np.arange(m), labels]=1 

np.random.seed(42)
W = np.random.randn(784, 10) * 0.01 
b = np.zeros(10) 
miu=0.05
epochs=300
batch_size=128
for epoch in range(epochs):
     permutation=np.random.permutation(m)
     X1=X[permutation]
     T1=Target[permutation]

     for i in range(0, m, batch_size):
        X_batch=X1[i:i+batch_size]
        Target_batch=T1[i:i+batch_size]

        Z = X_batch.dot(W) + b
        Y = np.exp(Z- np.max(Z, axis=1, keepdims=True)) 
        Y /= np.sum(Y, axis=1, keepdims=True) 
      
        cross_entropy = -np.sum(Target_batch * np.log(Y + 1e-8)) / X_batch.shape[0]

        W_correction=X_batch.T.dot(Target_batch-Y) 
        W+=miu*W_correction/batch_size
        b_correction = np.sum(Target_batch - Y, axis=0) 
        b += miu * b_correction/batch_size



X=np.array(test_data,  dtype=np.float32) / 255.0  
Z = X.dot(W) + b 
Y = np.exp(Z- np.max(Z, axis=1, keepdims=True)) 
Y /= np.sum(Y, axis=1, keepdims=True) 
pred_class=np.argmax(Y, axis=1) 
print(pred_class)



predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(pred_class):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)

