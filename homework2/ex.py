import pickle
import os
import pandas as pd
import numpy as np
train_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_train.pkl"
test_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_test.pkl"

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
# You must use NumPy to implement from scratch
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(40,),
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.2
)
mlp.fit(train_data, train_labels)

mlp.score(train_data, train_labels)
Iteration 1, loss = 137752.80018828
Iteration 2, loss = 145223.46597493
Iteration 3, loss = 145136.35835661
Iteration 4, loss = 145049.30311513
Iteration 5, loss = 144962.30041570
Iteration 6, loss = 144875.34986284
Iteration 7, loss = 144788.45135822
Iteration 8, loss = 144701.60512960
Iteration 9, loss = 144614.81068528
Iteration 10, loss = 144528.06842535
Iteration 11, loss = 144441.37829004
Iteration 12, loss = 144354.74019188
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
0.11236666666666667
predictions = mlp.predict(test_data)
# This is how you prepare a submission for the competition
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)