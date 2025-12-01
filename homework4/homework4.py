import pickle
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.ndimage import rotate, shift, zoom
from tqdm import tqdm
import multiprocessing


class ExtendedMNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        file = "extended_mnist_train.pkl" if train else "extended_mnist_test.pkl"
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def preprocess_and_augment(data, train=True):
    processed = []
    for img in data:
        x = np.array(img[0], dtype=np.float32).reshape(28, 28)  
        x = (x / 255.0 - 0.1307) / 0.3081

        if train:
            angle = np.random.uniform(-12, 12)
            x = rotate(x, angle, reshape=False, order=1, mode="nearest")
          
            sx, sy = np.random.uniform(-2, 2, 2)
            x = shift(x, shift=(sx, sy), order=1, mode="nearest")
            
            scale = np.random.uniform(0.9, 1.1)
            zs = zoom(x, scale, order=1)
            
            if zs.shape[0] > 28:
                start = (zs.shape[0] - 28) // 2
                zs = zs[start:start+28, start:start+28]
            else:
                pad = 28 - zs.shape[0]
                p1 = pad // 2
                p2 = pad - p1
                zs = np.pad(zs, ((p1, p2), (p1, p2)), mode="edge")
            x = zs
            
            x += np.random.normal(0, 0.015, x.shape)

        processed.append(x.flatten())

    return torch.tensor(np.array(processed), dtype=torch.float32)


class MNISTModel(nn.Module):
    def __init__(self, input_size=784, h1=1024, h2=512, h3=256, h4=128, out=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.fc4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)
        self.fc5 = nn.Linear(h4, out)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: Tensor):
        x = self.dropout(self.bn1(self.fc1(x)).relu())
        x = self.dropout(self.bn2(self.fc2(x)).relu())
        x = self.dropout(self.bn3(self.fc3(x)).relu())
        x = self.dropout(self.bn4(self.fc4(x)).relu())
        return self.fc5(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    mean_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        mean_loss += loss.item()
    return mean_loss / len(dataloader)

@torch.inference_mode()
def validate(model, dataloader, criterion, device):
    model.eval()
    mean_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        mean_loss += loss.item()
    return mean_loss / len(dataloader)

@torch.inference_mode()
def predict_tta(models, X, device, n=5):
    preds = 0.0
    for _ in range(n):
        noise = 0.015 * torch.randn_like(X)
        X_aug = X + noise
        p = sum(m(X_aug.to(device)).softmax(1) for m in models)
        preds += p / len(models)
    return (preds / n).argmax(1).cpu()


def main():
    NUM_WORKERS = 0
    device = get_device()


    train_raw = [img for img in ExtendedMNISTDataset(train=True)]
    train_labels = [label for _, label in ExtendedMNISTDataset(train=True)]
    test_raw = [img for img in ExtendedMNISTDataset(train=False)]

    X_train = preprocess_and_augment(train_raw, train=True)
    y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test = preprocess_and_augment(test_raw, train=False)


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=NUM_WORKERS)

   
    model1 = MNISTModel().to(device)
    model2 = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)


    epochs = 38
    steps_per_epoch = len(train_loader)
    scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=0.004, epochs=epochs, steps_per_epoch=steps_per_epoch)
    scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer2, max_lr=0.004, epochs=epochs, steps_per_epoch=steps_per_epoch)

  
    for ep in range(epochs):
        l1 = train(model1, train_loader, criterion, optimizer1, device, scheduler1)
        l2 = train(model2, train_loader, criterion, optimizer2, device, scheduler2)
        print(f"Epoch {ep+1}")


    preds = predict_tta([model1, model2], X_test, device, n=5)

    df = pd.DataFrame({"ID": range(len(preds)), "target": preds.tolist()})
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
