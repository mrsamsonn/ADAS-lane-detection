### train.py
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from models import LaneNet
from utils import LaneDataset

def main():
    train_images = pickle.load(open("full_CNN_train.p", "rb"))
    labels = pickle.load(open("full_CNN_labels.p", "rb"))
    train_images = np.array(train_images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32) / 255.0
    train_images, labels = shuffle(train_images, labels)
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
    train_dataset = LaneDataset(X_train, y_train)
    val_dataset = LaneDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()
