import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
)
        
def forward(self, x):
    x = self.flatten(x)
    return self.network(x)
##Chargement des données
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="data/raw",
    train=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data/raw",
    train=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader (
    dataset=test_dataset,
    batch_size=64
)

##Entraînement du modèle
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)

model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1} "
        f"Loss: {average_loss:.4f}"
    )