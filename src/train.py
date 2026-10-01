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