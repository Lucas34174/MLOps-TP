import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

mlflow.set_experiment(
    "Fashion-MNIST"
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
##Définition du modèle
class NeuralNetwork(nn.Module):
    def __init__(self, hidden1=128, hidden2=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10)
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

print("Train :", len(train_dataset))
print("Test  :", len(test_dataset))

##Fonction d'entrainement
def train_experiment(
    learning_rate=0.001,
    batch_size=64,
    epochs=3,
    hidden1=128,
    hidden2=64
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader (
        dataset=test_dataset,
        batch_size=batch_size
    )

    ##Entraînement du modèle

    print("Device :", device)

    model = NeuralNetwork(
        hidden1=hidden1,
        hidden2=hidden2
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    
    ##Ajout du MLFlow
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate )
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden1", hidden1)
        mlflow.log_param("hidden2", hidden2)

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
            ##Enregistrement de la Loss
            mlflow.log_metric(
                "train_loss",
                average_loss,
                step=epoch
            )
            print(
                f"Epoch {epoch + 1} "
                f"Loss: {average_loss:.4f}"
            )
    ##evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (
                    predictions == labels
                ).sum().item()
        accuracy = correct / total
        ## Enregistrement de l'Accuracy
        mlflow.log_metric(
            "test_accuracy",
            accuracy
        )
        print(f"Accuracy : {accuracy:.4f}")
        ##Enregistrement du modèle
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle"
        )
        return accuracy