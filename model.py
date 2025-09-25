import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader

# Modèle d'encodage
class PinnaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        return x

# Générateur HRTF

class HRTFGenerator(nn.Module):
    def __init__(self, num_angles=19, num_freq_bins=129):
        super().__init__()
        self.encoder = PinnaEncoder()
        self.flatten_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_angles * num_freq_bins * 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        images = images.squeeze(3).float()  # Assurez-vous que les images sont en float32

        batch_size, num_ears, num_views, height, width = images.shape
        features = []
        for ear in range(num_ears):
            ear_features = []
            for view in range(num_views):
                x = images[:, ear, view, :, :].unsqueeze(1)
                x = self.encoder(x)
                ear_features.append(x)
            ear_features = torch.stack(ear_features, dim=1)
            ear_features = torch.mean(ear_features, dim=1)
            features.append(ear_features)
        features = torch.cat(features, dim=1)
       

        if self.flatten_size is None:
            self.flatten_size = features.view(batch_size, -1).shape[1]
            self.fc1 = nn.Linear(self.flatten_size, 512).to(features.device)

        features = features.view(batch_size, -1)
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        hrtf = x.view(batch_size, 19, 2, 129)
        return hrtf

# Entraîneur
class HRTFTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self, dataloader, accumulation_steps=4):
        self.model.train()
        total_loss = 0
        for i, (images, hrtfs) in enumerate(dataloader):
            images = images.to(self.device).float()  # Conversion explicite en float32
            hrtfs = hrtfs.to(self.device).float()  # Conversion explicite en float32

            # Convertir les données complexes en réels
            hrtfs_real = torch.abs(hrtfs)

            self.optimizer.zero_grad()
            predictions = self.model(images)

            # Conversion des prédictions en réels
            predictions_real = torch.abs(predictions)

            # Calcul de la perte
            loss = self.criterion(predictions_real, hrtfs_real)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            del images, hrtfs, predictions
            gc.collect()
            torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, hrtfs in dataloader:
                images = images.to(self.device).float()  # Conversion explicite en float32
                hrtfs = hrtfs.to(self.device).float()  # Conversion explicite en float32

                # Convertir les données complexes en réels
                hrtfs_real = torch.abs(hrtfs)

                predictions = self.model(images)

                # Conversion des prédictions en réels
                predictions_real = torch.abs(predictions)

                # Calcul de la perte
                loss = self.criterion(predictions_real, hrtfs_real)
                total_loss += loss.item()

                del images, hrtfs, predictions
                gc.collect()
                torch.cuda.empty_cache()
        return total_loss / len(dataloader)


# Entraînement du modèle
def train_model(train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRTFGenerator().to(device)
    trainer = HRTFTrainer(model)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}:')
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        gc.collect()
        torch.cuda.empty_cache()

    print("Entraînement terminé.")
    return model