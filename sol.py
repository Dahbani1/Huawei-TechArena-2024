from torch.utils.data import DataLoader
from utils import SonicomDatabase
from model import train_model

# Create dataloaders
train_data = SonicomDatabase("./data", training_data=True)
val_data = SonicomDatabase("./data", training_data=False)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Train the model
model = train_model(train_loader, val_loader)