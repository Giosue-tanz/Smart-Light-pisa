# create_dummy_model.py
import torch
import torch.nn as nn
import os

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 31)  # 10-40s = 31 azioni
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

# Crea cartella models se non esiste
os.makedirs("models", exist_ok=True)

# Crea modello vuoto
model = DQN()
torch.save(model.state_dict(), 'models/smartlight_dqn_pisa_v1.8.pth')
print("Modello DQN creato con successo!")
print("File salvato in: models/smartlight_dqn_pisa_v1.8.pth")