import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import json
from torch.utils.data import Dataset, DataLoader

class Skew(nn.Module):
    def __init__(self, vocab_size=100277):
        super(Skew, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.fc1 = nn.Linear(32 * 64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 29)
    
    def forward(self, x):
        x = x.long()
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class SkewDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)
        self.texts = list(self.data_dict.keys())
        self.labels = list(self.data_dict.values())
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_length = 32

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.enc.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

model = Skew()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = SkewDataset("data.json")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

epochs = 1000

print("Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

print("Training complete!")
torch.save(model.state_dict(), "skew_model.pth")