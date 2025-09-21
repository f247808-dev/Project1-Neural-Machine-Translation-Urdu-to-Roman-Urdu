# neural_machine_training.py
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================================
# Dataset preparation
# ==================================
DATA_DIR = "dataset"
DATA_ZIP = "dataset.zip"

if not os.path.exists(DATA_DIR) and os.path.exists(DATA_ZIP):
    with zipfile.ZipFile(DATA_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

class TransliterationDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        with open(src_file, "r", encoding="utf-8") as f:
            self.src = f.readlines()
        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt = f.readlines()
        self.data = list(zip(self.src, self.tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==================================
# Model definitions
# ==================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.bilstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim * 2,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, output_dim)

    def forward(self, trg, hidden, cell):
        embedded = self.embedding(trg)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        outputs, (hidden, cell) = self.decoder(trg, hidden, cell)
        return outputs

# ==================================
# Training loop
# ==================================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(dataloader, desc="Training"):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg.contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ==================================
# Main script
# ==================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example toy vocab size (replace with actual tokenizer later)
    INPUT_DIM = 5000
    OUTPUT_DIM = 5000

    encoder = Encoder(INPUT_DIM).to(device)
    decoder = Decoder(OUTPUT_DIM).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy dataset loader for now (replace with tokenized tensors)
    src_file = os.path.join(DATA_DIR, "train.ur")
    tgt_file = os.path.join(DATA_DIR, "train.ro")
    dataset = TransliterationDataset(src_file, tgt_file)

    # NOTE: You must convert text to tensors here
    # For now, let's simulate with random tensors
    X = torch.randint(0, INPUT_DIM, (len(dataset), 20))
    Y = torch.randint(0, OUTPUT_DIM, (len(dataset), 20))
    train_data = list(zip(X, Y))
    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Training
    for epoch in range(1, 6):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/seq2seq_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
