# 2_train_model_final_architecture.py
# This script trains our most advanced, deeper model architecture on the massive masked dataset.

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load and Prepare Data
print("Loading massive masked data...")
df = pd.read_csv('hangman_masked_training_data.csv').dropna().reset_index(drop=True)

print(f"Loaded {len(df)} training examples.")

# Character mapping
chars = sorted(list(set(''.join(df['pattern'])) | set(''.join(df['target_word']))))
char_to_int = {c: i + 1 for i, c in enumerate(chars)} # +1 for padding 0
VOCAB_SIZE = len(char_to_int) + 1
MAX_LEN = df['pattern'].str.len().max()
alphabet = sorted(list(set(''.join(df['target_word'])))) # a-z
ALPHABET_SIZE = len(alphabet)

class HangmanMaskedDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pattern_seq = [char_to_int.get(c, 0) for c in row['pattern']]
        pattern_tensor = torch.tensor(pattern_seq, dtype=torch.long)
        target_seq = [alphabet.index(c) for c in row['target_word']]
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return pattern_tensor, target_tensor

class HangmanDeepModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(HangmanDeepModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # Stacked BiLSTM layers
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        
        # Deeper final prediction network
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, pattern):
        embedded = self.embedding(pattern)
        embedded = embedded.permute(0, 2, 1)
        conv_out = F.relu(self.conv1d(embedded))
        conv_out = conv_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm1(conv_out)
        lstm_out = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout2(x)
        logits = self.fc2(x)
        
        return logits

train_df, test_df = train_test_split(df, test_size=0.05, random_state=42) # 5% for validation is enough for a huge dataset
train_dataset = HangmanMaskedDataset(train_df)
test_dataset = HangmanMaskedDataset(test_df)

def collate_fn(batch):
    patterns, targets = zip(*batch)
    padded_patterns = nn.utils.rnn.pad_sequence(patterns, batch_first=True, padding_value=0)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
    return padded_patterns, padded_targets

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HangmanDeepModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=128,
    hidden_dim=256,
    num_classes=ALPHABET_SIZE
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=1, verbose=True)

# training loop
print("--- Starting Training for Deep Model ---")
NUM_EPOCHS = 5 
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for patterns, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        patterns, targets = patterns.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(patterns)
        loss = criterion(outputs.view(-1, ALPHABET_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for patterns, targets in test_loader:
            patterns, targets = patterns.to(device), targets.to(device)
            outputs = model(patterns)
            
            val_loss = criterion(outputs.view(-1, ALPHABET_SIZE), targets.view(-1))
            total_val_loss += val_loss.item()

            _, predicted = torch.max(outputs, 2)
            mask = targets != -1
            total_correct += (predicted[mask] == targets[mask]).sum().item()
            total_samples += mask.sum().item()
            
    accuracy = 100 * total_correct / total_samples
    avg_val_loss = total_val_loss / len(test_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")
    
    # Update the learning rate scheduler
    scheduler.step(avg_val_loss)

# save the model
print("\nSaving final model and supporting files...")
torch.save(model.state_dict(), 'hangman_bilstm_model_new.pth')

metadata = {
    'max_len': MAX_LEN,
    'char_to_int': char_to_int,
    'alphabet': alphabet,
}
with open('model_metadata_new.json', 'w') as f:
    json.dump(metadata, f)

print("Training complete!")
