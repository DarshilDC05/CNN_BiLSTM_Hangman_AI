# 3_finetune_model.py
# This script loads the main trained model and fine-tunes it on a specialized
# dataset of difficult, early-game AND end-game scenarios to maximize performance.

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. Load and Prepare Data ---
print("Loading massive masked data for fine-tuning...")
df = pd.read_csv('hangman_masked_training_data.csv').dropna().reset_index(drop=True)
print(f"Loaded {len(df)} total training examples.")

# --- NEW: Create a specialized fine-tuning dataset for difficult scenarios ---
print("Creating specialized dataset for difficult scenarios (early-game, mid-game, and end-game)...")
# a) Early-game: 1 or 2 letters revealed
revealed_letters_count = df['pattern'].str.count('[a-zA-Z]')
early_game_df = df[(revealed_letters_count >= 1) & (revealed_letters_count <= 2)].copy()
print(f"Found {len(early_game_df)} early-game examples.")

# b) End-game: 1 or 2 blanks remaining
blank_letters_count = df['pattern'].str.count('_')
end_game_df = df[(blank_letters_count >= 1) & (blank_letters_count <= 2)].copy()
print(f"Found {len(end_game_df)} end-game examples.")

# c) Mid-game: A random sample of other patterns to prevent catastrophic forgetting
# Get the indices of the early and end game samples to avoid overlap
difficult_indices = pd.concat([early_game_df, end_game_df]).index.unique()
# Create a dataframe of mid-game samples by excluding the difficult ones
mid_game_df = df.drop(difficult_indices)
print(f"Found {len(mid_game_df)} mid-game examples.")

# Sample 5 million mid-game examples
num_mid_game_samples = 5000000
if len(mid_game_df) > num_mid_game_samples:
    mid_game_sample_df = mid_game_df.sample(n=num_mid_game_samples, random_state=42)
    print(f"Sampling {len(mid_game_sample_df)} mid-game examples.")
else:
    mid_game_sample_df = mid_game_df # Use all if less than 5M
    print(f"Using all {len(mid_game_sample_df)} mid-game examples.")

# d) Combine them into one powerful fine-tuning dataset
finetune_df = pd.concat([early_game_df, end_game_df, mid_game_sample_df]).drop_duplicates().reset_index(drop=True)
print(f"Created a combined specialized fine-tuning set with {len(finetune_df)} examples.")


# Character mapping from saved metadata
with open('model_metadata.json', 'r') as f:
    model_metadata = json.load(f)

char_to_int = model_metadata['char_to_int']
alphabet = model_metadata['alphabet']
MAX_LEN = model_metadata['max_len']
VOCAB_SIZE = len(char_to_int) + 1
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
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
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

train_df, test_df = train_test_split(finetune_df, test_size=0.05, random_state=42)
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

# Instantiate the model
model = HangmanDeepModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=128,
    hidden_dim=256,
    num_classes=ALPHABET_SIZE
).to(device)

print("Loading pre-trained model weights from 'hangman_bilstm_model.pth'...")
model.load_state_dict(torch.load('hangman_bilstm_model_new.pth', map_location=device))

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 10x smaller LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=1, verbose=True)

print("\n--- Starting Specialized Fine-Tuning Phase ---")
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for patterns, targets in tqdm(train_loader, desc=f"Fine-Tune Epoch {epoch+1}/{NUM_EPOCHS}"):
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
    print(f"Fine-Tune Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")
    
    scheduler.step(avg_val_loss)

# save the model
print("\nSaving final fine-tuned model...")
# overwrite the original model
torch.save(model.state_dict(), 'hangman_bilstm_model_new.pth')

print("Fine-tuning complete!")
