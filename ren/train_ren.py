# ren/train_ren.py

import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from ren_model import REN

# ------------------------------------------------
# 1. Load & Scale Dataset
# ------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("datasets/vrfb_dataset.csv")

feature_cols = [
    "voltage", "current", "temperature", "flow",
    "I_limit", "transport_ratio", "dVdt", "dIdt"
]

X_raw = df[feature_cols].values
y_raw = df["SOC_true"].values

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Ensure the directory exists and save the scaler
os.makedirs("ren", exist_ok=True)
joblib.dump(scaler, "ren/scaler.pkl")
print("Saved scaler to ren/scaler.pkl")

# ------------------------------------------------
# 2. Sequence Chunking (Truncated BPTT)
# ------------------------------------------------
SEQ_LENGTH = 1000  # Network learns 1000 seconds of history per batch
num_sequences = len(X_scaled) // SEQ_LENGTH

# Trim excess data that doesn't fit perfectly into a 1000-step block
X_seq = X_scaled[:num_sequences * SEQ_LENGTH].reshape(num_sequences, SEQ_LENGTH, 8)
y_seq = y_raw[:num_sequences * SEQ_LENGTH].reshape(num_sequences, SEQ_LENGTH, 1)

X = torch.tensor(X_seq, dtype=torch.float32)
y = torch.tensor(y_seq, dtype=torch.float32)

# ------------------------------------------------
# 3. Model & Optimizer
# ------------------------------------------------
model = REN(input_dim=8, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# ------------------------------------------------
# 4. Training Loop
# ------------------------------------------------
epochs = 30
print(f"Starting training on {num_sequences} sequences of length {SEQ_LENGTH}...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Initialize hidden state for the epoch
    batch_size = 1 
    z = torch.zeros(batch_size, model.hidden_dim)

    for i in range(num_sequences):
        x_batch = X[i].unsqueeze(0)  # Shape: (1, 1000, 8)
        y_batch = y[i].unsqueeze(0)  # Shape: (1, 1000, 1)

        # Forward pass
        pred, z = model(x_batch, z)
        loss = loss_fn(pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping prevents exploding gradients in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Detach hidden state so gradients don't trace back to previous sequences
        z = z.detach()

        total_loss += loss.item()

    avg_loss = total_loss / num_sequences
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.6f}")

# Save the trained model weights
torch.save(model.state_dict(), "ren/ren_soc_model.pth")
print("\nTraining complete. Model saved to ren/ren_soc_model.pth")