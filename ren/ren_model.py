# ren/ren_model.py

import torch
import torch.nn as nn

class REN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Custom recurrent equations
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, 1)

        self.activation = torch.tanh

    def forward(self, x, z=None):
        """
        x shape: (batch_size, sequence_length, features)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if z is None:
            z = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        
        # Unroll the sequence to build memory over time (BPTT)
        for t in range(seq_len):
            x_t = x[:, t, :]
            z = self.activation(self.W(z) + self.U(x_t))
            outputs.append(z.unsqueeze(1))

        # Combine all time steps into one tensor
        z_seq = torch.cat(outputs, dim=1)
        
        # Predict SOC for the whole sequence at once
        y = self.C(z_seq) 
        
        return y, z