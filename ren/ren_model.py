"""
Recurrent Equilibrium Network (REN) for VRFB State-of-Charge Estimation
=========================================================================
Based on: Revay et al. "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness" (IEEE TAC, 2023)

Improvements over v1
--------------------
  1. Accurate spectral norm
     Persistent u/v buffers warm-started each call with N_POWER_ITERS=10.

  2. A_bar cached at eval time
     Computed once on first eval forward(), cleared on model.train().

  3. Learned initial hidden state z0
     Trainable _z0_raw parameter used when z=None.

  4. Better default alpha=0.5
     sigma_max(A_bar) < 0.5 — better memory horizon than alpha=0.95.

  5. MC-dropout uncertainty
     predict_with_uncertainty() returns (mean, std) over n_samples passes.

  6. Direct feedthrough D removed (use_feedthrough=False default)
     y_t = C(z_next) + D.bias
     Removes the D(x_t) path that caused flow-rate sensitivity.

  7. Current-gated hidden state (use_current_gate=True default)
     ENFORCES CHARGE CONSERVATION BY DESIGN:
       gate = tanh(gate_k * |I_scaled|)
       z_next = gate * z_candidate + (1 - gate) * z
     When I=0: gate=0 → z frozen → output constant → dSOC/dt = 0  ✓
     When I large: gate→1 → normal REN dynamics  ✓
     Result: SOC cannot change when current is zero, regardless of
     what other inputs (flow, temperature) are doing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REN(nn.Module):
    """
    Recurrent Equilibrium Network — explicit contractive form with
    current-gated hidden state for physics-consistent SOC estimation.

    Parameters
    ----------
    input_dim        : feature dimension (default 7 for VRFB hybrid)
    hidden_dim       : hidden state dimension (default 128)
    output_dim       : output channels (default 1)
    alpha            : contraction margin; sigma_max(A_bar) < (1-alpha)
    dropout          : hidden-state dropout during training
    n_power_iters    : power iterations for spectral norm
    use_feedthrough  : include D(x_t) in output (False = recommended)
    use_current_gate : freeze z at I=0 (True = physically correct)
    current_feat_idx : index of current feature in input vector (default 1)
    """

    def __init__(
        self,
        input_dim:        int   = 7,
        hidden_dim:       int   = 128,
        output_dim:       int   = 1,
        alpha:            float = 0.5,
        dropout:          float = 0.1,
        n_power_iters:    int   = 10,
        use_feedthrough:  bool  = False,
        use_current_gate: bool  = True,
        current_feat_idx: int   = 1,
    ):
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha must be strictly in (0, 1)"

        self.hidden_dim       = hidden_dim
        self.output_dim       = output_dim
        self.alpha            = alpha
        self.n_power_iters    = n_power_iters
        self.use_feedthrough  = use_feedthrough
        self.use_current_gate = use_current_gate
        self.current_feat_idx = current_feat_idx

        # Free weight — projected to A_bar at runtime
        self.A_free = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        # Input -> state, state -> output, bias
        self.B   = nn.Linear(input_dim,  hidden_dim)
        self.C   = nn.Linear(hidden_dim, output_dim)
        self.D   = nn.Linear(input_dim,  output_dim)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

        # Learned initial hidden state
        self._z0_raw = nn.Parameter(torch.zeros(1, hidden_dim))

        # Learned gate sharpness — init=4.0 gives tanh(4*|I|):
        #   |I_scaled|=0.0 → gate=0.00  (I=0A, perfectly frozen)
        #   |I_scaled|=0.3 → gate=0.93  (moderate current, mostly active)
        #   |I_scaled|=1.0 → gate≈1.00  (high current, fully active)
        self.gate_k = nn.Parameter(torch.tensor(4.0))

        self.ln_z = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Persistent spectral norm buffers
        self.register_buffer("_sn_u", F.normalize(torch.randn(hidden_dim, 1), dim=0))
        self.register_buffer("_sn_v", F.normalize(torch.randn(hidden_dim, 1), dim=0))
        self._A_bar_cache = None

        # Initialisation
        nn.init.orthogonal_(self.A_free)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)
        nn.init.xavier_uniform_(self.D.weight)
        nn.init.zeros_(self.D.bias)

    @property
    def z0(self) -> torch.Tensor:
        return torch.tanh(self._z0_raw)

    def _contractive_A(self) -> torch.Tensor:
        if not self.training and self._A_bar_cache is not None:
            return self._A_bar_cache
        W = self.A_free
        u, v = self._sn_u, self._sn_v
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_power_iters):
                    v = F.normalize(W.t() @ u, dim=0)
                    u = F.normalize(W     @ v, dim=0)
                self._sn_u.copy_(u)
                self._sn_v.copy_(v)
        sigma = (u.t() @ W @ v).squeeze().clamp(min=1e-8)
        A_bar = (1.0 - self.alpha) * W / sigma
        if not self.training:
            self._A_bar_cache = A_bar.detach()
        return A_bar

    def train(self, mode: bool = True) -> "REN":
        super().train(mode)
        self._A_bar_cache = None
        return self

    def _step(
        self,
        x_t:   torch.Tensor,   # (batch, input_dim)
        z:     torch.Tensor,   # (batch, hidden_dim)
        A_bar: torch.Tensor,
        I_raw_t=None   # (hidden_dim, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single time-step update with current-gated z.

        Physics guarantee (when use_current_gate=True):
          I = 0  →  gate = 0  →  z_next = z  →  output unchanged
          dSOC/dt = 0 when I = 0, regardless of flow or temperature.
        """
        # Candidate hidden state (standard REN update)
        pre     = z @ A_bar.t() + self.B(x_t) + self.b_z
        z_cand  = torch.tanh(self.ln_z(pre))
        z_cand  = self.drop(z_cand)

        # Current gate — enforces charge conservation
        if self.use_current_gate:
            # |I_scaled| = 0 when I=0, > 0 when current flows
            if hasattr(self, "use_raw_current") and self.use_raw_current:
                I_abs = I_raw_t.abs()
            else:
                I_abs = x_t[:, self.current_feat_idx:self.current_feat_idx+1].abs()
            
            gate   = torch.tanh(self.gate_k.abs() * I_abs)   # (batch, 1)
            z_next = gate * z_cand + (1.0 - gate) * z
        else:
            z_next = z_cand

        # Output — D(x_t) excluded unless use_feedthrough requested
        if self.use_feedthrough:
            y_t = self.C(z_next) + self.D(x_t)
        else:
            y_t = self.C(z_next) + self.D.bias

        return y_t, z_next

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if z is None:
            z = self.z0.expand(batch_size, -1).contiguous()
        A_bar   = self._contractive_A()
        outputs = []
        for t in range(seq_len):
            I_raw_t = x_raw[:, t, self.current_feat_idx:self.current_feat_idx+1] if x_raw is not None else None
            y_t, z = self._step(x[:, t, :], z, A_bar, I_raw_t)
            outputs.append(y_t.unsqueeze(1))
        y_seq = torch.cat(outputs, dim=1)
        y_seq = torch.sigmoid(y_seq)
        return y_seq, z

    @torch.no_grad()
    def predict(self, x, z=None):
        self.eval()
        return self.forward(x, z)

    def predict_with_uncertainty(self, x, z=None, n_samples=30):
        was_training = self.training
        self.train()
        preds, z_last = [], None
        with torch.no_grad():
            for _ in range(n_samples):
                y_seq, z_last = self.forward(x, z)
                preds.append(y_seq)
        if not was_training:
            self.eval()
        stack = torch.stack(preds, dim=0)
        return stack.mean(dim=0), stack.std(dim=0), z_last

    @torch.no_grad()
    def contraction_rate(self) -> float:
        W = self.A_free
        u, v = self._sn_u.clone(), self._sn_v.clone()
        for _ in range(30):
            v = F.normalize(W.t() @ u, dim=0)
            u = F.normalize(W     @ v, dim=0)
        sigma_A = (u.t() @ W @ v).squeeze().clamp(min=1e-8)
        A_bar = (1.0 - self.alpha) * W / sigma_A
        for _ in range(10):
            v = F.normalize(A_bar.t() @ u, dim=0)
            u = F.normalize(A_bar     @ v, dim=0)
        return (u.t() @ A_bar @ v).squeeze().item()

    @torch.no_grad()
    def z0_norm(self) -> float:
        return self.z0.norm().item()

    @torch.no_grad()
    def gate_value_at(self, I_scaled_abs: float) -> float:
        """Diagnostic: what gate value for a given scaled |I|?"""
        return torch.tanh(self.gate_k.abs() * torch.tensor(I_scaled_abs)).item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)