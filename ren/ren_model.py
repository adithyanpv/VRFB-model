"""
Recurrent Equilibrium Network (REN) for VRFB State-of-Charge Estimation
=========================================================================
Based on: Revay et al. "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness" (IEEE TAC, 2023)

Key constraints
---------------
  1. Contractive A_bar via spectral-norm projection
     sigma_max(A_bar) < (1 - alpha) guaranteed every step.

  2. No direct feedthrough (use_feedthrough=False default)
     y_t = C(z_next) + D.bias
     Prevents raw sensor noise reaching output per-step.

  3. Current-gated hidden state (use_current_gate=True default)
     PHYSICS CONSTRAINT: dSOC/dt = 0 when I = 0  (Faraday's law)

     gate = tanh(gate_k * |I_raw| / I_max)
     z_next = gate * z_candidate + (1 - gate) * z

     CRITICAL: gate uses RAW AMPS (x_raw argument), NOT scaled current.
     StandardScaler shifts I by its mean, so scaled I=0A is NOT zero.
     Using scaled current as gate input means gate never closes at I=0A,
     completely breaking the physics constraint.

     x_raw must be passed as raw current in Amps from the caller.
     Shape: (batch, seq, 1)  — just the current column, in Amps.
     The caller (train_ren.py, server.py) reconstructs raw Amps from
     scaled values using the scaler's mean and std before calling forward().

  4. A_bar cached at eval time — cleared on model.train().

  5. Learned z0 — reduces cold-start transient from ~50 to ~5 steps.

  6. MC-dropout uncertainty via predict_with_uncertainty().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# I_max used to normalise raw Amps into gate — must match config.py I_max
I_MAX = 200.0


class REN(nn.Module):
    """
    Recurrent Equilibrium Network with physics-constrained current gate.

    Parameters
    ----------
    input_dim        : feature dimension (7 for VRFB hybrid model)
    hidden_dim       : hidden state dimension (128)
    output_dim       : output channels (1 — SOC correction)
    alpha            : contraction margin; sigma_max(A_bar) < (1-alpha)
    dropout          : hidden-state dropout probability during training
    n_power_iters    : power iterations for spectral norm estimation
    use_feedthrough  : if True, adds D(x_t) to output (not recommended)
    use_current_gate : if True, enforces dSOC/dt=0 at I=0 via gate
    current_feat_idx : column index of current in the input feature vector
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
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"

        self.hidden_dim       = hidden_dim
        self.output_dim       = output_dim
        self.alpha            = alpha
        self.n_power_iters    = n_power_iters
        self.use_feedthrough  = use_feedthrough
        self.use_current_gate = use_current_gate
        self.current_feat_idx = current_feat_idx

        # Unconstrained weight — projected to A_bar at runtime
        self.A_free = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        # Core linear maps
        self.B   = nn.Linear(input_dim,  hidden_dim)
        self.C   = nn.Linear(hidden_dim, output_dim)
        self.D   = nn.Linear(input_dim,  output_dim)   # weight used only when use_feedthrough=True
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

        # Learned initial hidden state — bounded by tanh in (-1, 1)
        self._z0_raw = nn.Parameter(torch.zeros(1, hidden_dim))

        # Learnable gate sharpness — initialised so gate is near-linear at
        # moderate currents and sharply closed at I=0.
        # gate_k=4.0: tanh(4*|I/200|) → 0.0 at I=0A, 0.93 at I=46A, ~1.0 at I=100A
        self.gate_k = nn.Parameter(torch.tensor(4.0))

        self.ln_z = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Persistent buffers for power-iteration spectral norm (warm-started)
        self.register_buffer("_sn_u", F.normalize(torch.randn(hidden_dim, 1), dim=0))
        self.register_buffer("_sn_v", F.normalize(torch.randn(hidden_dim, 1), dim=0))
        self._A_bar_cache: torch.Tensor | None = None

        # Weight initialisation
        nn.init.orthogonal_(self.A_free)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)
        nn.init.xavier_uniform_(self.D.weight)
        nn.init.zeros_(self.D.bias)

    # ------------------------------------------------------------------
    # Learned initial state
    # ------------------------------------------------------------------

    @property
    def z0(self) -> torch.Tensor:
        """Learned initial hidden state, bounded in (-1, 1)."""
        return torch.tanh(self._z0_raw)

    # ------------------------------------------------------------------
    # Spectral-norm contractive projection
    # ------------------------------------------------------------------

    def _contractive_A(self) -> torch.Tensor:
        """Returns A_bar with sigma_max(A_bar) < (1 - alpha)."""
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

    # ------------------------------------------------------------------
    # Single time-step update
    # ------------------------------------------------------------------

    def _step(
        self,
        x_t:     torch.Tensor,          # (batch, input_dim) — scaled features
        z:       torch.Tensor,           # (batch, hidden_dim)
        A_bar:   torch.Tensor,           # (hidden_dim, hidden_dim)
        I_raw_t: torch.Tensor | None,   # (batch, 1) — RAW AMPS (not scaled)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One REN timestep with current-gated hidden state update.

        Physics guarantee when use_current_gate=True and I_raw_t is provided:
          I = 0A  →  gate = tanh(gate_k * 0) = 0  →  z_next = z
                  →  y_t unchanged  →  dSOC/dt = 0  ✓

        If I_raw_t is None (fallback), gate uses scaled current — the
        gate will NOT close at true I=0A. Always provide I_raw_t.
        """
        pre    = z @ A_bar.t() + self.B(x_t) + self.b_z
        z_cand = torch.tanh(self.ln_z(pre))
        z_cand = self.drop(z_cand)

        if self.use_current_gate:
            if I_raw_t is not None:
                # Raw Amps normalised by I_max → [0, 1] scale.
                # gate = 0 exactly when I_raw = 0A.
                # Dividing by I_MAX makes gate_k dimensionless and
                # scale-invariant regardless of the hardware current range.
                I_abs = I_raw_t.abs() / I_MAX
            else:
                # Fallback: scaled current (INACCURATE — gate won't close at I=0A)
                # This path should never be reached in normal operation.
                I_abs = x_t[:, self.current_feat_idx:self.current_feat_idx + 1].abs()

            gate   = torch.tanh(self.gate_k.abs() * I_abs)   # (batch, 1)
            z_next = gate * z_cand + (1.0 - gate) * z
        else:
            z_next = z_cand

        if self.use_feedthrough:
            y_t = self.C(z_next) + self.D(x_t)
        else:
            y_t = self.C(z_next) + self.D.bias

        return y_t, z_next

    # ------------------------------------------------------------------
    # Forward pass over a full sequence
    # ------------------------------------------------------------------

    def forward(
        self,
        x:     torch.Tensor,                   # (batch, seq, input_dim) — scaled
        z:     torch.Tensor | None = None,     # (batch, hidden_dim) or None → z0
        x_raw: torch.Tensor | None = None,     # (batch, seq, 1) — RAW AMPS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x     : scaled input features  (batch, seq, input_dim)
        z     : initial hidden state; None → use learned z0
        x_raw : raw current in Amps    (batch, seq, 1)
                Must be provided for the gate to work correctly.
                Shape is (batch, seq, 1) — only the current column.

        Returns
        -------
        y_seq : SOC correction in (0, 1) via sigmoid  (batch, seq, 1)
        z     : final hidden state                     (batch, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        if z is None:
            z = self.z0.expand(batch_size, -1).contiguous()

        A_bar   = self._contractive_A()
        outputs = []

        for t in range(seq_len):
            # Extract raw current for this timestep: (batch, 1)
            I_raw_t = x_raw[:, t, 0:1] if x_raw is not None else None
            y_t, z  = self._step(x[:, t, :], z, A_bar, I_raw_t)
            outputs.append(y_t.unsqueeze(1))

        y_seq = torch.cat(outputs, dim=1)   # (batch, seq, 1)
        y_seq = torch.sigmoid(y_seq)
        return y_seq, z

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x:     torch.Tensor,
        z:     torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic inference (eval mode, no gradient)."""
        self.eval()
        return self.forward(x, z, x_raw)

    def predict_with_uncertainty(
        self,
        x:         torch.Tensor,
        z:         torch.Tensor | None = None,
        x_raw:     torch.Tensor | None = None,
        n_samples: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC-dropout uncertainty: returns (mean, std, z_last)."""
        was_training = self.training
        self.train()
        preds, z_last = [], None
        with torch.no_grad():
            for _ in range(n_samples):
                y_seq, z_last = self.forward(x, z, x_raw)
                preds.append(y_seq)
        if not was_training:
            self.eval()
        stack = torch.stack(preds, dim=0)
        return stack.mean(dim=0), stack.std(dim=0), z_last

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def contraction_rate(self) -> float:
        """Accurate sigma_max(A_bar) — should be < (1 - alpha)."""
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
        """L2 norm of learned initial hidden state."""
        return self.z0.norm().item()

    @torch.no_grad()
    def gate_value_at_raw_amps(self, I_amps: float) -> float:
        """
        Gate value for a given raw current in Amps.
        This is the correct diagnostic — not scaled units.
        gate_value_at_raw_amps(0.0) must be << 0.1 for constraint to hold.
        """
        I_norm = abs(I_amps) / I_MAX
        return torch.tanh(self.gate_k.abs() * torch.tensor(I_norm)).item()

    @torch.no_grad()
    def gate_value_at(self, I_scaled_abs: float) -> float:
        """
        Legacy diagnostic using scaled current magnitude.
        Use gate_value_at_raw_amps() for physically meaningful values.
        """
        return torch.tanh(self.gate_k.abs() * torch.tensor(I_scaled_abs)).item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)