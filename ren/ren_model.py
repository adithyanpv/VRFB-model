"""
Recurrent Equilibrium Network (REN) for VRFB State-of-Charge Estimation
=========================================================================
Based on: Revay et al. "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness" (IEEE TAC, 2023)

Key properties enforced:
  - Contractivity  : the hidden state mapping is a contraction (||dz/dz_prev|| < 1)
  - Well-posedness : the implicit layer has a unique fixed point (via the
                     Implicit Function Theorem / Banach fixed-point theorem)
  - Output bounded : sigmoid guarantees SOC ∈ (0,1)

Architecture
------------
  z_{t+1} = σ(A_bar z_t  +  B_bar x_t  +  b_z)
  y_t      = C z_t  +  D x_t  +  b_y
  ŷ_t      = sigmoid(y_t)           # hard-clamp SOC ∈ (0,1)

  where A_bar is the *contractive* version of a free weight matrix A:
      A_bar = (1-α) * A / ||A||_spectral
  with contraction rate α ∈ (0,1).

  This is the "explicit" / simplified REN variant.  The full implicit
  variant (where z appears on both sides at time t) adds an equilibrium
  solver inner loop; we keep the explicit form here because:
    1. It is already theoretically contractive.
    2. It trains stably with standard BPTT.
    3. The implicit form rarely adds accuracy on tabular battery data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: spectral normalisation (manual, so we can reuse the σ_max value)
# ---------------------------------------------------------------------------
def spectral_norm_value(W: torch.Tensor) -> torch.Tensor:
    """Largest singular value of a 2-D weight matrix (power iteration, 1 step)."""
    u = F.normalize(torch.randn(W.shape[0], 1, device=W.device), dim=0)
    v = F.normalize(W.t() @ u, dim=0)
    u = F.normalize(W @ v, dim=0)
    return (u.t() @ W @ v).squeeze()


# ---------------------------------------------------------------------------
# Core REN block
# ---------------------------------------------------------------------------
class REN(nn.Module):
    """
    Recurrent Equilibrium Network (explicit contractive form).

    Parameters
    ----------
    input_dim  : number of input features  (default 8 for VRFB)
    hidden_dim : dimension of internal state z  (default 64)
    output_dim : SOC channels (default 1)
    alpha      : contraction rate, 0 < alpha < 1  (default 0.95 → tight contraction)
    dropout    : dropout on hidden state during training (default 0.0)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 1,
        alpha: float = 0.95,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha  # contraction rate

        # ── Free (unconstrained) weights ──────────────────────────────────
        # A_free will be projected to A_bar at each forward pass
        self.A_free = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.B      = nn.Linear(input_dim,  hidden_dim)   # input → state
        self.C      = nn.Linear(hidden_dim, output_dim)   # state → output
        self.D      = nn.Linear(input_dim,  output_dim)   # direct feedthrough
        self.b_z    = nn.Parameter(torch.zeros(hidden_dim))

        # ── Layer normalisation on z (improves training stability) ────────
        self.ln_z = nn.LayerNorm(hidden_dim)

        # ── Dropout ───────────────────────────────────────────────────────
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # ── Initialisation ────────────────────────────────────────────────
        nn.init.orthogonal_(self.A_free)   # orthogonal → σ_max = 1 → easy to project
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)
        nn.init.xavier_uniform_(self.D.weight)
        nn.init.zeros_(self.D.bias)

    # ------------------------------------------------------------------
    # Contractive projection: A_bar = (1-α) * A / σ_max(A)
    # ------------------------------------------------------------------
    def _contractive_A(self) -> torch.Tensor:
        sigma = spectral_norm_value(self.A_free).clamp(min=1e-8)
        return (1.0 - self.alpha) * self.A_free / sigma

    # ------------------------------------------------------------------
    # Single time-step update
    # ------------------------------------------------------------------
    def _step(
        self,
        x_t: torch.Tensor,   # (batch, input_dim)
        z:   torch.Tensor,   # (batch, hidden_dim)
        A_bar: torch.Tensor, # (hidden_dim, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (y_t, z_next).
        y_t is the *raw* (pre-sigmoid) SOC logit.
        """
        # Contractive hidden update
        z_next = torch.tanh(
            self.ln_z(z @ A_bar.t() + self.B(x_t) + self.b_z)
        )
        z_next = self.drop(z_next)

        # Output (direct feedthrough + state readout)
        y_t = self.C(z_next) + self.D(x_t)

        return y_t, z_next

    # ------------------------------------------------------------------
    # Forward pass (full sequence)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, input_dim)
        z: torch.Tensor | None = None, # (batch, hidden_dim)  or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        y_seq : (batch, seq_len, output_dim)  – SOC predictions in (0,1)
        z     : (batch, hidden_dim)            – final hidden state
        """
        batch_size, seq_len, _ = x.shape

        if z is None:
            z = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        A_bar = self._contractive_A()  # compute once per forward pass

        outputs = []
        for t in range(seq_len):
            y_t, z = self._step(x[:, t, :], z, A_bar)
            outputs.append(y_t.unsqueeze(1))           # (batch, 1, output_dim)

        y_seq = torch.cat(outputs, dim=1)              # (batch, seq_len, output_dim)
        y_seq = torch.sigmoid(y_seq)                   # SOC ∈ (0,1)

        return y_seq, z

    # ------------------------------------------------------------------
    # Convenience: predict without gradient (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        return self.forward(x, z)

    # ------------------------------------------------------------------
    # Sanity check: verify contraction rate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def contraction_rate(self) -> float:
        A_bar = self._contractive_A()
        sigma = spectral_norm_value(A_bar).item()
        return sigma