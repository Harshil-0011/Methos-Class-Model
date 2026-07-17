from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nslt.ssm_scan import selective_scan, get_scan_mode, set_scan_mode

logger = logging.getLogger(__name__)


class SSMCompressionEngine(nn.Module):
    """
    Layer 1: State-Space Neural Compression Engine (O(1) Memory)

    Maps arbitrary-length context sequences into a fixed-size hidden state
    vector via a structured state-space model with selective scan (Mamba-style).

    Mathematical core (discretized SSM with zero-order hold):
        A_bar_t = exp(Delta_t * A)                  # [batch, seq_len, d_state]
        B_bar_t = (A_bar_t - I) * A^{-1} * Delta_t * B_proj(x_t)
        h_t = A_bar_t * h_{t-1} + B_bar_t @ x_t    # [batch, d_state]
        y_t = C_proj(x_t) @ h_t                     # [batch, d_model]

    Memory: O(1) — fixed d_state=hidden regardless of sequence length.

    Args:
        d_model: Input token embedding dimension
        d_state: Compressed state dimension (fixed, independent of seq_len)
        dt_rank: Rank of the step-parameterization projection
        expand_factor: Expansion factor for the inner SSM dimension
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        d_model: int = 7168,
        d_state: int = 2048,
        dt_rank: int = 256,
        expand_factor: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = int(d_model * expand_factor)

        # Selective scan projections — all input-dependent (Mamba 2024)
        # Project input to inner dimension
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        # [batch, seq_len, d_inner * 2]

        # Convolution before SSM for local context mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3,
            bias=True,
        )
        # [batch, d_inner, seq_len + padding]

        # Activation normalization
        self.act = nn.SiLU()

        # SSM parameters — diagonal structured A
        self.dt_proj = nn.Linear(self.d_inner, dt_rank, bias=bias)
        # [batch, seq_len, dt_rank]
        self.dt_bias = nn.Parameter(torch.randn(dt_rank) * 0.01)

        # A is diagonal with negative half-plane initialization for stability
        A_log = torch.log(torch.rand(d_state, dtype=torch.float32) * 0.5 + 0.001)
        self.A_log = nn.Parameter(A_log)  # [d_state]

        # Input-dependent B and C projections
        self.B_proj = nn.Linear(self.d_inner, dt_rank, bias=bias)
        # Projects to dt_rank, then expanded to d_state per step
        self.C_proj = nn.Linear(self.d_inner, dt_rank, bias=bias)

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,         # [batch, seq_len, d_model]
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Normalize input
        x = self.norm(x)

        # Input projection to inner dimension
        # [batch, seq_len, d_inner * 2]
        x_proj = self.in_proj(x)
        x_inner, x_gate = x_proj.chunk(2, dim=-1)
        # x_inner: [batch, seq_len, d_inner], x_gate: [batch, seq_len, d_inner]

        # 1D convolution for local context mixing
        # Permute to [batch, d_inner, seq_len] for Conv1d
        x_conv = self.conv1d(x_inner.permute(0, 2, 1))
        # Remove padding — take first seq_len positions
        x_conv = x_conv[:, :, :seq_len].permute(0, 2, 1)
        # [batch, seq_len, d_inner]
        x_conv = self.act(x_conv)

        # Compute SSM parameters from the convolved input
        # Stepsize delta [batch, seq_len, dt_rank]
        delta = F.softplus(self.dt_proj(x_conv) + self.dt_bias)

        # A parameter (diagonal) [d_state]
        A = -torch.exp(self.A_log)

        # B and C projections [batch, seq_len, dt_rank]
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        # Selective scan dispatches to sequential (CPU), vectorized (CUDA),
        # or Triton kernel based on get_scan_mode().
        y, h_final = selective_scan(
            x_conv, delta, A, B, C, self.dt_rank, mode=get_scan_mode()
        )

        # Gate with SiLU-activated input
        y = y * self.act(x_gate)

        # Output projection back to d_model
        output = self.out_proj(y)
        # [batch, seq_len, d_model]

        if return_state:
            return output, h_final

        return output, h_final.detach()
