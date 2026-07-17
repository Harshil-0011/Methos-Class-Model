from __future__ import annotations

import math
import logging
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) cell — continuous-time neural ODE unit.

    The time constant tau is itself a learnable function of the input and hidden
    state, enabling dynamic adaptation to task complexity:

        tau(z, x) = sigmoid(U_tau @ [z, x]) * (tau_max - tau_min) + tau_min
        dz/dt = -(1/tau) * z + f(z, x)

    where f(z, x) = tanh(W_f @ [z, x] + b_f) is the liquid network.

    Args:
        d_state: Dimension of the compressed state from Layer 1
        d_hidden: Hidden dimension of the liquid ODE
        tau_min: Minimum time constant (fast dynamics)
        tau_max: Maximum time constant (slow dynamics)
    """

    def __init__(
        self,
        d_state: int = 2048,
        d_hidden: int = 7168,
        tau_min: float = 0.001,
        tau_max: float = 5.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_hidden = d_hidden
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Time constant network: tau = f(z, x_proj)
        # z is [batch, d_hidden], x_proj is [batch, d_hidden]
        self.tau_net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Liquid network: f(z, x) = tanh(W_f @ [z, x] + b_f)
        self.f_net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )

        # Output projection from liquid hidden (reserved for future use)
        self.out_proj = nn.Linear(d_hidden, d_hidden)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def ode_func(
        self,
        t: torch.Tensor,
        z: torch.Tensor,         # [batch, d_hidden]
        x: torch.Tensor,         # [batch, d_state]
    ) -> torch.Tensor:           # [batch, d_hidden]
        """
        ODE function: dz/dt = -(1/tau) * z + f(z, x)

        tau is dynamically computed from the current state and input,
        enabling variable-speed dynamics depending on input complexity.
        """
        # Concatenate state and input
        zx = torch.cat([z, x], dim=-1)  # [batch, d_hidden + d_state]

        # Compute time constant
        tau_raw = self.tau_net(zx)  # [batch, d_hidden]
        tau = torch.sigmoid(tau_raw) * (self.tau_max - self.tau_min) + self.tau_min

        # Compute liquid network output
        f_out = self.f_net(zx)  # [batch, d_hidden]

        # ODE: dz/dt = -(1/tau) * z + f_out
        dz_dt = -(1.0 / (tau + 1e-8)) * z + torch.tanh(f_out)

        return dz_dt


class LTCRoutingLayer(nn.Module):
    """
    Layer 2: Liquid Time-Constant Routing Layer

    Takes the O(1) compressed state from Layer 1 and evolves it through a
    continuous-time ODE with input-dependent time constants. Uses the adjoint
    method for memory-efficient backpropagation through the ODE solve.

    The adjoint method (Chen et al. 2018, NeurIPS) avoids storing intermediate
    ODE states by solving an augmented ODE backwards in time:

        da/dt = -a^T * ∂f/∂z(t), a(T) = ∂L/∂z(T)
        dL/dθ = -∫_{t0}^{t1} a(t)^T * ∂f/∂θ dt

    Args:
        d_state: Compressed state dimension from Layer 1
        d_hidden: Hidden dimension for liquid dynamics
        n_ode_steps: Number of Euler steps for ODE solve
        solver: ODE solver type ('euler', 'rk4', or 'adjoint')
        use_adjoint: Use adjoint method for memory-efficient gradients
    """

    def __init__(
        self,
        d_state: int = 2048,
        d_hidden: int = 7168,
        n_ode_steps: int = 8,
        solver: str = "rk4",
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_hidden = d_hidden
        self.n_ode_steps = n_ode_steps
        self.solver = solver
        self.use_adjoint = use_adjoint

        # Liquid cell
        self.cell = LTCCell(d_state=d_state, d_hidden=d_hidden)

        # Input projection from compressed state to liquid hidden
        self.input_proj = nn.Linear(d_state, d_hidden, bias=False)

        # Projection from compressed state to initial ODE state
        self.initial_proj = nn.Linear(d_state, d_hidden, bias=False)

        # Norm
        self.norm = nn.LayerNorm(d_hidden, eps=1e-6)

    def _ode_solve_euler(
        self,
        func: Callable,
        z0: torch.Tensor,       # [batch, d_hidden]
        x: torch.Tensor,        # [batch, d_state]
        t_span: torch.Tensor,   # [n_steps]
    ) -> List[torch.Tensor]:
        """Simple Euler solver for the ODE."""
        z = z0
        trajectory = [z]
        dt = t_span[1] - t_span[0]
        for i in range(1, len(t_span)):
            dz = func(t_span[i-1], z, x)
            z = z + dt * dz
            trajectory.append(z)
        return trajectory

    def _ode_solve_rk4(
        self,
        func: Callable,
        z0: torch.Tensor,       # [batch, d_hidden]
        x: torch.Tensor,        # [batch, d_state]
        t_span: torch.Tensor,   # [n_steps]
    ) -> List[torch.Tensor]:
        """Classical RK4 solver for higher accuracy."""
        z = z0
        trajectory = [z]
        dt = t_span[1] - t_span[0]
        for i in range(1, len(t_span)):
            t = t_span[i-1]
            k1 = func(t, z, x)
            k2 = func(t + 0.5 * dt, z + 0.5 * dt * k1, x)
            k3 = func(t + 0.5 * dt, z + 0.5 * dt * k2, x)
            k4 = func(t + dt, z + dt * k3, x)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(z)
        return trajectory

    def _ode_solve_adjoint(
        self,
        func: Callable,
        z0: torch.Tensor,       # [batch, d_hidden]
        x: torch.Tensor,        # [batch, d_state]
        t_span: torch.Tensor,   # [n_steps]
    ) -> List[torch.Tensor]:
        """
        ODE solve with adjoint sensitivity for memory-efficient gradients.

        The adjoint method recomputes the trajectory backward in time during
        backward pass, avoiding storage of all intermediate states.

        Implementation note: For production, replace with torchdiffeq's
        odeint_adjoint. Here we implement Euler solve + manual gradient
        checkpointing for the adjoint state.
        """
        # Forward pass with gradient checkpointing
        z = z0
        xs = [z]
        dt = t_span[1] - t_span[0]

        for i in range(1, len(t_span)):
            t = t_span[i-1]
            # Gradient checkpoint: don't store intermediates for all steps
            def _step(z_prev):
                dz = func(t, z_prev, x)
                return z_prev + dt * dz

            z = torch.utils.checkpoint.checkpoint(_step, z, use_reentrant=False)
            xs.append(z)

        return xs

    def forward(
        self,
        h_compressed: torch.Tensor,  # [batch, d_state] — O(1) compressed context
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        batch = h_compressed.shape[0]

        # Project compressed state
        x_proj = self.input_proj(h_compressed)  # [batch, d_hidden]
        z0 = self.initial_proj(h_compressed)    # [batch, d_hidden]

        # Time span for ODE integration
        t_span = torch.linspace(
            0, 1, self.n_ode_steps + 1, device=h_compressed.device, dtype=h_compressed.dtype
        )

        # Solve ODE
        if self.solver == "euler":
            trajectory = self._ode_solve_euler(self.cell.ode_func, z0, x_proj, t_span)
        elif self.solver == "rk4":
            trajectory = self._ode_solve_rk4(self.cell.ode_func, z0, x_proj, t_span)
        elif self.solver == "adjoint":
            trajectory = self._ode_solve_adjoint(self.cell.ode_func, z0, x_proj, t_span)
        else:
            trajectory = self._ode_solve_rk4(self.cell.ode_func, z0, x_proj, t_span)

        # Final state = last ODE state
        z_final = trajectory[-1]  # [batch, d_hidden]

        # Normalize
        z_final = self.norm(z_final)

        if return_trajectory:
            return z_final, trajectory

        return z_final, None
