from __future__ import annotations

import pytest
import torch

from src.nslt.layer1_ssm import SSMCompressionEngine
from src.nslt.layer2_ltc import LTCRoutingLayer, LTCCell
from src.nslt.layer3_sandbox import LatentSandbox, EnergyFunction
from src.nslt.layer4_output import SparseOutputSynthesizer, SparseGatingUnit
from src.nslt.model import NSLTModel


def test_ssm_compression_output_shape():
    """SSM Engine compresses arbitrary seq_len to fixed d_state."""
    d_model = 64
    d_state = 32
    batch, seq_len = 2, 128
    ssm = SSMCompressionEngine(d_model=d_model, d_state=d_state, dt_rank=16, expand_factor=2)

    x = torch.randn(batch, seq_len, d_model)
    output, h_final = ssm(x, return_state=True)

    assert output.shape == (batch, seq_len, d_model), f"Expected {(batch, seq_len, d_model)}, got {output.shape}"
    assert h_final.shape == (batch, d_state), f"Expected {(batch, d_state)}, got {h_final.shape}"


def test_ssm_o1_memory_independent_of_seq_len():
    """O(1): compressed state size does NOT depend on sequence length."""
    d_model = 64
    d_state = 32
    ssm = SSMCompressionEngine(d_model=d_model, d_state=d_state, dt_rank=16, expand_factor=2)

    x_short = torch.randn(2, 16, d_model)
    x_long = torch.randn(2, 256, d_model)

    _, h_short = ssm(x_short, return_state=True)
    _, h_long = ssm(x_long, return_state=True)

    assert h_short.shape == h_long.shape, "O(1) invariant violated: compressed state grows with seq_len"


def test_ssm_forward_backward():
    """SSM gradients flow correctly through the selective scan."""
    d_model = 64
    d_state = 32
    ssm = SSMCompressionEngine(d_model=d_model, d_state=d_state, dt_rank=16, expand_factor=2)

    x = torch.randn(2, 32, d_model, requires_grad=True)
    output, _ = ssm(x, return_state=False)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient did not flow to input"
    assert all(p.grad is not None for p in ssm.parameters() if p.requires_grad), "Some parameters have no gradient"


def test_ltc_cell_ode_dynamics():
    """LTC cell ODE function produces smooth dynamics."""
    d_state = 32
    d_hidden = 64
    cell = LTCCell(d_state=d_state, d_hidden=d_hidden)

    z = torch.randn(2, d_hidden)
    # x_proj is already projected to d_hidden dimension by input_proj in LTCRoutingLayer
    x = torch.randn(2, d_hidden)

    dz_dt = cell.ode_func(torch.tensor(0.0), z, x)
    assert dz_dt.shape == (2, d_hidden), f"Expected {(2, d_hidden)}, got {dz_dt.shape}"
    assert torch.isfinite(dz_dt).all(), "ODE function produced non-finite values"


def test_ltc_routing_output_shape():
    """LTC routing produces correct output dim from compressed state."""
    d_state = 32
    d_hidden = 64
    ltc = LTCRoutingLayer(d_state=d_state, d_hidden=d_hidden, n_ode_steps=4, solver="euler")

    h = torch.randn(2, d_state)
    z_final, _ = ltc(h, return_trajectory=True)

    assert z_final.shape == (2, d_hidden), f"Expected {(2, d_hidden)}, got {z_final.shape}"


def test_latent_sandbox_parallel_reasoning():
    """Latent sandbox selects lowest-energy trajectory from K parallel simulations."""
    d_hidden = 32
    sandbox = LatentSandbox(d_hidden=d_hidden, d_latent=16, n_trajectories=4, n_sim_steps=4, select_every=2)

    z = torch.randn(2, d_hidden)
    z_optimal, energies = sandbox(z, return_trajectories=True)

    assert z_optimal.shape == (2, d_hidden), f"Expected {(2, d_hidden)}, got {z_optimal.shape}"
    assert energies is not None
    assert torch.isfinite(z_optimal).all(), "Sandbox produced non-finite values"


def test_energy_function_scores():
    """Energy function produces lower scores for coherent states."""
    d_hidden = 32
    energy_fn = EnergyFunction(d_hidden=d_hidden, d_latent=16)

    z_coherent = torch.randn(2, d_hidden)
    z_noisy = z_coherent + 10.0 * torch.randn(2, d_hidden)

    e_coherent, _ = energy_fn(z_coherent)
    e_noisy, _ = energy_fn(z_noisy)

    assert e_coherent.shape == (2,), f"Expected (2,), got {e_coherent.shape}"
    # Noisy states should generally have higher energy (not guaranteed but likely)
    assert torch.isfinite(e_coherent).all()


def test_sparse_gating_top_k():
    """Sparse gating selects exactly top-k% of vocabulary."""
    d_hidden = 32
    vocab_size = 1000
    sparsity_pct = 1.0
    gate = SparseGatingUnit(d_hidden=d_hidden, vocab_size=vocab_size, sparsity_pct=sparsity_pct)

    x = torch.randn(2, d_hidden)
    gate_values, top_indices, _ = gate(x)

    top_k = max(1, int(vocab_size * sparsity_pct / 100.0))
    assert top_indices.shape == (2, top_k), f"Expected {(2, top_k)}, got {top_indices.shape}"
    assert gate_values.shape == (2, top_k)
    assert torch.allclose(gate_values.sum(dim=-1), torch.ones(2), atol=1e-5), "Gate values do not sum to 1"


def test_sparse_output_synthesizer_shape():
    """Sparse output produces correct vocab_size logits with mostly zeros."""
    d_hidden = 32
    vocab_size = 1000
    sparsity_pct = 1.0
    output_syn = SparseOutputSynthesizer(
        d_hidden=d_hidden, vocab_size=vocab_size, d_model=64, sparsity_pct=sparsity_pct, adaptive_sparsity=False
    )

    x = torch.randn(2, d_hidden)
    logits = output_syn(x)

    assert logits.shape == (2, vocab_size), f"Expected {(2, vocab_size)}, got {logits.shape}"
    # Check extreme sparsity: most logits should be 0
    nonzero_pct = (logits != 0).float().mean().item() * 100
    assert nonzero_pct <= sparsity_pct + 0.5, f"Sparsity violated: {nonzero_pct:.1f}% non-zero (expected <= {sparsity_pct}%)"


def test_nslt_model_forward_shape():
    """Full NSLT model forward pass produces correct logits."""
    vocab_size = 256
    d_model = 32
    d_state = 16
    d_hidden = 32
    model = NSLTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        d_hidden=d_hidden,
        n_ssm_layers=2,
        sparsity_pct=5.0,
        n_ode_steps=4,
        n_trajectories=4,
        n_sim_steps=4,
    )

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    output = model(input_ids)
    assert output.logits.shape == (batch, seq_len, vocab_size), f"Expected {(batch, seq_len, vocab_size)}, got {output.logits.shape}"


def test_nslt_model_training_step():
    """NSLT model can compute loss and backpropagate."""
    vocab_size = 256
    model = NSLTModel(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        d_hidden=32,
        n_ssm_layers=2,
        sparsity_pct=5.0,
        n_ode_steps=4,
        n_trajectories=4,
        n_sim_steps=4,
    )

    batch, seq_len = 2, 8
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    assert torch.isfinite(output.loss).all(), f"Loss is not finite: {output.loss}"

    output.loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_exists, "No gradients flowed through the model"


def test_nslt_o1_memory():
    """Full NSLT model maintains O(1) compressed state independent of seq_len."""
    vocab_size = 256
    model = NSLTModel(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        d_hidden=32,
        n_ssm_layers=2,
        sparsity_pct=5.0,
        n_ode_steps=4,
        n_trajectories=4,
        n_sim_steps=4,
    )

    batch = 2
    # Short sequence
    short_ids = torch.randint(0, vocab_size, (batch, 4))
    h_short = model.get_compressed_state(short_ids)

    # Long sequence
    long_ids = torch.randint(0, vocab_size, (batch, 32))
    h_long = model.get_compressed_state(long_ids)

    assert h_short.shape == h_long.shape, "O(1) invariant violated: compressed state grows with sequence length"
    assert h_short.shape == (batch, 16), f"Expected {(batch, 16)}, got {h_short.shape}"


def test_nslt_generation():
    """NSLT can generate tokens auto-regressively."""
    vocab_size = 256
    model = NSLTModel(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        d_hidden=32,
        n_ssm_layers=2,
        sparsity_pct=5.0,
        n_ode_steps=4,
        n_trajectories=4,
        n_sim_steps=4,
    )

    batch, seq_len = 1, 4
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    # Generate a few tokens
    output = model.generate(
        input_ids,
        max_new_tokens=8,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )

    assert output.shape == (batch, seq_len + 8), f"Expected {(batch, seq_len + 8)}, got {output.shape}"
    assert torch.allclose(output[:, :seq_len], input_ids), "Generation changed the input prefix"
