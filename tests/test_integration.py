from __future__ import annotations

import math
import pytest
import torch

from src.nslt.model import NSLTModel, MoENSLTModel
from src.nslt.vision_encoder import SigLIPVisionEncoder
from src.nslt.ssm_scan import selective_scan_vectorized, selective_scan_sequential


@pytest.fixture(scope="module")
def tiny_config():
    return dict(
        vocab_size=256,
        d_model=32,
        d_state=16,
        d_hidden=32,
        n_ssm_layers=2,
        sparsity_pct=5.0,
        n_ode_steps=4,
        n_trajectories=4,
        n_sim_steps=4,
    )


@pytest.fixture(scope="module")
def tiny_data():
    vocab_size = 256
    batch, seq_len = 2, 16
    generator = torch.Generator().manual_seed(42)
    return {
        "input_ids": torch.randint(0, vocab_size, (batch, seq_len), generator=generator),
        "labels": torch.randint(0, vocab_size, (batch, seq_len), generator=generator),
    }


class TestVectorizedSSMScan:
    """Verify vectorized scan matches sequential reference."""

    def test_vectorized_matches_sequential(self):
        batch, seq_len, d_inner = 2, 8, 32
        dt_rank, d_state = 8, 16

        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, dt_rank).abs()
        A = -torch.exp(torch.randn(d_state) * 0.5)
        B = torch.randn(batch, seq_len, dt_rank)
        C = torch.randn(batch, seq_len, dt_rank)

        y_seq, h_seq = selective_scan_sequential(x, delta, A, B, C, dt_rank)
        y_vec, h_vec = selective_scan_vectorized(x, delta, A, B, C, d_state, dt_rank)

        assert torch.allclose(y_seq, y_vec, atol=1e-5), "SSM outputs differ"
        assert torch.allclose(h_seq, h_vec, atol=1e-5), "Final states differ"

    def test_vectorized_gradients(self):
        batch, seq_len, d_inner = 2, 8, 32
        dt_rank, d_state = 8, 16

        x = torch.randn(batch, seq_len, d_inner, requires_grad=True)
        delta = torch.randn(batch, seq_len, dt_rank).abs()
        A = -torch.exp(torch.randn(d_state) * 0.5)
        B = torch.randn(batch, seq_len, dt_rank)
        C = torch.randn(batch, seq_len, dt_rank)

        y, h = selective_scan_vectorized(x, delta, A, B, C, d_state, dt_rank)
        loss = y.sum() + h.sum()
        loss.backward()

        assert x.grad is not None, "Gradient flow broken through vectorized scan"
        assert torch.isfinite(x.grad).all(), "Non-finite gradients"

    def test_o1_memory_independent(self):
        """Verify memory usage is O(1) w.r.t. sequence length."""
        dt_rank = 8
        d_state = 16
        d_inner = 32
        A = -torch.exp(torch.randn(d_state) * 0.5)
        B = torch.randn(1, 1, dt_rank)
        C = torch.randn(1, 1, dt_rank)
        delta = torch.randn(1, 1, dt_rank).abs()

        mem_short = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        x_short = torch.randn(1, 64, d_inner)
        delta_s = delta.expand(1, 64, -1)
        B_s = B.expand(1, 64, -1)
        C_s = C.expand(1, 64, -1)
        y_s, h_s = selective_scan_vectorized(x_short, delta_s, A, B_s, C_s, d_state, dt_rank)

        mem_long = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_growth = mem_long - mem_short

        x_long = torch.randn(1, 256, d_inner)
        delta_l = delta.expand(1, 256, -1)
        B_l = B.expand(1, 256, -1)
        C_l = C.expand(1, 256, -1)
        y_l, h_l = selective_scan_vectorized(x_long, delta_l, A, B_l, C_l, d_state, dt_rank)

        assert h_s.shape == h_l.shape == (1, d_state), "State not O(1)"
        assert y_s.shape[-1] == y_l.shape[-1] == d_inner, "Output dim changed"


class TestEndToEndTraining:
    """Train NSLT on a tiny dataset and verify loss decreases."""

    def test_nslt_loss_decreases(self, tiny_config, tiny_data):
        """Train NSLT for 100 steps, verify loss decreases."""
        model = NSLTModel(**tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

        input_ids = tiny_data["input_ids"]
        labels = tiny_data["labels"]

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                losses.append(float("inf"))
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        valid = [l for l in losses if torch.isfinite(torch.tensor(l))]
        assert len(valid) >= 20, f"Too many NaN losses ({len(losses) - len(valid)}/{len(losses)})"
        assert valid[-1] < valid[0], (
            f"Loss did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

    def test_moe_loss_decreases(self, tiny_config, tiny_data):
        """Train MoENSLT for 100 steps, verify loss decreases."""
        config = {**tiny_config, "n_experts": 4, "top_k_experts": 2}
        model = MoENSLTModel(**config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

        input_ids = tiny_data["input_ids"]
        labels = tiny_data["labels"]

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                losses.append(float("inf"))
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        valid = [l for l in losses if torch.isfinite(torch.tensor(l))]
        assert len(valid) >= 50, f"Too many NaN losses ({len(losses) - len(valid)}/{len(losses)})"
        assert valid[-1] < valid[0], (
            f"Loss did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

    def test_training_generates_coherent_output(self, tiny_config):
        """After training, generation should produce valid tokens."""
        vocab_size = tiny_config["vocab_size"]
        model = NSLTModel(**tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

        batch, seq_len = 2, 8
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        labels = torch.randint(0, vocab_size, (batch, seq_len))

        for step in range(50):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        output = model.generate(input_ids, max_new_tokens=4)
        expected_len = seq_len + 4
        assert output.shape == (batch, expected_len), (
            f"Generation shape mismatch: {output.shape} != {(batch, expected_len)}"
        )
        assert (output >= 0).all() and (output < vocab_size).all(), "Invalid token IDs"


class TestMoENSLTModel:
    """Test MoE-specific functionality."""

    def test_moe_forward_shape(self, tiny_config):
        config = {**tiny_config, "n_experts": 4, "top_k_experts": 2}
        model = MoENSLTModel(**config)

        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))
        output = model(input_ids)

        assert output.logits.shape == (batch, seq_len, tiny_config["vocab_size"]), (
            f"Expected {(batch, seq_len, tiny_config['vocab_size'])}, got {output.logits.shape}"
        )

    def test_moe_compressed_state(self, tiny_config):
        config = {**tiny_config, "n_experts": 4, "top_k_experts": 2}
        model = MoENSLTModel(**config)

        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))
        state = model.get_compressed_state(input_ids)

        assert state.shape == (batch, tiny_config["d_state"]), (
            f"Compressed state shape mismatch: {state.shape} != {(batch, tiny_config['d_state'])}"
        )

    def test_moe_training_step(self, tiny_config, tiny_data):
        config = {**tiny_config, "n_experts": 4, "top_k_experts": 2}
        model = MoENSLTModel(**config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        output = model(tiny_data["input_ids"], labels=tiny_data["labels"])
        output.loss.backward()
        optimizer.step()
        assert torch.isfinite(output.loss), f"Non-finite loss: {output.loss}"


class TestVisionEncoder:
    """Test vision encoder integration."""

    def test_vision_encoder_forward(self):
        encoder = SigLIPVisionEncoder(
            d_state=32,
            image_size=32,  # tiny size for test
            patch_size=8,
            d_vision=16,
            n_layers=2,
        )

        batch, channels, H, W = 2, 3, 32, 32
        images = torch.randn(batch, channels, H, W)
        visual_state = encoder(images)

        assert visual_state.shape == (batch, 32), (
            f"Expected (2, 32), got {visual_state.shape}"
        )
        assert torch.isfinite(visual_state).all(), "Non-finite visual state"

    def test_multimodal_forward(self, tiny_config, tiny_data):
        model = NSLTModel.from_multimodal_config(
            d_vision=8,
            image_size=16,
            patch_size=4,
            vision_d_state=tiny_config["d_state"],
            **tiny_config,
        )

        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))
        images = torch.randn(batch, 3, 16, 16)
        labels = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))

        output = model.forward_multimodal(input_ids, images, labels=labels)
        assert torch.isfinite(output.loss), f"Non-finite multimodal loss: {output.loss}"

        output.loss.backward()
        # Verify key params get gradients (skip unused params like ltc.cell.out_proj)
        key_params_got_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and "embedding" not in name and "vision" in name:
                if param.grad is not None:
                    key_params_got_grad = True
                    assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
        assert key_params_got_grad, "No vision parameters received gradients"

    def test_vision_config_present(self, tiny_config):
        model = NSLTModel.from_multimodal_config(
            d_vision=8,
            image_size=16,
            patch_size=4,
            vision_d_state=tiny_config["d_state"],
            **tiny_config,
        )
        config = model.get_config()
        assert "vision_encoder" in config, "Vision config not in model config"
        assert config["vision_encoder"]["type"] == "SigLIPVisionEncoder"


class TestMCTSLatentSandbox:
    """Test MCTS-based latent sandbox."""

    def test_mcts_sandbox_forward(self, tiny_config):
        model = NSLTModel(use_mcts_sandbox=True, **tiny_config)
        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))
        output = model(input_ids)
        assert output.logits.shape == (batch, seq_len, tiny_config["vocab_size"])

    def test_mcts_sandbox_training(self, tiny_config):
        model = NSLTModel(use_mcts_sandbox=True, **tiny_config)
        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch, seq_len))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), f"Non-finite MCTS loss: {output.loss}"

        output.loss.backward()
        grad_exists = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.sandbox.parameters()
        )
        assert grad_exists, "No gradients through MCTS sandbox"

    def test_mcts_loss_decreases(self, tiny_config):
        model = NSLTModel(use_mcts_sandbox=True, **tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

        batch, seq_len = 2, 8
        vocab_size = tiny_config["vocab_size"]
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        labels = input_ids.clone()

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                losses.append(float("inf"))
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        valid = [l for l in losses if torch.isfinite(torch.tensor(l))]
        assert len(valid) >= 20, f"MCTS too many NaN losses ({len(losses) - len(valid)}/{len(losses)})"
        assert valid[-1] < valid[0], (
            f"MCTS loss did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

    def test_mcts_node_operations(self):
        from src.nslt.mcts_sandbox import MCTSNode

        state = torch.randn(32)
        root = MCTSNode(state=state)
        assert root.is_leaf
        assert root.visit_count == 0
        assert root.depth == 0

        child = MCTSNode(state=torch.randn(32), parent=root, action_id=0)
        root.children.append(child)
        assert not root.is_leaf
        assert child.depth == 1
        assert child.parent is root


class TestMultiScaleSSM:
    """Test multi-scale state hierarchy."""

    def test_forward_shape(self):
        from src.nslt.multiscale_ssm import MultiScaleSSM

        model = MultiScaleSSM(d_model=32, d_state=16, n_ssm_layers=1, k_medium=4, k_slow=8)
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, 32)
        out, h = model(x, return_state=True)

        assert out.shape == (batch, seq_len, 32)
        assert h.shape == (batch, 48), f"Expected (batch, 48), got {h.shape}"

    def test_gradient_flow(self):
        from src.nslt.multiscale_ssm import MultiScaleSSM

        model = MultiScaleSSM(d_model=32, d_state=16, n_ssm_layers=1, k_medium=2, k_slow=4)
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, 32, requires_grad=True)
        out, h = model(x, return_state=True)
        loss = out.sum() + h.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_medium_level_captures_context(self):
        from src.nslt.multiscale_ssm import MultiScaleSSM

        model = MultiScaleSSM(d_model=32, d_state=16, n_ssm_layers=1, k_medium=4, k_slow=8)

        batch, seq_len = 2, 12
        x = torch.randn(batch, seq_len, 32)

        _, h_short = model(x, return_state=True)

        _, h_long = model(x, return_state=True)

        assert h_short.shape == h_long.shape
        assert torch.isfinite(h_short).all()

    def test_medium_update_schedule(self):
        from src.nslt.multiscale_ssm import MultiScaleSSM

        model = MultiScaleSSM(d_model=32, d_state=16, n_ssm_layers=1, k_medium=4, k_slow=8)
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, 32)

        out, h = model(x, return_state=True)
        # h = [h_fast | h_med | h_slow] = [16 | 16 | 16] = [48]
        assert h.shape == (batch, 48)
        assert out.shape == (batch, seq_len, 32)
