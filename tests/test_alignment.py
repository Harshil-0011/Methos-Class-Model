from __future__ import annotations

import torch

from src.alignment.dpo_trainer import DPOTrainer, ORPOTrainer, SimPOTrainer


def _dummy_logps(batch_size: int = 4) -> tuple:
    return (
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
    )


class TestDPO:
    def test_dpo_loss_shape(self):
        chosen_lps = torch.tensor([-2.0, -1.0, -3.0, -0.5])
        rejected_lps = torch.tensor([-4.0, -3.0, -5.0, -2.0])
        ref_chosen = torch.tensor([-2.5, -1.5, -3.5, -1.0])
        ref_rejected = torch.tensor([-3.5, -2.5, -4.5, -1.5])

        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.beta = 0.1

        loss, chosen_reward, rejected_reward = trainer.dpo_loss(
            chosen_lps, rejected_lps, ref_chosen, ref_rejected,
        )
        assert loss.ndim == 0
        assert loss > 0
        assert chosen_reward > rejected_reward

    def test_dpo_prefers_chosen(self):
        chosen_lps = torch.tensor([-1.0, -0.5])
        rejected_lps = torch.tensor([-5.0, -4.0])
        ref_chosen = torch.tensor([-1.5, -1.0])
        ref_rejected = torch.tensor([-4.5, -3.5])

        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.beta = 0.1
        loss, _, _ = trainer.dpo_loss(chosen_lps, rejected_lps, ref_chosen, ref_rejected)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestORPO:
    def test_orpo_loss_lower_for_preferred(self):
        trainer = ORPOTrainer.__new__(ORPOTrainer)
        trainer.beta = 0.05
        preferred = torch.tensor([-1.0])
        dispreferred = torch.tensor([-5.0])
        loss_good = trainer.orpo_loss(preferred, dispreferred)

        preferred_bad = torch.tensor([-5.0])
        dispreferred_bad = torch.tensor([-1.0])
        loss_bad = trainer.orpo_loss(preferred_bad, dispreferred_bad)
        assert loss_good < loss_bad

    def test_orpo_loss_finite(self):
        trainer = ORPOTrainer.__new__(ORPOTrainer)
        trainer.beta = 0.05
        loss = trainer.orpo_loss(torch.tensor([-2.0, -1.0]), torch.tensor([-4.0, -3.0]))
        assert torch.isfinite(loss).all()


class TestSimPO:
    def test_simpo_loss_lower_for_preferred(self):
        trainer = SimPOTrainer.__new__(SimPOTrainer)
        trainer.gamma = 0.5
        trainer.beta = 2.0
        preferred = torch.tensor([-1.0])
        dispreferred = torch.tensor([-5.0])
        loss_good = trainer.simpo_loss(preferred, dispreferred)

        preferred_bad = torch.tensor([-5.0])
        dispreferred_bad = torch.tensor([-1.0])
        loss_bad = trainer.simpo_loss(preferred_bad, dispreferred_bad)
        assert loss_good < loss_bad

    def test_simpo_loss_finite(self):
        trainer = SimPOTrainer.__new__(SimPOTrainer)
        trainer.gamma = 0.5
        trainer.beta = 2.0
        loss = trainer.simpo_loss(torch.tensor([-2.0]), torch.tensor([-4.0]))
        assert torch.isfinite(loss)
