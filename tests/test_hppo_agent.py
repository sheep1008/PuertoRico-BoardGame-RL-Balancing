import unittest
import torch
import numpy as np
from agents.ppo_agent import HierarchicalAgent, PHASE_TO_HEAD, HEAD_HIDDEN_DIMS


class TestHierarchicalAgent(unittest.TestCase):
    """Verify HierarchicalAgent shapes, phase routing, and parameter count."""

    def setUp(self):
        self.obs_dim = 210
        self.action_dim = 200
        self.agent = HierarchicalAgent(obs_dim=self.obs_dim, action_dim=self.action_dim)

    def test_output_shapes(self):
        """All outputs have correct shapes for a mixed-phase batch."""
        batch = 4
        obs = torch.randn(batch, self.obs_dim)
        mask = torch.ones(batch, self.action_dim)
        phase_ids = torch.tensor([0, 2, 5, 8])  # Settler, Builder, Captain, EndRound

        action, logprob, entropy, value = self.agent.get_action_and_value(obs, mask, phase_ids)

        self.assertEqual(action.shape, (batch,))
        self.assertEqual(logprob.shape, (batch,))
        self.assertEqual(entropy.shape, (batch,))
        self.assertEqual(value.shape, (batch, 1))

    def test_single_sample_all_phases(self):
        """Single sample works for every phase ID 0..8."""
        obs = torch.randn(1, self.obs_dim)
        mask = torch.ones(1, self.action_dim)

        for phase_id in range(9):
            phase = torch.tensor([phase_id])
            action, logprob, entropy, value = self.agent.get_action_and_value(obs, mask, phase)
            self.assertEqual(action.shape, (1,))
            self.assertFalse(torch.isnan(logprob).any(), f"NaN logprob at phase {phase_id}")
            self.assertFalse(torch.isnan(value).any(), f"NaN value at phase {phase_id}")

    def test_get_value_shape(self):
        """get_value returns (B, 1) tensor."""
        obs = torch.randn(8, self.obs_dim)
        phases = torch.randint(0, 9, (8,))
        value = self.agent.get_value(obs, phases)
        self.assertEqual(value.shape, (8, 1))

    def test_different_heads_produce_different_logits(self):
        """Different phases route to different heads → different logits."""
        obs = torch.randn(1, self.obs_dim)
        phase_settler = torch.tensor([0])
        phase_captain = torch.tensor([5])

        with torch.no_grad():
            features_s = self.agent._shared_features(obs, phase_settler)
            features_c = self.agent._shared_features(obs, phase_captain)

            logits_settler = self.agent.phase_heads["settler"](features_s)
            logits_captain = self.agent.phase_heads["captain"](features_c)

        # Different heads + different phase embeddings → different logits
        self.assertFalse(torch.allclose(logits_settler, logits_captain),
                         "Settler and Captain heads should produce different logits")

    def test_action_masking(self):
        """Invalid actions (mask=0) should never be selected."""
        obs = torch.randn(100, self.obs_dim)
        mask = torch.zeros(100, self.action_dim)
        mask[:, 15] = 1  # Only action 15 (pass) is valid
        phases = torch.full((100,), 7)  # Prospector

        with torch.no_grad():
            actions, _, _, _ = self.agent.get_action_and_value(obs, mask, phases)

        self.assertTrue((actions == 15).all(),
                         "All actions should be 15 when only pass is valid")

    def test_parameter_count(self):
        """Total parameters should be approximately 3.4M (significantly more than flat 2.3M)."""
        total = sum(p.numel() for p in self.agent.parameters())
        print(f"  HierarchicalAgent params: {total:,}")
        self.assertGreater(total, 2_500_000, "Should have more params than flat agent")
        self.assertLess(total, 5_000_000, "Should not be unreasonably large")

    def test_all_heads_exist(self):
        """All expected phase heads are present."""
        expected_heads = set(HEAD_HIDDEN_DIMS.keys())
        actual_heads = set(self.agent.phase_heads.keys())
        self.assertEqual(expected_heads, actual_heads)

    def test_phase_to_head_coverage(self):
        """Every phase ID (0-8) maps to a valid head key."""
        for phase_id in range(9):
            head_key = PHASE_TO_HEAD[phase_id]
            self.assertIn(head_key, self.agent.phase_heads,
                          f"Phase {phase_id} maps to unknown head '{head_key}'")

    def test_backward_pass(self):
        """Gradient flows correctly through all heads."""
        obs = torch.randn(8, self.obs_dim)
        mask = torch.ones(8, self.action_dim)
        phases = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8])  # All different phases

        action, logprob, entropy, value = self.agent.get_action_and_value(obs, mask, phases)
        loss = -logprob.mean() + value.mean()
        loss.backward()

        # Check that all phase heads received gradients
        heads_with_grad = set()
        for head_key, head in self.agent.phase_heads.items():
            for p in head.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    heads_with_grad.add(head_key)
                    break

        # At least the heads used in this batch should have gradients
        expected_used = {"settler", "mayor", "builder", "craftsman",
                         "trader", "captain", "role_select"}
        self.assertEqual(heads_with_grad, expected_used,
                         f"Missing grad for heads: {expected_used - heads_with_grad}")


if __name__ == "__main__":
    unittest.main()
