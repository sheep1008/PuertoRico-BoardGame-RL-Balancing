import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """Pre-norm residual block: x + MLP(LayerNorm(x))"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Agent(nn.Module):
    """
    PPO Actor-Critic with shared ResidualMLP trunk.
    Architecture: Embed(LayerNorm+ReLU) -> 3x ResidualBlock -> separate Actor/Critic heads.
    """
    def __init__(self, obs_dim: int, action_dim: int = 200, hidden_dim: int = 512, num_res_blocks: int = 3):
        super(Agent, self).__init__()

        # Embedding layer normalizes raw observations of different scales
        self.embed = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Shared trunk with residual connections for stable deep gradient flow
        self.shared_trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Separate heads — actor uses small init std for initial uniform-ish policy
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _shared_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_trunk(self.embed(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._shared_features(x))

    def get_action_and_value(self, x: torch.Tensor, action_mask: torch.Tensor, action: torch.Tensor = None):
        features = self._shared_features(x)
        logits = self.actor_head(features)

        # Invalid actions get logits of -1e8 so softmax drives their probability to ~0
        huge_negative = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask > 0.5, logits, huge_negative)

        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic_head(features)


# ---------- Phase-to-Head mapping (Phase IntEnum → head key) ----------
# Phase: SETTLER=0, MAYOR=1, BUILDER=2, CRAFTSMAN=3, TRADER=4,
#         CAPTAIN=5, CAPTAIN_STORE=6, PROSPECTOR=7, END_ROUND=8
PHASE_TO_HEAD = {
    0: "settler",
    1: "mayor",
    2: "builder",
    3: "craftsman",
    4: "trader",
    5: "captain",
    6: "captain",       # CAPTAIN_STORE shares captain head
    7: "role_select",   # PROSPECTOR → only pass action
    8: "role_select",   # END_ROUND → role selection
}

# Adaptive hidden dimensions by decision complexity
HEAD_HIDDEN_DIMS = {
    "role_select": 512,   # Strategic role choice — most important
    "settler":     256,   # ~8 options (plantations + quarry + pass)
    "builder":     512,   # 23 buildings — complex long-term investment
    "mayor":       512,   # Combinatorial colonist placement, high data volume
    "craftsman":   128,   # 5 goods at most — simplest decision, fewest samples
    "trader":      256,   # ~6 options (5 goods + pass)
    "captain":     512,   # ship×good combos + wharf + store — multi-step
}

PHASE_EMBED_DIM = 16


class PhaseActorHead(nn.Module):
    """Single phase-specific actor head: LayerNorm → Linear → ReLU → Linear(→action_dim)"""
    def __init__(self, trunk_dim: int, head_hidden: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            layer_init(nn.Linear(trunk_dim, head_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(head_hidden, action_dim), std=0.01),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class HierarchicalAgent(nn.Module):
    """
    Hierarchical PPO Actor-Critic with phase-conditioned sub-policies.

    Architecture:
      obs(210) + phase_embed(16) → Embed(226→512) → 3× ResidualBlock
        → Phase-routed actor heads (7 heads, adaptive sizes)
        → Shared critic head (phase-aware via embedding)

    Paper contribution: Puerto Rico's natural phase structure provides
    option boundaries for hierarchical RL without manual option design.
    """
    NUM_PHASES = 9  # Phase IntEnum 0..8

    def __init__(self, obs_dim: int, action_dim: int = 200,
                 hidden_dim: int = 512, num_res_blocks: int = 3):
        super(HierarchicalAgent, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Phase embedding: small learnable vector per phase
        self.phase_embed = nn.Embedding(self.NUM_PHASES, PHASE_EMBED_DIM)

        # Embedding layer: obs + phase_embed → hidden_dim
        embed_input_dim = obs_dim + PHASE_EMBED_DIM
        self.embed = nn.Sequential(
            layer_init(nn.Linear(embed_input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Shared trunk with residual connections
        self.shared_trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Phase-specific actor heads with adaptive hidden dimensions
        self.phase_heads = nn.ModuleDict({
            name: PhaseActorHead(hidden_dim, head_hidden, action_dim)
            for name, head_hidden in HEAD_HIDDEN_DIMS.items()
        })

        # Shared critic head (phase-aware through trunk features)
        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _shared_features(self, x: torch.Tensor, phase_ids: torch.Tensor) -> torch.Tensor:
        """Embed observation concatenated with phase embedding, then pass through trunk."""
        phase_emb = self.phase_embed(phase_ids)                # (B, 16)
        combined = torch.cat([x, phase_emb], dim=-1)           # (B, obs_dim+16)
        return self.shared_trunk(self.embed(combined))          # (B, 512)

    def get_value(self, x: torch.Tensor, phase_ids: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._shared_features(x, phase_ids))

    def get_action_and_value(self, x: torch.Tensor, action_mask: torch.Tensor,
                             phase_ids: torch.Tensor, action: torch.Tensor = None):
        """
        Forward pass with phase-routed actor heads.

        Args:
            x: (B, obs_dim) observations
            action_mask: (B, action_dim) binary mask
            phase_ids: (B,) int tensor of Phase IntEnum values
            action: (B,) optional pre-selected actions (for PPO update)

        Returns:
            action, log_prob, entropy, value
        """
        features = self._shared_features(x, phase_ids)           # (B, 512)
        value = self.critic_head(features)                        # (B, 1)

        # Build full logits by routing each sample to its phase head
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.action_dim,
                             device=x.device, dtype=x.dtype)

        # Group indices by head to minimize forward passes (max 7 groups)
        head_groups: dict[str, list[int]] = {}
        for i in range(batch_size):
            pid = phase_ids[i].item()
            head_key = PHASE_TO_HEAD.get(pid, "role_select")
            head_groups.setdefault(head_key, []).append(i)

        for head_key, indices in head_groups.items():
            idx_tensor = torch.tensor(indices, device=x.device, dtype=torch.long)
            head_features = features[idx_tensor]                  # (G, 512)
            head_logits = self.phase_heads[head_key](head_features)  # (G, 200)
            logits[idx_tensor] = head_logits

        # Apply action mask
        huge_negative = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask > 0.5, logits, huge_negative)

        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value
