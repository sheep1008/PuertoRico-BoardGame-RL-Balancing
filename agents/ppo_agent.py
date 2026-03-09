import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 200, hidden_dim: int = 256):
        super(Agent, self).__init__()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_mask, action=None):
        logits = self.actor(x)
        
        # Apply Action Masking
        # Mask shape should match logits (batch_size, action_dim)
        # We set invalid action logits to a very small number (~ -1e8)
        huge_negative = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask > 0.5, logits, huge_negative)
        
        probs = Categorical(logits=masked_logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
