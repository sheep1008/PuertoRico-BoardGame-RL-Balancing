import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent

# Hyperparameters
NUM_PLAYERS = 3
TOTAL_TIMESTEPS = 10000000
LEARNING_RATE = 2.5e-4
NUM_STEPS = 1000 # Steps per PPO rollout phase
BATCH_SIZE = NUM_STEPS * NUM_PLAYERS
MINIBATCH_SIZE = 125
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    return PuertoRicoEnv(num_players=NUM_PLAYERS)

def train():
    run_name = f"PPO_PuertoRico_{NUM_PLAYERS}P_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    env = make_env()
    env.reset()
    
    obs_space = env.observation_space(env.possible_agents[0])['observation']
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = env.action_space(env.possible_agents[0]).n
    
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Replay Buffers
    obs_buffers = {p: torch.zeros((NUM_STEPS, obs_dim)).to(DEVICE) for p in env.possible_agents}
    mask_buffers = {p: torch.zeros((NUM_STEPS, action_dim)).to(DEVICE) for p in env.possible_agents}
    action_buffers = {p: torch.zeros((NUM_STEPS,)).to(DEVICE) for p in env.possible_agents}
    logprob_buffers = {p: torch.zeros((NUM_STEPS,)).to(DEVICE) for p in env.possible_agents}
    reward_buffers = {p: torch.zeros((NUM_STEPS,)).to(DEVICE) for p in env.possible_agents}
    done_buffers = {p: torch.zeros((NUM_STEPS,)).to(DEVICE) for p in env.possible_agents}
    value_buffers = {p: torch.zeros((NUM_STEPS,)).to(DEVICE) for p in env.possible_agents}

    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    
    print(f"Starting Training on {DEVICE}. Total Updates: {num_updates}")
    
    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow
        
        step_count = 0
        total_steps = NUM_STEPS * len(env.possible_agents)
        idx_count = {p: 0 for p in env.possible_agents}
        
        while step_count < total_steps:
            # We must instantiate agent_iter repeatedly because it exhausts when game ends
            for agent_name in env.agent_iter():
                idx = idx_count[agent_name]
                
                # If buffer is full, we must still step to not break PettingZoo, but we discard data.
                obs, reward, termination, truncation, info = env.last()
                
                if idx > 0 and idx <= NUM_STEPS:
                    reward_buffers[agent_name][idx - 1] = reward
                    
                if termination or truncation:
                    if idx < NUM_STEPS:
                        done_buffers[agent_name][idx] = 1.0
                    env.step(None)
                    
                    # If all players are dead, PettingZoo iterator will naturally exhaust on this turn.
                    # We just log the score BEFORE resetting.
                    if all(env.terminations.values()): 
                        for p in env.possible_agents:
                            if p in env.infos and "player_rewards" in env.infos[p]:
                                writer.add_scalar(f"charts/episodic_return_{p}", env.infos[p]["player_rewards"], global_step)
                        env.reset()
                    continue
                    
                # Normal Step
                obs_dict = obs["observation"]
                mask = obs["action_mask"]
                
                flat_obs = flatten_dict_observation(obs_dict, obs_space)
                obs_tensor = torch.Tensor(flat_obs).to(DEVICE).unsqueeze(0)
                mask_tensor = torch.Tensor(mask).to(DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    action_sample, logprob, _, value = agent.get_action_and_value(obs_tensor, mask_tensor)
                    
                action_idx = action_sample.item()
                
                if idx < NUM_STEPS:
                    obs_buffers[agent_name][idx] = obs_tensor.squeeze(0)
                    mask_buffers[agent_name][idx] = mask_tensor.squeeze(0)
                    action_buffers[agent_name][idx] = action_sample.squeeze(0)
                    logprob_buffers[agent_name][idx] = logprob.squeeze(0)
                    value_buffers[agent_name][idx] = value.squeeze(0)
                    done_buffers[agent_name][idx] = 0.0
                    
                    idx_count[agent_name] += 1
                    step_count += 1
                    
                env.step(action_idx)
                
                if step_count >= total_steps:
                    break
                    
            # Iterator exhausted (Game ended). Reset environment and start next loop seamlessly.
            if step_count < total_steps:
                env.reset()
                
        print(f"Update {update}/{num_updates} - Data Collection Complete")
        
        # --- GAE Advantage Calculation & Actor-Critic Updates ---
        b_obs = torch.cat(list(obs_buffers.values()), dim=0)
        b_masks = torch.cat(list(mask_buffers.values()), dim=0)
        b_actions = torch.cat(list(action_buffers.values()), dim=0)
        b_logprobs = torch.cat(list(logprob_buffers.values()), dim=0)
        b_rewards = torch.cat(list(reward_buffers.values()), dim=0)
        b_dones = torch.cat(list(done_buffers.values()), dim=0)
        b_values = torch.cat(list(value_buffers.values()), dim=0)
        
        b_advantages = torch.zeros_like(b_rewards).to(DEVICE)
        
        # We calculate simplistic advantage since PettingZoo multi-agent tracking 
        # requires complex graph unwrapping for precise next-state value mapping.
        # For this baseline, we use Monte-Carlo style returns approximated per agent buffer.
        for p in env.possible_agents:
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - done_buffers[p][t]
                    nextvalues = 0 # Approximate
                else:
                    nextnonterminal = 1.0 - done_buffers[p][t + 1]
                    nextvalues = value_buffers[p][t + 1]
                    
                delta = reward_buffers[p][t] + GAMMA * nextvalues * nextnonterminal - value_buffers[p][t]
                b_advantages_p = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                lastgaelam = b_advantages_p
                # Assign back to flattened buffer
                idx_offset = env.possible_agents.index(p) * NUM_STEPS
                b_advantages[idx_offset + t] = b_advantages_p
        
        b_returns = b_advantages + b_values
        
        # Optimize Policy and Value networks
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue.squeeze(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        
        print(f"Update {update}/{num_updates} | Value Loss: {v_loss.item():.4f} | Policy Loss: {pg_loss.item():.4f}")
        
        if update % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(), f"models/ppo_agent_update_{update}.pth")

if __name__ == "__main__":
    train()
