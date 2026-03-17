import os
import copy
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter

from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import HierarchicalAgent, PHASE_TO_HEAD

# --- Hyperparameters ---
_TEST_MODE = os.environ.get("PPO_TEST_MODE", "0") == "1"

NUM_PLAYERS = 3
TOTAL_TIMESTEPS = 500_000 if _TEST_MODE else 10_000_000
LEARNING_RATE = 2.5e-4
NUM_STEPS = 500 if _TEST_MODE else 4096  # Steps per PPO rollout (learning player only)
BATCH_SIZE = NUM_STEPS
MINIBATCH_SIZE = 128 if _TEST_MODE else 256
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Self-play opponent pool settings
SNAPSHOT_INTERVAL = 10
OPPONENT_POOL_SIZE = 20
LATEST_POLICY_PROB = 0.8
LEARNING_PLAYER_IDX = 0


def make_env():
    return PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=2000)


def sample_opponent_weights(opponent_pool: list, agent: HierarchicalAgent) -> dict:
    """Sample opponent weights: 80% latest policy, 20% from historical pool."""
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return agent.state_dict()
    return random.choice(opponent_pool)


def extract_phase_id(obs_dict) -> int:
    """Extract current_phase int from the raw observation dict."""
    return int(obs_dict["global_state"]["current_phase"])


def collect_rollout(env, agent, opponent_agent, opponent_pool,
                    obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf,
                    phase_buf, obs_space, obs_dim, writer, global_step):
    """Collect NUM_STEPS transitions for the learning player.
    Now also records phase_id for hierarchical head routing.
    """
    learning_agent_name = f"player_{LEARNING_PLAYER_IDX}"
    step_idx = 0
    games_completed = 0
    win_count = 0
    total_score = 0.0
    game_lengths = []
    current_game_steps = 0

    # Load opponent weights at game start
    opp_weights = sample_opponent_weights(opponent_pool, agent)
    opponent_agent.load_state_dict(opp_weights)
    opponent_agent.eval()

    while step_idx < NUM_STEPS:
        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            player_idx = int(agent_name.split("_")[1])
            is_learner = (player_idx == LEARNING_PLAYER_IDX)

            # Record reward from previous step for the learning player
            if is_learner and step_idx > 0 and step_idx <= NUM_STEPS:
                rew_buf[step_idx - 1] = reward

            if termination or truncation:
                if is_learner and step_idx < NUM_STEPS:
                    done_buf[step_idx] = 1.0
                env.step(None)

                if all(env.terminations.values()):
                    games_completed += 1
                    current_game_steps_copy = current_game_steps

                    if learning_agent_name in env.infos and "final_scores" in env.infos[learning_agent_name]:
                        scores = env.infos[learning_agent_name]["final_scores"]
                        learner_score = scores[LEARNING_PLAYER_IDX][0]
                        max_opp_score = max(scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX)
                        total_score += learner_score
                        if learner_score >= max_opp_score:
                            win_count += 1
                        writer.add_scalar("charts/episodic_score", learner_score, global_step)
                        game_lengths.append(current_game_steps_copy)

                    env.reset()
                    current_game_steps = 0
                    opp_weights = sample_opponent_weights(opponent_pool, agent)
                    opponent_agent.load_state_dict(opp_weights)
                    opponent_agent.eval()
                continue

            # Prepare observation
            obs_dict = obs["observation"]
            mask = obs["action_mask"]
            flat_obs = flatten_dict_observation(obs_dict, obs_space)
            obs_tensor = torch.Tensor(flat_obs).to(DEVICE).unsqueeze(0)
            mask_tensor = torch.Tensor(mask).to(DEVICE).unsqueeze(0)

            # Extract phase ID for hierarchical routing
            phase_id = extract_phase_id(obs_dict)
            phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=DEVICE)

            if is_learner:
                with torch.no_grad():
                    action_sample, logprob, _, value = agent.get_action_and_value(
                        obs_tensor, mask_tensor, phase_tensor
                    )

                action_idx = action_sample.item()

                if step_idx < NUM_STEPS:
                    obs_buf[step_idx] = obs_tensor.squeeze(0)
                    mask_buf[step_idx] = mask_tensor.squeeze(0)
                    act_buf[step_idx] = action_sample.squeeze(0)
                    logp_buf[step_idx] = logprob.squeeze(0)
                    val_buf[step_idx] = value.squeeze(0)
                    done_buf[step_idx] = 0.0
                    phase_buf[step_idx] = phase_id
                    step_idx += 1
                    global_step += 1
            else:
                with torch.no_grad():
                    action_sample, _, _, _ = opponent_agent.get_action_and_value(
                        obs_tensor, mask_tensor, phase_tensor
                    )
                action_idx = action_sample.item()

            env.step(action_idx)
            current_game_steps += 1

            if step_idx >= NUM_STEPS:
                break

        # Iterator exhausted (game ended mid-collection). Reset and continue.
        if step_idx < NUM_STEPS:
            env.reset()
            current_game_steps = 0
            opp_weights = sample_opponent_weights(opponent_pool, agent)
            opponent_agent.load_state_dict(opp_weights)
            opponent_agent.eval()

    stats = {
        "games": games_completed,
        "wins": win_count,
        "avg_score": total_score / max(games_completed, 1),
        "avg_length": np.mean(game_lengths) if game_lengths else 0,
    }
    return global_step, stats


def compute_gae(agent, obs_buf, rew_buf, done_buf, val_buf, phase_buf):
    """Compute GAE advantages with proper bootstrap at trajectory end."""
    advantages = torch.zeros_like(rew_buf).to(DEVICE)

    with torch.no_grad():
        last_value = agent.get_value(
            obs_buf[-1].unsqueeze(0), phase_buf[-1:].long()
        ).squeeze()

    lastgaelam = 0.0
    for t in reversed(range(NUM_STEPS)):
        if t == NUM_STEPS - 1:
            nextnonterminal = 1.0 - done_buf[t]
            nextvalues = last_value * nextnonterminal
        else:
            nextnonterminal = 1.0 - done_buf[t + 1]
            nextvalues = val_buf[t + 1] * nextnonterminal

        delta = rew_buf[t] + GAMMA * nextvalues - val_buf[t]
        advantages[t] = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        lastgaelam = advantages[t]

    returns = advantages + val_buf
    return advantages, returns


def ppo_update(agent, optimizer, obs_buf, mask_buf, act_buf, logp_buf,
               advantages, returns, val_buf, phase_buf, writer, global_step):
    """Perform PPO policy and value updates with phase-conditioned heads."""
    b_inds = np.arange(BATCH_SIZE)
    clipfracs = []

    # Phase-specific entropy tracking
    phase_entropies = defaultdict(list)

    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            mb_phase = phase_buf[mb_inds].long()

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                obs_buf[mb_inds], mask_buf[mb_inds], mb_phase, act_buf[mb_inds]
            )
            logratio = newlogprob - logp_buf[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > CLIP_COEF).float().mean().item())

                # Track entropy per phase head
                for i, idx in enumerate(mb_inds):
                    pid = phase_buf[idx].item()
                    head_key = PHASE_TO_HEAD.get(int(pid), "role_select")
                    phase_entropies[head_key].append(entropy[i].item())

            mb_advantages = advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Clipped policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Clipped value loss
            newvalue = newvalue.squeeze(-1)
            v_clipped = val_buf[mb_inds] + torch.clamp(
                newvalue - val_buf[mb_inds], -VF_CLIP_COEF, VF_CLIP_COEF
            )
            v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
            v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    # Log training metrics
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

    # Log per-phase entropy
    for head_key, ent_list in phase_entropies.items():
        if ent_list:
            writer.add_scalar(f"entropy/{head_key}", np.mean(ent_list), global_step)

    return v_loss.item(), pg_loss.item()


def train():
    run_name = f"HPPO_PuertoRico_{NUM_PLAYERS}P_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    env = make_env()
    env.reset()

    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = env.action_space(env.possible_agents[0]).n

    # Learning agent (Hierarchical)
    agent = HierarchicalAgent(obs_dim=obs_dim, action_dim=action_dim).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Opponent agent (same hierarchical architecture, no optimizer)
    opponent_agent = HierarchicalAgent(obs_dim=obs_dim, action_dim=action_dim).to(DEVICE)
    opponent_agent.load_state_dict(agent.state_dict())
    opponent_agent.eval()

    # Opponent pool for self-play diversity
    opponent_pool = []

    # Replay buffers (learning player only)
    obs_buf = torch.zeros((NUM_STEPS, obs_dim)).to(DEVICE)
    mask_buf = torch.zeros((NUM_STEPS, action_dim)).to(DEVICE)
    act_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)
    logp_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)
    rew_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)
    done_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)
    val_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)
    phase_buf = torch.zeros((NUM_STEPS,)).to(DEVICE)  # Phase ID per step

    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    win_history = deque(maxlen=100)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Starting Hierarchical PPO Training on {DEVICE}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  NUM_STEPS={NUM_STEPS}, BATCH_SIZE={BATCH_SIZE}, Updates={num_updates}")
    print(f"  Network params: {total_params:,} (Hierarchical)")
    print(f"  Phase heads: {list(agent.phase_heads.keys())}")

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # Linear LR annealing
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        # --- Data Collection ---
        global_step, stats = collect_rollout(
            env, agent, opponent_agent, opponent_pool,
            obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf,
            phase_buf, obs_space, obs_dim, writer, global_step
        )

        # Track win history
        if stats["games"] > 0:
            for _ in range(stats["wins"]):
                win_history.append(1)
            for _ in range(stats["games"] - stats["wins"]):
                win_history.append(0)

        # --- GAE + PPO Update ---
        advantages, returns = compute_gae(agent, obs_buf, rew_buf, done_buf, val_buf, phase_buf)
        v_loss, pg_loss = ppo_update(
            agent, optimizer, obs_buf, mask_buf, act_buf, logp_buf,
            advantages, returns, val_buf, phase_buf, writer, global_step
        )

        # --- Self-Play Opponent Pool Management ---
        if update % SNAPSHOT_INTERVAL == 0:
            snapshot = copy.deepcopy(agent.state_dict())
            opponent_pool.append(snapshot)
            if len(opponent_pool) > OPPONENT_POOL_SIZE:
                opponent_pool.pop(0)

        # --- Logging ---
        sps = NUM_STEPS / (time.time() - update_start)
        win_rate = np.mean(win_history) if win_history else 0.0
        writer.add_scalar("charts/win_rate", win_rate, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/opponent_pool_size", len(opponent_pool), global_step)

        # Phase distribution in this rollout
        phase_counts = {}
        for head_key in agent.phase_heads.keys():
            count = sum(1 for p in phase_buf[:NUM_STEPS]
                        if PHASE_TO_HEAD.get(int(p.item()), "role_select") == head_key)
            phase_counts[head_key] = count
            writer.add_scalar(f"phase_dist/{head_key}", count, global_step)

        if stats["games"] > 0:
            writer.add_scalar("charts/avg_game_length", stats["avg_length"], global_step)
            writer.add_scalar("charts/avg_score", stats["avg_score"], global_step)

        print(
            f"Update {update}/{num_updates} | "
            f"VLoss: {v_loss:.4f} | PLoss: {pg_loss:.4f} | "
            f"WinRate: {win_rate:.2%} | "
            f"Games: {stats['games']} | "
            f"SPS: {sps:.0f} | "
            f"Step: {global_step}"
        )

        # --- Checkpoint ---
        if update % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "opponent_pool_size": len(opponent_pool),
                "architecture": "hierarchical",
            }, f"models/hppo_checkpoint_update_{update}.pth")

    writer.close()
    print("Hierarchical PPO Training complete!")


if __name__ == "__main__":
    train()
