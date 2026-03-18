import os
import copy
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# 기존 파일 구조를 따름
from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent

# --- Hyperparameters (i9-13900F + RTX 3070 + 32GB RAM 최적화) ---
_TEST_MODE = os.environ.get("PPO_TEST_MODE", "0") == "1"

NUM_PLAYERS = 3 # 기본 게임 인원 [cite: 14]
NUM_ENVS = 16 
TOTAL_TIMESTEPS = 500_000 if _TEST_MODE else 50_000_000
LEARNING_RATE = 2.5e-4
STEPS_PER_ENV = 1024 
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV  # 16,384
MINIBATCH_SIZE = 2048 
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_CLIP_COEF = 0.2
# 푸에르토리코의 다양한 전략(Role 선택) 탐색을 위해 탐색율 유지 [cite: 8, 87]
ENT_COEF = 0.03 
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play settings
SNAPSHOT_INTERVAL = 10
OPPONENT_POOL_SIZE = 20
LATEST_POLICY_PROB = 0.7 
LEARNING_PLAYER_IDX = 0

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    # 환경 초기화 (최대 게임 스텝 설정으로 무한 루프 방지)
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1500)
    env.reset()
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    
    local_agent.load_state_dict(shared_weights_dict)
    local_agent.eval()
    
    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
    local_opponent.load_state_dict(opp_weights)
    local_opponent.eval()

    obs_buf = np.zeros((steps_per_env, obs_dim), dtype=np.float32)
    mask_buf = np.zeros((steps_per_env, action_dim), dtype=np.float32)
    act_buf = np.zeros((steps_per_env,), dtype=np.float32)
    logp_buf = np.zeros((steps_per_env,), dtype=np.float32)
    rew_buf = np.zeros((steps_per_env,), dtype=np.float32)
    done_buf = np.zeros((steps_per_env,), dtype=np.float32)
    val_buf = np.zeros((steps_per_env,), dtype=np.float32)
    
    step_idx = 0
    games_completed = 0
    win_count = 0
    total_score = 0.0
    
    learning_agent_name = f"player_{LEARNING_PLAYER_IDX}"

    while step_idx < steps_per_env:
        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            player_idx = int(agent_name.split("_")[1])
            is_learner = (player_idx == LEARNING_PLAYER_IDX)

            if is_learner and step_idx > 0 and step_idx <= steps_per_env:
                rew_buf[step_idx - 1] = reward

            if termination or truncation:
                if is_learner and step_idx < steps_per_env:
                    done_buf[step_idx] = 1.0
                env.step(None)

                if all(env.terminations.values()):
                    games_completed += 1
                    if learning_agent_name in env.infos and "final_scores" in env.infos[learning_agent_name]:
                        scores = env.infos[learning_agent_name]["final_scores"]
                        learner_score = scores[LEARNING_PLAYER_IDX][0]
                        opp_scores = [scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX]
                        total_score += learner_score
                        # 가장 높은 승점을 얻은 플레이어가 승리 [cite: 13, 18, 360]
                        if learner_score >= max(opp_scores):
                            win_count += 1
                    
                    env.reset()
                    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
                    local_opponent.load_state_dict(opp_weights)
                continue

            obs_dict = obs["observation"]
            mask = obs["action_mask"]
            flat_obs = flatten_dict_observation(obs_dict, obs_space)
            
            with torch.no_grad():
                obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

                if is_learner:
                    action_sample, logprob, _, value = local_agent.get_action_and_value(obs_tensor, mask_tensor)
                    action_idx = action_sample.item()
                    
                    if step_idx < steps_per_env:
                        obs_buf[step_idx] = flat_obs
                        mask_buf[step_idx] = mask
                        act_buf[step_idx] = action_idx
                        logp_buf[step_idx] = logprob.item()
                        val_buf[step_idx] = value.item()
                        done_buf[step_idx] = 0.0
                        step_idx += 1
                else:
                    action_sample, _, _, _ = local_opponent.get_action_and_value(obs_tensor, mask_tensor)
                    action_idx = action_sample.item()

            env.step(action_idx)
            if step_idx >= steps_per_env: break

    with torch.no_grad():
        last_obs = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
        last_mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        _, _, _, next_val = local_agent.get_action_and_value(last_obs, last_mask)
        next_val = next_val.item()

    return_queue.put({
        "obs": obs_buf, "mask": mask_buf, "act": act_buf, 
        "logp": logp_buf, "rew": rew_buf, "done": done_buf, 
        "val": val_buf, "next_val": next_val,
        "stats": {"games": games_completed, "wins": win_count, "score": total_score}
    })

def compute_gae_batched(rew_buf, done_buf, val_buf, next_val_arr, gamma, gae_lambda):
    advantages = np.zeros_like(rew_buf)
    num_envs, num_steps = rew_buf.shape
    lastgaelam = np.zeros(num_envs)
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - done_buf[:, t]
            nextvalues = next_val_arr * nextnonterminal
        else:
            nextnonterminal = 1.0 - done_buf[:, t + 1]
            nextvalues = val_buf[:, t + 1] * nextnonterminal

        delta = rew_buf[:, t] + gamma * nextvalues - val_buf[:, t]
        advantages[:, t] = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        lastgaelam = advantages[:, t]

    returns = advantages + val_buf
    return advantages, returns

def train():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device}] 학습 최적화 및 자동 저장 버전 실행 중...")

    run_name = f"PPO_PR_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    # 모델 저장용 디렉토리 생성
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_space = temp_env.observation_space(temp_env.possible_agents[0])["observation"]
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    win_history = deque(maxlen=200)

    for update in range(1, num_updates + 1):
        update_start = time.time()
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = []

        for i in range(NUM_ENVS):
            p = mp.Process(target=rollout_worker, args=(i, shared_weights, opponent_pool, STEPS_PER_ENV, obs_dim, action_dim, return_queue))
            p.start()
            processes.append(p)

        results = [return_queue.get() for _ in range(NUM_ENVS)]
        for p in processes: p.join()

        obs_batch = np.stack([r["obs"] for r in results])
        mask_batch = np.stack([r["mask"] for r in results])
        act_batch = np.stack([r["act"] for r in results])
        logp_batch = np.stack([r["logp"] for r in results])
        rew_batch = np.stack([r["rew"] for r in results])
        done_batch = np.stack([r["done"] for r in results])
        val_batch = np.stack([r["val"] for r in results])
        next_val_arr = np.array([r["next_val"] for r in results])
        
        tot_games = sum(r["stats"]["games"] for r in results)
        tot_wins = sum(r["stats"]["wins"] for r in results)
        if tot_games > 0: win_history.extend([1]*tot_wins + [0]*(tot_games - tot_wins))

        adv_batch, ret_batch = compute_gae_batched(rew_batch, done_batch, val_batch, next_val_arr, GAMMA, GAE_LAMBDA)
        
        obs_tensor = torch.as_tensor(obs_batch.reshape(-1, obs_dim), device=device)
        mask_tensor = torch.as_tensor(mask_batch.reshape(-1, action_dim), device=device)
        act_tensor = torch.as_tensor(act_batch.reshape(-1), device=device)
        logp_tensor = torch.as_tensor(logp_batch.reshape(-1), device=device)
        adv_tensor = torch.as_tensor(adv_batch.reshape(-1), device=device)
        ret_tensor = torch.as_tensor(ret_batch.reshape(-1), device=device)

        global_step += BATCH_SIZE

        b_inds = np.arange(BATCH_SIZE)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb_inds = b_inds[start:start+MINIBATCH_SIZE]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_tensor[mb_inds], mask_tensor[mb_inds], act_tensor[mb_inds])
                
                ratio = (newlogprob - logp_tensor[mb_inds]).exp()
                mb_advantages = adv_tensor[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1-CLIP_COEF, 1+CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - ret_tensor[mb_inds]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + v_loss * VF_COEF

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # 체크포인트 저장 및 Opponent Pool 업데이트
        if update % SNAPSHOT_INTERVAL == 0:
            current_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
            opponent_pool.append(copy.deepcopy(current_weights))
            if len(opponent_pool) > OPPONENT_POOL_SIZE: opponent_pool.pop(0)
            
            # 파일 저장
            ckpt_path = os.path.join(model_dir, f"{run_name}_step_{global_step}.pth")
            torch.save(agent.state_dict(), ckpt_path)
            print(f">>> 체크포인트 저장 완료: {ckpt_path}")

        sps = int(BATCH_SIZE / (time.time() - update_start))
        win_rate = np.mean(win_history) if win_history else 0.0
        
        print(f"업데이트 {update}/{num_updates} | Loss(V:{v_loss.item():.3f}/P:{pg_loss.item():.3f}) | "
              f"승률: {win_rate:.1%} | 게임수: {tot_games} | SPS: {sps} | 스텝: {global_step}")

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step) 
        writer.add_scalar("charts/win_rate", win_rate, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

    # 최종 모델 저장
    final_path = os.path.join(model_dir, f"{run_name}_final.pth")
    torch.save(agent.state_dict(), final_path)
    print(f"========================================")
    print(f"학습 종료! 최종 모델 저장 완료: {final_path}")
    print(f"========================================")
    writer.close()

if __name__ == "__main__":
    train()