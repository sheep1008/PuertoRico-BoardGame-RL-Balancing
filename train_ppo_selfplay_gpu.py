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

from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent

# --- Hyperparameters ---
_TEST_MODE = os.environ.get("PPO_TEST_MODE", "0") == "1"

NUM_PLAYERS = 3
NUM_ENVS = 8  # 동시 실행할 멀티프로세싱 워커(환경) 수 - CPU 코어 수에 맞게 조절 (예: 8~16)
TOTAL_TIMESTEPS = 500_000 if _TEST_MODE else 10_000_000
LEARNING_RATE = 2.5e-4
STEPS_PER_ENV = 500 if _TEST_MODE else 2048  # 각 워커가 수집할 스텝 수
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV        # 총 수집되는 대규모 배치 사이즈
MINIBATCH_SIZE = 128 if _TEST_MODE else 1024 # GPU 효율을 위한 큰 미니배치
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play settings
SNAPSHOT_INTERVAL = 10
OPPONENT_POOL_SIZE = 20
LATEST_POLICY_PROB = 0.8
LEARNING_PLAYER_IDX = 0

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    """
    각 독립된 프로세스에서 실행되는 워커.
    자신의 환경을 돌며 할당된 스텝만큼 데이터를 수집하여 반환합니다.
    """
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=2000)
    env.reset()
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    # 워커용 로컬 에이전트 (CPU 연산으로 수집 속도 최적화)
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    
    # 최신 글로벌 가중치 동기화
    local_agent.load_state_dict(shared_weights_dict)
    local_agent.eval()
    
    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
    local_opponent.load_state_dict(opp_weights)
    local_opponent.eval()

    # 버퍼 초기화
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
    game_lengths = []
    current_game_steps = 0
    
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
                    game_lengths.append(current_game_steps)
                    if learning_agent_name in env.infos and "final_scores" in env.infos[learning_agent_name]:
                        scores = env.infos[learning_agent_name]["final_scores"]
                        learner_score = scores[LEARNING_PLAYER_IDX][0]
                        max_opp_score = max(scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX)
                        total_score += learner_score
                        if learner_score >= max_opp_score:
                            win_count += 1
                    
                    env.reset()
                    current_game_steps = 0
                    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
                    local_opponent.load_state_dict(opp_weights)
                continue

            # State 준비
            obs_dict = obs["observation"]
            mask = obs["action_mask"]
            flat_obs = flatten_dict_observation(obs_dict, obs_space)
            obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

            with torch.no_grad():
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
            current_game_steps += 1

            if step_idx >= steps_per_env:
                break

    # Next Value for Bootstrap GAE
    with torch.no_grad():
        _, _, _, next_val = local_agent.get_action_and_value(obs_tensor, mask_tensor)
        next_val = next_val.item()

    stats = {
        "games": games_completed,
        "wins": win_count,
        "score": total_score,
        "lengths": sum(game_lengths)
    }
    
    # 큐를 통해 메인 프로세스로 데이터 전송 (Numpy 배열 전송)
    return_queue.put({
        "obs": obs_buf, "mask": mask_buf, "act": act_buf, 
        "logp": logp_buf, "rew": rew_buf, "done": done_buf, 
        "val": val_buf, "next_val": next_val, "stats": stats
    })

def compute_gae_batched(rew_buf, done_buf, val_buf, next_val_arr, gamma, gae_lambda):
    """벡터화된 버퍼에 대해 GAE 연산"""
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
    # Multiprocessing Spawn 설정 (CUDA 환경에서의 필수 설정)
    mp.set_start_method('spawn')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device}] 병렬 분산 수집 기반 GPU PPO 훈련 시작...")

    run_name = f"PPO_GPU_PR_{NUM_PLAYERS}P_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # 환경 차원 파악을 위한 임시 환경
    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_space = temp_env.observation_space(temp_env.possible_agents[0])["observation"]
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    # 메인 GPU 에이전트
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    win_history = deque(maxlen=200)

    print(f"  워커 수(NUM_ENVS): {NUM_ENVS}")
    print(f"  배치 사이즈(BATCH_SIZE): {BATCH_SIZE}")

    for update in range(1, num_updates + 1):
        update_start = time.time()
        
        # 선형 LR 감소
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        # 1. CPU 멀티프로세스를 이용한 병렬 데이터 수집
        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = []

        for i in range(NUM_ENVS):
            p = mp.Process(target=rollout_worker, args=(
                i, shared_weights, opponent_pool, STEPS_PER_ENV, 
                obs_dim, action_dim, return_queue
            ))
            p.start()
            processes.append(p)

        # 2. 결과 수합
        results = []
        for _ in range(NUM_ENVS):
            results.append(return_queue.get())
        
        for p in processes:
            p.join()

        # 3. 데이터 통합 (B, T, ...) -> (B * T, ...)
        obs_batch = np.stack([r["obs"] for r in results])
        mask_batch = np.stack([r["mask"] for r in results])
        act_batch = np.stack([r["act"] for r in results])
        logp_batch = np.stack([r["logp"] for r in results])
        rew_batch = np.stack([r["rew"] for r in results])
        done_batch = np.stack([r["done"] for r in results])
        val_batch = np.stack([r["val"] for r in results])
        next_val_arr = np.array([r["next_val"] for r in results])
        
        # 승률 및 통계 계산
        tot_games = sum(r["stats"]["games"] for r in results)
        tot_wins = sum(r["stats"]["wins"] for r in results)
        avg_score = sum(r["stats"]["score"] for r in results) / max(tot_games, 1)
        
        if tot_games > 0:
            win_history.extend([1]*tot_wins + [0]*(tot_games - tot_wins))

        # GAE 계산
        adv_batch, ret_batch = compute_gae_batched(rew_batch, done_batch, val_batch, next_val_arr, GAMMA, GAE_LAMBDA)
        
        # Flatten for GPU processing
        obs_tensor = torch.tensor(obs_batch.reshape(-1, obs_dim), device=device)
        mask_tensor = torch.tensor(mask_batch.reshape(-1, action_dim), device=device)
        act_tensor = torch.tensor(act_batch.reshape(-1), device=device)
        logp_tensor = torch.tensor(logp_batch.reshape(-1), device=device)
        val_tensor = torch.tensor(val_batch.reshape(-1), device=device)
        adv_tensor = torch.tensor(adv_batch.reshape(-1), device=device)
        ret_tensor = torch.tensor(ret_batch.reshape(-1), device=device)

        global_step += BATCH_SIZE

        # 4. GPU 대규모 PPO 학습 업데이트
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []

        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_tensor[mb_inds], mask_tensor[mb_inds], act_tensor[mb_inds]
                )
                
                logratio = newlogprob - logp_tensor[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > CLIP_COEF).float().mean().item())

                mb_advantages = adv_tensor[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.squeeze(-1)
                v_clipped = val_tensor[mb_inds] + torch.clamp(
                    newvalue - val_tensor[mb_inds], -VF_CLIP_COEF, VF_CLIP_COEF
                )
                v_loss_unclipped = (newvalue - ret_tensor[mb_inds]) ** 2
                v_loss_clipped = (v_clipped - ret_tensor[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # 5. 모델 저장 및 통계 기록
        if update % SNAPSHOT_INTERVAL == 0:
            opponent_pool.append(copy.deepcopy({k: v.cpu() for k, v in agent.state_dict().items()}))
            if len(opponent_pool) > OPPONENT_POOL_SIZE:
                opponent_pool.pop(0)

        sps = int(BATCH_SIZE / (time.time() - update_start))
        win_rate = np.mean(win_history) if win_history else 0.0
        
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/win_rate", win_rate, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        print(f"업데이트 {update}/{num_updates} | Loss(V:{v_loss:.3f}/P:{pg_loss:.3f}) | "
              f"승률: {win_rate:.1%} | 게임수: {tot_games} | SPS: {sps} | 스텝: {global_step}")

        if update % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(), f"models/ppo_gpu_update_{update}.pth")

    writer.close()
    print("훈련 완료!")

if __name__ == "__main__":
    train()