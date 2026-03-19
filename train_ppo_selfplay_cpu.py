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

# 프로젝트 모듈 임포트
from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent
from configs.constants import Role, BuildingType, Good

# --- CPU 환경 최적화 하이퍼파라미터 ---
NUM_PLAYERS = 3
NUM_ENVS = 4          # 노트북 CPU 코어 수에 맞춰 조절 (보통 4~8 권장)
STEPS_PER_ENV = 256   # 한 워커당 수집할 스텝 (CPU 환경에선 작게 설정하여 빠른 순환 유도)
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV  # 총 배치 사이즈 (예: 1024)
MINIBATCH_SIZE = 64   # CPU 연산 효율을 위해 작은 미니배치 사용
UPDATE_EPOCHS = 4
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.03       # 전략 탐색 유지
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play 설정
SNAPSHOT_INTERVAL = 10
OPPONENT_POOL_SIZE = 10
LATEST_POLICY_PROB = 0.7 
LEARNING_PLAYER_IDX = 0

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    # CPU 환경에서는 가급적 무거운 렌더링이나 복잡한 로직을 최소화한 환경 사용
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1000)
    env.reset()
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    
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
    
    # 텐서보드용 상세 지표 수집
    stats = {
        "games": 0, "wins": 0, "total_score": 0.0,
        "vp_chips": 0.0, "building_vp": 0.0,
        "role_counts": np.zeros(8),
        "building_counts": np.zeros(23),
        "produced_goods": np.zeros(5)
    }
    
    step_idx = 0
    while step_idx < steps_per_env:
        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            p_idx = int(agent_name.split("_")[1])
            is_learner = (p_idx == LEARNING_PLAYER_IDX)

            if is_learner and step_idx > 0 and step_idx <= steps_per_env:
                rew_buf[step_idx - 1] = reward

            if termination or truncation:
                if is_learner and step_idx < steps_per_env:
                    done_buf[step_idx] = 1.0
                env.step(None)

                if all(env.terminations.values()):
                    stats["games"] += 1
                    final_scores = env.game.get_scores()
                    learner_score = final_scores[LEARNING_PLAYER_IDX][0]
                    stats["total_score"] += learner_score
                    
                    # 승리 판정: 가장 높은 점수를 얻었는지 확인
                    max_opp = max([final_scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX])
                    if learner_score >= max_opp: stats["wins"] += 1
                    
                    # 건물/상품/VP 분석
                    p_obj = env.game.players[LEARNING_PLAYER_IDX]
                    stats["vp_chips"] += p_obj.vp_chips
                    stats["building_vp"] += (learner_score - p_obj.vp_chips)
                    for b in p_obj.city_board:
                        if b.building_type.value < 23: stats["building_counts"][b.building_type.value] += 1
                    for g in range(5): stats["produced_goods"][g] += p_obj.goods[Good(g)]
                    
                    env.reset()
                    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
                    local_opponent.load_state_dict(opp_weights)
                continue

            flat_obs = flatten_dict_observation(obs["observation"], obs_space)
            mask = obs["action_mask"]
            
            with torch.no_grad():
                obs_t = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

                if is_learner:
                    action, logp, _, val = local_agent.get_action_and_value(obs_t, mask_t)
                    act_idx = action.item()
                    if step_idx < steps_per_env:
                        obs_buf[step_idx], mask_buf[step_idx], act_buf[step_idx] = flat_obs, mask, act_idx
                        logp_buf[step_idx], val_buf[step_idx] = logp.item(), val.item()
                        done_buf[step_idx] = 0.0
                        if 0 <= act_idx <= 7: stats["role_counts"][act_idx] += 1
                        step_idx += 1
                else:
                    action, _, _, _ = local_opponent.get_action_and_value(obs_t, mask_t)
                    act_idx = action.item()

            env.step(act_idx)
            if step_idx >= steps_per_env: break

    # Next value for GAE
    with torch.no_grad():
        _, _, _, next_val = local_agent.get_action_and_value(torch.as_tensor(flat_obs).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0))

    return_queue.put({"stats": stats, "next_val": next_val.item(), "data": (obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf)})

def train():
    # Windows/macOS 호환성을 위해 spawn 사용 (CPU에서도 안전함)
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    device = torch.device("cpu")
    run_name = f"PPO_PR_CPU_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"[{device}] CPU 최적화 훈련 모드 시작...")

    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_dim = get_flattened_obs_dim(temp_env.observation_space(temp_env.possible_agents[0])["observation"])
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    win_history = deque(maxlen=100)

    for update in range(1, (10_000_000 // BATCH_SIZE) + 1):
        # 1. 병렬 수집
        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = [mp.Process(target=rollout_worker, args=(i, shared_weights, opponent_pool, STEPS_PER_ENV, obs_dim, action_dim, return_queue)) for i in range(NUM_ENVS)]
        for p in processes: p.start()
        results = [return_queue.get() for _ in range(NUM_ENVS)]
        for p in processes: p.join()

        # 2. 데이터 수합 및 통계 기록
        total_games = sum(r["stats"]["games"] for r in results)
        if total_games > 0:
            avg_win = sum(r["stats"]["wins"] for r in results) / total_games
            win_history.append(avg_win)
            writer.add_scalar("Performance/WinRate", np.mean(win_history), global_step)
            writer.add_scalar("Strategy/VP_Shipping", sum(r["stats"]["vp_chips"] for r in results) / total_games, global_step)
            # 건물별 선호도 기록
            bldg_dist = np.sum([r["stats"]["building_counts"] for r in results], axis=0)
            for b_idx in range(23):
                writer.add_scalar(f"Buildings/{BuildingType(b_idx).name}", bldg_dist[b_idx] / total_games, global_step)

        # 3. PPO 업데이트 (CPU 환경에선 큰 텐서 연산 시 메모리 주의)
        # (GAE 및 업데이트 로직은 이전과 동일하되 device=device 반영)
        # ... (이하 PPO 업데이트 로직 중략, GPU 버전과 동일하나 CPU 장치 사용) ...
        
        global_step += BATCH_SIZE
        print(f"업데이트 {update} | 승률: {np.mean(win_history):.1%} | 스텝: {global_step}")

    writer.close()

if __name__ == "__main__":
    train()