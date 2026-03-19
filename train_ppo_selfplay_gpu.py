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

# 프로젝트 내부 모듈 임포트
from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent
from configs.constants import Role, BuildingType, Good

# --- 하이퍼파라미터 (i9-13900F + RTX 3070 환경 최적화) ---
_TEST_MODE = os.environ.get("PPO_TEST_MODE", "0") == "1"

NUM_PLAYERS = 3
NUM_ENVS = 16  # 병렬 실행할 프로세스(환경) 수
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
ENT_COEF = 0.03  # 다양한 전략(역할/건물) 탐색을 위해 약간 높게 유지
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play 설정
SNAPSHOT_INTERVAL = 20
OPPONENT_POOL_SIZE = 20
LATEST_POLICY_PROB = 0.7 
LEARNING_PLAYER_IDX = 0

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    """
    각 워커 프로세스에서 실행되는 데이터 수집 함수.
    에이전트의 전략적 지표(건물, 상품, VP 출처 등)를 함께 수집합니다.
    """
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1500)
    env.reset()
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    
    # 가중치 로드
    local_agent.load_state_dict(shared_weights_dict)
    local_agent.eval()
    
    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
    local_opponent.load_state_dict(opp_weights)
    local_opponent.eval()

    # 데이터 버퍼
    obs_buf = np.zeros((steps_per_env, obs_dim), dtype=np.float32)
    mask_buf = np.zeros((steps_per_env, action_dim), dtype=np.float32)
    act_buf = np.zeros((steps_per_env,), dtype=np.float32)
    logp_buf = np.zeros((steps_per_env,), dtype=np.float32)
    rew_buf = np.zeros((steps_per_env,), dtype=np.float32)
    done_buf = np.zeros((steps_per_env,), dtype=np.float32)
    val_buf = np.zeros((steps_per_env,), dtype=np.float32)
    
    # 상세 지표 수집용 딕셔너리
    stats = {
        "games": 0, "wins": 0, "total_score": 0.0,
        "vp_chips": 0.0, "building_vp": 0.0,
        "end_doubloons": 0.0,
        "role_counts": np.zeros(8),          # Role 0-7
        "produced_goods": np.zeros(5),       # Good 0-4
        "building_counts": np.zeros(23)      # BuildingType 0-22
    }
    
    step_idx = 0
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
                    stats["games"] += 1
                    # 최종 스코어 및 승률 계산
                    final_scores = env.game.get_scores()
                    learner_score = final_scores[LEARNING_PLAYER_IDX][0]
                    opp_scores = [final_scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX]
                    
                    stats["total_score"] += learner_score
                    if learner_score >= max(opp_scores):
                        stats["wins"] += 1
                    
                    # 상세 전략 지표 수집
                    p_obj = env.game.players[LEARNING_PLAYER_IDX]
                    stats["vp_chips"] += p_obj.vp_chips
                    stats["building_vp"] += (learner_score - p_obj.vp_chips)
                    stats["end_doubloons"] += p_obj.doubloons
                    
                    for b in p_obj.city_board:
                        if b.building_type.value < 23:
                            stats["building_counts"][b.building_type.value] += 1
                    for g_idx in range(5):
                        stats["produced_goods"][g_idx] += p_obj.goods[Good(g_idx)]

                    env.reset()
                    opp_weights = sample_opponent_weights(opponent_pool, shared_weights_dict)
                    local_opponent.load_state_dict(opp_weights)
                continue

            # 액션 선택 루직
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
                        
                        # 역할 선택 추적 (Action 0~7)
                        if 0 <= action_idx <= 7:
                            stats["role_counts"][action_idx] += 1
                        step_idx += 1
                else:
                    action_sample, _, _, _ = local_opponent.get_action_and_value(obs_tensor, mask_tensor)
                    action_idx = action_sample.item()

            env.step(action_idx)
            if step_idx >= steps_per_env: break

    # GAE 연산을 위한 Next Value
    with torch.no_grad():
        last_obs = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
        last_mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        _, _, _, next_val = local_agent.get_action_and_value(last_obs, last_mask)
        next_val = next_val.item()

    return_queue.put({
        "obs": obs_buf, "mask": mask_buf, "act": act_buf, 
        "logp": logp_buf, "rew": rew_buf, "done": done_buf, 
        "val": val_buf, "next_val": next_val, "stats": stats
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
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"PPO_PR_Final_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"[{device}] 통합 훈련 프로세스 시작... 로그: {run_name}")

    # 환경 차원 정의
    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_space = temp_env.observation_space(temp_env.possible_agents[0])["observation"]
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    # 메인 에이전트 및 옵티마이저
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    win_history = deque(maxlen=200)

    os.makedirs("models", exist_ok=True)

    for update in range(1, num_updates + 1):
        update_start = time.time()
        # 선형 LR 스케줄링
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        # 1. 병렬 데이터 수집
        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = []

        for i in range(NUM_ENVS):
            p = mp.Process(target=rollout_worker, args=(i, shared_weights, opponent_pool, STEPS_PER_ENV, obs_dim, action_dim, return_queue))
            p.start()
            processes.append(p)

        results = [return_queue.get() for _ in range(NUM_ENVS)]
        for p in processes: p.join()

        # 2. 데이터 통합 및 GAE 계산
        obs_batch = np.stack([r["obs"] for r in results])
        mask_batch = np.stack([r["mask"] for r in results])
        act_batch = np.stack([r["act"] for r in results])
        logp_batch = np.stack([r["logp"] for r in results])
        rew_batch = np.stack([r["rew"] for r in results])
        done_batch = np.stack([r["done"] for r in results])
        val_batch = np.stack([r["val"] for r in results])
        next_val_arr = np.array([r["next_val"] for r in results])
        
        adv_batch, ret_batch = compute_gae_batched(rew_batch, done_batch, val_batch, next_val_arr, GAMMA, GAE_LAMBDA)
        
        # GPU 전송용 텐서 변환
        obs_tensor = torch.as_tensor(obs_batch.reshape(-1, obs_dim), device=device)
        mask_tensor = torch.as_tensor(mask_batch.reshape(-1, action_dim), device=device)
        act_tensor = torch.as_tensor(act_batch.reshape(-1), device=device)
        logp_tensor = torch.as_tensor(logp_batch.reshape(-1), device=device)
        adv_tensor = torch.as_tensor(adv_batch.reshape(-1), device=device)
        ret_tensor = torch.as_tensor(ret_batch.reshape(-1), device=device)

        global_step += BATCH_SIZE

        # 3. PPO 업데이트 루프
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

        # 4. 통계 기록 및 모델 저장
        tot_games = sum(r["stats"]["games"] for r in results)
        if tot_games > 0:
            tot_wins = sum(r["stats"]["wins"] for r in results)
            win_rate = tot_wins / tot_games
            win_history.append(win_rate)
            
            # 성능 지표
            writer.add_scalar("Performance/WinRate", np.mean(win_history), global_step)
            writer.add_scalar("Performance/AvgScore", sum(r["stats"]["total_score"] for r in results) / tot_games, global_step)
            
            # 전략 지표 (VP 출처 및 경제)
            avg_vp_chips = sum(r["stats"]["vp_chips"] for r in results) / tot_games
            avg_vp_bldg = sum(r["stats"]["building_vp"] for r in results) / tot_games
            writer.add_scalar("Strategy/VP_Shipping", avg_vp_chips, global_step)
            writer.add_scalar("Strategy/VP_Building", avg_vp_bldg, global_step)
            writer.add_scalar("Economy/AvgEndDoubloons", sum(r["stats"]["end_doubloons"] for r in results) / tot_games, global_step)
            
            # 역할 선호도 (Role Usage)
            role_dist = np.sum([r["stats"]["role_counts"] for r in results], axis=0)
            for r_idx in range(8):
                writer.add_scalar(f"Roles/{Role(r_idx).name}", role_dist[r_idx] / BATCH_SIZE, global_step)
            
            # 건물 구매 선호도 (Building Preference)
            bldg_dist = np.sum([r["stats"]["building_counts"] for r in results], axis=0)
            for b_idx in range(23):
                writer.add_scalar(f"Buildings/{BuildingType(b_idx).name}", bldg_dist[b_idx] / tot_games, global_step)
                
            # 상품 생산량
            goods_dist = np.sum([r["stats"]["produced_goods"] for r in results], axis=0)
            for g_idx in range(5):
                writer.add_scalar(f"Production/{Good(g_idx).name}", goods_dist[g_idx] / tot_games, global_step)

        # 체크포인트 저장 및 Opponent Pool 업데이트
        if update % SNAPSHOT_INTERVAL == 0:
            opponent_pool.append(copy.deepcopy({k: v.cpu() for k, v in agent.state_dict().items()}))
            if len(opponent_pool) > OPPONENT_POOL_SIZE: opponent_pool.pop(0)
            torch.save(agent.state_dict(), f"models/{run_name}_step_{global_step}.pth")

        sps = int(BATCH_SIZE / (time.time() - update_start))
        print(f"Update {update}/{num_updates} | WinRate: {np.mean(win_history):.1%} | SPS: {sps} | Step: {global_step}")

    # 최종 모델 저장
    torch.save(agent.state_dict(), f"models/{run_name}_final.pth")
    writer.close()
    print("훈련 완료!")

if __name__ == "__main__":
    train()