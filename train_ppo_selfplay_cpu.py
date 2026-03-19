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

# --- CPU 환경 최적화 하이퍼파라미터 ---
NUM_PLAYERS = 3
NUM_ENVS = 4         
STEPS_PER_ENV = 512 
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV 
MINIBATCH_SIZE = 128 
UPDATE_EPOCHS = 4
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.03      
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play 및 저장 설정
TOTAL_TIMESTEPS = 10_000_000
SNAPSHOT_INTERVAL = 50 
OPPONENT_POOL_SIZE = 10
LATEST_POLICY_PROB = 0.7 
LEARNING_PLAYER_IDX = 0

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    """데이터 수집 워커: 개별 건물 구매 현황 및 종료 원인 수집"""
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1200)
    env.reset()
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_agent.load_state_dict(shared_weights_dict)
    
    local_agent.eval()
    local_opponent.eval()

    obs_buf = np.zeros((steps_per_env, obs_dim), dtype=np.float32)
    mask_buf = np.zeros((steps_per_env, action_dim), dtype=np.float32)
    act_buf = np.zeros((steps_per_env,), dtype=np.float32)
    logp_buf = np.zeros((steps_per_env,), dtype=np.float32)
    rew_buf = np.zeros((steps_per_env,), dtype=np.float32)
    done_buf = np.zeros((steps_per_env,), dtype=np.float32)
    val_buf = np.zeros((steps_per_env,), dtype=np.float32)
    
    stats = {
        "games": 0, "wins": 0, "total_score": 0.0,
        "vp_chips": 0.0, "building_vp": 0.0,
        "role_counts": np.zeros(8),
        "building_counts": np.zeros(23),
        "end_reason_shipping": 0,
        "end_reason_building": 0,
        "end_reason_colonists": 0
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
                    # 종료 원인 판단
                    if env.game.vp_chips <= 0: stats["end_reason_shipping"] += 1
                    elif any(p.empty_city_spaces == 0 for p in env.game.players): stats["end_reason_building"] += 1
                    elif getattr(env.game, '_colonists_ship_underfilled', False): stats["end_reason_colonists"] += 1
                    
                    final_scores = env.game.get_scores()
                    learner_score = final_scores[LEARNING_PLAYER_IDX][0]
                    stats["total_score"] += learner_score
                    
                    max_opp = max([final_scores[j][0] for j in range(NUM_PLAYERS) if j != LEARNING_PLAYER_IDX])
                    if learner_score >= max_opp: stats["wins"] += 1
                    
                    p_obj = env.game.players[LEARNING_PLAYER_IDX]
                    stats["vp_chips"] += p_obj.vp_chips
                    stats["building_vp"] += (learner_score - p_obj.vp_chips)
                    
                    # 건물 구매 기록 (23종 개별 수집)
                    for b in p_obj.city_board:
                        if b.building_type.value < 23: 
                            stats["building_counts"][b.building_type.value] += 1
                    
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

    return_queue.put({"stats": stats, "next_val": val.item(), "data": (obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf)})

def train():
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    device = torch.device("cpu")
    run_name = f"PPO_PR_Individual_Buildings_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_dim = get_flattened_obs_dim(temp_env.observation_space(temp_env.possible_agents[0])["observation"])
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    win_history = deque(maxlen=100)
    os.makedirs("models", exist_ok=True)

    for update in range(1, (TOTAL_TIMESTEPS // BATCH_SIZE) + 1):
        update_start = time.time()
        frac = 1.0 - (update - 1.0) / (TOTAL_TIMESTEPS // BATCH_SIZE)
        optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        # 1. 데이터 수집
        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = [mp.Process(target=rollout_worker, args=(i, shared_weights, opponent_pool, STEPS_PER_ENV, obs_dim, action_dim, return_queue)) for i in range(NUM_ENVS)]
        for p in processes: p.start()
        results = [return_queue.get() for _ in range(NUM_ENVS)]
        for p in processes: p.join()

        # 2. 데이터 통합 및 GAE 계산
        obs_b = np.stack([r["data"][0] for r in results]).reshape(-1, obs_dim)
        mask_b = np.stack([r["data"][1] for r in results]).reshape(-1, action_dim)
        act_b = np.stack([r["data"][2] for r in results]).reshape(-1)
        logp_b = np.stack([r["data"][3] for r in results]).reshape(-1)
        rew_b = np.stack([r["data"][4] for r in results])
        done_b = np.stack([r["data"][5] for r in results])
        val_b = np.stack([r["data"][6] for r in results])
        next_val_arr = np.array([r["next_val"] for r in results])

        advantages = np.zeros_like(rew_b)
        lastgaelam = 0
        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                nextnonterminal = 1.0 - done_b[:, t]
                nextvalues = next_val_arr
            else:
                nextnonterminal = 1.0 - done_b[:, t+1]
                nextvalues = val_b[:, t+1]
            delta = rew_b[:, t] + GAMMA * nextvalues * nextnonterminal - val_b[:, t]
            advantages[:, t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns_b = advantages + val_b

        obs_t = torch.as_tensor(obs_b, device=device)
        mask_t = torch.as_tensor(mask_b, device=device)
        act_t = torch.as_tensor(act_b, device=device)
        logp_t = torch.as_tensor(logp_b, device=device)
        adv_t = torch.as_tensor(advantages.reshape(-1), device=device)
        ret_t = torch.as_tensor(returns_b.reshape(-1), device=device)

        # 3. PPO 업데이트
        losses_pg, losses_v, losses_ent = [], [], []
        b_inds = np.arange(BATCH_SIZE)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb = b_inds[start:start+MINIBATCH_SIZE]
                _, newlogp, entropy, newval = agent.get_action_and_value(obs_t[mb], mask_t[mb], act_t[mb])
                ratio = (newlogp - logp_t[mb]).exp()
                mb_adv = (adv_t[mb] - adv_t[mb].mean()) / (adv_t[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1-CLIP_COEF, 1+CLIP_COEF)).mean()
                v_loss = 0.5 * ((newval.view(-1) - ret_t[mb])**2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + v_loss * VF_COEF
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                losses_pg.append(pg_loss.item()); losses_v.append(v_loss.item()); losses_ent.append(entropy.mean().item())

        # 4. 개별 건물 그룹화 로깅 (핵심 수정 부분)
        global_step += BATCH_SIZE
        total_games = sum(r["stats"]["games"] for r in results)
        
        # 기본 훈련 지표 기록
        writer.add_scalar("Loss/PolicyLoss", np.mean(losses_pg), global_step)
        writer.add_scalar("Loss/ValueLoss", np.mean(losses_v), global_step)
        writer.add_scalar("Loss/Entropy", np.mean(losses_ent), global_step)

        if total_games > 0:
            writer.add_scalar("Performance/WinRate", sum(r["stats"]["wins"] for r in results) / total_games, global_step)
            writer.add_scalar("Strategy/VP_Shipping", sum(r["stats"]["vp_chips"] for r in results) / total_games, global_step)
            writer.add_scalar("Strategy/VP_Building", sum(r["stats"]["building_vp"] for r in results) / total_games, global_step)
            
            # 건물 통계 통합 데이터
            bldg_dist = np.sum([r["stats"]["building_counts"] for r in results], axis=0)
            
            # 1. 생산 건물 토글 (0~5)
            # Small Indigo, Small Sugar, Indigo Plant, Sugar Mill, Tobacco Storage, Coffee Roaster
            for i in range(0, 6):
                name = BuildingType(i).name
                writer.add_scalar(f"Buildings_Production/{name}", bldg_dist[i] / total_games, global_step)
                
            # 2. 상업 건물 토글 (6~17)
            # Market, Hacienda, Construction Hut, Warehouse, Hospice, Office, Factory, University, Harbor, Wharf 등
            for i in range(6, 18):
                name = BuildingType(i).name
                writer.add_scalar(f"Buildings_Commercial/{name}", bldg_dist[i] / total_games, global_step)
                
            # 3. 대형 건물 토글 (18~22)
            # Guildhall, Residence, Fortress, Customs House, City Hall
            for i in range(18, 23):
                name = BuildingType(i).name
                writer.add_scalar(f"Buildings_Large/{name}", bldg_dist[i] / total_games, global_step)

            # 역할 및 종료 사유 로깅
            role_dist = np.sum([r["stats"]["role_counts"] for r in results], axis=0)
            for r_idx in range(8):
                writer.add_scalar(f"Roles/{Role(r_idx).name}", role_dist[r_idx] / total_games, global_step)
            writer.add_scalar("End_Reason/Shipping_Limit", sum(r["stats"]["end_reason_shipping"] for r in results) / total_games, global_step)
            writer.add_scalar("End_Reason/Building_Full", sum(r["stats"]["end_reason_building"] for r in results) / total_games, global_step)

        if update % SNAPSHOT_INTERVAL == 0:
            opponent_pool.append(copy.deepcopy(shared_weights))
            torch.save(agent.state_dict(), f"models/{run_name}_step_{global_step}.pth")

        print(f"Update {update} | Step: {global_step} | SPS: {int(BATCH_SIZE/(time.time()-update_start))}")

    writer.close()

if __name__ == "__main__":
    train()