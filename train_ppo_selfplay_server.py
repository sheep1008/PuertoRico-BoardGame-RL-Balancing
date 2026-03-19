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

# --- 서버 최적화 하이퍼파라미터 ---
NUM_PLAYERS = 3
NUM_ENVS = 32
STEPS_PER_ENV = 512 
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV 
MINIBATCH_SIZE = 512 
UPDATE_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
INITIAL_ENT_COEF = 0.03 # 초기 엔트로피 계수 (수정됨)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Self-play 및 저장 설정
TOTAL_TIMESTEPS = 10_000_000
SNAPSHOT_INTERVAL = 50 
OPPONENT_POOL_SIZE = 10
LATEST_POLICY_PROB = 0.7 
# LEARNING_PLAYER_IDX = 0  <-- 전역 고정 변수 삭제 (Fix 1)

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(worker_id, shared_weights_dict, opponent_pool, steps_per_env, obs_dim, action_dim, return_queue):
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1200)
    env.reset()
    
    # 에피소드 시작 시 학습 플레이어 위치 랜덤 배정 (Fix 1)
    learning_player_idx = random.randint(0, NUM_PLAYERS - 1)
    
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
            is_learner = (p_idx == learning_player_idx) # 동적 할당된 학습자 인덱스 사용

            if is_learner and step_idx > 0 and step_idx <= steps_per_env:
                rew_buf[step_idx - 1] = reward

            if termination or truncation:
                if is_learner and step_idx < steps_per_env:
                    done_buf[step_idx] = 1.0
                env.step(None)

                if all(env.terminations.values()):
                    stats["games"] += 1
                    
                    if env.game.vp_chips <= 0: stats["end_reason_shipping"] += 1
                    elif any(p.empty_city_spaces == 0 for p in env.game.players): stats["end_reason_building"] += 1
                    elif getattr(env.game, '_colonists_ship_underfilled', False): stats["end_reason_colonists"] += 1
                    
                    final_scores = env.game.get_scores()
                    learner_score = final_scores[learning_player_idx][0]
                    stats["total_score"] += learner_score
                    
                    max_opp = max([final_scores[j][0] for j in range(NUM_PLAYERS) if j != learning_player_idx])
                    if learner_score >= max_opp: stats["wins"] += 1
                    
                    p_obj = env.game.players[learning_player_idx]
                    stats["vp_chips"] += p_obj.vp_chips
                    stats["building_vp"] += (learner_score - p_obj.vp_chips)
                    for b in p_obj.city_board:
                        if b.building_type.value < 23: stats["building_counts"][b.building_type.value] += 1
                    
                    env.reset()
                    # 리셋 후 다음 게임의 학습자 위치도 다시 랜덤 배정
                    learning_player_idx = random.randint(0, NUM_PLAYERS - 1)
                    
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
    
    # GPU 0번 혹은 1번 중 선택 (nvidia-smi에서 확인한 한가한 번호)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    run_name = f"PPO_PR_CPU_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_dim = get_flattened_obs_dim(temp_env.observation_space(temp_env.possible_agents[0])["observation"])
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    opponent_pool = []
    global_step = 0
    os.makedirs("models", exist_ok=True)

    total_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    train_start_time = time.time()

    for update in range(1, total_updates + 1):
        update_start = time.time()
        
        # 학습률(LR) 및 엔트로피 계수 선형 감소 (Fix 3)
        frac = 1.0 - (update - 1.0) / total_updates
        optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE
        current_ent_coef = INITIAL_ENT_COEF * frac 

        shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        return_queue = mp.Queue()
        processes = [mp.Process(target=rollout_worker, args=(i, shared_weights, opponent_pool, STEPS_PER_ENV, obs_dim, action_dim, return_queue)) for i in range(NUM_ENVS)]
        for p in processes: p.start()
        results = [return_queue.get() for _ in range(NUM_ENVS)]
        for p in processes: p.join()

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
        
        # CleanRL 기반 올바른 GAE 구현 (Fix 2)
        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                nextnonterminal = 1.0 - done_b[:, t]
                nextvalues = next_val_arr
            else:
                nextnonterminal = 1.0 - done_b[:, t] 
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
                
                # 선형 감소된 엔트로피 계수 적용
                loss = pg_loss - current_ent_coef * entropy.mean() + v_loss * VF_COEF
                
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM); optimizer.step()
                losses_pg.append(pg_loss.item()); losses_v.append(v_loss.item()); losses_ent.append(entropy.mean().item())

        global_step += BATCH_SIZE
        total_games = sum(r["stats"]["games"] for r in results)
        
        # 훈련 지표 기록
        writer.add_scalar("Loss/PolicyLoss", np.mean(losses_pg), global_step)
        writer.add_scalar("Loss/ValueLoss", np.mean(losses_v), global_step)
        writer.add_scalar("Loss/Entropy", np.mean(losses_ent), global_step)
        writer.add_scalar("Hyperparameters/Ent_Coef", current_ent_coef, global_step)

        if total_games > 0:
            writer.add_scalar("Performance/WinRate", sum(r["stats"]["wins"] for r in results) / total_games, global_step)
            writer.add_scalar("Strategy/VP_Shipping", sum(r["stats"]["vp_chips"] for r in results) / total_games, global_step)
            writer.add_scalar("Strategy/VP_Building", sum(r["stats"]["building_vp"] for r in results) / total_games, global_step)
            
            writer.add_scalar("End_Reason/Shipping_Limit", sum(r["stats"]["end_reason_shipping"] for r in results) / total_games, global_step)
            writer.add_scalar("End_Reason/Building_Full", sum(r["stats"]["end_reason_building"] for r in results) / total_games, global_step)
            writer.add_scalar("End_Reason/Colonist_Empty", sum(r["stats"]["end_reason_colonists"] for r in results) / total_games, global_step)

            bldg_dist = np.sum([r["stats"]["building_counts"] for r in results], axis=0)
            for i in range(0, 6): writer.add_scalar(f"Buildings_Production/{BuildingType(i).name}", bldg_dist[i] / total_games, global_step)
            for i in range(6, 18): writer.add_scalar(f"Buildings_Commercial/{BuildingType(i).name}", bldg_dist[i] / total_games, global_step)
            for i in range(18, 23): writer.add_scalar(f"Buildings_Large/{BuildingType(i).name}", bldg_dist[i] / total_games, global_step)
            for i in range(8): writer.add_scalar(f"Role_Selection/{Role(i).name}", sum(r["stats"]["role_counts"][i] for r in results) / total_games, global_step)

        # 터미널 진행 상황 및 ETA 출력 로직
        elapsed_time = time.time() - train_start_time
        fps = global_step / elapsed_time if elapsed_time > 0 else 1
        remaining_steps = TOTAL_TIMESTEPS - global_step
        eta_seconds = remaining_steps / fps

        # 시간을 보기 좋게 HH:MM:SS 포맷으로 변환
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        # 지표 계산 (total_games가 0일 때의 예외 처리 포함)
        if total_games > 0:
            win_rate = sum(r["stats"]["wins"] for r in results) / total_games
            vp_ship = sum(r["stats"]["vp_chips"] for r in results) / total_games
            vp_bldg = sum(r["stats"]["building_vp"] for r in results) / total_games
        else:
            win_rate, vp_ship, vp_bldg = 0, 0, 0

        # 콘솔 출력
        print(f"[Update {update}/{total_updates}] Progress: {update/total_updates*100:.1f}% | "
              f"Step: {global_step}/{TOTAL_TIMESTEPS} | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str} | FPS: {int(fps)}")
        print(f" └─ WinRate: {win_rate:.2f} | VP(Ship/Bldg): {vp_ship:.1f}/{vp_bldg:.1f} | "
              f"Loss(P/V/Ent): {np.mean(losses_pg):.3f}/{np.mean(losses_v):.3f}/{np.mean(losses_ent):.3f}\n")

        if update % SNAPSHOT_INTERVAL == 0:
            opponent_pool.append(copy.deepcopy(shared_weights))
            # 메모리 누수 방지 (Fix 3)
            if len(opponent_pool) > OPPONENT_POOL_SIZE:
                opponent_pool.pop(0)
            torch.save(agent.state_dict(), f"models/{run_name}_step_{global_step}.pth")

    writer.close()

if __name__ == "__main__":
    train()