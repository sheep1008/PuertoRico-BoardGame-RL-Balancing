import os
import copy
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 프로젝트 내부 모듈 임포트
from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent
from configs.constants import Role, BuildingType

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
INITIAL_ENT_COEF = 0.03
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

TOTAL_TIMESTEPS = 10_000_000
SNAPSHOT_INTERVAL = 50 
OPPONENT_POOL_SIZE = 10
LATEST_POLICY_PROB = 0.7 

# 명령 상수를 정의하여 파이프 통신 효율화
CMD_STEP = 0
CMD_SET_WEIGHTS = 1
CMD_CLOSE = 2

def sample_opponent_weights(opponent_pool: list, current_weights: dict) -> dict:
    if not opponent_pool or random.random() < LATEST_POLICY_PROB:
        return current_weights
    return random.choice(opponent_pool)

def rollout_worker(rank, conn, shared_bufs, obs_dim, action_dim, opponent_pool):
    """
    지속적으로 살아있으며 메인 프로세스의 명령을 대기하는 워커.
    공유 메모리에 데이터를 직접 기록하여 직렬화 오버헤드를 제거함.
    """
    env = PuertoRicoEnv(num_players=NUM_PLAYERS, max_game_steps=1200)
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    
    local_agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_opponent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    local_agent.eval()
    local_opponent.eval()

    # 공유 버퍼 참조 (Shared Memory)
    s_obs, s_mask, s_act, s_logp, s_rew, s_done, s_val, s_next_val = shared_bufs

    while True:
        cmd, data = conn.recv()
        
        if cmd == CMD_SET_WEIGHTS:
            local_agent.load_state_dict(data)
            continue
            
        elif cmd == CMD_CLOSE:
            break
            
        elif cmd == CMD_STEP:
            # 데이터 수집 로직 시작
            env.reset()
            learning_player_idx = random.randint(0, NUM_PLAYERS - 1)
            
            # 상대방 모델 결정 (Self-play)
            if not opponent_pool or random.random() < LATEST_POLICY_PROB:
                opp_weights = None # 최신 가중치 사용 시 별도 로드 불필요 (이미 local_agent와 유사)
            else:
                local_opponent.load_state_dict(random.choice(opponent_pool))

            stats = {
                "games": 0, "wins": 0, "total_score": 0.0,
                "vp_chips": 0.0, "building_vp": 0.0,
                "role_counts": np.zeros(8),
                "building_counts": np.zeros(23),
                "end_reason_shipping": 0, "end_reason_building": 0, "end_reason_colonists": 0
            }
            
            step_idx = 0
            while step_idx < STEPS_PER_ENV:
                for agent_name in env.agent_iter():
                    obs, reward, termination, truncation, info = env.last()
                    p_idx = int(agent_name.split("_")[1])
                    is_learner = (p_idx == learning_player_idx)

                    if is_learner and step_idx > 0 and step_idx <= STEPS_PER_ENV:
                        s_rew[rank, step_idx - 1] = reward

                    if termination or truncation:
                        if is_learner and step_idx < STEPS_PER_ENV:
                            s_done[rank, step_idx] = 1.0
                        env.step(None)

                        if all(env.terminations.values()):
                            stats["games"] += 1
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
                            learning_player_idx = random.randint(0, NUM_PLAYERS - 1)
                        continue

                    flat_obs = flatten_dict_observation(obs["observation"], obs_space)
                    mask = obs["action_mask"]
                    
                    with torch.no_grad():
                        obs_t = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

                        if is_learner:
                            action, logp, _, val = local_agent.get_action_and_value(obs_t, mask_t)
                            if step_idx < STEPS_PER_ENV:
                                s_obs[rank, step_idx] = torch.from_numpy(flat_obs)
                                s_mask[rank, step_idx] = torch.from_numpy(mask)
                                s_act[rank, step_idx] = action.item()
                                s_logp[rank, step_idx] = logp.item()
                                s_val[rank, step_idx] = val.item()
                                s_done[rank, step_idx] = 0.0
                                step_idx += 1
                        else:
                            # 상대방은 local_opponent 혹은 학습 중인 최신 모델 사용
                            agent_to_use = local_opponent if opp_weights else local_agent
                            action, _, _, _ = agent_to_use.get_action_and_value(obs_t, mask_t)

                    env.step(action.item())
                    if step_idx >= STEPS_PER_ENV: break

            s_next_val[rank] = val.item()
            conn.send({"stats": stats})

def train():
    # 1. 멀티프로세싱 시작 방식 설정 (서버 환경 필수)
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_name = f"PPO_PR_Server_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # 환경 정보 추출
    temp_env = PuertoRicoEnv(num_players=NUM_PLAYERS)
    obs_dim = get_flattened_obs_dim(temp_env.observation_space(temp_env.possible_agents[0])["observation"])
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    del temp_env

    # 2. 공유 메모리 버퍼 할당 (Shared Memory)
    # 프로세스 간 데이터 복사 오버헤드를 제거하여 속도를 비약적으로 향상시킵니다.
    shared_bufs = (
        torch.zeros((NUM_ENVS, STEPS_PER_ENV, obs_dim)).share_memory_(),      # obs
        torch.zeros((NUM_ENVS, STEPS_PER_ENV, action_dim)).share_memory_(),   # mask
        torch.zeros((NUM_ENVS, STEPS_PER_ENV)).share_memory_(),               # actions
        torch.zeros((NUM_ENVS, STEPS_PER_ENV)).share_memory_(),               # log_probs
        torch.zeros((NUM_ENVS, STEPS_PER_ENV)).share_memory_(),               # rewards
        torch.zeros((NUM_ENVS, STEPS_PER_ENV)).share_memory_(),               # dones
        torch.zeros((NUM_ENVS, STEPS_PER_ENV)).share_memory_(),               # values
        torch.zeros(NUM_ENVS).share_memory_(),                                # next_values
    )

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # 3. 지속적 워커(Persistent Workers) 생성
    # 매 업데이트마다 프로세스를 새로 띄우지 않고 파이프(Pipe)로 명령만 전달합니다.
    opponent_pool = mp.Manager().list()
    processes = []
    conns = []
    for i in range(NUM_ENVS):
        parent_conn, child_conn = mp.Pipe()
        # 주의: rollout_worker 함수도 위에서 제안한 최적화 버전(CMD 대응형)으로 교체되어 있어야 합니다.
        p = mp.Process(target=rollout_worker, args=(i, child_conn, shared_bufs, obs_dim, action_dim, opponent_pool))
        p.start()
        processes.append(p)
        conns.append(parent_conn)

    global_step = 0
    total_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    train_start_time = time.time()

    try:
        for update in range(1, total_updates + 1):
            update_start = time.time()
            
            # 학습률 및 엔트로피 계수 선형 감소
            frac = 1.0 - (update - 1.0) / total_updates
            optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE
            current_ent_coef = INITIAL_ENT_COEF * frac 

            # 최신 가중치 전송 및 데이터 수집 명령
            shared_weights = {k: v.cpu() for k, v in agent.state_dict().items()}
            for conn in conns:
                conn.send((CMD_SET_WEIGHTS, shared_weights))
                conn.send((CMD_STEP, None))
            
            # 워커들로부터 통계 정보(stats) 수집 대기
            results = [conn.recv() for conn in conns]

            # 공유 메모리 버퍼 참조 (clone은 계산 안정성을 위해 사용)
            obs_b, mask_b, act_b, logp_b, rew_b, done_b, val_b, next_val_arr = [b.clone() for b in shared_bufs]

            # 4. GAE(Generalized Advantage Estimation) 계산
            advantages = torch.zeros_like(rew_b)
            lastgaelam = 0
            for t in reversed(range(STEPS_PER_ENV)):
                nextnonterminal = 1.0 - done_b[:, t]
                nextvalues = next_val_arr if t == STEPS_PER_ENV - 1 else val_b[:, t+1]
                delta = rew_b[:, t] + GAMMA * nextvalues * nextnonterminal - val_b[:, t]
                advantages[:, t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns_b = advantages + val_b

            # 데이터를 GPU로 이동
            obs_t = obs_b.reshape(-1, obs_dim).to(device)
            mask_t = mask_b.reshape(-1, action_dim).to(device)
            act_t = act_b.reshape(-1).to(device)
            logp_t = logp_b.reshape(-1).to(device)
            adv_t = advantages.reshape(-1).to(device)
            ret_t = returns_b.reshape(-1).to(device)

            # 5. PPO 업데이트 루프
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
                    loss = pg_loss - current_ent_coef * entropy.mean() + v_loss * VF_COEF
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    
                    losses_pg.append(pg_loss.item())
                    losses_v.append(v_loss.item())
                    losses_ent.append(entropy.mean().item())

            global_step += BATCH_SIZE
            total_games = sum(r["stats"]["games"] for r in results)
            
            # 6. 기존 TensorBoard 차트 기록 유지
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

            # 터미널 출력 로직
            elapsed_time = time.time() - train_start_time
            fps = global_step / elapsed_time if elapsed_time > 0 else 1
            remaining_steps = TOTAL_TIMESTEPS - global_step
            eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_steps / fps))
            
            print(f"[Update {update}/{total_updates}] Step: {global_step}/{TOTAL_TIMESTEPS} | FPS: {int(fps)} | ETA: {eta_str}")
            if total_games > 0:
                win_rate = sum(r["stats"]["wins"] for r in results) / total_games
                print(f" └─ WinRate: {win_rate:.2f} | Loss(P/V/E): {np.mean(losses_pg):.3f}/{np.mean(losses_v):.3f}/{np.mean(losses_ent):.3f}\n")

            if update % SNAPSHOT_INTERVAL == 0:
                opponent_pool.append(copy.deepcopy(shared_weights))
                if len(opponent_pool) > OPPONENT_POOL_SIZE: opponent_pool.pop(0)
                torch.save(agent.state_dict(), f"models/{run_name}_step_{global_step}.pth")

    finally:
        # 워커 안전 종료
        for conn in conns: conn.send((CMD_CLOSE, None))
        for p in processes: p.join()
        writer.close()

if __name__ == "__main__":
    train()