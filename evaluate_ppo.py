import torch
import numpy as np
from env.pr_env import PuertoRicoEnv
from utils.env_wrappers import flatten_dict_observation, get_flattened_obs_dim
from agents.ppo_agent import Agent

def evaluate(model_path="models/ppo_agent_update_100.pth", num_games=3):
    env = PuertoRicoEnv(num_players=3)
    env.reset()
    
    # Needs to match training dimensions
    obs_space = env.observation_space(env.possible_agents[0])['observation']
    obs_dim = get_flattened_obs_dim(obs_space)
    action_dim = env.action_space(env.possible_agents[0]).n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    # Load trained weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Starting Evaluation for {num_games} games...\n")

    for game_idx in range(num_games):
        env.reset()
        step_count = 0
        saved_infos = {}
        
        print(f"=== Game {game_idx + 1} ===")
        
        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                saved_infos[agent_name] = info
                env.step(None)
                continue
                
            obs_dict = obs["observation"]
            mask = obs["action_mask"]
            
            flat_obs = flatten_dict_observation(obs_dict, obs_space)
            obs_tensor = torch.Tensor(flat_obs).to(device).unsqueeze(0)
            mask_tensor = torch.Tensor(mask).to(device).unsqueeze(0)
            
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                huge_negative = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
                masked_logits = torch.where(mask_tensor > 0.5, logits, huge_negative)
                
                # Use stochastic sampling since the model is only at update 100 and argmax can get stuck
                from torch.distributions.categorical import Categorical
                probs = Categorical(logits=masked_logits)
                best_action = probs.sample().item()
                
            env.step(best_action)
            step_count += 1
            
            if step_count > 5000:
                print("Game got stuck in an infinite loop! (Likely due to underdeveloped policy repeatedly passing). Breaking out.")
                break
            
        print(f"Game finished in {step_count} steps.")
        print("Final Scores:")
        for p in env.possible_agents:
            p_info = saved_infos.get(p, {})
            if "player_rewards" in p_info:
                score = p_info["player_rewards"]
            elif "final_scores" in p_info:
                score = p_info["final_scores"]
            else:
                err = p_info.get("error", "No error trace")
                score = f"Unknown (Early Termination) -> {err}"
            print(f"- {p}: {score}")
        print("\n")

if __name__ == "__main__":
    evaluate()
