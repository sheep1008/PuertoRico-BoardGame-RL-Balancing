import numpy as np
import pprint
from env.pr_env import PuertoRicoEnv

def test_pr_env():
    env = PuertoRicoEnv(num_players=3)
    
    print("Resetting env...")
    obs, info = env.reset()
    print("Initial Phase:", info['current_phase'])
    
    mask = env.valid_action_mask()
    valid_actions = np.where(mask)[0]
    print(f"Valid actions at start: {valid_actions}")
    
    if len(valid_actions) > 0:
        action = np.random.choice(valid_actions)
        print(f"Taking action: {action}")
        obs, reward, done, _, info = env.step(action)
        print(f"New Phase: {info['current_phase']}")
        print(f"Done: {done}, Reward: {reward}")
    else:
        print("Error: No valid actions!")
        
if __name__ == "__main__":
    test_pr_env()
