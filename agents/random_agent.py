import numpy as np
from env.pr_env import PuertoRicoEnv
import time

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def select_action(self, obs, valid_mask):
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for the agent!")
        return np.random.choice(valid_actions)

def run_simulation(num_games=10):
    env = PuertoRicoEnv(num_players=4)
    agent = RandomAgent(env.action_space)
    
    wins = [0] * env.num_players
    start_time = time.time()
    
    for i in range(num_games):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            mask = env.valid_action_mask()
            action = agent.select_action(obs, mask)
            obs, reward, done, _, info = env.step(action)
            steps += 1
            
            if steps > 5000:
                print(f"Game {i+1} got stuck after 5000 steps! Phase: {info['current_phase']}")
                break
                
        scores = env.game.get_scores()
        winner = np.argmax(scores)
        wins[winner] += 1
        
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{num_games} games. Winner dist: {wins}")
            
    print(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")
    print(f"Final Wins: {wins}")

if __name__ == "__main__":
    run_simulation(num_games=100)
