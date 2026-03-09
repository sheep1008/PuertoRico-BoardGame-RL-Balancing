import unittest
import numpy as np
from pettingzoo.test import api_test

from env.pr_env import PuertoRicoEnv
from configs.constants import Phase

class TestPuertoRicoAECEnv(unittest.TestCase):
    def test_pettingzoo_api(self):
        """Run the standard PettingZoo API test to ensure compliance"""
        env = PuertoRicoEnv(num_players=2)
        api_test(env, num_cycles=100)
        
    def test_random_rollout(self):
        """Run a minimal random rollout using the standard AEC Env loop"""
        env = PuertoRicoEnv(num_players=3)
        env.reset()
        
        step_count = 0
        max_steps = 1000
        
        for agent in env.agent_iter():
            step_count += 1
            if step_count > max_steps:
                break
                
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                valid_actions = np.where(mask == 1)[0]
                self.assertGreater(len(valid_actions), 0, "Agent must have at least 1 valid action")
                action = int(np.random.choice(valid_actions))
                
            env.step(action)
            
            if all(env.terminations.values()):
                break

if __name__ == "__main__":
    unittest.main()
