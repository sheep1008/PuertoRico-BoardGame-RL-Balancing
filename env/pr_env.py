import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.engine import PuertoRicoGame
from configs.constants import Phase, Role, Good, TileType, BuildingType

class PuertoRicoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = 4):
        super(PuertoRicoEnv, self).__init__()
        self.num_players = num_players
        self.game = None
        
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _define_action_space(self) -> spaces.Discrete:
        # Total flat discrete actions
        # 0-7: Pick Role
        # 8-14: Settler (0-5: Face up, 6: Quarry, 7: Hacienda)
        # 15: Pass
        # 16-38: Builder (23 buildings)
        # 39-43: Trader (5 goods)
        # 44-58: Captain Load (5 goods * 3 ships)
        # 59-63: Captain Store Privilege (Which 1 Good to keep as free)
        # 64-68: Captain Store WH (Small/Large warehouse selections could be complex)
        # For RL, a Dict action space is often cleaner if using custom algorithms,
        # but standard RL libs like stable-baselines3 prefer Discrete or MultiDiscrete.
        # Given the combinatorial nature of Mayor phase, we might need a large Discrete space
        # or MultiDiscrete. Let's use MultiDiscrete to pack different action types for now,
        # or a large flat Discrete and map it.
        # Let's start with a simpler flat representation for prototyping.
        return spaces.Discrete(200)

    def _define_observation_space(self) -> spaces.Dict:
        obs_dict = {
            "global_state": spaces.Dict({
                "vp_chips": spaces.Discrete(123),
                "colonists_supply": spaces.Discrete(100),
                "colonists_ship": spaces.Discrete(10),
                "goods_supply": spaces.MultiDiscrete([10, 10, 11, 12, 12]), # Coffee, Tobacco, Corn, Sugar, Indigo
                "cargo_ships_good": spaces.MultiDiscrete([6, 6, 6]), # 0-4 for good, 5 for None
                "cargo_ships_load": spaces.MultiDiscrete([10, 10, 10]),
                "trading_house": spaces.MultiDiscrete([6, 6, 6, 6]),
                "role_doubloons": spaces.MultiDiscrete([10] * 8), # Doubloons per role, 9 meaning taken
                "face_up_plantations": spaces.MultiDiscrete([6] * 6),
                "quarry_stack": spaces.Discrete(9),
                "current_player": spaces.Discrete(self.num_players),
                "current_phase": spaces.Discrete(8)
            }),
            "players": spaces.Tuple([
                spaces.Dict({
                    "doubloons": spaces.Discrete(50),
                    "vp_chips": spaces.Discrete(100),
                    "goods": spaces.MultiDiscrete([10, 10, 11, 12, 12]),
                    "island_tiles": spaces.MultiDiscrete([6] * 12),
                    "island_occupied": spaces.MultiBinary(12),
                    "city_buildings": spaces.MultiDiscrete([24] * 12),
                    "city_colonists": spaces.MultiDiscrete([4] * 12),
                    "unplaced_colonists": spaces.Discrete(20)
                }) for _ in range(self.num_players)
            ])
        }
        return spaces.Dict(obs_dict)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = PuertoRicoGame(self.num_players)
        self.game.start_game()
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Decode flat action to specific engine calls
        player_idx = self.game.current_player_idx
        
        # Example naive decoding:
        if 0 <= action <= 7:
            # Pick Role
            role = Role(action)
            self.game.select_role(player_idx, role)
        elif action == 15:
            # Pass
            self._handle_pass(player_idx)
        else:
            # Other actions
            # Need to implement the translation layer here
            pass

        # Check end
        done = self.game.check_game_end()
        reward = self._calculate_reward() if done else 0.0

        return self._get_obs(), reward, done, False, self._get_info()

    def _handle_pass(self, player_idx: int):
        if self.game.current_phase == Phase.SETTLER:
            self.game.action_settler(player_idx, tile_choice=-2)
        elif self.game.current_phase == Phase.BUILDER:
            self.game.action_builder(player_idx, building_choice=None)
        elif self.game.current_phase == Phase.TRADER:
            self.game.action_trader(player_idx, sell_good=None)
        else:
            raise ValueError(f"Cannot pass in phase {self.game.current_phase.name}")

    def _get_obs(self):
        # Map game state to numpy arrays
        # ... logic to serialize self.game ...
        # Dummy return to satisfy interface during drafting
        return self.observation_space.sample()

    def _get_info(self):
        return {"current_phase": self.game.current_phase.name if self.game.current_phase else "INIT"}

    def _calculate_reward(self):
        # Calculate final VPs to assign rewards
        # +1 for winning, -1 for losing, etc.
        return 0.0

    def valid_action_mask(self):
        # Return a boolean array of size 200 indicating valid actions for current player
        mask = np.zeros(200, dtype=bool)
        # Calculate based on self.game state
        return mask
