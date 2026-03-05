import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.engine import PuertoRicoGame
from configs.constants import Phase, Role, Good, TileType, BuildingType, BUILDING_DATA

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
        obs_space = {
            "global_state": spaces.Dict({
                "vp_chips": spaces.Discrete(123),
                "colonists_supply": spaces.Discrete(100),
                "colonists_ship": spaces.Discrete(10),
                "goods_supply": spaces.MultiDiscrete([10, 10, 11, 12, 12]), # Coffee, Tobacco, Corn, Sugar, Indigo
                "cargo_ships_good": spaces.MultiDiscrete([6, 6, 6]), # 0-4 for good, 5 for None
                "cargo_ships_load": spaces.MultiDiscrete([10, 10, 10]),
                "trading_house": spaces.MultiDiscrete([6, 6, 6, 6]), # 0-4 for good, 5 for empty
                "role_doubloons": spaces.MultiDiscrete([10] * 8), # Doubloons per role
                "roles_available": spaces.MultiBinary(8), # 1 if available, 0 if taken
                "face_up_plantations": spaces.MultiDiscrete([7] * 6), # 0-5 tile types, 6 for empty slot
                "quarry_stack": spaces.Discrete(9),
                "current_player": spaces.Discrete(self.num_players),
                "current_phase": spaces.Discrete(10)
            }),
            "players": spaces.Tuple([
                spaces.Dict({
                    "doubloons": spaces.Discrete(50),
                    "vp_chips": spaces.Discrete(100), # Assuming max 100 for safety
                    "goods": spaces.MultiDiscrete([10, 10, 11, 12, 12]), # Inventory
                    "island_tiles": spaces.MultiDiscrete([7] * 12),      # 0-5 plantations, 6 empty
                    "island_occupied": spaces.MultiBinary(12),
                    "city_buildings": spaces.MultiDiscrete([24] * 12),   # 0-22 buildings, 23 empty
                    "city_colonists": spaces.MultiDiscrete([4] * 12),    # Up to 3 per building, 0-3 range (size 4)
                    "unplaced_colonists": spaces.Discrete(20)
                }) for _ in range(self.num_players)
            ])
        }
        return spaces.Dict(obs_space)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = PuertoRicoGame(self.num_players)
        self.game.start_game()
        return self._get_obs(), self._get_info()

    def step(self, action):
        player_idx = self.game.current_player_idx
        p = self.game.players[player_idx]
        
        try:
            if 0 <= action <= 7:
                self.game.select_role(player_idx, Role(action))
                
            elif 8 <= action <= 14:
                # Settler Phase
                if action <= 13:
                    self.game.action_settler(player_idx, tile_choice=action-8)
                else:
                    self.game.action_settler(player_idx, tile_choice=-1) # Quarry (action 14)
                    
            elif action == 15:
                # Use Hacienda
                self.game.action_settler(player_idx, tile_choice=-2, use_hacienda=True)
                
            elif action == 16:
                # Pass
                self._handle_pass(player_idx)
                
            elif 17 <= action <= 39:
                # Builder Phase (23 buildings)
                b_type = BuildingType(action - 17)
                self.game.action_builder(player_idx, building_choice=b_type)
                
            elif 40 <= action <= 44:
                # Trader Phase
                g_type = Good(action - 40)
                self.game.action_trader(player_idx, sell_good=g_type)
                
            elif action == 45:
                # Craftsman Privilege (Choose extra good)
                # For RL simplicity, we'll auto-resolve Craftsman if no privilege.
                # If privilege, they use 40-44 mapping? Let's give specific actions.
                # Actually, in Craftsman, the engine handles it silently. We just need to give them a choice
                pass # Revisit if we add craftsman choice actions
                
            elif 46 <= action <= 60:
                # Captain Load (5 goods * 3 ships)
                idx = action - 46
                ship_idx = idx // 5
                g_type = Good(idx % 5)
                self.game.action_captain_load(player_idx, ship_idx, g_type)
                
            elif 61 <= action <= 65:
                # Captain Load Wharf
                g_type = Good(action - 61)
                self.game.action_captain_load(player_idx, -1, g_type)
                
            elif 66 <= action <= 80:
                # Captain Store (simplified: keep 1 distinct good)
                g_type = Good(action - 66)
                self.game.action_captain_store(player_idx, {g_type: 1})
                # Note: Full warehouse combinations are complex. We will simplify for the baseline RL agent
                # to just picking 1 good to store, or passing (action 16) to store nothing.
                
            elif 81 <= action <= 104:
                # Mayor Toggle
                # 81-92: Toggle Island (12 slots)
                # 93-104: Toggle City (12 slots)
                if action <= 92:
                    idx = action - 81
                    if idx < len(p.island_board):
                        p.island_board[idx].is_occupied = not p.island_board[idx].is_occupied
                else:
                    idx = action - 93
                    if idx < len(p.city_board):
                        # toggle 0 -> 1 -> max -> 0
                        b = p.city_board[idx]
                        max_cap = BUILDING_DATA[b.building_type][2]
                        if max_cap > 0:
                            b.colonists = (b.colonists + 1) % (max_cap + 1)
                            
        except ValueError as e:
            # Invalid action taken, though mask should prevent this.
            # Penalize heavily if happens
            return self._get_obs(), -50.0, True, False, {"error": str(e)}

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
        elif self.game.current_phase == Phase.CAPTAIN:
            self.game.action_captain_pass(player_idx)
        elif self.game.current_phase == Phase.CAPTAIN_STORE:
            self.game.action_captain_store(player_idx, {})
        elif self.game.current_phase == Phase.MAYOR:
            # During Mayor pass, we must solidify the board and call the engine pass
            p = self.game.players[player_idx]
            island_assgn = [t.is_occupied for t in p.island_board]
            city_assgn = [b.colonists for b in p.city_board]
            self.game.action_mayor_pass(player_idx, island_assgn, city_assgn)
        else:
            raise ValueError(f"Cannot pass in phase {self.game.current_phase.name if self.game.current_phase else 'INIT'}")

    def _get_obs(self):
        game = self.game

        # Global State
        cargo_good = []
        cargo_load = []
        for ship in game.cargo_ships:
            cargo_good.append(ship.good_type if ship.good_type is not None else 5)
            cargo_load.append(ship.current_load)
        # Pad cargo ships up to 3 (for 3-5 players, there are exactly 3 ships)
        
        trading_house = [g for g in game.trading_house]
        trading_house += [5] * (4 - len(trading_house))
        
        role_doubloons = []
        roles_available = []
        for i in range(8):
            try:
                role = Role(i)
                doubloons = game.role_doubloons.get(role, 0)
                available = 1 if role in game.available_roles else 0
            except ValueError:
                # E.g. prospectors in 3 player games
                doubloons = 0
                available = 0
            role_doubloons.append(doubloons)
            roles_available.append(available)
            
        face_up_plantations = [t for t in game.face_up_plantations]
        face_up_plantations += [6] * (6 - len(face_up_plantations))
        
        global_state = {
            "vp_chips": game.vp_chips,
            "colonists_supply": game.colonists_supply,
            "colonists_ship": game.colonists_ship,
            "goods_supply": np.array([game.goods_supply[Good(i)] for i in range(5)], dtype=np.int64),
            "cargo_ships_good": np.array(cargo_good, dtype=np.int64),
            "cargo_ships_load": np.array(cargo_load, dtype=np.int64),
            "trading_house": np.array(trading_house, dtype=np.int64),
            "role_doubloons": np.array(role_doubloons, dtype=np.int64),
            "roles_available": np.array(roles_available, dtype=np.int8),
            "face_up_plantations": np.array(face_up_plantations, dtype=np.int64),
            "quarry_stack": game.quarry_stack,
            "current_player": game.current_player_idx,
            "current_phase": game.current_phase if game.current_phase is not None else 9
        }

        # Player States
        players_obs = []
        for i in range(self.num_players):
            p = game.players[i]
            
            island_tiles = []
            island_occ = []
            for t in p.island_board:
                island_tiles.append(t.tile_type)
                island_occ.append(1 if t.is_occupied else 0)
            island_tiles += [6] * (12 - len(island_tiles))
            island_occ += [0] * (12 - len(island_occ))
            
            city_buildings = []
            city_col = []
            for b in p.city_board:
                city_buildings.append(b.building_type)
                city_col.append(b.colonists)
            city_buildings += [23] * (12 - len(city_buildings))
            city_col += [0] * (12 - len(city_col))
            
            player_dict = {
                "doubloons": p.doubloons,
                "vp_chips": p.vp_chips,
                "goods": np.array([p.goods[Good(g)] for g in range(5)], dtype=np.int64),
                "island_tiles": np.array(island_tiles, dtype=np.int64),
                "island_occupied": np.array(island_occ, dtype=np.int8),
                "city_buildings": np.array(city_buildings, dtype=np.int64),
                "city_colonists": np.array(city_col, dtype=np.int64),
                "unplaced_colonists": p.unplaced_colonists
            }
            players_obs.append(player_dict)

        return {
            "global_state": global_state,
            "players": tuple(players_obs)
        }

    def _get_info(self):
        return {"current_phase": self.game.current_phase.name if self.game.current_phase else "INIT"}

    def _calculate_reward(self):
        """
        Calculates reward at the end of the game based on final scores.
        Winner gets +1, others get 0 or negative relative depending on score difference.
        Let's give the actual scores for now and we can normalize later.
        """
        scores = self.game.get_scores()
        # The agent's reward is its own score. 
        # Optional: shape it so winning gives massive bonus.
        return float(scores[self.game.current_player_idx])

    def valid_action_mask(self):
        mask = np.zeros(200, dtype=bool)
        game = self.game
        p = game.players[game.current_player_idx]
        phase = game.current_phase

        if phase == Phase.END_ROUND or phase is None:
            for r in game.available_roles:
                mask[r.value] = True
                
        elif phase == Phase.SETTLER:
            mask[16] = True # Pass
            for i in range(len(game.face_up_plantations)):
                if p.empty_island_spaces > 0:
                    mask[8 + i] = True
            
            can_quarry = (game.current_player_idx == game.active_role_player_idx()) or p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)
            if can_quarry and game.quarry_stack > 0 and p.empty_island_spaces > 0:
                mask[14] = True
                
            if p.is_building_occupied(BuildingType.HACIENDA) and game.plantation_stack:
                mask[15] = True
                
        elif phase == Phase.BUILDER:
            mask[16] = True # Pass
            for b_type, count in game.building_supply.items():
                if count > 0 and not p.has_building(b_type):
                    # Rough cost check (ignoring quarry discounts for pure mask, allowing engine to reject if really too poor,
                    # but let's do safe base mask)
                    if p.doubloons + 4 >= BUILDING_DATA[b_type][0]: # At least structurally possible
                        mask[17 + b_type.value] = True
                        
        elif phase == Phase.TRADER:
            mask[16] = True # Pass
            if len(game.trading_house) < 4:
                has_office = p.is_building_occupied(BuildingType.OFFICE)
                for g in Good:
                    if p.goods[g] > 0:
                        if g not in game.trading_house or has_office:
                            mask[40 + g.value] = True
                            
        elif phase == Phase.CAPTAIN:
            # Need to find valid ship/good combos
            can_load_anything = False
            for ship_idx, ship in enumerate(game.cargo_ships):
                if not ship.is_full:
                    for g in Good:
                        if p.goods[g] > 0:
                            # Is good allowed on this ship?
                            allowed = False
                            if ship.good_type is None:
                                # Check if no OTHER ship has this good
                                other_has_it = any(os.good_type == g for i, os in enumerate(game.cargo_ships) if i != ship_idx)
                                if not other_has_it:
                                    allowed = True
                            elif ship.good_type == g:
                                allowed = True
                                
                            if allowed:
                                mask[46 + (ship_idx * 5) + g.value] = True
                                can_load_anything = True
                                
            # Wharf
            if p.is_building_occupied(BuildingType.WHARF) and not game._wharf_used.get(game.current_player_idx, False):
                for g in Good:
                    if p.goods[g] > 0:
                        mask[61 + g.value] = True
                        can_load_anything = True
                        
            # Pass only allowed if cannot load anything
            if not can_load_anything:
                mask[16] = True
                
        elif phase == Phase.CAPTAIN_STORE:
            mask[16] = True # Can always store nothing
            for g in Good:
                if p.goods[g] >= 1:
                    mask[66 + g.value] = True
                    
        elif phase == Phase.MAYOR:
            # Toggle actions
            mask[16] = True # Pass (Submit)
            for i in range(len(p.island_board)):
                if p.island_board[i].tile_type != TileType.EMPTY:
                    mask[81 + i] = True
            for i in range(len(p.city_board)):
                if p.city_board[i].building_type != BuildingType.EMPTY:
                    mask[93 + i] = True

        elif phase == Phase.CRAFTSMAN:
            # Since auto-producing, the only action is to pass (or pick privilege good if implemented)
            # For now just let them pass.
            mask[16] = True
            
        elif phase == Phase.PROSPECTOR:
            mask[16] = True

        return mask
