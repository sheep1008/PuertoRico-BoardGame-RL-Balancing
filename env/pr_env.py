from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np

from env.engine import PuertoRicoGame
from configs.constants import Phase, Role, Good, TileType, BuildingType, BUILDING_DATA

# Discount factor for potential-based reward shaping (must match PPO gamma)
SHAPING_GAMMA = 0.99

class PuertoRicoEnv(AECEnv):
    metadata = {'render.modes': ['human'], 'name': 'puerto_rico_v0'}

    def __init__(self, num_players: int = 4, max_game_steps: int = 2000):
        super(PuertoRicoEnv, self).__init__()
        self.num_players = num_players
        self.max_game_steps = max_game_steps
        self.game = None
        
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.num_players))))
        
        self._action_spaces = {agent: self._define_action_space() for agent in self.possible_agents}
        self._observation_spaces = {agent: self._define_observation_space() for agent in self.possible_agents}

    def _define_action_space(self) -> spaces.Discrete:
        # === Action Mapping ===
        # 0-7:     Pick Role (Role.SETTLER=0 ~ Role.PROSPECTOR_2=7)
        # 8-13:    Settler - Face up plantation (index 0~5)
        # 14:      Settler - Take Quarry
        # 15:      Pass (phase-dependent: Settler/Builder/Trader/Captain/Mayor/Store/Craftsman)
        # 16-38:   Builder - Build building (BuildingType 0~22)
        # 39-43:   Trader - Sell good (Good 0~4)
        # 44-58:   Captain - Load (ship_idx * 5 + good_type)
        # 59-63:   Captain - Load via Wharf (Good 0~4)
        # 64-68:   Captain Store Windrose - Keep Good (Good 0~4)
        # 69-80:   Mayor - Toggle island slot (0~11)
        # 81-92:   Mayor - Toggle city slot (0~11)
        # 93-97:   Craftsman - Privilege good selection (Good 0~4)
        # 98-103:  (Deprecated) Settler WITH Hacienda - Face up plantation (index 0~5)
        # 104:     (Deprecated) Settler WITH Hacienda - Take Quarry
        # 105:     Settler WITH Hacienda - Pass (Only take Hacienda tile)
        # 106-110: Captain Store Warehouse (Good 0~4)
        # 111-199: (Reserved for future use)
        return spaces.Discrete(200)

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def _define_observation_space(self) -> spaces.Dict:
        obs_space = {
            "observation": spaces.Dict({
                "global_state": spaces.Dict({
                    "vp_chips": spaces.Box(low=-50, high=200, shape=(), dtype=np.int64),
                    "colonists_supply": spaces.Discrete(100),
                    "colonists_ship": spaces.Discrete(30),
                    "goods_supply": spaces.MultiDiscrete([15, 15, 15, 15, 15]), # Coffee, Tobacco, Corn, Sugar, Indigo
                    "cargo_ships_good": spaces.MultiDiscrete([6, 6, 6]), # 0-4 for good, 5 for None
                    "cargo_ships_load": spaces.MultiDiscrete([15, 15, 15]),
                    "trading_house": spaces.MultiDiscrete([6, 6, 6, 6]), # 0-4 for good, 5 for empty
                    "role_doubloons": spaces.MultiDiscrete([20] * 8), # Doubloons per role
                    "roles_available": spaces.MultiBinary(8), # 1 if available, 0 if taken
                    "face_up_plantations": spaces.MultiDiscrete([7] * (self.num_players + 1)), # 0-5 tile types, 6 for empty slot
                    "quarry_stack": spaces.Discrete(9),
                    "governor_idx": spaces.Discrete(self.num_players),
                    "current_player": spaces.Discrete(self.num_players),
                    "current_phase": spaces.Discrete(10)
                }),
                "players": spaces.Dict({
                    f"player_{i}": spaces.Dict({
                        "doubloons": spaces.Discrete(100),
                        "vp_chips": spaces.Box(low=0, high=200, shape=(), dtype=np.int64),
                        "goods": spaces.MultiDiscrete([15, 15, 15, 15, 15]), # Inventory
                        "island_tiles": spaces.MultiDiscrete([7] * 12),      # 0-5 plantations, 6 empty
                        "island_occupied": spaces.MultiBinary(12),
                        "city_buildings": spaces.MultiDiscrete([25] * 12),   # 0-23 buildings, 24 empty
                        "city_colonists": spaces.MultiDiscrete([4] * 12),    # Up to 3 per building, 0-3 range (size 4)
                        "unplaced_colonists": spaces.Discrete(20)
                    }) for i in range(self.num_players)
                })
            }),
            "action_mask": spaces.Box(low=0, high=1, shape=(200,), dtype=np.int8)
        }
        return spaces.Dict(obs_space)

    def reset(self, seed=None, options=None):
        if seed is not None:
            # Note: We should ideally pass seed to the engine, but for now we'll rely on global random
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game = PuertoRicoGame(self.num_players)
        self.game.start_game()
        self._game_step_count = 0
        
        # Determine starting player based on engine
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = f"player_{self.game.current_player_idx}"
        
        # Initialize potential tracking for reward shaping
        self._prev_potentials = {
            f"player_{i}": self._compute_potential(i) for i in range(self.num_players)
        }
        
        # Populate initial info
        for agent in self.agents:
            self.infos[agent] = self._get_info()

    def observe(self, agent: str):
        obs = self._get_obs()
        
        # Action mask should be generated for the requested agent
        # If it's not their turn, technically their mask is all 0.
        # But commonly we just return the full obs dictionary.
        agent_idx = self.agent_name_mapping[agent]
        if agent_idx == self.game.current_player_idx and not (self.terminations[agent] or self.truncations[agent]):
            mask = self.valid_action_mask().astype(np.int8)
        else:
            mask = np.zeros(200, dtype=np.int8)
            
        # PettingZoo standard convention requires action_mask inside the top-level Dict
        return {
            "observation": obs,
            "action_mask": mask
        }

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        p = self.game.players[player_idx]
        
        try:
            if 0 <= action <= 7:
                self.game.select_role(player_idx, Role(action))
                
            elif 8 <= action <= 14:
                # Settler Phase (No Hacienda)
                if action <= 13:
                    self.game.action_settler(player_idx, tile_choice=action-8)
                else:
                    self.game.action_settler(player_idx, tile_choice=-1) # Quarry
                    
            elif action == 15:
                # Pass
                self._handle_pass(player_idx)
                
            elif 16 <= action <= 38:
                # Builder Phase (23 buildings)
                b_type = BuildingType(action - 16)
                self.game.action_builder(player_idx, building_choice=b_type)
                
            elif 39 <= action <= 43:
                # Trader Phase
                g_type = Good(action - 39)
                self.game.action_trader(player_idx, sell_good=g_type)
                
            elif 44 <= action <= 58:
                # Captain Load (5 goods * 3 ships)
                idx = action - 44
                ship_idx = idx // 5
                g_type = Good(idx % 5)
                self.game.action_captain_load(player_idx, ship_idx, g_type)
                
            elif 59 <= action <= 63:
                # Captain Load Wharf
                g_type = Good(action - 59)
                self.game.action_captain_load(player_idx, -1, g_type)
                
            elif 64 <= action <= 68:
                # Captain Store Windrose
                g_type = Good(action - 64)
                self.game.action_captain_store_windrose(player_idx, g_type)

            elif 106 <= action <= 110:
                # Captain Store Warehouse
                g_type = Good(action - 106)
                self.game.action_captain_store_warehouse(player_idx, g_type)
                
            elif 69 <= action <= 92:
                # Mayor Toggle
                if action <= 80:
                    idx = action - 69
                    if idx < len(p.island_board):
                        tile = p.island_board[idx]
                        if tile.tile_type != TileType.EMPTY:
                            if not tile.is_occupied and p.unplaced_colonists > 0:
                                tile.is_occupied = True
                                p.unplaced_colonists -= 1
                            elif tile.is_occupied:
                                tile.is_occupied = False
                                p.unplaced_colonists += 1
                else:
                    idx = action - 81
                    if idx < len(p.city_board):
                        b = p.city_board[idx]
                        if b.building_type != BuildingType.EMPTY and b.building_type != BuildingType.OCCUPIED_SPACE:
                            max_cap = BUILDING_DATA[b.building_type][2]
                            if max_cap > 0:
                                old_c = b.colonists
                                new_c = (old_c + 1) % (max_cap + 1)
                                diff = new_c - old_c
                                if diff > 0: # Adding a colonist
                                    if p.unplaced_colonists >= diff:
                                        b.colonists = new_c
                                        p.unplaced_colonists -= diff
                                else: # Wrapped around, removing colonists
                                    b.colonists = new_c
                                    p.unplaced_colonists += (-diff)
                            
            elif 93 <= action <= 97:
                # Craftsman Privilege
                g_type = Good(action - 93)
                self.game.action_craftsman(player_idx, privilege_good=g_type)
                
            elif 98 <= action <= 104:
                # Settler Phase WITH Hacienda
                raise ValueError("Deprecated action combination. Agent should use action 105 to draw Hacienda first.")
                
            elif action == 105:
                # Hacienda Draw (face-down tile)
                self.game.action_hacienda_draw(player_idx)
                            
        except ValueError as e:
            # Invalid action penalty (reduced to -10 to avoid value function distortion)
            self.rewards[agent] = -10.0
            for a in self.agents:
                self.terminations[a] = True
                self.infos[a]["error"] = str(e)
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        self._game_step_count += 1
        done = self.game.check_game_end()
        truncated = False

        if done:
            # Natural game end
            all_rewards = self._calculate_all_rewards()
            for idx, r in enumerate(all_rewards):
                agent_name = f"player_{idx}"
                self.rewards[agent_name] = r
                self.terminations[agent_name] = True
            
            final_scores = self.game.get_scores()
            for a in self.agents:
                self.infos[a]["final_scores"] = final_scores

        elif self._game_step_count >= self.max_game_steps:
            # Truncation: game took too long (likely Mayor toggle loop with random policy)
            truncated = True
            all_rewards = self._calculate_all_rewards()
            for idx, r in enumerate(all_rewards):
                agent_name = f"player_{idx}"
                self.rewards[agent_name] = r * 0.5  # Discount truncated rewards
                self.truncations[agent_name] = True

            final_scores = self.game.get_scores()
            for a in self.agents:
                self.infos[a]["final_scores"] = final_scores
                self.infos[a]["truncated"] = True

        else:
            # Dense reward shaping: acting player gets ΔΦ = Φ(s') - Φ(s)
            # Clipped to prevent large cumulative accumulation through PettingZoo AEC mechanics.
            acting_idx = player_idx
            new_potential = self._compute_potential(acting_idx)
            old_potential = self._prev_potentials[f"player_{acting_idx}"]
            shaping_reward = max(-0.5, min(0.5, new_potential - old_potential))
            self.rewards[f"player_{acting_idx}"] = shaping_reward

            # Update potentials for all players (state may have changed globally)
            for i in range(self.num_players):
                a_name = f"player_{i}"
                self._prev_potentials[a_name] = self._compute_potential(i)
                if i != acting_idx:
                    self.rewards[a_name] = 0.0
                self.infos[a_name] = self._get_info()
                # Expose phase ID as int for hierarchical agent routing
                self.infos[a_name]["current_phase_id"] = int(
                    self.game.current_phase if self.game.current_phase is not None else 9
                )

        self.agent_selection = f"player_{self.game.current_player_idx}"
        self._accumulate_rewards()

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
            self.game.action_captain_store_pass(player_idx)
        elif self.game.current_phase == Phase.MAYOR:
            # During Mayor pass, we must solidify the board and call the engine pass
            p = self.game.players[player_idx]
            island_assgn = [t.is_occupied for t in p.island_board]
            city_assgn = [b.colonists for b in p.city_board]
            self.game.action_mayor_pass(player_idx, island_assgn, city_assgn)
        elif self.game.current_phase == Phase.CRAFTSMAN:
            self.game.action_craftsman(player_idx, privilege_good=None)
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
        # Pad cargo ships up to 3 (for 2-5 players, there are exactly 3 ships max)
        cargo_good += [5] * (3 - len(cargo_good))
        cargo_load += [0] * (3 - len(cargo_load))
        
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
        max_face_up = self.num_players + 1
        face_up_plantations += [6] * (max_face_up - len(face_up_plantations))
        
        global_state = {
            "vp_chips": np.array(game.vp_chips, dtype=np.int64),
            "colonists_supply": np.array(game.colonists_supply, dtype=np.int64),
            "colonists_ship": np.array(game.colonists_ship, dtype=np.int64),
            "goods_supply": np.array([game.goods_supply[Good(i)] for i in range(5)], dtype=np.int64),
            "cargo_ships_good": np.array(cargo_good, dtype=np.int64),
            "cargo_ships_load": np.array(cargo_load, dtype=np.int64),
            "trading_house": np.array(trading_house, dtype=np.int64),
            "role_doubloons": np.array(role_doubloons, dtype=np.int64),
            "roles_available": np.array(roles_available, dtype=np.int8),
            "face_up_plantations": np.array(face_up_plantations, dtype=np.int64),
            "quarry_stack": np.array(game.quarry_stack, dtype=np.int64),
            "governor_idx": np.array(game.governor_idx, dtype=np.int64),
            "current_player": np.array(game.current_player_idx, dtype=np.int64),
            "current_phase": np.array(game.current_phase if game.current_phase is not None else 9, dtype=np.int64)
        }

        # Player States
        players_obs = {}
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
            city_buildings += [24] * (12 - len(city_buildings))
            city_col += [0] * (12 - len(city_col))
            
            player_dict = {
                "doubloons": np.array(p.doubloons, dtype=np.int64),
                "vp_chips": np.array(p.vp_chips, dtype=np.int64),
                "goods": np.array([p.goods[Good(g)] for g in range(5)], dtype=np.int64),
                "island_tiles": np.array(island_tiles, dtype=np.int64),
                "island_occupied": np.array(island_occ, dtype=np.int8),
                "city_buildings": np.array(city_buildings, dtype=np.int64),
                "city_colonists": np.array(city_col, dtype=np.int64),
                "unplaced_colonists": np.array(p.unplaced_colonists, dtype=np.int64)
            }
            players_obs[f"player_{i}"] = player_dict

        return {
            "global_state": global_state,
            "players": players_obs
        }

    def _get_info(self):
        info = {"current_phase": self.game.current_phase.name if self.game.current_phase else "INIT"}
        if getattr(self, 'game', None) and self.game.check_game_end():
            info["final_scores"] = self.game.get_scores()
            if hasattr(self, '_final_rewards'):
                info["player_rewards"] = self._final_rewards
        return info

    def _compute_potential(self, player_idx: int) -> float:
        """Potential function for reward shaping (Ng et al., 1999).
        Higher potential = more developed player state.
        Provably preserves optimal policy when used as: gamma * Phi(s') - Phi(s).
        """
        p = self.game.players[player_idx]
        phi = 0.0
        phi += p.vp_chips * 0.05                  # VP chip accumulation
        phi += p.doubloons * 0.01                  # Economic power
        # Occupied production buildings contribute to future goods production
        occupied_production = sum(
            1 for b in p.city_board
            if BUILDING_DATA[b.building_type][5] is not None and b.colonists > 0
        )
        phi += occupied_production * 0.03
        phi += sum(p.goods.values()) * 0.005       # Current stockpile
        # Occupied island tiles = productive land
        occupied_island = sum(1 for t in p.island_board if t.is_occupied)
        phi += occupied_island * 0.01
        return phi

    def _calculate_all_rewards(self) -> list[float]:
        """Terminal reward: competitive win/loss + score margin.
        Dense shaping is handled separately via potentials in step().
        """
        scores = self.game.get_scores()
        totals = [vp + tb * 0.0001 for vp, tb in scores]
        
        rewards = []
        for i in range(self.num_players):
            my_total = totals[i]
            max_opp_total = max(totals[j] for j in range(self.num_players) if j != i)
            
            is_winner = my_total >= max_opp_total
            win_reward = 1.0 if is_winner else -1.0
            # Increased margin weight for better score differentiation
            margin = (my_total - max_opp_total) * 0.02
            
            rewards.append(win_reward + margin)
            
        return rewards

    def valid_action_mask(self):
        mask = np.zeros(200, dtype=bool)
        game = self.game
        p = game.players[game.current_player_idx]
        phase = game.current_phase

        if phase == Phase.END_ROUND or phase is None:
            for r in game.available_roles:
                mask[r.value] = True
                
        elif phase == Phase.SETTLER:
            mask[15] = True # Pass
            can_hacienda = p.is_building_occupied(BuildingType.HACIENDA) and game.plantation_stack and getattr(game, '_hacienda_used', False) is False
            
            for i in range(len(game.face_up_plantations)):
                if p.empty_island_spaces > 0:
                    mask[8 + i] = True
            
            can_quarry = (game.current_player_idx == game.active_role_player_idx()) or p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)
            if can_quarry and game.quarry_stack > 0 and p.empty_island_spaces > 0:
                mask[14] = True
                    
            if can_hacienda and p.empty_island_spaces > 0:
                mask[105] = True # Standalone Hacienda Draw
                
        elif phase == Phase.BUILDER:
            mask[15] = True # Pass
            has_privilege = (game.current_player_idx == game.active_role_player_idx())
            active_quarries = sum(1 for t in p.island_board if t.tile_type == TileType.QUARRY and t.is_occupied)
                
            for b_type, count in game.building_supply.items():
                if count > 0 and not p.has_building(b_type):
                    base_cost = BUILDING_DATA[b_type][0]
                    # Max reduction depends on building column. Base VP equals the column number (1 to 4).
                    max_q = BUILDING_DATA[b_type][1]
                        
                    quarry_discount = min(active_quarries, max_q)
                    privilege_discount = 1 if has_privilege else 0
                    final_cost = max(0, base_cost - quarry_discount - privilege_discount)
                    
                    if p.doubloons >= final_cost: 
                        mask[16 + b_type.value] = True
                        
        elif phase == Phase.TRADER:
            mask[15] = True # Pass
            if len(game.trading_house) < 4:
                has_office = p.is_building_occupied(BuildingType.OFFICE)
                for g in Good:
                    if p.goods[g] > 0:
                        if g not in game.trading_house or has_office:
                            mask[39 + g.value] = True
                            
        elif phase == Phase.CAPTAIN:
            # Need to find valid ship/good combos
            can_load_anything = False
            
            # For each good, find the maximum loadable amount across all valid ships
            max_loadable_for_good = {g: 0 for g in Good}
            allowed_ships_for_good = {g: [] for g in Good}
            
            for ship_idx, ship in enumerate(game.cargo_ships):
                if not ship.is_full:
                    for g in Good:
                        if p.goods[g] > 0:
                            allowed = False
                            if ship.good_type is None:
                                other_has_it = any(os.good_type == g for i, os in enumerate(game.cargo_ships) if i != ship_idx)
                                if not other_has_it:
                                    allowed = True
                            elif ship.good_type == g:
                                allowed = True
                                
                            if allowed:
                                potential_load = min(p.goods[g], ship.capacity - ship.current_load)
                                allowed_ships_for_good[g].append((ship_idx, potential_load))
                                max_loadable_for_good[g] = max(max_loadable_for_good[g], potential_load)
                                
            for g, ships in allowed_ships_for_good.items():
                max_load = max_loadable_for_good[g]
                for ship_idx, potential_load in ships:
                    if potential_load == max_load:
                        mask[44 + (ship_idx * 5) + g.value] = True
                        can_load_anything = True
                                
            # Wharf
            if p.is_building_occupied(BuildingType.WHARF) and not game._wharf_used.get(game.current_player_idx, False):
                for g in Good:
                    if p.goods[g] > 0:
                        mask[59 + g.value] = True
                        

            # Pass only allowed if cannot load anything
            if not can_load_anything:
                mask[15] = True
                
        elif phase == Phase.CAPTAIN_STORE:
            assign = game._storage_assignments[game.current_player_idx]
            unstored_types = [g for g in Good if p.goods[g] > 0 and g != assign['windrose'] and g not in assign['warehouses']]
            
            max_wh_slots = 0
            if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE): max_wh_slots += 1
            if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE): max_wh_slots += 2
            
            has_empty_windrose = (assign['windrose'] is None)
            has_empty_wh = len(assign['warehouses']) < max_wh_slots
            
            can_pass = True
            if len(unstored_types) > 0:
                if has_empty_windrose or has_empty_wh:
                    can_pass = False
                    
            if can_pass:
                mask[15] = True
                
            for g in Good:
                if p.goods[g] > 0:
                    if has_empty_windrose and assign['windrose'] is None and g not in assign['warehouses']:
                        mask[64 + g.value] = True
                    if has_empty_wh and g != assign['windrose'] and g not in assign['warehouses']:
                        mask[106 + g.value] = True
                    
        elif phase == Phase.MAYOR:
            # Toggle actions
            
            # Check if passing is allowed
            total_placed = sum([1 for t in p.island_board if t.is_occupied]) + sum([b.colonists for b in p.city_board])
            empty_island = sum([1 for t in p.island_board if t.tile_type != TileType.EMPTY and not t.is_occupied])
            empty_city = 0
            for b in p.city_board:
                if b.building_type != BuildingType.EMPTY and b.building_type != BuildingType.OCCUPIED_SPACE:
                    max_cap = BUILDING_DATA[b.building_type][2]
                    empty_city += (max_cap - b.colonists)
                    
            leftover_colonists = p.total_colonists_owned - total_placed
            can_pass = (leftover_colonists == 0) or (empty_island == 0 and empty_city == 0)
            
            if can_pass:
                mask[15] = True # Pass (Submit)
                
            for i in range(len(p.island_board)):
                if p.island_board[i].tile_type != TileType.EMPTY:
                    mask[69 + i] = True
            for i in range(len(p.city_board)):
                if p.city_board[i].building_type != BuildingType.EMPTY and p.city_board[i].building_type != BuildingType.OCCUPIED_SPACE:
                    mask[81 + i] = True

        elif phase == Phase.CRAFTSMAN:
            has_privilege = (game.current_player_idx == game.active_role_player_idx())
            mask[15] = True # Can always pass 
            if has_privilege:
                for g in getattr(game, '_craftsman_produced_kinds', []):
                    if game.goods_supply[g] > 0:
                        mask[93 + g.value] = True
            
        elif phase == Phase.PROSPECTOR:
            mask[15] = True

        return mask
