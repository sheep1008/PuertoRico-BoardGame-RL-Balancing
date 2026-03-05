import random
from typing import List, Dict, Optional

from configs.constants import (
    Phase, Role, Good, TileType, BuildingType,
    PLANTATION_COUNTS, QUARRY_COUNT, VP_CHIPS_SETUP,
    COLONIST_SHIP_SETUP, COLONIST_SUPPLY_SETUP,
    STARTING_DOUBLOONS, CARGO_SHIPS_SETUP, BUILDING_DATA
)
from env.components import CargoShip
from env.player import Player

class PuertoRicoGame:
    def __init__(self, num_players: int):
        if num_players not in [3, 4, 5]:
            raise ValueError("Puerto Rico only supports 3, 4, or 5 players.")
        
        self.num_players = num_players
        self.players: List[Player] = [Player(i) for i in range(num_players)]
        
        # Global State Variables
        self.vp_chips = VP_CHIPS_SETUP[num_players]
        self.colonists_supply = COLONIST_SUPPLY_SETUP[num_players]
        self.colonists_ship = COLONIST_SHIP_SETUP[num_players]
        
        # Track goods available in supply
        self.goods_supply = {
            Good.COFFEE: 9,
            Good.TOBACCO: 9,
            Good.CORN: 10,
            Good.SUGAR: 11,
            Good.INDIGO: 11
        }
        
        # Cargo ships setup
        self.cargo_ships: List[CargoShip] = [
            CargoShip(capacity=cap) for cap in CARGO_SHIPS_SETUP[num_players]
        ]
        
        # Trading house (holds up to 4 goods)
        self.trading_house: List[Good] = []
        
        # Tile supplies
        self.quarry_stack = QUARRY_COUNT
        self.plantation_stack: List[TileType] = self._init_plantation_stack()
        self.face_up_plantations: List[TileType] = []
        
        # Building supplies
        self.building_supply: Dict[BuildingType, int] = {
            b_type: data[3] for b_type, data in BUILDING_DATA.items()
        }
        
        # Roles and Doubloons
        self.available_roles: List[Role] = self._init_roles()
        self.role_doubloons: Dict[Role, int] = {role: 0 for role in self.available_roles}
        self.roles_in_play: List[Role] = [] # Roles currently picked by players this round
        
        # State Machine Tracking
        self.governor_idx = random.randint(0, num_players - 1)
        self.current_player_idx = self.governor_idx
        self.current_phase: Optional[Phase] = None
        self.active_role: Optional[Role] = None
        self.active_role_player: int = -1
        
        # Phase-specific tracking flags
        self._captain_privilege_used = False
        self._wharf_used = {i: False for i in range(num_players)}
        self._captain_passed_players = set()
        self._storage_assignments = {i: {'windrose': None, 'warehouses': []} for i in range(self.num_players)}
        self._hacienda_used = False
        self._colonists_ship_underfilled = False # Evaluated at end of Mayor phase
        
        # For sequential tracking within a phase
        self._phase_turn_offset = 0  # 0 to num_players-1, denotes who goes next in the action circle
        self.players_taken_action = 0

        # Discarded plantations
        self.plantation_discard: List[TileType] = []

        self._setup_players()

    def _init_plantation_stack(self) -> List[TileType]:
        stack = []
        for t_type, count in PLANTATION_COUNTS.items():
            stack.extend([t_type] * count)
        random.shuffle(stack)
        return stack

    def _init_roles(self) -> List[Role]:
        roles = [
            Role.SETTLER, Role.MAYOR, Role.BUILDER, 
            Role.CRAFTSMAN, Role.TRADER, Role.CAPTAIN
        ]
        if self.num_players >= 4:
            roles.append(Role.PROSPECTOR_1)
        if self.num_players >= 5:
            roles.append(Role.PROSPECTOR_2)
        return roles

    def _setup_players(self):
        # Deal initial money
        start_money = STARTING_DOUBLOONS[self.num_players]
        for p in self.players:
            p.add_doubloons(start_money)
            
        # Deal initial plantations
        # 3 players: governor=Indigo, 2nd=Indigo, 3rd=Corn
        # 4 players: gov=Indigo, 2nd=Indigo, 3rd=Corn, 4th=Corn
        # 5 players: gov=Indigo, 2nd=Indigo, 3rd=Indigo, 4th=Corn, 5th=Corn
        
        num_indigo = 2 if self.num_players < 5 else 3
        
        for i in range(self.num_players):
            actual_idx = (self.governor_idx + i) % self.num_players
            p = self.players[actual_idx]
            
            if i < num_indigo:
                p.place_plantation(TileType.INDIGO_PLANTATION)
                self.plantation_stack.remove(TileType.INDIGO_PLANTATION)
            else:
                p.place_plantation(TileType.CORN_PLANTATION)
                self.plantation_stack.remove(TileType.CORN_PLANTATION)

    def _deal_face_up_plantations(self):
        target = self.num_players + 1
        needed = target - len(self.face_up_plantations)
        
        for _ in range(needed):
            if not self.plantation_stack and self.plantation_discard:
                random.shuffle(self.plantation_discard)
                self.plantation_stack = self.plantation_discard
                self.plantation_discard = []
                
            if self.plantation_stack:
                self.face_up_plantations.append(self.plantation_stack.pop())
            else:
                break

    def start_game(self):
        """Starts the game by dealing face up plantations and setting the initial phase."""
        self._deal_face_up_plantations()
        self.current_phase = Phase.END_ROUND # Force role selection
        self.active_role = None

    def get_current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def _next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def _end_phase(self):
        """Called when all players have acted in a phase."""
        self.active_role = None
        
        # Who is next to choose a role?
        # Roles are chosen in clockwise order starting from governor
        if len(self.roles_in_play) < self.num_players:
            # Not everyone has chosen a role yet this round
            # Next player to choose is the one left of whoever chose last
            # Actually, turn offset doesn't help here. We need to find the next player
            # who hasn't chosen a role yet. Since it's clockwise:
            first_chooser = self.governor_idx
            next_chooser_offset = len(self.roles_in_play)
            next_player = (first_chooser + next_chooser_offset) % self.num_players
            self.current_player_idx = next_player
            self.current_phase = Phase.END_ROUND # Technically "Wait for role select"
        else:
            # All players have chosen a role. End the round.
            self._end_round()

    def _end_round(self):
        """Ends the current round, updates governor, adds doubloons to unpicked roles."""
        # Add 1 doubloon to each unused role
        for role in self.available_roles:
            self.role_doubloons[role] += 1
            
        # Return used roles
        self.available_roles.extend(self.roles_in_play)
        self.roles_in_play.clear()
        
        # Check game end condition loosely here, but we will have a explicit check
        # Pass Governor
        self.governor_idx = (self.governor_idx + 1) % self.num_players
        self.current_player_idx = self.governor_idx
        self.current_phase = Phase.END_ROUND

    def check_game_end(self) -> bool:
        """Returns True if any game ending condition is met at the end of the round."""
        # A round is fully over when roles_in_play has been cleared by _end_round.
        if self.current_phase != Phase.END_ROUND or len(self.roles_in_play) > 0:
            return False
            
        if self._colonists_ship_underfilled:
            return True
        if self.vp_chips <= 0:
            return True
        for p in self.players:
            if p.empty_city_spaces == 0:
                return True
        return False

    def get_scores(self) -> List[int]:
        """Calculates final scores for all players."""
        scores = []
        for p in self.players:
            score = p.vp_chips
            
            # Base building VPs
            for b in p.city_board:
                score += BUILDING_DATA[b.building_type][1]
                
            # Large Building Bonuses (must be occupied)
            for b in p.city_board:
                b_data = BUILDING_DATA[b.building_type]
                is_large = b_data[4]
                if is_large and b.colonists > 0:
                    b_type = b.building_type
                    
                    if b_type == BuildingType.GUILDHALL:
                        for other_b in p.city_board:
                            ob_data = BUILDING_DATA[other_b.building_type]
                            produces_good = ob_data[5] is not None
                            if produces_good:
                                # Small vs Large production
                                if other_b.building_type in (BuildingType.SMALL_INDIGO_PLANT, BuildingType.SMALL_SUGAR_MILL):
                                    score += 1
                                else:
                                    score += 2
                                    
                    elif b_type == BuildingType.RESIDENCE:
                        # Depends on filled island spaces (plantations + quarries)
                        filled_island = len(p.island_board)
                        if filled_island <= 9:
                            score += 4
                        elif filled_island == 10:
                            score += 5
                        elif filled_island == 11:
                            score += 6
                        elif filled_island == 12:
                            score += 7
                            
                    elif b_type == BuildingType.FORTRESS:
                        total_col = p.total_colonists_owned
                        score += total_col // 3
                        
                    elif b_type == BuildingType.CUSTOMS_HOUSE:
                        score += p.vp_chips // 4
                        
                    elif b_type == BuildingType.CITY_HALL:
                        # 1 VP per violet building
                        violet_count = sum(1 for v_b in p.city_board if BUILDING_DATA[v_b.building_type][5] is None)
                        score += violet_count
                        
            scores.append(score)
        return scores

    def select_role(self, player_idx: int, role: Role):
        """Player selects a role. Transitions game to that phase."""
        if player_idx != self.current_player_idx:
            raise ValueError("Not this player's turn to pick a role!")
        if role not in self.available_roles:
            raise ValueError("Role not available!")
            
        p = self.players[player_idx]
        
        # Give doubloons
        earned_doubloons = self.role_doubloons[role]
        if earned_doubloons > 0:
            p.add_doubloons(earned_doubloons)
            self.role_doubloons[role] = 0
            
        self.available_roles.remove(role)
        self.roles_in_play.append(role)
        self.active_role = role
        
        self.active_role_player = player_idx # The player who picked acting first
        self.players_taken_action = 0
        self.current_player_idx = player_idx 
        
        if role == Role.SETTLER:
            self.current_phase = Phase.SETTLER
        elif role == Role.MAYOR:
            self.current_phase = Phase.MAYOR
            # Privilege: take 1 from supply
            if self.colonists_supply > 0:
                p.unplaced_colonists += 1
                self.colonists_supply -= 1
            # Distribute colonists on ship immediately
            idx = player_idx
            while self.colonists_ship > 0:
                self.players[idx].unplaced_colonists += 1
                self.colonists_ship -= 1
                idx = (idx + 1) % self.num_players
        elif role == Role.BUILDER:
            self.current_phase = Phase.BUILDER
        elif role == Role.CRAFTSMAN:
            self.current_phase = Phase.CRAFTSMAN
        elif role == Role.TRADER:
            self.current_phase = Phase.TRADER
        elif role == Role.CAPTAIN:
            self.current_phase = Phase.CAPTAIN
            self._captain_passed_players = set()
            self._wharf_used = {i: False for i in range(self.num_players)}
        elif role in (Role.PROSPECTOR_1, Role.PROSPECTOR_2):
            self.current_phase = Phase.PROSPECTOR
            # Privilege: 1 doubloon
            p.add_doubloons(1)
            # Immediately end phase as prospector has no action for others
            self._end_phase()
            
    def _advance_phase_turn(self):
        """Advances turn to next player in the current phase."""
        if self.current_phase == Phase.CAPTAIN:
            if len(self._captain_passed_players) >= self.num_players:
                self.current_phase = Phase.CAPTAIN_STORE
                self.players_taken_action = 0
                self.current_player_idx = self.active_role_player_idx()
                self._storage_assignments = {i: {'windrose': None, 'warehouses': []} for i in range(self.num_players)}
            else:
                self._next_player()
                while self.current_player_idx in self._captain_passed_players:
                    self._next_player()
        else:
            self.players_taken_action += 1
            self._hacienda_used = False
            if self.players_taken_action >= self.num_players:
                # Phase is over
                self._execute_phase_cleanup()
                self._end_phase()
            else:
                self._next_player()



    def active_role_player_idx(self) -> int:
        """Returns the idx of the player who picked the current active role."""
        return self.active_role_player

    def action_craftsman(self, player_idx: int, privilege_good: Optional[Good] = None):
        """
        Auto produces goods. If privilege holder, they must specify `privilege_good` to take 1 extra.
        If they can't produce anything, privilege_good is ignored.
        """
        if self.current_phase != Phase.CRAFTSMAN or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Craftsman phase.")
            
        p = self.players[player_idx]
        has_privilege = (player_idx == self.active_role_player_idx())
        
        production = {g: 0 for g in Good}
        corn_plantations = sum(1 for t in p.island_board if t.tile_type == TileType.CORN_PLANTATION and t.is_occupied)
        production[Good.CORN] = corn_plantations
        
        raw_indigo = sum(1 for t in p.island_board if t.tile_type == TileType.INDIGO_PLANTATION and t.is_occupied)
        raw_sugar = sum(1 for t in p.island_board if t.tile_type == TileType.SUGAR_PLANTATION and t.is_occupied)
        raw_tobacco = sum(1 for t in p.island_board if t.tile_type == TileType.TOBACCO_PLANTATION and t.is_occupied)
        raw_coffee = sum(1 for t in p.island_board if t.tile_type == TileType.COFFEE_PLANTATION and t.is_occupied)
        
        cap_indigo = sum(b.colonists for b in p.city_board if b.building_type in (BuildingType.SMALL_INDIGO_PLANT, BuildingType.INDIGO_PLANT))
        cap_sugar = sum(b.colonists for b in p.city_board if b.building_type in (BuildingType.SMALL_SUGAR_MILL, BuildingType.SUGAR_MILL))
        cap_tobacco = sum(b.colonists for b in p.city_board if b.building_type == BuildingType.TOBACCO_STORAGE)
        cap_coffee = sum(b.colonists for b in p.city_board if b.building_type == BuildingType.COFFEE_ROASTER)
        
        production[Good.INDIGO] = min(raw_indigo, cap_indigo)
        production[Good.SUGAR] = min(raw_sugar, cap_sugar)
        production[Good.TOBACCO] = min(raw_tobacco, cap_tobacco)
        production[Good.COFFEE] = min(raw_coffee, cap_coffee)
        
        kinds_produced = []
        for g in Good:
            amount = min(production[g], self.goods_supply[g])
            if amount > 0:
                p.add_good(g, amount)
                self.goods_supply[g] -= amount
                kinds_produced.append(g)
                
        if p.is_building_occupied(BuildingType.FACTORY):
            kp = len(kinds_produced)
            if kp == 2: p.add_doubloons(1)
            elif kp == 3: p.add_doubloons(2)
            elif kp == 4: p.add_doubloons(3)
            elif kp == 5: p.add_doubloons(5)
            
        if has_privilege and kinds_produced and privilege_good in kinds_produced:
            if self.goods_supply[privilege_good] > 0:
                p.add_good(privilege_good, 1)
                self.goods_supply[privilege_good] -= 1
                
        self._advance_phase_turn()

    def action_trader(self, player_idx: int, sell_good: Optional[Good]):
        """
        sell_good: Good to sell, or None to pass.
        """
        if self.current_phase != Phase.TRADER or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Trader phase.")
            
        if sell_good is not None:
            p = self.players[player_idx]
            has_privilege = (player_idx == self.active_role_player_idx())
            
            if p.goods[sell_good] == 0:
                raise ValueError("Player does not have this good.")
            if len(self.trading_house) >= 4:
                raise ValueError("Trading house is full.")
                
            has_office = p.is_building_occupied(BuildingType.OFFICE)
            if sell_good in self.trading_house and not has_office:
                raise ValueError("Trading house already has this good and player has no Office.")
                
            # Process sale
            from configs.constants import GOOD_PRICES
            price = GOOD_PRICES[sell_good]
            
            if has_privilege:
                price += 1
            if p.is_building_occupied(BuildingType.SMALL_MARKET):
                price += 1
            if p.is_building_occupied(BuildingType.LARGE_MARKET):
                price += 2
                
            p.remove_good(sell_good, 1)
            p.add_doubloons(price)
            self.trading_house.append(sell_good)
            
        self._advance_phase_turn()

    # --- Phase Action Handlers --- 
    # These functions take the player index and their chosen action parameters,
    # validate the action, apply state changes, and then call _advance_phase_turn()

    def action_hacienda_draw(self, player_idx: int):
        """
        Allows a player with an occupied Hacienda to draw a face-down plantation tile.
        This does not end their turn. They must still call action_settler afterwards.
        """
        if self.current_phase != Phase.SETTLER or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Settler phase.")
        p = self.players[player_idx]
        if not p.is_building_occupied(BuildingType.HACIENDA):
            raise ValueError("Player does not have an active Hacienda.")
        if self._hacienda_used:
            raise ValueError("Hacienda already used this turn.")
        if p.empty_island_spaces <= 0:
            raise ValueError("Player has no empty island spaces.")
            
        if self.plantation_stack:
            drawn_tile = self.plantation_stack.pop()
            p.place_plantation(drawn_tile)
            # Note: Hospice does NOT apply to Hacienda-drawn tiles according to rules!
        self._hacienda_used = True

    def action_settler(self, player_idx: int, tile_choice: int):
        """
        tile_choice: index in self.face_up_plantations (0-len). 
        Set to -1 to take Quarry (if privilege or construction hut).
        Set to -2 to pass.
        """
        if self.current_phase != Phase.SETTLER or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Settler phase.")
            
        p = self.players[player_idx]
        has_privilege = (player_idx == self.active_role_player_idx())
        
        if p.empty_island_spaces <= 0 and tile_choice != -2:
            raise ValueError("Player has filled all island spaces and must pass.")
            
        tile_placed = False
        
        if tile_choice == -2: # Pass
            pass
        elif tile_choice == -1: # Quarry
            can_quarry = has_privilege or p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)
            if not can_quarry:
                raise ValueError("Cannot take quarry without privilege or construction hut.")
            if self.quarry_stack > 0 and p.empty_island_spaces > 0:
                p.place_plantation(TileType.QUARRY)
                self.quarry_stack -= 1
                tile_placed = True
        else: # Face up plantation
            if 0 <= tile_choice < len(self.face_up_plantations):
                tile = self.face_up_plantations.pop(tile_choice)
                if p.empty_island_spaces > 0:
                    p.place_plantation(tile)
                    tile_placed = True
            else:
                raise ValueError("Invalid plantation choice.")
                
        # Hospice check ONLY applies to the officially drafted tile
        if tile_placed and p.is_building_occupied(BuildingType.HOSPICE):
            if self.colonists_supply > 0:
                p.island_board[-1].is_occupied = True
                self.colonists_supply -= 1
                
        self._advance_phase_turn()

    def action_mayor_pass(self, player_idx: int, island_assignment: List[bool], city_assignment: List[int]):
        """
        Since Mayor phase involves combinatorial placement, the agent submits the final configuration.
        island_assignment: boolean list matching length of player's island_board
        city_assignment: integer list matching length of player's city_board (count of colonists in each)
        """
        if self.current_phase != Phase.MAYOR or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Mayor phase.")
            
        p = self.players[player_idx]
        
        # Validate total colonists
        total_placed = sum(island_assignment) + sum(city_assignment)
        if total_placed > p.total_colonists_owned:
            raise ValueError("Attempting to place more colonists than owned.")
            
        # Validate capacities
        if len(island_assignment) != len(p.island_board):
            raise ValueError("Mismatch in island board length.")
        if len(city_assignment) != len(p.city_board):
            raise ValueError("Mismatch in city board length.")
            
        for i, val in enumerate(island_assignment):
            p.island_board[i].is_occupied = val
            
        for i, val in enumerate(city_assignment):
            b_type = p.city_board[i].building_type
            max_cap = BUILDING_DATA[b_type][2]
            if val < 0 or val > max_cap:
                raise ValueError(f"Invalid colonist count for building {b_type.name}")
            p.city_board[i].colonists = val
            
        # Any remaining are returned to unplaced (San Juan)
        p.unplaced_colonists = p.total_colonists_owned - total_placed
        
        self._advance_phase_turn()

    def action_builder(self, player_idx: int, building_choice: Optional[BuildingType]):
        """
        building_choice: The building to build. None to pass.
        """
        if self.current_phase != Phase.BUILDER or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Builder phase.")
            
        if building_choice is not None:
            p = self.players[player_idx]
            has_privilege = (player_idx == self.active_role_player_idx())
            
            if self.building_supply.get(building_choice, 0) <= 0:
                raise ValueError("Building out of stock.")
            if p.has_building(building_choice):
                raise ValueError("Player already has this building.")
                
            base_cost = BUILDING_DATA[building_choice][0]
            
            # Privilege reduces cost by 1
            if has_privilege:
                base_cost = max(0, base_cost - 1)
                
            # Quarries reduce cost
            active_quarries = sum(1 for t in p.island_board if t.tile_type == TileType.QUARRY and t.is_occupied)
            # Max reduction depends on building column (cost category roughly). 
            # 1-3: 1, 4-6: 2, 7-9: 3, 10: 4
            if BUILDING_DATA[building_choice][0] <= 3:
                max_q_reduction = 1
            elif BUILDING_DATA[building_choice][0] <= 6:
                max_q_reduction = 2
            elif BUILDING_DATA[building_choice][0] <= 9:
                max_q_reduction = 3
            else:
                max_q_reduction = 4
                
            actual_q_reduction = min(active_quarries, max_q_reduction)
            final_cost = max(0, base_cost - actual_q_reduction)
            
            p.pay_doubloons(final_cost)
            p.build_building(building_choice)
            self.building_supply[building_choice] -= 1
            
            # University Check
            if p.is_building_occupied(BuildingType.UNIVERSITY):
                if self.colonists_supply > 0:
                    p.city_board[-1].colonists += 1
                    self.colonists_supply -= 1
                    
        self._advance_phase_turn()

    def action_captain_load(self, player_idx: int, ship_idx: int, good_type: Good):
        """
        Loads all goods of `good_type` from player onto `cargo_ships[ship_idx]`.
        The engine then computes VPs.
        If player has Wharf, ship_idx can be -1 (imaginary ship).
        """
        if self.current_phase != Phase.CAPTAIN or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain phase.")
            
        p = self.players[player_idx]
        has_privilege = (player_idx == self.active_role_player_idx())
        
        amount = p.goods[good_type]
        if amount == 0:
            raise ValueError("Player has no goods of this type.")
            
        is_wharf = False
        if ship_idx == -1:
            if not p.is_building_occupied(BuildingType.WHARF):
                raise ValueError("Player does not have occupied Wharf for imaginary ship.")
            # Assume wharf is single use per captain phase. This needs tracking if we want to be strict,
            # but usually you just use it once. Let's assume it's valid.
            loaded_amount = amount
            is_wharf = True
        else:
            ship = self.cargo_ships[ship_idx]
            if ship.is_full:
                raise ValueError("Ship is full.")
            if ship.good_type is not None and ship.good_type != good_type:
                raise ValueError("Ship already contains a different good.")
                
            # Check other ships to see if they hold this good
            for i, other_ship in enumerate(self.cargo_ships):
                if i != ship_idx and other_ship.good_type == good_type:
                    raise ValueError("Another ship is already loading this kind of good.")
                    
            loaded_amount = min(amount, ship.capacity - ship.current_load)
            ship.current_load += loaded_amount
            ship.good_type = good_type
            
        # VPs
        points_earned = loaded_amount
        if has_privilege and not getattr(self, '_captain_privilege_used', False):
            points_earned += 1
            self._captain_privilege_used = True
            
        if is_wharf:
            if self._wharf_used.get(player_idx, False):
                raise ValueError("Wharf can only be used once per Captain phase.")
            self._wharf_used[player_idx] = True
            
        if p.is_building_occupied(BuildingType.HARBOR):
            points_earned += 1
            
        self.vp_chips -= points_earned # May go negative, which indicates triggered game end
        p.add_vp(points_earned) # Players earn VP even if VP chips are exhausted
        
        p.remove_good(good_type, loaded_amount)
        if is_wharf:
            self.goods_supply[good_type] += loaded_amount # Wharf goods return to supply immediately
        
        # In actual Puerto Rico, a player MUST load if they can.
        # So instead of immediately advancing, the player's turn continues until they can't load.
        # But for RL MDP, it's easier to: Action -> State update. 
        # Player MUST pick a valid load action, or a PASS action. If they pick PASS while they can load,
        # the env raises an error or penalizes them.
        self._advance_phase_turn()

    def action_captain_pass(self, player_idx: int):
        """
        Player indicates they can no longer load any goods.
        """
        if self.current_phase != Phase.CAPTAIN or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain phase.")
            
        p = self.players[player_idx]
        
        # Verify if player TRULY cannot load anything
        can_load_anything = False
        for ship_idx, ship in enumerate(self.cargo_ships):
            if not ship.is_full:
                for g in Good:
                    if p.goods[g] > 0:
                        allowed = False
                        if ship.good_type is None:
                            other_has_it = any(os.good_type == g for i, os in enumerate(self.cargo_ships) if i != ship_idx)
                            if not other_has_it:
                                allowed = True
                        elif ship.good_type == g:
                            allowed = True
                            
                        if allowed:
                            can_load_anything = True
                            break
            if can_load_anything:
                break
                
        # Wharf is 100% voluntary. We do NOT force can_load_anything to be True just because they have a Wharf.
        
        if can_load_anything:
            raise ValueError("Rule Violation: Player MUST load if they have valid goods and ship capacity.")
            
        self._captain_passed_players.add(player_idx)
        self._advance_phase_turn()

    def action_captain_store_windrose(self, player_idx: int, good_type: Good):
        """
        Assign a single barrel of a good to the Windrose.
        """
        if self.current_phase != Phase.CAPTAIN_STORE or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain Store phase.")
            
        p = self.players[player_idx]
        if p.goods[good_type] == 0:
            raise ValueError(f"Player does not have {good_type.name}.")
            
        if self._storage_assignments[player_idx]['windrose'] is not None:
            raise ValueError("Windrose slot is already occupied.")
            
        if good_type in self._storage_assignments[player_idx]['warehouses']:
            raise ValueError("Good is already assigned to a warehouse.")
            
        self._storage_assignments[player_idx]['windrose'] = good_type

    def action_captain_store_warehouse(self, player_idx: int, good_type: Good):
        """
        Assign all barrels of a good to a Warehouse.
        """
        if self.current_phase != Phase.CAPTAIN_STORE or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain Store phase.")
            
        p = self.players[player_idx]
        if p.goods[good_type] == 0:
            raise ValueError(f"Player does not have {good_type.name}.")
            
        max_wh_slots = 0
        if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE):
            max_wh_slots += 1
        if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE):
            max_wh_slots += 2
            
        if len(self._storage_assignments[player_idx]['warehouses']) >= max_wh_slots:
            raise ValueError("All warehouse slots are already used.")
            
        if good_type in self._storage_assignments[player_idx]['warehouses']:
            raise ValueError("Good is already assigned to a warehouse.")
            
        if self._storage_assignments[player_idx]['windrose'] == good_type:
            raise ValueError("Good is already assigned to the windrose.")
            
        self._storage_assignments[player_idx]['warehouses'].append(good_type)

    def action_captain_store_pass(self, player_idx: int):
        """
        Finalize storage. Discard unstored goods.
        Must not voluntarily discard if space is available.
        """
        if self.current_phase != Phase.CAPTAIN_STORE or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain Store phase.")
            
        p = self.players[player_idx]
        assign = self._storage_assignments[player_idx]
        
        unstored_types = [g for g in Good if p.goods[g] > 0 and g != assign['windrose'] and g not in assign['warehouses']]
        
        max_wh_slots = 0
        if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE): max_wh_slots += 1
        if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE): max_wh_slots += 2
        
        has_empty_windrose = (assign['windrose'] is None)
        has_empty_wh = len(assign['warehouses']) < max_wh_slots
        
        if len(unstored_types) > 0:
            if has_empty_windrose or has_empty_wh:
                raise ValueError("Rule Violation: Cannot voluntarily discard goods if you have empty storage space.")
                
        # Apply storage
        for g in Good:
            if g in assign['warehouses']:
                keep = p.goods[g]
            elif g == assign['windrose']:
                keep = 1
            else:
                keep = 0
                
            discard = p.goods[g] - keep
            if discard > 0:
                p.remove_good(g, discard)
                self.goods_supply[g] += discard
                
        self._advance_phase_turn()

    def _execute_phase_cleanup(self):
        """Phase specific cleanup before ending the phase."""
        if self.current_phase == Phase.SETTLER:
            self.plantation_discard.extend(self.face_up_plantations)
            self.face_up_plantations.clear()
            self._deal_face_up_plantations()
            
        elif self.current_phase == Phase.MAYOR:
            capacity = 0
            for p in self.players:
                for b in p.city_board:
                    capacity += BUILDING_DATA[b.building_type][2] - b.colonists
                    
            refill = max(capacity, self.num_players)
            if refill > self.colonists_supply:
                self._colonists_ship_underfilled = True
                
            actual_refill = min(refill, self.colonists_supply)
            self.colonists_ship = actual_refill
            self.colonists_supply -= actual_refill
            
        elif self.current_phase == Phase.TRADER:
            if len(self.trading_house) == 4:
                for good in self.trading_house:
                    self.goods_supply[good] += 1
                self.trading_house.clear()
                
        elif self.current_phase == Phase.CAPTAIN_STORE:
            # Empty full ships
            for ship in self.cargo_ships:
                if ship.is_full:
                    if ship.good_type is not None:
                        self.goods_supply[ship.good_type] += ship.current_load
                    ship.current_load = 0
                    ship.good_type = None
            # Reset captain privilege
            self._captain_privilege_used = False

