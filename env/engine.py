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
        if self.colonists_supply <= 0:
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
        elif role == Role.BUILDER:
            self.current_phase = Phase.BUILDER
        elif role == Role.CRAFTSMAN:
            self.current_phase = Phase.CRAFTSMAN
        elif role == Role.TRADER:
            self.current_phase = Phase.TRADER
        elif role == Role.CAPTAIN:
            self.current_phase = Phase.CAPTAIN
        elif role in (Role.PROSPECTOR_1, Role.PROSPECTOR_2):
            self.current_phase = Phase.PROSPECTOR
            # Privilege: 1 doubloon
            p.add_doubloons(1)
            # Immediately end phase as prospector has no action for others
            self._end_phase()
            
    def _advance_phase_turn(self):
        """Advances turn to next player in the current phase."""
        self.players_taken_action += 1
        if self.players_taken_action >= self.num_players:
            # Phase is over
            self._execute_phase_cleanup()
            self._end_phase()
        else:
            self._next_player()

    def _execute_phase_cleanup(self):
        """Phase specific cleanup before ending the phase."""
        if self.current_phase == Phase.SETTLER:
            # Discard remaining face up tiles and deal new ones
            self.plantation_discard.extend(self.face_up_plantations)
            self.face_up_plantations.clear()
            self._deal_face_up_plantations()
            
        elif self.current_phase == Phase.MAYOR:
            # Distribute colonists on ship
            # This is complex in engine logic but basically:
            # deal them out one by one starting from Mayor
            idx = self.active_role_player_idx()
            while self.colonists_ship > 0:
                self.players[idx].unplaced_colonists += 1
                self.colonists_ship -= 1
                idx = (idx + 1) % self.num_players
                
            # Then Mayor fills the ship
            empty_spots = sum(p.empty_city_spaces for p in self.players) # Wait, it's empty circles on BUILDINGS on player boards!
            # Let's count properly: Total capacities of all buildings - current colonists in them
            capacity = 0
            for p in self.players:
                for b in p.city_board:
                    capacity += BUILDING_DATA[b.building_type][3] - b.colonists
                    
            refill = max(capacity, self.num_players)
            actual_refill = min(refill, self.colonists_supply)
            self.colonists_ship = actual_refill
            self.colonists_supply -= actual_refill
            
        elif self.current_phase == Phase.TRADER:
            if len(self.trading_house) == 4:
                self.trading_house.clear()

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

    def action_settler(self, player_idx: int, tile_choice: int, use_hacienda: bool = False):
        """
        tile_choice: index in self.face_up_plantations (0-len). 
        Set to -1 to take Quarry (if privilege or construction hut).
        Set to -2 to pass.
        use_hacienda: boolean, if True and player has active Hacienda, draws from stack first.
        """
        if self.current_phase != Phase.SETTLER or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Settler phase.")
            
        p = self.players[player_idx]
        has_privilege = (player_idx == self.active_role_player_idx())
        
        if use_hacienda:
            if p.is_building_occupied(BuildingType.HACIENDA):
                if self.plantation_stack:
                    drawn_tile = self.plantation_stack.pop()
                    if p.empty_island_spaces > 0:
                        p.place_plantation(drawn_tile)
            else:
                raise ValueError("Player does not have an active Hacienda.")
                
        if tile_choice == -2: # Pass
            pass
        elif tile_choice == -1: # Quarry
            can_quarry = has_privilege or p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)
            if not can_quarry:
                raise ValueError("Cannot take quarry without privilege or construction hut.")
            if self.quarry_stack > 0 and p.empty_island_spaces > 0:
                p.place_plantation(TileType.QUARRY)
                self.quarry_stack -= 1
        else: # Face up plantation
            if 0 <= tile_choice < len(self.face_up_plantations):
                tile = self.face_up_plantations.pop(tile_choice)
                if p.empty_island_spaces > 0:
                    p.place_plantation(tile)
            else:
                raise ValueError("Invalid plantation choice.")
                
        # Hospice check
        if tile_choice != -2 and p.is_building_occupied(BuildingType.HOSPICE):
            # Place 1 colonist on the newly placed tile
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
            max_cap = BUILDING_DATA[b_type][3]
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
            
        if p.is_building_occupied(BuildingType.HARBOR):
            points_earned += 1
            
        actual_points = min(points_earned, self.vp_chips)
        self.vp_chips = max(0, self.vp_chips - points_earned) # Negative indicates triggered game end
        p.add_vp(actual_points)
        
        p.remove_good(good_type, loaded_amount)
        
        # In actual Puerto Rico, a player MUST load if they can.
        # So instead of immediately advancing, the player's turn continues until they can't load.
        # But for RL MDP, it's easier to: Action -> State update. 
        # Player MUST pick a valid load action, or a PASS action. If they pick PASS while they can load,
        # the env raises an error or penalizes them.
        self._advance_phase_turn()

    def action_captain_store(self, player_idx: int, store_goods: Dict[Good, int]):
        """
        At end of captain phase, players store 1 good, plus large/small warehouses.
        Extra goods are discarded.
        store_goods is dict of {Good: amount} they wish to keep.
        """
        if self.current_phase != Phase.CAPTAIN or self.current_player_idx != player_idx:
            raise ValueError("Not this player's turn in Captain phase.")
            
        p = self.players[player_idx]
        
        # Verify they actually own what they want to store
        for g, amt in store_goods.items():
            if p.goods[g] < amt:
                raise ValueError(f"Player trying to store {amt} {g.name} but only has {p.goods[g]}")
                
        # Validate storage limits
        total_types = sum(1 for g, v in store_goods.items() if v > 0)
        
        allowed_types_full_store = 0
        if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE):
            allowed_types_full_store += 1
        if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE):
            allowed_types_full_store += 2
            
        # Is this configuration legal?
        # A legal config has at most 'allowed_types_full_store' types where amount > 1
        # AND at most 1 additional single barrel of any type NOT covered by warehouses
        
        types_needing_warehouse = 0
        single_barrel_used = False
        is_valid = True
        
        for g, amt in store_goods.items():
            if amt > 1:
                types_needing_warehouse += 1
            elif amt == 1:
                # Can either use a warehouse or the 1 free windrose spot
                pass
                
        # To strictly check: sort amounts descending
        amounts = sorted([v for v in store_goods.values() if v > 0], reverse=True)
        # First `allowed_types_full_store` items are covered by warehouses regardless of size
        remaining_amounts = amounts[allowed_types_full_store:]
        # What's left can be AT MOST one item, and its value must be EXACTLY 1
        if len(remaining_amounts) > 1:
            is_valid = False
        elif len(remaining_amounts) == 1 and remaining_amounts[0] > 1:
            is_valid = False
            
        if not is_valid:
            raise ValueError("Invalid storage configuration.")
            
        # Execute storage (discard the rest)
        for g in Good:
            keep = store_goods.get(g, 0)
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
            idx = self.active_role_player_idx()
            while self.colonists_ship > 0:
                self.players[idx].unplaced_colonists += 1
                self.colonists_ship -= 1
                idx = (idx + 1) % self.num_players
                
            capacity = 0
            for p in self.players:
                for b in p.city_board:
                    capacity += BUILDING_DATA[b.building_type][3] - b.colonists
                    
            refill = max(capacity, self.num_players)
            actual_refill = min(refill, self.colonists_supply)
            self.colonists_ship = actual_refill
            self.colonists_supply -= actual_refill
            
        elif self.current_phase == Phase.TRADER:
            if len(self.trading_house) == 4:
                self.trading_house.clear()
                
        elif self.current_phase == Phase.CAPTAIN:
            # Empty full ships
            for ship in self.cargo_ships:
                if ship.is_full:
                    if ship.good_type is not None:
                        self.goods_supply[ship.good_type] += ship.current_load
                    ship.current_load = 0
                    ship.good_type = None
            # Reset captain privilege
            self._captain_privilege_used = False

