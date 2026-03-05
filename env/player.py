from typing import List, Dict
from configs.constants import TileType, BuildingType, Good
from env.components import IslandTile, CityBuilding

class Player:
    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self.doubloons = 0
        self.vp_chips = 0
        self.unplaced_colonists = 0
        
        # 12 Island Spaces
        self.island_board: List[IslandTile] = []
        
        # 12 City Spaces
        self.city_board: List[CityBuilding] = []
        
        # Goods on windrose
        self.goods: Dict[Good, int] = {
            Good.COFFEE: 0,
            Good.TOBACCO: 0,
            Good.CORN: 0,
            Good.SUGAR: 0,
            Good.INDIGO: 0
        }

    @property
    def empty_island_spaces(self) -> int:
        return 12 - len(self.island_board)

    @property
    def empty_city_spaces(self) -> int:
        # Note: Large buildings take 2 spaces, handled during placement
        occupied_spaces = sum(2 if self._is_large_building(b.building_type) else 1 for b in self.city_board)
        return 12 - occupied_spaces

    def _is_large_building(self, b_type: BuildingType) -> bool:
        from configs.constants import BUILDING_DATA
        return BUILDING_DATA[b_type][4] # is_large boolean at index 4

    def add_doubloons(self, amount: int):
        self.doubloons += amount
        
    def pay_doubloons(self, amount: int):
        if amount > self.doubloons:
            raise ValueError("Not enough doubloons")
        self.doubloons -= amount

    def add_vp(self, amount: int):
        self.vp_chips += amount

    def add_good(self, good: Good, amount: int = 1):
        self.goods[good] += amount

    def remove_good(self, good: Good, amount: int = 1):
        if amount > self.goods[good]:
            raise ValueError(f"Not enough {good.name}")
        self.goods[good] -= amount

    def place_plantation(self, tile_type: TileType):
        if self.empty_island_spaces <= 0:
            raise ValueError("No empty island spaces")
        self.island_board.append(IslandTile(tile_type=tile_type, is_occupied=False))

    def build_building(self, building_type: BuildingType):
        spaces_needed = 2 if self._is_large_building(building_type) else 1
        if self.empty_city_spaces < spaces_needed:
            raise ValueError("Not enough empty city spaces")
        self.city_board.append(CityBuilding(building_type=building_type, colonists=0))

    def has_building(self, building_type: BuildingType) -> bool:
        return any(b.building_type == building_type for b in self.city_board)

    def is_building_occupied(self, building_type: BuildingType) -> bool:
        for b in self.city_board:
            if b.building_type == building_type:
                return b.is_occupied
        return False
    
    @property
    def total_colonists_owned(self) -> int:
        count = self.unplaced_colonists
        count += sum(1 for t in self.island_board if t.is_occupied)
        count += sum(b.colonists for b in self.city_board)
        return count
