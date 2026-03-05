import unittest
from env.engine import PuertoRicoGame
from configs.constants import Phase, Role, Good, TileType, BuildingType

class TestPuertoRicoGame(unittest.TestCase):

    def setUp(self):
        self.game = PuertoRicoGame(num_players=4)

    def test_setup(self):
        self.assertEqual(len(self.game.players), 4)
        self.assertEqual(self.game.vp_chips, 100)
        self.assertEqual(self.game.colonists_supply, 75)
        self.assertEqual(self.game.colonists_ship, 4)
        
        # Check initial doubloons
        for p in self.game.players:
            self.assertEqual(p.doubloons, 3)
            
        # Check plantations
        gov_idx = self.game.governor_idx
        self.assertEqual(self.game.players[gov_idx].island_board[0].tile_type, TileType.INDIGO_PLANTATION)
        self.assertEqual(self.game.players[(gov_idx + 1) % 4].island_board[0].tile_type, TileType.INDIGO_PLANTATION)
        self.assertEqual(self.game.players[(gov_idx + 2) % 4].island_board[0].tile_type, TileType.CORN_PLANTATION)
        self.assertEqual(self.game.players[(gov_idx + 3) % 4].island_board[0].tile_type, TileType.CORN_PLANTATION)

    def test_start_game(self):
        self.game.start_game()
        self.assertEqual(len(self.game.face_up_plantations), 5)
        self.assertEqual(self.game.current_phase, Phase.END_ROUND)

    def test_role_selection(self):
        self.game.start_game()
        player_idx = self.game.current_player_idx
        self.game.select_role(player_idx, Role.SETTLER)
        
        self.assertEqual(self.game.current_phase, Phase.SETTLER)
        self.assertEqual(self.game.active_role, Role.SETTLER)
        self.assertNotIn(Role.SETTLER, self.game.available_roles)
        self.assertEqual(self.game.active_role_player_idx(), player_idx)

    def test_settler_phase(self):
        self.game.start_game()
        player_idx = self.game.current_player_idx
        
        # Pick Settler
        self.game.select_role(player_idx, Role.SETTLER)
        
        # Take Quarry (Privilege)
        self.game.action_settler(player_idx, tile_choice=-1)
        self.assertEqual(self.game.players[player_idx].island_board[-1].tile_type, TileType.QUARRY)
        
        # Next player takes a face up plantation
        next_player = self.game.current_player_idx
        self.game.action_settler(next_player, tile_choice=0)
        self.assertEqual(len(self.game.players[next_player].island_board), 2)
        
        # Others pass
        for _ in range(2):
            self.game.action_settler(self.game.current_player_idx, tile_choice=-2)
            
        # Phase should end, next player in line to pick role
        self.assertEqual(self.game.current_phase, Phase.END_ROUND)
        self.assertEqual(self.game.current_player_idx, next_player)

if __name__ == '__main__':
    unittest.main()
