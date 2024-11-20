import random

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        return random.choice(self.available_pieces)

    def place_piece(self, selected_piece):
        available_places = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    available_places.append((row, col))
        
        return random.choice(available_places)