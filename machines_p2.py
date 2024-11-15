from itertools import product
import copy
import math
import numpy as np
import random

MCTS_ITERATIONS = 1000
BOARD_ROWS = 4
BOARD_COLS = 4
flag = "select_piece"
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

class P2():
    def __init__(self, board, available_pieces):
        self.board = Board(board) # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.move_history = []

    def select_piece(self):
        return self.mcts_go("select_piece")

    def place_piece(self):
        return self.mcts_go("place_piece")

    def mcts_go(self, operation, iterations=MCTS_ITERATIONS, debug=False):
        tree = [Node()]
        if operation == "select_piece":
            for place in self.board.get_available_places():
                new_node = Node(parent=tree[0], move_to=place)
                tree[0].children.append(new_node)
                tree.append(new_node)

        elif operation == "place_piece":
            for piece in self.available_pieces:
                new_node = Node(parent=tree[0], move_to=piece)
                tree[0].children.append(new_node)
                tree.append(new_node)
        else:
            raise TypeError("Opration is invaild.")

        for _ in range(iterations):
            current_node = tree[0]
            while not current_node.is_leaf():
                children_scores = tuple(map(lambda x: x.ucb1(), current_node.children))
                current_node = current_node.children[children_scores.index(max(children_scores))]

            board_updates = 0
            if operation == "select_piece" and current_node.moves_to:
                piece = random.choice(current_node.moves_to)
                self.board.select(piece)
            elif operation == "place_piece" and current_node.moves_to:
                move = random.choice(current_node.moves_to)
                self.board.place(move(move[0], move[1], 1))
                if self.board.check_win():
                    is_terminated = True
            else:
                raise TypeError("current_node.moves_to is EMPTY")

            if not current_node.visits and not is_terminated: # not visited and not terminated
                rollout_res = next_round(self.board, 1)

            elif current_node.visits and not is_terminated: # visited and not terminated
                if operation == "select_piece":
                    for place in self.board.get_available_places():
                        new_node = Node(parent=current_node, move_to=place)
                        tree[0].children.append(new_node)
                        tree.append(new_node)
                else:
                    for piece in self.available_pieces:
                        new_node = Node(parent=tree[0], move_to=piece)
                        tree[0].children.append(new_node)
                        tree.append(new_node)

                if not current_node.children:
                    rollout_res = 0
                else:
                    current_node = current_node.children[0]
                    board_updates += 1
                    self.board.move(current_node.moves_to[-1][0], current_node.moves_to[-1][1], 1)
                    rollout_res = next_round(self.board, 1)

            #revert board
            for _ in range(board_updates):
                self.board.undo()

            #backpropogate the rollout
            while current_node.parent: #not None. only the top node has None as a parent
                current_node.visits += 1
                current_node.score += rollout_res
                current_node = current_node.parent
            current_node.visits += 1 #for the mother node

        #pick the move with the most visits
        current_node = tree[0]
        visit_map = tuple(map(lambda x: x.visits, current_node.children))
        best_move = visit_map.index(max(visit_map))
        if operation == "select_piece":
            return self.board.get_available_places()[best_move]
        else:
            return self.available_pieces[best_move]
        
    def rollout(self, team = None):
        "Rollout a game"
        check_win = check_win()
        if check_win:   
            return (check_win + 1) // 2
        #make a random move
        while True:
            row = random.randint(0, BOARD_ROWS - 1)
            col = random.randint(0, BOARD_COLS - 1)
            if (row, col) not in self.move_history:
                self.board[row][col] = team
                break
        return 0.5 #draw

class Board:
    def __init__(self, board):
        self.__board = board
        self.path = [] # backtracking
        self.selected_piece = None

    def check_line(self, line):
        if 0 in line:
            return False  # Incomplete line
        characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True
        return False

    def check_2x2_subgrid_win(self):
        for r in range(BOARD_ROWS - 1):
            for c in range(BOARD_COLS - 1):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [pieces[idx - 1] for idx in subgrid]
                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True
        return False

    def check_win(self):
        # Check rows, columns, and diagonals
        for col in range(BOARD_COLS):
            if self.check_line([self.board[row][col] for row in range(BOARD_ROWS)]):
                return True
        
        for row in range(BOARD_ROWS):
            if self.check_line([self.board[row][col] for col in range(BOARD_COLS)]):
                return True
            
        if self.check_line([self.board[i][i] for i in range(BOARD_ROWS)]) or self.check_line([self.board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win():
            return True
        
        return False
    
    def available_square(self, row, col):
        return self.board[row][col] == 0
    
    def get_available_places(self):
        available_positions = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.available_square(row, col):
                    available_positions.append((row, col))
        return available_positions
    
    def is_board_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def place(self, row, col):
        if self.__board.available_square(row, col):
            self.__board[row][col] = self.selected_piece
            self.selected_piece = None

    def select(self, piece):
        self.selected_piece = piece

    def undo(self):
        if self.path:
            self.__board[self.path[-1][0]][self.path[-1][1]] = 0
            self.path.pop()
        else:
            raise IndexError("No moves have been played.")

def next_round(real_board, player):
    board = copy.deepcopy(real_board)
    while not board.is_board_full():
        check_win = board.check_win()
        if check_win: # not draw
            if check_win == player:     #win
                return 1
            else:                       #lose
                return 
            
        while True:
            row = random.randint(0, BOARD_ROWS - 1)
            col = random.randint(0, BOARD_COLS - 1)
            if board.available_square(row, col):
                board.move(row, col, player)
                break
    
    return 0.5

class Node:
    def __init__(self, parent=None, move_to=None):
        self.parent = parent #the object
        if parent and not move_to:
            raise TypeError("A parent is provided with no move_to paramenter.")
        elif parent:
            self.moves_to = copy.deepcopy(self.parent.moves_to)
            self.moves_to.append(move_to)
        else:
            self.moves_to = []
        self.score = 0
        self.visits = 0
        self.children = []

    def is_leaf(self):
        return not bool(self.children)

    def ucb1(self):
        try:
            return self.score / self.visits + 2 * math.sqrt(math.log(self.parent.visits)
                                                            / self.visits)
        except ZeroDivisionError:
            #equivalent to infinity
            #assuming log(parent visits) / visits will not exceed 100
            return 10000