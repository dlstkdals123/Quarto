import numpy as np
import random
from collections import defaultdict
from itertools import product
import math
import copy

# Algorithms parameters
MCTS_ITERATIONS = 50
SWITCH_POINT = 8

# Constants
BOARD_ROWS = 4
BOARD_COLS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

# Variables
PLAYER = 1
isFirst = False # P2인 경우 True로 바꿔주세요.

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.available_places = self.get_available_places()
    
    def select_piece(self):
        global isFirst

        #첫 선택은 무작위
        if isFirst:
            isFirst = False
            return random.choice(self.pieces)
        
        
        if len(self.available_pieces) > SWITCH_POINT: # piece가 기준점보다 많다면 MCTS
            # MCTS
            tree = MCTS()
            board = Board(self.board, PLAYER, None, self.available_places, self.available_pieces)
            node = Node(board)
            
            # 상대가 이길 수 있는 piece은 선택에서 제외
            tree.children[node] = []
            for piece in self.available_pieces:
                is_opponent_win = False
                for row, col in self.available_places:
                    if check_win_with_piece(board, piece, row, col):
                        is_opponent_win = True
                        break
                
                if not is_opponent_win:
                    next_board = copy.deepcopy(board)
                    next_board.select(piece)
                    next_node = Node(next_board)
                    tree.children[node].append(next_node)

            # 상대가 모두 이기는 경우 -> 랜덤 선택
            if not tree.children[node]: # lose
                return random.choice(self.available_pieces)
            
            # 상대가 한 개를 제외하고 모두 이기는 경우 -> 남은 1가지 무조건 선택
            if len(tree.children[node]) == 1: #only one piece
                return tree.children[node][0].board_state.selected_piece

            # 첫 확장 simulate & backpropagate
            reward = tree._simulate(node)
            tree._backpropagate([node], reward)

            # MCTS search
            for i in range(MCTS_ITERATIONS):
                tree.do_rollout(node)

            best_node = tree.choose(node)
            return best_node.board_state.selected_piece

        else: # Minimax
            best_piece = None
            best_value = -float('inf')

            for piece in self.available_pieces:
                eval = self.minmax_alpha_beta(self.board, [p for p in self.available_pieces if p != piece], -float('inf'), float('inf'), False, piece, None)
                if eval > best_value:
                    best_value = eval
                    best_piece = piece

            return best_piece

    def place_piece(self, selected_piece):

        if len(self.available_pieces) > SWITCH_POINT: # piece가 기준점보다 많다면 MCTS
            tree = MCTS()
            board = Board(self.board, PLAYER, selected_piece, self.available_places, self.available_pieces)
            node = Node(board)

            # 바로 이길 수 있다면 -> 그것을 선택
            for row, col in self.available_places:
                if check_win_with_piece(board, selected_piece, row, col):
                    return row, col
            
            # MCTS search
            for i in range(MCTS_ITERATIONS):
                tree.do_rollout(node)

            best_node = tree.choose(node)
            
            # 가장 좋은 row, col값 return
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    if best_node.board_state[row][col] == self.pieces.index(selected_piece) + 1:  # 실제 값으로 비교
                        return row, col
        else: # Minimax
            best_move = None
            best_value = -float('inf')

            for row, col in product(range(4), range(4)):
                if self.board[row][col] == 0:
                    self.board[row][col] = self.pieces.index(selected_piece) + 1
                    eval = self.minmax_alpha_beta(self.board, self.available_pieces, -float('inf'), float('inf'), True, None, (row, col))
                    self.board[row][col] = 0

                    if eval > best_value:
                        best_value = eval
                        best_move = (row, col)
            return best_move
    
    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    available_places.append((row, col))
        return available_places
        
    def minmax_alpha_beta(self, board, available_pieces, alpha, beta, is_maximizing, selected_piece, log = None): # log는 place_piece을 하였다면 (row, col) select_piece라면 None
        if selected_piece is None: # 전의 player가 place_piece한 경우 -> player가 변하지 않음 = is_maximizing도 변하지 않음
            row, col = log
            if check_win(board, row, col):
                if is_maximizing:
                    return 10
                else:
                    return -10

        if len(available_pieces) == 0:
            return 0

        if is_maximizing:
            max_eval = -float('inf')
            if selected_piece is None: # select_piece
                for piece in available_pieces:
                    eval = self.minmax_alpha_beta(board, [p for p in available_pieces if p != piece], alpha, beta, not is_maximizing, piece)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break

            else: # place_piece
                for row, col in product(range(4), range(4)):
                    if board[row][col] == 0:
                        board[row][col] = self.pieces.index(selected_piece) + 1
                        eval = self.minmax_alpha_beta(board, available_pieces, alpha, beta, is_maximizing, None, (row, col))
                        board[row][col] = 0
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break

            return max_eval
        
        else:
            min_eval = float('inf')
            if selected_piece is None: # select_piece
                for piece in available_pieces:
                    eval = self.minmax_alpha_beta(board, [p for p in available_pieces if p != piece], alpha, beta, not is_maximizing, piece)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            
            else: #place_piece
                for row, col in product(range(4), range(4)):
                    if board[row][col] == 0:
                        board[row][col] = self.pieces.index(selected_piece) + 1
                        eval = self.minmax_alpha_beta(board, available_pieces, alpha, beta, is_maximizing, None, (row, col))
                        board[row][col] = 0
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break

            return min_eval

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if self.children[node] is None:
            raise ValueError("Cannot choose from unexplored node")
                
        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # Node is either unexplored or terminal
                return path

            # Explore unexplored nodes
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            # Descend deeper using UCT
            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        
        deleted_nodes = 0
        node_list = node.find_children()
        self.children[node] = []
        for currnet_node in node_list:
            if currnet_node not in self.children.keys():
                self.children[node].append(currnet_node)
            else:
                deleted_nodes += 1

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        depth = 1
        while True:
            if node.is_terminal():
                return node.reward(depth)
            node = node.find_random_child()
            depth += 1

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

# =========================================
# MCTS Node
# MCTS에서는 메모리에 저장하기 때문에 따로 저장합니다.
# Minimax에서는 메모리에 저장하지 않고, 백트래킹을 이용합니다.
# =========================================

class Node():
    def __init__(self, board):
        self.board_state = board
        self.children = []

    def find_children(self):
        result = []
        # 플레이어 순서를 고려하여 자식 노드의 상태를 전환
        if self.board_state.selected_piece is None: # Select_piece
            for piece in self.board_state.available_pieces:
                next_board = copy.deepcopy(self.board_state)
                next_board.select(piece)
                next_node = Node(next_board)
                result.append(next_node)

        else: # Place_piece
            for row, col in self.board_state.available_places:
                next_board = copy.deepcopy(self.board_state)
                next_board.place(row, col)
                next_node = Node(next_board)
                result.append(next_node)

        self.children = result
        return result

    def find_random_child(self):
        return random.choice(self.children)

    def is_terminal(self):
        return not self.children

    def reward(self, depth):
        copy_board = copy.deepcopy(self.board_state)
        "Assumes self is terminal node. 1=win, 0=loss, .5=draw"

        while not is_board_full(copy_board.available_places):
            if copy_board.selected_piece is None: # Select_piece
                copy_board.random_select()
            else: # Place_piece
                row, col = copy_board.get_random_place()
                copy_board.place(row, col)
                if check_win(copy_board.get_board(), row, col):
                    return (copy_board.player == PLAYER) / depth

        return 0.5

    # =========================================
    # Special Methods
    # =========================================

    def __hash__(self):
        return hash(self.board_state)

    def __eq__(self, other):
        return self.board_state == other.board_state
    
    def __str__(self):
        return str(self.board_state)
    

# =========================================
# MCTS Board
# =========================================

class Board:
    def __init__(self, board, player, selected_piece, available_places, available_pieces):        
        self.__board = copy.deepcopy(board)
        self.player = player
        self.selected_piece = selected_piece
        self.available_places = copy.deepcopy(available_places)
        self.available_pieces = copy.deepcopy(available_pieces)
        if selected_piece in self.available_pieces:
            self.available_pieces.remove(selected_piece)
    
    def get_board(self):
        return self.__board

    def random_select(self):
        if self.selected_piece is not None:
            raise TypeError(f"Now is 'place_piece' state")

        selected_piece = random.choice(self.available_pieces)

        self.select(selected_piece)

    def select(self, piece):
        if piece not in self.available_pieces:
            raise ValueError(f"The selected piece {piece} is not available")

        self.player = -self.player
        self.selected_piece = piece
        self.available_pieces.remove(piece)
    
    def get_random_place(self):
        if self.selected_piece is None:
            raise TypeError("Now is 'select_piece' state")
        
        selected_place = random.choice(self.available_places)
        return selected_place[0], selected_place[1]
    
    def place(self, row, col):
        if self.selected_piece is None:
            raise TypeError("Now is 'select_piece' state")

        self.player = self.player
        self.__board[row][col] = get_piece_idx(self.selected_piece)
        self.available_places.remove((row, col))
        self.selected_piece = None

    # =========================================
    # Special Methods
    # =========================================

    def __str__(self):
        # Convert the board into a readable string
        if self.current_state == "place_piece":
            return f"\nPlayer: {self.player}, Selected_piece: {self.selected_piece}\n"
        else:
            board_str = ''
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    piece = self.get(row, col)
                    if piece is None:
                        piece_text = '....'
                    else:
                        piece_text = f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"
                    board_str += f"{piece_text} "
                board_str += '\n'
            return f"\nPlayer: {self.player}, Current_state: {self.current_state}, Selected_piece: {self.selected_piece}\n{board_str}\n"
        
    def __getitem__(self, index):
        return self.__board[index]
    
    def __hash__(self):
        return hash(self.__board.tobytes()) ^ hash(self.selected_piece)

    def __eq__(self, other):
        # Equality check for hash compatibility
        if not isinstance(other, Board):
            return False

        return (
            all(
                self.__board[row][col] == other.__board[row][col]
                for row in range(BOARD_ROWS)
                for col in range(BOARD_COLS)
            ) and
            self.selected_piece == other.selected_piece 
        )

# =========================================
# General Functions
# =========================================

def get(board, x, y):
    piece_idx = board[x][y] - 1
    if piece_idx == -1:
        return None
    else:
        return pieces[piece_idx]

def get_piece_idx(piece):
    return pieces.index(piece) + 1

def is_board_full(available_places):
    return not available_places

def check_win_with_piece(board, piece, x, y):
    # Place the piece temporarily
    if piece not in pieces:
        raise ValueError(f"Piece {piece} is not available")
    board[x][y] = get_piece_idx(piece)
    flag = check_win(board, x, y)
    board[x][y] = 0
    return flag
    
def check_win(board, x, y):
    def check_equal_attributes(pieces):
        if len(pieces) < 4:
            return False
        for i in range(4):
            if all(piece[i] == pieces[0][i] for piece in pieces):
                return True
        
        return False

    # Check row
    pieces = []
    for j in range(BOARD_COLS):
        piece = get(board, x, j)
        if not piece:
            break
        else:
            pieces.append(piece)
    
    if check_equal_attributes(pieces):
        return True
    
    # Check column
    pieces = []
    for i in range(BOARD_ROWS):
        piece = get(board, i, y)
        if not piece:
            break
        else:
            pieces.append(piece)
    
    if check_equal_attributes(pieces):
        return True

    # Check main diagonal (top-left to bottom-right)
    if x == y:  # Only check if the piece is on the diagonal
        pieces = []
        for i in range(BOARD_ROWS):
            piece = get(board, i, i)
            if not piece:
                break
            else:
                pieces.append(piece)
        
        if check_equal_attributes(pieces):
            return True

    # Check anti-diagonal (top-right to bottom-left)
    if x + y == BOARD_ROWS - 1:  # Only check if the piece is on the diagonal
        pieces = []
        for i in range(BOARD_ROWS):
            piece = get(board, i, BOARD_ROWS - 1 - i)
            if not piece:
                break
            else:
                pieces.append(piece)
        
        if check_equal_attributes(pieces):
            return True

    # Check 2x2 groups
    for i in range(max(0, x - 1), min(BOARD_ROWS - 2, x) + 1):
        for j in range(max(0, y - 1), min(BOARD_COLS - 2, y) + 1):
            pieces = [
                get(board, i, j), get(board, i, j + 1),
                get(board, i + 1, j), get(board, i + 1, j + 1)
            ]

            if None not in pieces and check_equal_attributes(pieces):
                return True

    return False