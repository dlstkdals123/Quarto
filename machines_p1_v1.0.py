import numpy as np
import random
from collections import defaultdict
import math
import copy

MCTS_ITERATIONS = 1000
BOARD_ROWS = 4
BOARD_COLS = 4

PLAYER = 1

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        return 

    def place_piece(self):
        return 
    
class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, debug=False):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.debug = debug

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

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
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        if self.debug:
            print(f"Expanding node: {node}")
        self.children[node] = node.find_children()
        if self.debug:
            print(f"Updated MCTS children for {node}: {self.children[node]}")

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

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


class Node():
    def __init__(self, board_state, player, current_state, debug=False):
        self.board_state = Board(board_state)
        self.player = player
        self.current_state = current_state
        self.children = []
        self.debug = debug

    def find_children(self):
        if self.current_state == "selected_piece":
            result = self.board_state.available_pieces
        elif self.current_state == "place_piece":
            result = self.board_state.available_places
        else:
            raise TypeError(f'current_state({self.current_state}) is invalid')
        
        if self.debug:
            print(f"Node {self} generated children: {result}")
        return result

    def find_random_child(self):
        return random.choice(self.children)

    def is_terminal(self):
        return not bool(self.children)

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        if self.board_state.check_win():
            return -1 * self.player == PLAYER
        elif self.board_state.is_board_full():
            return 0.5
        else:
            raise ValueError(f'board_state is not terminated')

    def __hash__(self):
        return hash((tuple(self.board_state), self.player, self.current_state))

    def __eq__(self, other):
        # 다른 객체가 같은 클래스의 인스턴스인지 확인
        if not isinstance(other, Node):
            return False
        
        # 보드 상태, 플레이어, 현재 상태 비교
        return (self.board_state == other.board_state and
                self.player == other.player and
                self.current_state == other.current_state)

class Board:
    def __init__(self, board):
        self.__board = copy.deepcopy(board)
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.available_places = self.get_available_places()
        self.available_pieces = self.get_available_pieces()

    def is_board_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    return False
        return True

    def check_line(self, line):
        if 0 in line:
            return False  # Incomplete line
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True
        return False

    def check_2x2_subgrid_win(self):
        for r in range(BOARD_ROWS - 1):
            for c in range(BOARD_COLS - 1):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
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

    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    available_places.append((row, col))
        return available_places

    def get_available_pieces(self):
        all_pieces = set(range(1, 17))  # 1부터 16까지의 전체 인덱스
        used_pieces = set()

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.available_square(row, col):
                    self.available_positions.append((row, col))
                num = self.__board[row][col]
                if num != 0:
                    used_pieces.add(num)
                    
        available_indices = list(all_pieces - used_pieces)
        return [self.pieces[idx] for idx in available_indices]
    
    def place(self, row, col):
        if self.__board.available_square(row, col):
            self.__board[row][col] = self.selected_piece
            self.selected_piece = None

    def select(self, piece):
        self.selected_piece = piece