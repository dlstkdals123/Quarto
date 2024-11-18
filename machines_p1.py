import numpy as np
import random
from collections import defaultdict
import math
import copy

MCTS_ITERATIONS = 100
BOARD_ROWS = 4
BOARD_COLS = 4

PLAYER = 1

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        tree = MCTS(debug=False)
        node = Node(self.board, PLAYER, "select_piece", debug=tree.debug)
                
        for i in range(MCTS_ITERATIONS):

            tree.do_rollout(node)
            
            # 진행 상태를 25%씩 출력 (i / MCTS_ITERATIONS * 100)
            progress = (i + 1) / MCTS_ITERATIONS * 100
            if progress % 25 == 0:  # 25% 단위로 출력
                print(f"Progress: {int(progress)}%")

        best_node = tree.choose(node)
        return best_node.board_state.selected_piece

    def place_piece(self, selected_piece):
        tree = MCTS(debug=False)
        node = Node(self.board, PLAYER, "place_piece", selected_piece, debug=tree.debug)
        tree.do_rollout(node)
        if node in tree.children and tree.children[node]:
            # Check if any child is an immediate win
            for child in tree.children[node]:
                if child.board_state.player == PLAYER and child.board_state.check_win():
                    for row in range(BOARD_ROWS):
                        for col in range(BOARD_COLS):
                            if child.board_state[row][col] == self.pieces.index(selected_piece) + 1:  # 실제 값으로 비교
                                print(f"is win in ({row}, {col})")
                                return row, col
                            
        for i in range(MCTS_ITERATIONS):
            tree.do_rollout(node)

            # 진행 상태를 25% 단위로 출력 (i / MCTS_ITERATIONS * 100)
            progress = (i + 1) / MCTS_ITERATIONS * 100
            if progress % 25 == 0:  # 25% 단위로 출력
                print(f"Progress: {int(progress)}%")

        best_node = tree.choose(node)
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if best_node.board_state[row][col] == self.pieces.index(selected_piece) + 1:  # 실제 값으로 비교
                    return row, col
    
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
        self.children[node] = node.find_children()

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
    def __init__(self, board_state, player, current_state, selected_piece = None, debug=False):
        self.board_state = Board(board_state, player, current_state, selected_piece, debug)
        self.children = []
        self.debug = debug

    def find_children(self):
        result = []
        # 플레이어 순서를 고려하여 자식 노드의 상태를 전환
        if self.board_state.current_state == "select_piece":
            # select_piece 이후는 상대가 piece를 배치할 차례
            for selected_piece in self.board_state.available_pieces:
                next_board_state = copy.deepcopy(self.board_state)
                next_board_state.select(selected_piece)
                next_node = Node(next_board_state.get_board(), next_board_state.player, next_board_state.current_state, next_board_state.selected_piece, next_board_state.debug)
                result.append(next_node)

        elif self.board_state.current_state == "place_piece":
            for selected_place in self.board_state.available_places:
                next_board_state = copy.deepcopy(self.board_state)
                next_board_state.place(selected_place[0], selected_place[1])
                next_node = Node(next_board_state.get_board(), next_board_state.player, next_board_state.current_state, next_board_state.selected_piece, next_board_state.debug)
                result.append(next_node)
        else:
            raise TypeError(f'current_state({self.self.board_state.current_state}) is invalid')
        
        if self.debug:
            print(f"Node {self} generated children {len(result)}:")
            print()

        self.children = result
        return result

    def find_random_child(self):
        return random.choice(self.children)

    def is_terminal(self):
        return not bool(self.children)

    def __hash__(self):
        return hash(self.board_state)

    def __eq__(self, other):
        return self.board_state == other.board_state
    
    def __str__(self):
        return str(self.board_state)
    
    def reward(self):
        "Assumes self is terminal node. 1=win, 0=loss, .5=tie, etc"
        current_board = copy.deepcopy(self.board_state)

        while not current_board.check_win() and not current_board.is_board_full():
            # Randomly select a piece to give to the opponent
            if current_board.current_state == "select_piece":
                current_board.random_select()
            
            # Randomly place the piece on the board
            elif current_board.current_state == "place_piece":
                current_board.random_place()
            
            else:
                raise TypeError(f'current_state({current_board.current_state}) is invalid')
        
        if current_board.check_win():
            return -1 * current_board.player == PLAYER
        elif current_board.is_board_full():
            return 0.5
        else:
            raise TypeError('current_board is not terminal')
            
            
class Board:
    def __init__(self, board, player, current_state, selected_piece = None, debug=False):
        self.__board = board
        self.player = player
        self.current_state = current_state
        self.selected_piece = selected_piece
        self.debug = debug
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.available_places = self.get_available_places()
        self.available_pieces = self.get_available_pieces()

    def is_board_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.__board[row][col] == 0:
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
                subgrid = [self.__board[r][c], self.__board[r][c+1], self.__board[r+1][c], self.__board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True
        return False

    def check_win(self):
        # Check rows, columns, and diagonals
        for col in range(BOARD_COLS):
            if self.check_line([self.__board[row][col] for row in range(BOARD_ROWS)]):
                return True
        
        for row in range(BOARD_ROWS):
            if self.check_line([self.__board[row][col] for col in range(BOARD_COLS)]):
                return True
            
        if self.check_line([self.__board[i][i] for i in range(BOARD_ROWS)]) or self.check_line([self.__board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win():
            return True
        
        return False

    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.__board[row][col] == 0:
                    available_places.append((row, col))
        return available_places

    def get_available_pieces(self):
        all_pieces = set(range(1, 17))  # 1부터 16까지의 전체 인덱스
        used_pieces = set()

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                num = self.__board[row][col]
                if num != 0:
                    used_pieces.add(num)
                    
        available_indices = list(all_pieces - used_pieces)
        available_pieces = [self.pieces[idx - 1] for idx in available_indices]
        if self.selected_piece is not None and self.selected_piece in available_pieces:
            available_pieces.remove(self.selected_piece)
        return available_pieces
    
    def get_board(self):
        return self.__board
    
    def __getitem__(self, index):
        # Delegate subscript access to __board
        return self.__board[index]
    
    def __str__(self):
        # Convert the board into a readable string
        board_str = '\n'.join(' '.join(map(str, row)) for row in self.__board)
        return f"Board:\n{board_str}\nPlayer: {self.player}\nState: {self.current_state}"
    
    def random_select(self):
        if self.current_state != "select_piece":
            raise TypeError(f"Now is {self.current_state} state")
        
        # Select a random available place
        selected_piece = random.choice(self.available_pieces)
        
        self.select(selected_piece)
    
    def random_place(self):
        if self.current_state != "place_piece":
            raise TypeError("Now is select_piece state")
        
        # Place a random available piece
        selected_place = random.choice(self.available_places)
        
        self.place(selected_place[0], selected_place[1])
    
    def place(self, row, col):
        self.current_state = "select_piece"
        self.player = self.player
        self.__board[row][col] = self.pieces.index(self.selected_piece) + 1
        self.available_places.remove((row, col))
        self.selected_piece = None

    def select(self, piece):
        self.current_state = "place_piece"
        self.player = -self.player
        self.selected_piece = piece
        self.available_pieces.remove(piece)

    def board_to_string(self):
        return ','.join(map(str, sum(self.board_state._board, [])))
    
    def __hash__(self):
    # Hash based on __board, player, and current_state
        return hash((
            tuple(tuple(row) for row in self.__board),  # Convert __board to immutable tuple
            self.player,
            self.current_state
        ))

    def __eq__(self, other):
        # Equality check for hash compatibility
        if not isinstance(other, Board):
            return False
        return (
            all(self.__board[row][col] == other.__board[row][col] for row in range(BOARD_ROWS) for col in range(BOARD_COLS)) and
            self.player == other.player and
            self.current_state == other.current_state
        )
