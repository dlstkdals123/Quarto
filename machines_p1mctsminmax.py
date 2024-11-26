import numpy as np
import random
from collections import defaultdict
import math
import copy

MCTS_ITERATIONS = 500
BOARD_ROWS = 4
BOARD_COLS = 4
PLAYER = 1
isFirst = True


class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # 총 16개의 조각
        self.board = board
        self.available_pieces = available_pieces
        self.cache = {}  # Minimax용 캐시

    def reset_cache(self):
        """Minimax 캐시 초기화"""
        self.cache.clear()

    def select_piece(self):
        global isFirst

        # 첫 선택은 무작위
        if isFirst:
            isFirst = False
            return random.choice(self.pieces)

        empty_cells = sum(1 for row in self.board for col in row if col == 0)

        if empty_cells > 10:
            # MCTS 기반 탐색
            tree = MCTS(debug=False)
            node = Node(self.board, PLAYER, "select_piece", debug=tree.debug)

            for i in range(MCTS_ITERATIONS):
                #if (i + 1) % (MCTS_ITERATIONS // 10) == 0 or i + 1 == MCTS_ITERATIONS:
                    #print(f"Progress (MCTS): {(i + 1) / MCTS_ITERATIONS * 100:.0f}%")
                tree.do_rollout(node)

            best_node = tree.choose(node)
            return best_node.board_state.selected_piece
        else:
            # Minimax 기반 탐색
            best_piece = None
            best_score = -float('inf')

            for piece in self.available_pieces:
                score = self.evaluate_future(self.board, piece)
                if score > best_score:
                    best_score = score
                    best_piece = piece

            return best_piece

    def place_piece(self, selected_piece):
        empty_cells = sum(1 for row in self.board for col in row if col == 0)

        if empty_cells > 10:
            # MCTS 기반 탐색
            tree = MCTS(debug=False)
            node = Node(self.board, PLAYER, "place_piece", selected_piece, debug=tree.debug)

            # 승리 조건 조기 확인
            for row, col in node.board_state.available_places:
                if node.board_state.mcts_check_win(selected_piece, row, col):
                    return row, col

            for i in range(MCTS_ITERATIONS):
                if (i + 1) % (MCTS_ITERATIONS // 10) == 0 or i + 1 == MCTS_ITERATIONS:
                    print(f"Progress (MCTS): {(i + 1) / MCTS_ITERATIONS * 100:.0f}%")
                tree.do_rollout(node)

            best_node = tree.choose(node)
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    if best_node.board_state[row][col] == self.pieces.index(selected_piece) + 1:
                        return row, col
        else:
            # Minimax 기반 탐색
            best_move = None
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')

            available_locs = [(row, col) for row in range(4) for col in range(4) if self.board[row][col] == 0]

            for row, col in available_locs:
                self.board[row][col] = self.pieces.index(selected_piece) + 1
                score = self.minimax(self.board, self.available_pieces, is_maximizing=False, alpha=alpha, beta=beta)
                self.board[row][col] = 0
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

            return best_move

    def minimax(self, board, available_pieces, is_maximizing, alpha, beta):
        # 캐싱: 동일 상태 재계산 방지
        board_tuple = tuple(map(tuple, board))
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        # 종료 조건: 승리, 패배, 무승부
        if self.minimax_check_win(board):
            return 1e9 if is_maximizing else -1e9
        if all(board[row][col] != 0 for row in range(BOARD_ROWS) for col in range(BOARD_COLS)):
            return 0  # 무승부

        # 최대화 플레이어의 턴
        if is_maximizing:
            max_eval = -float('inf')
            available_locs = [(row, col) for row in range(BOARD_ROWS) for col in range(BOARD_COLS) if board[row][col] == 0]

            for row, col in available_locs:
                board[row][col] = 1  # 임시로 말을 놓음
                eval = self.minimax(board, available_pieces, False, alpha, beta)
                board[row][col] = 0  # 복구
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:  # 알파-베타 가지치기
                    break

            self.cache[board_tuple] = max_eval
            return max_eval

        else:
            min_eval = float('inf')

            for piece in available_pieces:
                eval = self.minimax(board, available_pieces, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:  # 알파-베타 가지치기
                    break

            self.cache[board_tuple] = min_eval
            return min_eval

    def evaluate_future(self, board, piece):
        score = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    board[row][col] = self.pieces.index(piece) + 1
                    if self.minimax_check_win(board):
                        score += 1
                    board[row][col] = 0
        return score

    def evaluate_board(self, board):
        score = 0
        # 각 행, 열, 대각선을 평가
        for row in range(4):
            score += self.evaluate_line([board[row][col] for col in range(4)])
        for col in range(4):
            score += self.evaluate_line([board[row][col] for row in range(4)])
        score += self.evaluate_line([board[i][i] for i in range(4)])
        score += self.evaluate_line([board[i][3 - i] for i in range(4)])

        # 2x2 평가
        score += self.evaluate_2x2_subgrids(board)

        return score
    def evaluate_line(self, line):
        """한 줄(행, 열, 대각선)을 평가"""
        if 0 in line:  # 빈 칸이 있으면 평가하지 않음
            return 0
        # 각 특성을 확인하여 동일한 특성이 몇 개 일치하는지 계산
        characteristics = [self.pieces[piece_idx - 1] for piece_idx in line if piece_idx != 0]
        shared_attributes = sum(len(set(attr)) == 1 for attr in zip(*characteristics))
        return shared_attributes * 10  # 각 특성 일치마다 10점 부여
    

    def minimax_check_win(self, board):
        """승리 조건이 충족되었는지 확인"""
        # 가로, 세로, 대각선 체크
        for row in range(4):
            if self.evaluate_line([board[row][col] for col in range(4)]) == 40:
                return True
        for col in range(4):
            if self.evaluate_line([board[row][col] for row in range(4)]) == 40:
                return True
        if self.evaluate_line([board[i][i] for i in range(4)]) == 40:
            return True
        if self.evaluate_line([board[i][3 - i] for i in range(4)]) == 40:
            return True

        # 2x2 승리 조건 체크
        if self.check_2x2_subgrid_win(board):
            return True
        
        return False
    def check_2x2_subgrid_win(self, board):
        """2x2에서 승리 조건을 확인 (40점)"""
        for row in range(3):
            for col in range(3):
                subgrid = [
                    board[row][col], board[row][col + 1],
                    board[row + 1][col], board[row + 1][col + 1]
                ]
                if 0 not in subgrid:  # 모든 칸이 채워졌다면
                    subgrid_pieces = [self.pieces[idx - 1] for idx in subgrid]
                    total_score = 0
                    for i in range(4):  # 각 특성에 대해 일치하는지 확인
                        if len(set(attr[i] for attr in subgrid_pieces)) == 1:
                            total_score += 10  # 각 일치 특성에 대해 10점 추가

                    # 만약 4개 특성이 모두 일치하면 40점이 되어 승리
                    if total_score == 40:
                        return True
        return False

    def evaluate_2x2_subgrids(self, board):
        """2x2에서 특성 일치 평가"""
        score = 0
        for row in range(3):  # 2x2 영역을 찾아서
            for col in range(3):
                subgrid = [
                    board[row][col], board[row][col + 1],
                    board[row + 1][col], board[row + 1][col + 1]
                ]
                if 0 not in subgrid:  # 모든 칸이 채워졌다면
                    subgrid_pieces = [self.pieces[idx - 1] for idx in subgrid]
                    total_score = 0
                    for i in range(4):  # 각 특성에 대해 일치하는지 확인
                        if len(set(attr[i] for attr in subgrid_pieces)) == 1:
                            total_score += 10  # 각 일치 특성에 대해 10점 추가

                    # 만약 4개 특성이 모두 일치하면 40점이 되어 승리
                    if total_score == 40:
                        score += 40  # 2x2 승리하면 40점 추가
        return score

    def is_full(self, board):
        """보드가 가득 찼는지 확인"""
        return all(board[row][col] != 0 for row in range(4) for col in range(4))
 
    
class MCTS:
    def __init__(self, exploration_weight=1, debug=False):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.debug = debug

    def choose(self, node):
        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        if self.debug:
            scores = {child: score(child) for child in self.children[node]}
            for child, s in scores.items():
                print(f"Score of {child}: {s}")

        return max(self.children[node], key=score)

    def do_rollout(self, node):
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
                return path

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        invert_reward = True
        depth = 1
        while True:
            if node.is_terminal():
                reward = node.reward(depth)
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            depth += 1
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward

    def _uct_select(self, node):
        log_N_vertex = math.log(self.N[node])

        def uct(n):
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
            for key, pieces in self.board_state.available_pieces.items():
                next_board_state = copy.deepcopy(self.board_state)
                next_board_state.select(next(iter(pieces)))
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
    
    def reward(self, depth):
        "Assumes self is terminal node. 1=win, 0=loss, .5=tie, etc"
        current_board = copy.deepcopy(self.board_state)

        while not current_board.is_board_full():
            # Randomly select a piece to give to the opponent
            if current_board.current_state == "select_piece":
                current_board.random_select()
            
            # Randomly place the piece on the board
            elif current_board.current_state == "place_piece":
                row, col = current_board.random_place()
                selected_piece = current_board.selected_piece
                if current_board.mcts_check_win(selected_piece, row, col):
                    return (current_board.player == PLAYER) / depth
                else:
                    current_board.place(row, col)
            
            else:
                raise TypeError(f'current_state({current_board.current_state}) is invalid')
            
            depth += 1
        
        return 0.5
    

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

    def mcts_check_win(self, piece, x, y):
        """
        Check if placing the given piece at (x, y) results in a win.
        This includes row, column, diagonal, and 2x2 square checks.
        """
        # Place the piece temporarily
        original_value = self.__board[x][y]
        self.__board[x][y] = self.pieces.index(piece) + 1

        def has_common_attribute(line):
            """Check if all pieces in a line share at least one common attribute."""
            attributes = [0, 0, 0, 0]  # Attributes: (bit0, bit1, bit2, bit3)
            for cell in line:
                if cell == 0:  # Skip empty cells
                    return False
                for i in range(4):  # Check each bit
                    attributes[i] += cell[i]
            return any(attr == len(line) or attr == 0 for attr in attributes)

        # Check rows, columns, and diagonals
        row = [self.pieces[self.__board[x][j] - 1] for j in range(BOARD_ROWS)]
        col = [self.pieces[self.__board[i][y] - 1] for i in range(BOARD_COLS)]
        diag1 = [self.pieces[self.__board[i][i] - 1] for i in range(BOARD_COLS)] if x == y else []  # Main diagonal
        diag2 = [self.pieces[self.__board[i][BOARD_ROWS - 1 - i] - 1] for i in range(BOARD_COLS)] if x + y == BOARD_ROWS - 1 else []

        # Check 2x2 squares
        squares = []
        for i in range(max(0, x - 1), min(BOARD_ROWS - 2, x) + 1):
            for j in range(max(0, y - 1), min(BOARD_ROWS - 2, y) + 1):
                square = [
                    self.pieces[self.__board[i][j] - 1],
                    self.pieces[self.__board[i][j + 1] - 1],
                    self.pieces[self.__board[i + 1][j] - 1],
                    self.pieces[self.__board[i + 1][j + 1] - 1]
                ]
                squares.append(square)

        # Evaluate win conditions
        try:
            is_win = any(
                has_common_attribute(line)
                for line in [row, col, diag1, diag2] if line
            ) or any(
                has_common_attribute(square)
                for square in squares if len(square) == 4
            )
        finally:
            # Restore the board in case of mutable object issues
            self.__board[x][y] = original_value

        return is_win

    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.__board[row][col] == 0:
                    available_places.append((row, col))
        return available_places

    def get_available_pieces(self):
        all_pieces = set(range(1, 17))  # 1부터 16까지의 전체 인덱스
        if self.selected_piece is not None:
            all_pieces.remove(self.pieces.index(self.selected_piece) + 1)
        used_pieces = set()

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                num = self.__board[row][col]
                if num != 0:
                    used_pieces.add(num)
                    all_pieces.remove(num)

        grouped_pieces = defaultdict(set)
        
        for piece1 in all_pieces:
            difference_counts = []
            for piece2 in used_pieces:
                difference_counts.append(sum(1 for a, b in zip(self.pieces[piece1 - 1], self.pieces[piece2 - 1]) if a != b))
            
            sorted_differences = sorted(difference_counts)

            grouped_pieces[tuple(sorted_differences)].add(self.pieces[piece1 - 1])
        # print(grouped_pieces)
        return grouped_pieces
    
    def get_board(self):
        return self.__board
    
    def __getitem__(self, index):
        # Delegate subscript access to __board
        return self.__board[index]
    
    def __str__(self):
        # Convert the board into a readable string
        if self.current_state == "place_piece":
            return f"\nPlayer: {self.player}, Selected_piece: {self.selected_piece}\n"
        else:
            board_str = '\n'.join(' '.join(map(str, row)) for row in self.__board)
            return f"\nPlayer: {self.player}, Board:\n{board_str}\n"
    
    def random_select(self):
        if self.current_state != "select_piece":
            raise TypeError(f"Now is {self.current_state} state")
        
        # Select a random available place
        random_key = random.choice(list(self.available_pieces.keys()))

        selected_piece = random.choice(list(self.available_pieces[random_key]))

        self.select(selected_piece)
    
    def random_place(self):
        if self.current_state != "place_piece":
            raise TypeError("Now is select_piece state")
        
        # Place a random available piece
        selected_place = random.choice(self.available_places)
        return selected_place[0], selected_place[1]
    
    def place(self, row, col):
        self.current_state = "select_piece"
        self.player = self.player
        self.__board[row][col] = self.pieces.index(self.selected_piece) + 1
        self.available_places.remove((row, col))
        self.selected_piece = None

    def select(self, piece):
        if not any(piece in s for s in self.available_pieces.values()):
            raise ValueError(f"The selected piece {piece} is not available")

        self.current_state = "place_piece"
        self.player = -self.player
        self.selected_piece = piece

        for key in list(self.available_pieces.keys()):  # 키 목록을 복사
            if piece in self.available_pieces[key]:
                self.available_pieces[key].remove(piece)
                if not self.available_pieces[key]:  # 값이 비었다면 키 삭제
                    del self.available_pieces[key]

    def board_to_string(self):
        return ','.join(map(str, sum(self.board_state._board, [])))
    
    def __hash__(self):
        return hash(self.__board.tobytes()) ^ hash(self.player) ^ hash(self.current_state) ^ hash(self.selected_piece)

    def __eq__(self, other):
        # Equality check for hash compatibility
        if not isinstance(other, Board):
            return False
        
        if self.available_pieces.keys() != other.available_pieces.keys():
            return False
        
        for key in self.available_pieces.keys():
            if self.available_pieces[key] != other.available_pieces[key]:
                return False
            
        return (
            self.player == other.player and
            self.current_state == other.current_state and
            self.selected_piece == other.selected_piece 
        )
