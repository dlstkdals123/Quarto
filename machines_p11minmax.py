import numpy as np
from itertools import product
import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board  # Board with piece indices. 0: empty, 1~16: piece index
        self.available_pieces = available_pieces  # Currently available pieces in tuple form (e.g., (1, 0, 1, 0))

    def check_win(self, board): #승패 확인
        def check_line(line): #행,열,대각선 확인
            if 0 in line:
                return False  # Line is incomplete
            characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
            for i in range(4):  # Check each characteristic
                if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                    return True
            return False

        def check_2x2_subgrid_win(): #2x2 확인
            for r in range(3):  # BOARD_ROWS - 1
                for c in range(3):  # BOARD_COLS - 1
                    subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                    if 0 not in subgrid:
                        characteristics = [self.pieces[idx - 1] for idx in subgrid]
                        for i in range(4):  # Check each characteristic
                            if len(set(char[i] for char in characteristics)) == 1:
                                return True
            return False

        #행,열 확인
        for col in range(4):
            if check_line([board[row][col] for row in range(4)]):
                return True

        for row in range(4):
            if check_line([board[row][col] for col in range(4)]):
                return True

        #대각선 확인
        if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3 - i] for i in range(4)]):
            return True
        #2x2 확인
        if check_2x2_subgrid_win():
            return True

        return False

    def minmax_alpha_beta(self, board, available_pieces, depth, alpha, beta, is_maximizing, selected_piece): 
        #is_maximizing : 현재 노드가 최대화 플레이어의 턴인지 여부 // True -> 최대화 플레이어 턴 // False -> 최소화 플레이어 턴
        #depth 깊이제한 값
        if self.check_win(board): #노드가 승리 상태면 10 반환 /  패배 상태면 -10 반환
            return -10 if is_maximizing else 10

        if len(available_pieces) == 0 or depth == 0:
            return 0  # 무승부 또는 depth에 다다르면 0 반환

        if is_maximizing:
            max_eval = -float('inf')
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    board[row][col] = self.pieces.index(selected_piece) + 1
                    eval = self.minmax_alpha_beta(board, available_pieces, depth - 1, alpha, beta, False, None)
                    board[row][col] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for piece in available_pieces:
                eval = self.minmax_alpha_beta(board, [p for p in available_pieces if p != piece], depth - 1, alpha, beta, True, piece)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def select_piece(self):
        best_piece = None
        best_value = float('inf')

        for piece in self.available_pieces:
            eval = self.minmax_alpha_beta(self.board, [p for p in self.available_pieces if p != piece], 3, -float('inf'), float('inf'), True, piece)
            #임시로 깊이제한 3
            if eval < best_value:
                best_value = eval
                best_piece = piece

        return best_piece

    def place_piece(self, selected_piece):
        best_move = None
        best_value = -float('inf')

        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.pieces.index(selected_piece) + 1
                eval = self.minmax_alpha_beta(self.board, self.available_pieces, 3, -float('inf'), float('inf'), False, None)
                #임시로 깊이제한 3
                self.board[row][col] = 0

                if eval > best_value:
                    best_value = eval
                    best_move = (row, col)

        return best_move