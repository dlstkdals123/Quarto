import numpy as np
import random
from itertools import product
import time

class P1(): 
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # 총 16개의 말
        self.board = board  # 보드에 놓인 말들의 인덱스 (1~16)
        self.available_pieces = available_pieces  # 현재 사용할 수 있는 말들의 리스트
        self.cache = {}  # 보드 상태 평가 결과를 캐시하기 위한 딕셔너리

    def select_piece(self):
        best_piece = None
        best_score = -float('inf')

        # Minimax: 가능한 모든 말들을 평가하여 최적의 말을 선택
        for piece in self.available_pieces:
            score = self.evaluate_future(self.board, piece)  # 미래 예측을 통해 점수를 계산
            if score > best_score:
                best_score = score
                best_piece = piece
        return best_piece

    def place_piece(self, selected_piece):
        best_move = None
        best_score = -float('inf')
        depth = 3  # 탐색 깊이 (조정 가능 but,계산량 급증함)

        # 말을 놓을 수 있는 모든 위치를 찾아서
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]

        # Minimax: 가능한 모든 수를 탐색하고 최적의 위치를 선택
        for row, col in available_locs:
            self.board[row][col] = self.pieces.index(selected_piece) + 1  # 말을 보드에 놓음
            score = self.minimax(self.board, self.available_pieces, depth, True, -float('inf'), float('inf'))
            self.board[row][col] = 0  # 놓았던 말을 다시 원위치로 되돌림

            if score > best_score:
                best_score = score
                best_move = (row, col)

        return best_move

    def minimax(self, board, available_pieces, depth, is_maximizing, alpha, beta):
        board_tuple = tuple(map(tuple, board))
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        if self.check_win(board):  # 게임에서 이겼다면
            return 1e9 if is_maximizing else -1e9
        if depth == 0 or self.is_full(board):  # 탐색 깊이가 0이거나 보드가 가득 찼다면
            return self.evaluate_board(board)  # 보드 평가

        if is_maximizing:  # AI 턴
            max_eval = -float('inf')
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if board[row][col] == 0]
            for row, col in available_locs:
                board[row][col] = 1  # AI 말을 놓음
                eval = self.minimax(board, available_pieces, depth - 1, False, alpha, beta)  # 재귀적으로 평가
                board[row][col] = 0  # 놓았던 말을 다시 원위치로 되돌림
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:  # 알파-베타 가지치기
                    break
            self.cache[board_tuple] = max_eval
            return max_eval
        else:  # 상대 턴
            min_eval = float('inf')
            for piece in available_pieces:
                eval = self.minimax(board, available_pieces, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:  # 알파-베타 가지치기
                    break
            self.cache[board_tuple] = min_eval
            return min_eval

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

    def evaluate_future(self, board, piece):
        """말을 놓았을 때의 보드 상태를 평가"""
        score = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    board[row][col] = self.pieces.index(piece) + 1
                    if self.check_win(board):  # 승리 조건을 충족하는지 확인
                        score += 1
                    board[row][col] = 0  # 놓았던 말을 다시 원위치로 되돌림
        return score

    def check_win(self, board):
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
 