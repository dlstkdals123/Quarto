import sys
import numpy as np
import os
from datetime import datetime

from machines_p1 import P1
from machines_p2 import P2
import time

players = {
    1: P1,
    2: P2
}

BOARD_ROWS = 4
BOARD_COLS = 4

# Initialize board and pieces
board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

# MBTI Pieces (Binary Encoding: I/E = 0/1, N/S = 0/1, T/F = 0/1, P/J = 0/1)
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
available_pieces = pieces[:]

# Global variable for selected piece
selected_piece = None

def available_square(row, col):
    return board[row][col] == 0

def is_board_full():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 0:
                return False
    return True

def check_line(line):
    if 0 in line:
        return False  # Incomplete line
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

def check_2x2_subgrid_win():
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:  # All cells must be filled
                characteristics = [pieces[idx - 1] for idx in subgrid]
                for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                    if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                        return True
    return False

def check_win():
    # Check rows, columns, and diagonals
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
        
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True

    # Check 2x2 sub-grids
    if check_2x2_subgrid_win():
        return True
    
    return False

def restart_game():
    global board, available_pieces, selected_piece, player
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    available_pieces = pieces[:]
    selected_piece = None  # Reset selected piece

def second2hhmmss(seconds):
    if seconds >= 3600:
        hh = seconds//3600
        mm = (seconds-(hh*3600))//60
        ss = seconds-(hh*3600)-(mm*60)
        return f"{hh:.0f}h {mm:.0f}m {ss:.1f}s"
    elif seconds >= 60:
        mm = seconds//60
        ss = seconds-(mm*60)
        return f"{mm:.0f}m {ss:.1f}s"
    else:
        return f"{seconds:.1f}s"

# Game loop
turn = 1

ITERATION = 10

results = {"P1": {"wins": 0, "draws": 0, "loses": 0, "timeouts": 0, "total_time": 0},
           "P2": {"wins": 0, "draws": 0, "loses": 0, "timeouts": 0, "total_time": 0}}

for iteration in range(1, ITERATION + 1):
    print(f"iteration: {iteration} is running...")
    restart_game()
    game_over = False
    flag = "select_piece"
    total_time_consumption = {1: 0, 2: 0}
    winner = None

    while not game_over:
        if flag == "select_piece":
            begin = time.time()
            player = players[3 - turn](board=board, available_pieces=available_pieces)
            selected_piece = player.select_piece()
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    if board[row][col] == pieces.index(selected_piece) + 1:
                        raise TypeError(f"P{3 - turn}; wrong selection")
            finish = time.time()
            elapsed_time = finish - begin
            total_time_consumption[3 - turn] += elapsed_time

            flag = "place_piece"
        else:
            begin = time.time()
            player = players[turn](board=board, available_pieces=available_pieces)
            (board_row, board_col) = player.place_piece(selected_piece)
            finish = time.time()
            elapsed_time = finish - begin
            total_time_consumption[turn] += elapsed_time

            if available_square(board_row, board_col):
                board[board_row][board_col] = pieces.index(selected_piece) + 1
                available_pieces.remove(selected_piece)
                selected_piece = None

                if check_win():
                    game_over = True
                    winner = turn
                elif is_board_full():
                    game_over = True
                    winner = None
                else:
                    turn = 3 - turn
                    flag = "select_piece"
            else:
                raise TypeError(f"P{turn}; wrong selection")

    # 기록
    if winner:
        loser = 3 - winner
        results[f"P{winner}"]["wins"] += 1
        results[f"P{loser}"]["loses"] += 1
    else:
        results["P1"]["draws"] += 1
        results["P2"]["draws"] += 1

    results["P1"]["total_time"] += total_time_consumption[1]
    if total_time_consumption[1] >= 300:
        results["P1"]["timeouts"] += 1
    results["P2"]["total_time"] += total_time_consumption[2]
    if total_time_consumption[2] >= 300:
        results["P2"]["timeouts"] += 1

LOG_DIR = "log"
LOG_FILENAME = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%m%d_%H%M%S')}.txt")
os.makedirs(LOG_DIR, exist_ok=True)

with open(LOG_FILENAME, "w") as log_file:
# 종합 통계
    summary_lines = [f"Summary:\nTotal_Iteration: {ITERATION}\nMCTS_ITERATIONS: P1=(write this), P2=(write this)\n"]  # Summary 제목 추가
    for player in ["P1", "P2"]:
        total_games = ITERATION
        wins = results[player]["wins"]
        draws = results[player]["draws"]
        loses = results[player]["loses"]
        timeouts = results[player]["timeouts"]
        total_time = results[player]["total_time"]
        avg_time = total_time / total_games if total_games > 0 else 0
        win_rate = (wins / total_games) * 100
        summary_lines.append(
            f"{player}: [{wins}, {draws}, {loses}({timeouts})] {win_rate:.1f}% {avg_time:.1f}s\n"
        )

    log_file.writelines(summary_lines)