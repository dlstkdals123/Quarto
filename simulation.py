import sys
import numpy as np
import os
from datetime import datetime

import machines_p1
import machines_p2
import time

ITERATION = 100
P1 = machines_p1.P1
P2 = machines_p2.P2
P1_MCTS_ITERATIONS = machines_p1.MCTS_ITERATIONS
P2_MCTS_ITERATIONS = machines_p2.MCTS_ITERATIONS


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
    # Write the overall summary header
    summary_lines = [
        "Detailed Summary of Results:\n",
        f"Total Iterations: {ITERATION}\n",
        f"MCTS Iterations:\n",
        f"    P1: {P1_MCTS_ITERATIONS} iterations per move (Monte Carlo Tree Search for Player 1)\n",
        f"    P2: {P2_MCTS_ITERATIONS} iterations per move (Monte Carlo Tree Search for Player 2)\n",
        "-" * 50 + "\n"
    ]
    
    # Iterate over the players in the results dictionary
    for player, stats in results.items():
        # Fetch statistics for the player
        wins = stats["wins"]
        draws = stats["draws"]
        loses = stats["loses"]
        timeouts = stats["timeouts"]
        total_time = stats["total_time"]
        
        # Calculate derived statistics
        avg_time = total_time / ITERATION if ITERATION > 0 else 0
        win_rate = (wins / ITERATION) * 100 if ITERATION > 0 else 0
        
        # Append detailed stats for the player
        summary_lines.append(
            f"Player: {player}\n"
            f"    Wins: {wins}\n"
            f"    Draws: {draws}\n"
            f"    Losses: {loses} (Timeouts: {timeouts})\n"
            f"    Win Rate: {win_rate:.2f}%\n"
            f"    Average Time Per Game: {avg_time:.2f} seconds\n\n"
        )
    
    # Write the summary lines to the log file
    log_file.writelines(summary_lines)
