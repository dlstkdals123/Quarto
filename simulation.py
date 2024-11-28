import sys
import numpy as np
import os
from datetime import datetime

import machines_p11
import machines_p2
import time

ITERATION = 100
P1 = machines_p11.P1
P2 = machines_p2.P2
P1_MCTS_ITERATIONS = machines_p11.MCTS_ITERATIONS
P1_SWITCH_POINT = machines_p11.SWITCH_POINT
P2_MCTS_ITERATIONS = machines_p2.MCTS_ITERATIONS
P2_SWITCH_POINT = machines_p2.SWITCH_POINT

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

# results 초기화
results = {
    "P1": {
        "wins": 0, "draws": 0, "loses": 0, "timeouts": 0, "total_time": 0, "switches": 0,
        "switch_wins": 0, "switch_draws": 0, "switch_loses": 0, "minimax_time": 0
    },
    "P2": {
        "wins": 0, "draws": 0, "loses": 0, "timeouts": 0, "total_time": 0, "switches": 0,
        "switch_wins": 0, "switch_draws": 0, "switch_loses": 0, "minimax_time": 0
    }
}

# 메인 게임 루프
for iteration in range(1, ITERATION + 1):
    print(f"iteration: {iteration} is running...")
    restart_game()
    game_over = False
    flag = "select_piece"
    total_time_consumption = {1: 0, 2: 0}
    winner = None

    # MCTS -> Minimax 전환 여부 추적
    p1_switched = False
    p2_switched = False

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

            # MCTS -> Minimax 전환 체크 (available_pieces 길이 기준)
            if 3 - turn == 1 and not p1_switched and len(available_pieces) <= P1_SWITCH_POINT:
                p1_switched = True
            elif 3 - turn == 2 and not p2_switched and len(available_pieces) <= P2_SWITCH_POINT:
                p2_switched = True

            # Minimax 시간 추가
            if (3 - turn == 1 and p1_switched) or (3 - turn == 2 and p2_switched):
                results[f"P{3 - turn}"]["minimax_time"] += elapsed_time

            flag = "place_piece"
        else:
            begin = time.time()
            player = players[turn](board=board, available_pieces=available_pieces)
            (board_row, board_col) = player.place_piece(selected_piece)
            finish = time.time()
            elapsed_time = finish - begin
            total_time_consumption[turn] += elapsed_time

            if (turn == 1 and p1_switched) or (turn == 2 and p2_switched):
                results[f"P{turn}"]["minimax_time"] += elapsed_time

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
        # 전환을 사용한 판에 대한 기록 추가
        if p1_switched and winner == 1:
            results["P1"]["switch_wins"] += 1
        if p2_switched and winner == 2:
            results["P2"]["switch_wins"] += 1
        if p1_switched and loser == 1:
            results["P1"]["switch_loses"] += 1
        if p2_switched and loser == 2:
            results["P2"]["switch_loses"] += 1
    else:
        results["P1"]["draws"] += 1
        results["P2"]["draws"] += 1
        # 전환을 사용한 판에 대한 무승부 기록 추가
        if p1_switched:
            results["P1"]["switch_draws"] += 1
        if p2_switched:
            results["P2"]["switch_draws"] += 1

    results["P1"]["total_time"] += total_time_consumption[1]
    if total_time_consumption[1] >= 300:
        results["P1"]["timeouts"] += 1
    results["P2"]["total_time"] += total_time_consumption[2]
    if total_time_consumption[2] >= 300:
        results["P2"]["timeouts"] += 1

    # 한 판당 최대 한 번의 전환 기록
    if p1_switched:
        results["P1"]["switches"] += 1
    if p2_switched:
        results["P2"]["switches"] += 1

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
        f"    P1: Switch to Minimax after {P1_SWITCH_POINT} iterations per move (Player 1)\n",
        f"    P2: Switch to Minimax after {P2_SWITCH_POINT} iterations per move (Player 2)\n",
        "-" * 50 + "\n"
    ]
    
    for player, stats in results.items():
        # Fetch statistics for the player
        wins = stats["wins"]
        draws = stats["draws"]
        loses = stats["loses"]
        timeouts = stats["timeouts"]
        total_time = stats["total_time"]
        switches = stats["switches"]
        switch_wins = stats["switch_wins"]
        switch_draws = stats["switch_draws"]
        switch_loses = stats["switch_loses"]
        minimax_time = stats["minimax_time"]  # Minimax 시간 추가

        # Calculate derived statistics
        avg_time = total_time / ITERATION if ITERATION > 0 else 0
        avg_minimax_time = minimax_time / switches if switches > 0 else 0  # Minimax 평균 시간 계산

        # Append detailed stats for the player
        summary_lines.append(
            f"Player: {player}\n"
            f"    Wins: {wins}\n"
            f"    Draws: {draws}\n"
            f"    Losses: {loses} (Timeouts: {timeouts})\n"
            f"    Average Time Per Game: {avg_time:.2f} seconds\n"
            f"    Switches to Minimax: {switches}\n"
            f"    Total Minimax Time: {minimax_time:.2f} seconds\n"
            f"    Average Minimax Time Per Switch: {avg_minimax_time:.2f} seconds\n"
            f"    Stats when Minimax was used:\n"
            f"        Wins: {switch_wins}\n"
            f"        Draws: {switch_draws}\n"
            f"        Losses: {switch_loses}\n\n"
        )
    
    # Write the summary lines to the log file
    log_file.writelines(summary_lines)
