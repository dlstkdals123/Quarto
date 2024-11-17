import sys
import pygame
import numpy as np
import time

from machines_p1 import P1

# Player mapping
players = {
    1: P1,  # AI가 담당하는 Player
}

pygame.init()

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)  # Highlight color for selected piece

# Proportions & Sizes
WIDTH = 400
HEIGHT = 700  # Increased height to display all 16 pieces
LINE_WIDTH = 5
BOARD_ROWS = 4
BOARD_COLS = 4
SQUARE_SIZE = WIDTH // BOARD_COLS
PIECE_SIZE = SQUARE_SIZE // 2  # Size for the available pieces

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('MBTI Quarto')
screen.fill(BLACK)

# Initialize board and pieces
board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

# MBTI Pieces (Binary Encoding: I/E = 0/1, N/S = 0/1, T/F = 0/1, P/J = 0/1)
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
available_pieces = pieces[:]

# Global variable for selected piece
selected_piece = None

# Helper functions
def draw_lines(color=WHITE):
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, color, (0, SQUARE_SIZE * i), (WIDTH, SQUARE_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, color, (SQUARE_SIZE * i, 0), (SQUARE_SIZE * i, WIDTH), LINE_WIDTH)

def draw_pieces():
    font = pygame.font.Font(None, 40)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] != 0:
                piece_idx = board[row][col] - 1
                piece = pieces[piece_idx]
                piece_text = f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"
                text_surface = font.render(piece_text, True, WHITE)
                screen.blit(text_surface, (col * SQUARE_SIZE + 10, row * SQUARE_SIZE + 10))

def draw_available_pieces():
    global selected_piece
    font = pygame.font.Font(None, 30)
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, WIDTH, WIDTH, HEIGHT - WIDTH))
    
    for idx, piece in enumerate(available_pieces):
        col = idx % 4
        row = idx // 4
        piece_text = f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"
        if selected_piece == piece:
            text_surface = font.render(piece_text, True, YELLOW)
        else:
            text_surface = font.render(piece_text, True, BLUE)
        x_pos = col * SQUARE_SIZE + 10
        y_pos = WIDTH + (row * PIECE_SIZE) + 10
        screen.blit(text_surface, (x_pos, y_pos))

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
    for i in range(4):
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

def check_2x2_subgrid_win():
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:
                characteristics = [pieces[idx - 1] for idx in subgrid]
                for i in range(4):
                    if len(set(char[i] for char in characteristics)) == 1:
                        return True
    return False

def check_win():
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True
    if check_2x2_subgrid_win():
        return True
    return False

def display_message(message, color=WHITE):
    font = pygame.font.Font(None, 50)
    text_surface = font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 50))
    screen.blit(text_surface, text_rect)

# Game loop
turn = 1  # 시작은 P1이 선택
flag = "select_piece"
game_over = False
selected_piece = None

draw_lines()
draw_available_pieces()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if not game_over:
            if flag == "select_piece":
                if turn == 1:
                    # P1 (AI)가 조각을 선택
                    player = players[1](board=board, available_pieces=available_pieces)
                    selected_piece = player.select_piece()
                    flag = "place_piece"  # P2가 선택한 조각을 놓음
                else:
                    # P2 (사용자)가 조각을 선택
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if y > WIDTH:
                            col = x // SQUARE_SIZE
                            row = (y - WIDTH) // PIECE_SIZE
                            piece_idx = row * 4 + col
                            if piece_idx < len(available_pieces):
                                selected_piece = available_pieces[piece_idx]
                                flag = "place_piece"

            elif flag == "place_piece":
                if turn == 1:
                    # P2 (사용자)가 보드에 조각을 놓음
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if y <= WIDTH:
                            row = y // SQUARE_SIZE
                            col = x // SQUARE_SIZE
                            if available_square(row, col):
                                board[row][col] = pieces.index(selected_piece) + 1
                                available_pieces.remove(selected_piece)
                                selected_piece = None
                                if check_win():
                                    game_over = True
                                    winner = 2
                                elif is_board_full():
                                    game_over = True
                                    winner = None
                                else:
                                    flag = "select_piece"
                                    turn = 2  # 다음 턴에 P2가 선택

                else:
                    # P1 (AI)가 선택된 조각을 놓음
                    player = players[1](board=board, available_pieces=available_pieces)
                    (board_row, board_col) = player.place_piece(selected_piece)
                    if available_square(board_row, board_col):
                        board[board_row][board_col] = pieces.index(selected_piece) + 1
                        available_pieces.remove(selected_piece)
                        selected_piece = None
                        if check_win():
                            game_over = True
                            winner = 1
                        elif is_board_full():
                            game_over = True
                            winner = None
                        else:
                            flag = "select_piece"
                            turn = 1  # 다음 턴에 P1이 선택


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # 게임 재시작
                game_over = False
                turn = 1
                flag = "select_piece"
                board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
                available_pieces = pieces[:]
                selected_piece = None
                draw_available_pieces()
        
        screen.fill(BLACK)
        draw_lines()
        draw_pieces()
        draw_available_pieces()

        if game_over:
            if winner:
                display_message(f"Player {winner} Wins!", GREEN)
            else:
                display_message("Draw!", GRAY)
        elif flag == "select_piece":
            display_message(f"P{3-turn} selecting piece")
        else:
            display_message(f"P{turn} placing piece")

        pygame.display.update()

