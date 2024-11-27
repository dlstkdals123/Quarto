import numpy as np
board = [
    [11, 4, 1, 0],
    [0, 15, 5, 2],
    [9, 0, 14, 3],
    [0, 0, 0, 0]
]

pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

def check_win(board):
    def check_line(line):
        if 0 in line:
            return False
        characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):
            if len(set(characteristics[:, i])) == 1:
                return True
        return False

    def check_2x2_subgrid_win():
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in subgrid:
                    characteristics = [pieces[idx - 1] for idx in subgrid]
                    for i in range(4):
                        if len(set(char[i] for char in characteristics)) == 1:
                            return True
        return False

    for col in range(4):
        if check_line([board[row][col] for row in range(4)]):
            return True

    for row in range(4):
        if check_line([board[row][col] for col in range(4)]):
            return True

    if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3 - i] for i in range(4)]):
        return True

    if check_2x2_subgrid_win():
        return True

    return False

for row in range(4):
    for col in range(4):
        if board[row][col] == 0:
            board[row][col] = 7
            print(f"({row}, {col}) : {check_win(board)}")
            board[row][col] = 0