import random
import numpy as np
from deap import base, creator, tools, algorithms

BOARD_ROWS = 4
BOARD_COLS = 4

class P2:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board  # Current board state. 0: empty, 1~16: pieces
        self.available_pieces = available_pieces  # Currently available pieces as a list of tuples

    def fitness(self, individual):
        """Evaluate the fitness of a move based on the current board state."""
        position = individual[0]  # Position to place the piece
        selected_piece = individual[1]  # Piece to be placed
        x, y = position

        # Penalize invalid positions
        if self.board[x][y] != 0:
            return -100

        # Score for winning move
        if self.check_win(selected_piece, x, y):
            return 100

        # Score for blocking opponent's potential win
        if self.block_opponent(selected_piece, x, y):
            return 50

        # Additional heuristic scoring
        score = 0
        if (x, y) == (1, 1) or (x, y) == (2, 2):  # Prefer central positions
            score += 10
        score += self.evaluate_piece(selected_piece)
        return score

    def genetic_algorithm(self, mode, selected_piece=None):
        """
        Run genetic algorithm to find the best move.
        - mode == "place": Find the best position for a given piece
        - mode == "select": Select the best piece for the opponent
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        if mode == "place":
            toolbox.register("attr_position", lambda: (random.randint(0, 3), random.randint(0, 3)))
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_position, lambda: selected_piece), n=1)
        elif mode == "select":
            toolbox.register("attr_piece", lambda: random.choice(self.available_pieces))
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (lambda: (0, 0), toolbox.attr_piece), n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run GA
        population = toolbox.population(n=50)
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=False)

        # Return the best individual
        best_individual = tools.selBest(population, k=1)[0]
        return best_individual

    def place_piece(self, selected_piece):
        """Find the best position to place the given piece."""
        best_move = self.genetic_algorithm(mode="place", selected_piece=selected_piece)
        return best_move[0]  # Return the selected position

    def select_piece(self):
        """Select the best piece for the opponent to place."""
        best_move = self.genetic_algorithm(mode="select")
        return best_move[1]  # Return the selected piece

    def check_win(self, piece, x, y):
        """
        Check if placing the given piece at (x, y) results in a win.
        This includes row, column, diagonal, and 2x2 square checks.
        """
        # Place the piece temporarily
        original_value = self.board[x][y]
        self.board[x][y] = piece

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
        size = len(self.board)
        row = [self.board[x][j] for j in range(size)]
        col = [self.board[i][y] for i in range(size)]
        diag1 = [self.board[i][i] for i in range(size)] if x == y else []  # Main diagonal
        diag2 = [self.board[i][size - 1 - i] for i in range(size)] if x + y == size - 1 else []

        # Check 2x2 squares
        squares = []
        for i in range(max(0, x - 1), min(size - 1, x) + 1):
            for j in range(max(0, y - 1), min(size - 1, y) + 1):
                square = [
                    self.board[i][j],
                    self.board[i][j + 1],
                    self.board[i + 1][j],
                    self.board[i + 1][j + 1]
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
            self.board[x][y] = original_value

        return is_win


    def block_opponent(self, piece, x, y):
        """Check if placing the piece blocks the opponent from winning."""
        # TODO: Implement logic to identify and block opponent's winning moves
        return False

    def evaluate_piece(self, piece):
        """Evaluate the strategic value of a piece."""
        # TODO: Add heuristics for piece evaluation
        return random.randint(0, 10)  # Temporary random score