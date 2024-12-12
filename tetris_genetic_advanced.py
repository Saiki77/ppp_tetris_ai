import random
import copy
import sys
import argparse
import json
import tempfile
import subprocess
import os
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from statistics import variance

# Module-level constants for board dimensions
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Configuration Data Class
@dataclass
class Config:
    pop_size: int = 100
    generations: int = 500
    initial_mutation_rate: float = 0.05
    min_mutation_rate: float = 0.01
    episodes_per_evaluation: int = 10
    elite_size: int = 5
    max_pieces: int = None  # For Phase 4: Full game
    tournament_size: int = 3  # For Tournament Selection
    crossover_points: int = 2  # Number of crossover points
    hill_climb_iterations: int = 5  # Number of hill-climb steps per elite

    TETRIS_PIECES: dict = field(default_factory=lambda: {
        'I': [
            [(0,0),(1,0),(2,0),(3,0)],
            [(0,0),(0,1),(0,2),(0,3)]
        ],
        'O': [
            [(0,0),(1,0),(0,1),(1,1)]
        ],
        'T': [
            [(0,0),(1,0),(2,0),(1,1)],
            [(0,0),(0,1),(0,2),(1,1)],
            [(0,1),(1,1),(2,1),(1,0)],
            [(1,0),(1,1),(1,2),(0,1)]
        ],
        'S': [
            [(0,1),(1,1),(1,0),(2,0)],
            [(1,0),(1,1),(0,1),(0,2)]
        ],
        'Z': [
            [(0,0),(1,0),(1,1),(2,1)],
            [(1,1),(1,0),(0,0),(0,-1)]
        ],
        'J': [
            [(0,0),(0,1),(1,1),(2,1)],
            [(0,0),(1,0),(0,1),(0,2)],
            [(0,0),(1,0),(2,0),(2,-1)],
            [(1,0),(1,1),(1,2),(0,0)]
        ],
        'L': [
            [(2,0),(2,1),(1,1),(0,1)],
            [(0,0),(0,1),(0,2),(1,0)],
            [(0,0),(1,0),(2,0),(0,1)],
            [(0,2),(1,2),(1,1),(1,0)]
        ]
    })

    PIECE_TYPES: list = field(default_factory=lambda: ['I', 'O', 'T', 'S', 'Z', 'J', 'L'])

def get_config(args):
    """
    Returns the configuration for the Tetris AI.
    Modify parameters here to adjust AI behavior.
    """
    return Config(
        pop_size=args.pop_size,
        generations=args.generations,
        initial_mutation_rate=args.initial_mutation_rate,
        min_mutation_rate=args.min_mutation_rate,
        episodes_per_evaluation=args.episodes_per_evaluation,
        elite_size=args.elite_size,
        max_pieces=args.max_pieces,
        tournament_size=args.tournament_size,
        crossover_points=args.crossover_points,
        hill_climb_iterations=args.hill_climb_iterations
    )

# Tetris Environment
class TetrisEnv:
    def __init__(self, tetris_pieces, max_pieces=None):
        self.board = self.create_board()
        self.max_pieces = max_pieces
        self.done = False
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.last_piece = None  # To store last placed piece info
        self.tetris_pieces = tetris_pieces
        self.spawn_new_piece()

    def create_board(self):
        return [[0]*BOARD_WIDTH for _ in range(BOARD_HEIGHT)]

    def copy_board(self, board):
        return [row[:] for row in board]

    def is_valid_position(self, board, piece_coords):
        for (x, y) in piece_coords:
            if x < 0 or x >= BOARD_WIDTH:
                return False
            if y < 0 or y >= BOARD_HEIGHT:
                return False
            if board[y][x] != 0:
                return False
        return True

    def place_piece(self, board, piece_coords):
        for (x, y) in piece_coords:
            board[y][x] = 1

    def clear_lines(self, board):
        cleared = 0
        new_board = []
        for row in board:
            if all(cell == 1 for cell in row):
                cleared += 1
            else:
                new_board.append(row)
        while len(new_board) < BOARD_HEIGHT:
            new_board.insert(0, [0]*BOARD_WIDTH)
        return new_board, cleared

    def spawn_new_piece(self):
        ptype = random.choice(self.tetris_pieces['PIECE_TYPES'])
        rotations = self.tetris_pieces[ptype]
        rotation_idx = random.randint(0, len(rotations)-1)
        self.ptype = ptype
        self.rot_idx = rotation_idx
        self.piece = rotations[rotation_idx]

        # Place at top center
        min_x = min(x for x, y in self.piece)
        max_x = max(x for x, y in self.piece)
        w = max_x - min_x + 1
        start_x = BOARD_WIDTH//2 - w//2
        self.piece = self.translate_piece(self.piece, start_x, 0)

        # Ensure all y-coordinates are non-negative
        min_y = min(y for x, y in self.piece)
        if min_y < 0:
            self.piece = [(x, y - min_y) for (x, y) in self.piece]

        if not self.is_valid_position(self.board, self.piece):
            self.done = True
        else:
            # Store last piece info before placing
            self.last_piece = {
                'type': self.ptype,
                'rotation': self.rot_idx,
                'coordinates': self.piece
            }

    def step(self, coords):
        if self.done:
            return
        self.place_piece(self.board, coords)
        self.pieces_placed += 1
        self.board, cleared = self.clear_lines(self.board)
        self.lines_cleared += cleared

        if self.max_pieces is not None and self.pieces_placed >= self.max_pieces:
            # Game ends after reaching max pieces
            self.done = True
            return

        self.spawn_new_piece()

    def clone(self):
        env_copy = TetrisEnv(tetris_pieces=self.tetris_pieces, max_pieces=self.max_pieces)
        env_copy.board = self.copy_board(self.board)
        env_copy.done = self.done
        env_copy.lines_cleared = self.lines_cleared
        env_copy.pieces_placed = self.pieces_placed
        env_copy.ptype = self.ptype
        env_copy.rot_idx = self.rot_idx
        env_copy.piece = self.piece[:]
        env_copy.last_piece = copy.deepcopy(self.last_piece)
        return env_copy

    @staticmethod
    def translate_piece(coords, dx, dy):
        return [(x+dx, y+dy) for (x, y) in coords]

# Utility Functions
def get_all_moves(env: TetrisEnv):
    moves = []
    ptype = env.ptype
    rotations = env.tetris_pieces[ptype]

    for r in range(len(rotations)):
        test_piece = rotations[r]
        minx = min(x for x, y in test_piece)
        maxx = max(x for x, y in test_piece)
        width = maxx - minx + 1

        for col in range(BOARD_WIDTH - width + 1):
            shifted_piece = [(x - minx + col, y) for (x, y) in test_piece]
            final_coords = simulate_drop(env.board, shifted_piece)
            if env.is_valid_position(env.board, final_coords):
                moves.append((r, final_coords))
    return moves

def simulate_drop(board, piece_coords):
    while True:
        new_coords = [(x, y+1) for (x, y) in piece_coords]
        if not is_valid_position(board, new_coords):
            break
        piece_coords = new_coords
    return piece_coords

def is_valid_position(board, piece_coords):
    for (x, y) in piece_coords:
        if x < 0 or x >= BOARD_WIDTH:
            return False
        if y < 0 or y >= BOARD_HEIGHT:
            return False
        if board[y][x] != 0:
            return False
    return True

def get_aggregate_height(board):
    heights = []
    for x in range(BOARD_WIDTH):
        h = 0
        for y in range(BOARD_HEIGHT):
            if board[y][x] != 0:
                h = BOARD_HEIGHT - y
                break
        heights.append(h)
    return sum(heights)

def get_absolute_height(board):
    max_height = 0
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x] != 0:
                height = BOARD_HEIGHT - y
                if height > max_height:
                    max_height = height
                break
    return max_height

def get_number_of_holes(board):
    holes = 0
    for x in range(BOARD_WIDTH):
        block_found = False
        for y in range(BOARD_HEIGHT):
            if board[y][x] != 0:
                block_found = True
            elif block_found and board[y][x] == 0:
                holes += 1
    return holes

def get_bumpiness(board):
    heights = []
    for x in range(BOARD_WIDTH):
        h = 0
        for y in range(BOARD_HEIGHT):
            if board[y][x] != 0:
                h = BOARD_HEIGHT - y
                break
        heights.append(h)
    bumpiness = 0
    for i in range(len(heights)-1):
        bumpiness += abs(heights[i] - heights[i+1])
    return bumpiness

def get_four_deep_wells(board):
    wells = 0
    for x in range(BOARD_WIDTH):
        # A well is defined as a column where both adjacent columns are higher
        if x == 0 or x == BOARD_WIDTH -1:
            continue  # Cannot have wells on the edges
        left_height = 0
        for y in range(BOARD_HEIGHT):
            if board[y][x-1] != 0:
                left_height = BOARD_HEIGHT - y
                break
        right_height = 0
        for y in range(BOARD_HEIGHT):
            if board[y][x+1] != 0:
                right_height = BOARD_HEIGHT - y
                break
        column_height = 0
        for y in range(BOARD_HEIGHT):
            if board[y][x] != 0:
                column_height = BOARD_HEIGHT - y
                break
        # A well is where both adjacent heights are greater than the current
        if left_height > column_height and right_height > column_height:
            well_depth = min(left_height, right_height) - column_height
            if well_depth >=4:
                wells +=1
    return wells

def evaluate_board(board, lines_cleared_in_move, weights):
    """
    Calculates the fitness score based on the board state and weights.

    Args:
        board (list of lists): Current board state.
        lines_cleared_in_move (int): Number of lines cleared in the last move.
        weights (list of floats): Weight parameters.

    Returns:
        float: Fitness score.
    """
    # Unpack weights (6 parameters)
    a, b, c, d, e, f = weights
    agg_height = get_aggregate_height(board)
    holes = get_number_of_holes(board)
    bumpiness = get_bumpiness(board)
    abs_height = get_absolute_height(board)
    four_deep_wells = get_four_deep_wells(board)

    # Fitness calculation
    score = (a * lines_cleared_in_move) - (b * agg_height) - (c * holes) - (d * bumpiness) - (e * abs_height) - (f * four_deep_wells)
    return score

def play_game(weights, tetris_pieces, episodes, phase, max_pieces, num_weights):
    """
    Plays a number of Tetris games and returns the average lines cleared.

    Args:
        weights (list of floats): Weight parameters.
        tetris_pieces (dict): Tetris pieces configurations.
        episodes (int): Number of games to play.
        phase (int): Game phase (only Phase 4 is supported here).
        max_pieces (int or None): Maximum number of pieces.
        num_weights (int): Number of weights (should be 6).

    Returns:
        float: Average lines cleared across episodes.
    """
    total_lines = 0
    for _ in range(episodes):
        env = TetrisEnv(tetris_pieces=tetris_pieces, max_pieces=max_pieces)
        while not env.done:
            moves = get_all_moves(env)
            if not moves:
                env.done = True
                break
            best_score = -1e9
            best_coords = None
            for (r, coords) in moves:
                sim_env = env.clone()
                sim_env.place_piece(sim_env.board, coords)
                sim_env.board, cleared = sim_env.clear_lines(sim_env.board)
                score = evaluate_board(sim_env.board, cleared, weights)
                if score > best_score:
                    best_score = score
                    best_coords = coords
            env.step(best_coords)
        total_lines += env.lines_cleared
    return total_lines / episodes

def print_board(board):
    """
    Prints the current state of the board.

    Args:
        board (list of lists): Current board state.
    """
    for row in board:
        print(''.join(['X' if x != 0 else '.' for x in row]))
    print("-" * BOARD_WIDTH)

# Genetic Algorithm Functions with Elitism and Cross-Entropy-Based Mutation
def init_population(pop_size=100, num_weights=6):
    """
    Initializes the population with random weight vectors.

    Args:
        pop_size (int): Number of individuals in the population.
        num_weights (int): Number of weights per individual (should be 6).

    Returns:
        list of lists: Population of weight vectors.
    """
    return [[random.uniform(0,5) for _ in range(num_weights)] for _ in range(pop_size)]

def select_elites(pop, fitnesses, elite_size):
    """
    Selects the top-performing individuals as elites.

    Args:
        pop (list of lists): Current population.
        fitnesses (list of floats): Fitness scores of the population.
        elite_size (int): Number of elites to select.

    Returns:
        list of lists: Elite individuals.
    """
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [pop[i] for i in sorted_indices[:elite_size]]
    return elites

def tournament_selection(pop, fitnesses, tournament_size, num_parents):
    """
    Selects parents using tournament selection.

    Args:
        pop (list of lists): Current population.
        fitnesses (list of floats): Fitness scores.
        tournament_size (int): Number of individuals in each tournament.
        num_parents (int): Number of parents to select.

    Returns:
        list of lists: Selected parents.
    """
    parents = []
    pop_size = len(pop)
    for _ in range(num_parents):
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(pop_size), tournament_size)
        tournament = [pop[i] for i in tournament_indices]
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        # Select the best among them
        winner = tournament[tournament_fitnesses.index(max(tournament_fitnesses))]
        parents.append(copy.deepcopy(winner))
    return parents

def multi_point_crossover(p1, p2, crossover_points=2):
    """
    Performs multi-point crossover between two parents.

    Args:
        p1 (list of floats): Parent 1.
        p2 (list of floats): Parent 2.
        crossover_points (int): Number of crossover points.

    Returns:
        list of floats: Child individual.
    """
    if crossover_points >= len(p1):
        crossover_points = len(p1) - 1
    points = sorted(random.sample(range(1, len(p1)), crossover_points))
    child = []
    last = 0
    for i, point in enumerate(points):
        if i % 2 == 0:
            child += p1[last:point]
        else:
            child += p2[last:point]
        last = point
    # Add the remaining part
    if len(p1) > last:
        if len(points) % 2 == 0:
            child += p1[last:]
        else:
            child += p2[last:]
    return child

def inversion_mutation(chrom, mutation_rate=0.05):
    """
    Performs inversion mutation on a chromosome.

    Args:
        chrom (list of floats): Chromosome to mutate.
        mutation_rate (float): Probability of mutation per gene.

    Returns:
        list of floats: Mutated chromosome.
    """
    if random.random() < mutation_rate:
        if len(chrom) < 2:
            return chrom
        start = random.randint(0, len(chrom)-2)
        end = random.randint(start+1, len(chrom)-1)
        chrom[start:end+1] = reversed(chrom[start:end+1])
    return chrom

def mutate(chrom, mutation_rate=0.05):
    """
    Mutates a chromosome based on the mutation rate.

    Args:
        chrom (list of floats): Chromosome to mutate.
        mutation_rate (float): Mutation rate.

    Returns:
        list of floats: Mutated chromosome.
    """
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            chrom[i] += random.uniform(-0.5, 0.5)  # Increased mutation range
            chrom[i] = max(chrom[i], 0.0)  # Ensure non-negative
    # Apply inversion mutation
    chrom = inversion_mutation(chrom, mutation_rate)
    return chrom

def hill_climb(individual, tetris_pieces, episodes, phase, max_pieces, num_weights, iterations=5):
    """
    Performs hill-climbing optimization on an individual.

    Args:
        individual (list of floats): Individual to optimize.
        tetris_pieces (dict): Tetris pieces configurations.
        episodes (int): Number of episodes per evaluation.
        phase (int): Game phase.
        max_pieces (int or None): Maximum number of pieces.
        num_weights (int): Number of weights.
        iterations (int): Number of hill-climbing steps.

    Returns:
        list of floats: Optimized individual.
    """
    best_individual = copy.deepcopy(individual)
    best_fitness = play_game(best_individual, tetris_pieces, episodes, phase, max_pieces, num_weights)
    for _ in range(iterations):
        # Create a mutated copy
        mutated = copy.deepcopy(best_individual)
        mutated = mutate(mutated, mutation_rate=0.1)  # Higher mutation rate for hill-climbing
        fitness = play_game(mutated, tetris_pieces, episodes, phase, max_pieces, num_weights)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = mutated
    return best_individual

def breed_elitism_parents(parents, pop_size=100, crossover_points=2, mutation_rate=0.05, elite_size=5):
    """
    Breeds the next generation using elitism and advanced genetic operators.

    Args:
        parents (list of lists): Selected parents.
        pop_size (int): Population size.
        crossover_points (int): Number of crossover points.
        mutation_rate (float): Current mutation rate.
        elite_size (int): Number of elites.

    Returns:
        list of lists: Next generation population.
    """
    next_pop = []
    num_new = pop_size - elite_size
    for _ in range(num_new):
        p1, p2 = random.sample(parents, 2)
        child = multi_point_crossover(p1, p2, crossover_points)
        child = mutate(child, mutation_rate)
        next_pop.append(child)
    # Combine elites and new offspring
    return parents[:elite_size] + next_pop

def select_parents(pop, fitnesses, config: Config):
    """
    Selects parents using tournament selection.

    Args:
        pop (list of lists): Current population.
        fitnesses (list of floats): Fitness scores.
        config (Config): Configuration object.

    Returns:
        list of lists: Selected parents.
    """
    return tournament_selection(pop, fitnesses, config.tournament_size, config.pop_size)

def show_best_individual(weights, phase, game_id=1):
    """
    Shows one complete game final state for the best individual.

    Args:
        weights (list of floats): Weight parameters.
        phase (int): Game phase.
        game_id (int): Identifier for the game instance.
    """
    # Initialize environment
    tetris_pieces = {
        'PIECE_TYPES': ['I', 'O', 'T', 'S', 'Z', 'J', 'L'],
        **{k: v for k, v in Config().TETRIS_PIECES.items()}
    }
    env = TetrisEnv(tetris_pieces=tetris_pieces, max_pieces=None)
    while not env.done:
        moves = get_all_moves(env)
        if not moves:
            env.done = True
            break
        best_score = -1e9
        best_coords = None
        last_piece_info = None
        for (r, coords) in moves:
            sim_env = env.clone()
            sim_env.place_piece(sim_env.board, coords)
            sim_env.board, cleared = sim_env.clear_lines(sim_env.board)
            score = evaluate_board(sim_env.board, cleared, weights)
            if score > best_score:
                best_score = score
                best_coords = coords
                # Capture last piece info
                last_piece_info = {
                    'type': sim_env.last_piece['type'],
                    'rotation': sim_env.last_piece['rotation'],
                    'coordinates': sim_env.last_piece['coordinates']
                }
        env.step(best_coords)
    # After game ends, display final state
    print(f"--- Showing Final State of Game {game_id} for Phase {phase} ---")
    print_board(env.board)
    print(f"Lines Cleared: {env.lines_cleared}")
    print(f"Pieces Placed: {env.pieces_placed}")
    if env.last_piece:
        print(f"Last Placed Piece: Type={env.last_piece['type']}, Rotation={env.last_piece['rotation']}, Coordinates={env.last_piece['coordinates']}\n")
    print(f"--- End of Game {game_id} ---\n")

def show_best_individuals(weights, phase, num_games=1):
    """
    Shows multiple complete game final states for the best individual.

    Args:
        weights (list of floats): Weight parameters.
        phase (int): Game phase.
        num_games (int): Number of games to display.
    """
    for game_id in range(1, num_games + 1):
        show_best_individual(weights, phase, game_id)

# Top-Level Evaluation Function for Multiprocessing
def eval_individual(args):
    """
    Evaluates an individual in the population.
    Must be a top-level function for pickling.

    Args:
        args (tuple): A tuple containing (weights, tetris_pieces, episodes, phase, max_pieces, num_weights)

    Returns:
        float: Fitness score of the individual
    """
    weights, tetris_pieces, episodes, phase, max_pieces, num_weights = args
    return play_game(weights, tetris_pieces, episodes, phase, max_pieces, num_weights)

# Parallel Evaluation Function
def evaluate_population(population, phase, episodes, max_pieces, num_weights):
    """
    Evaluates the entire population in parallel.

    Args:
        population (list of lists): Current population.
        phase (int): Game phase.
        episodes (int): Number of episodes per evaluation.
        max_pieces (int or None): Maximum number of pieces.
        num_weights (int): Number of weights per individual.

    Returns:
        list of floats: Fitness scores.
    """
    config = Config()
    tetris_pieces = {
        'PIECE_TYPES': ['I', 'O', 'T', 'S', 'Z', 'J', 'L'],
        **{k: v for k, v in config.TETRIS_PIECES.items()}
    }
    args = [(weights, tetris_pieces, episodes, phase, max_pieces, num_weights) for weights in population]
    with ProcessPoolExecutor() as executor:
        # Map returns results in the order of the input
        fitnesses = list(executor.map(eval_individual, args))
    return fitnesses

# Function to spawn a new terminal window for visualization
def spawn_visualization(weights, phase, num_weights):
    """
    Spawns a new terminal window and runs the visualization game.

    Args:
        weights (list of floats): Weight parameters.
        phase (int): Game phase.
        num_weights (int): Number of weights (should be 6).
    """
    # Serialize weights to a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpfile:
        json.dump(weights, tmpfile)
        tmpfile_path = tmpfile.name

    # Determine the current script path
    script_path = os.path.abspath(sys.argv[0])

    # Build the command to run
    command = [
        'python3', script_path,
        '--visualize',
        '--weights_file', tmpfile_path,
        '--phase', str(phase)
    ]

    # Detect the operating system
    if os.name == 'nt':
        # For Windows, use 'start' with 'cmd /c' to run and close
        cmd_command = ' '.join(command)
        subprocess.Popen(['start', 'cmd', '/c', cmd_command], shell=True)
    elif sys.platform == 'darwin':
        # For macOS, use 'osascript' to open Terminal and run the command
        osa_command = f'''
        tell application "Terminal"
            activate
            do script "{" ".join(command)}"
        end tell
        '''
        subprocess.Popen(['osascript', '-e', osa_command])
    else:
        # For Linux, try 'gnome-terminal' or 'xterm' without 'hold'
        if shutil.which('gnome-terminal'):
            subprocess.Popen(['gnome-terminal', '--'] + command)
        elif shutil.which('xterm'):
            subprocess.Popen(['xterm', '-e'] + command)
        else:
            print("No compatible terminal emulator found. Cannot spawn visualization window.")

# Visualization Mode Handler
def handle_visualization(weights_file, phase):
    """
    Handles the visualization mode by reading weights and running a single game.

    Args:
        weights_file (str): Path to the weights JSON file.
        phase (int): Game phase.
    """
    # Load weights from the JSON file
    try:
        with open(weights_file, 'r') as f:
            weights = json.load(f)
    except Exception as e:
        print(f"Error loading weights file: {e}")
        sys.exit(1)

    # Ensure weights length is 6
    if len(weights) != 6:
        print(f"Error: Expected 6 weights, but got {len(weights)}.")
        sys.exit(1)

    # Run the visualization game
    visualize_game(weights, phase)

    # Optionally delete the temporary weights file
    try:
        os.remove(weights_file)
    except Exception as e:
        print(f"Error deleting temporary weights file: {e}")

# Main Genetic Algorithm Loop
def main():
    parser = argparse.ArgumentParser(description="Tetris AI with Genetic Algorithm")
    parser.add_argument('--visualize', action='store_true', help="Run in visualization mode")
    parser.add_argument('--weights_file', type=str, help="Path to weights JSON file for visualization")
    parser.add_argument('--phase', type=int, help="Phase number for visualization")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size")
    parser.add_argument('--generations', type=int, default=500, help="Number of generations")
    parser.add_argument('--initial_mutation_rate', type=float, default=0.05, help="Initial mutation rate")
    parser.add_argument('--min_mutation_rate', type=float, default=0.01, help="Minimum mutation rate")
    parser.add_argument('--episodes_per_evaluation', type=int, default=10, help="Episodes per evaluation")
    parser.add_argument('--elite_size', type=int, default=5, help="Number of elites to retain")
    parser.add_argument('--max_pieces', type=int, default=None, help="Maximum number of pieces (None for Phase 4)")
    parser.add_argument('--tournament_size', type=int, default=3, help="Tournament size for selection")
    parser.add_argument('--crossover_points', type=int, default=2, help="Number of crossover points")
    parser.add_argument('--hill_climb_iterations', type=int, default=5, help="Number of hill-climbing iterations per elite")
    
    args = parser.parse_args()

    # Initialize configuration
    config = get_config(args)

    if args.visualize:
        if not args.weights_file or not args.phase:
            print("Visualization mode requires --weights_file and --phase arguments.")
            sys.exit(1)
        # Ensure only Phase 4 is visualized
        if args.phase != 4:
            print("Visualization is only supported for Phase 4.")
            sys.exit(1)
        handle_visualization(args.weights_file, args.phase)
        sys.exit(0)

    # Training Mode
    population = init_population(pop_size=config.pop_size, num_weights=6)

    # To track generations for adaptive mutation
    best_fitness_history = []
    no_improvement_generations = 0
    fitness_threshold = 1e-3  # Threshold to consider as improvement
    mutation_rate = config.initial_mutation_rate

    visual_spawn_interval = 50  # Adjusted for Phase 4 training

    for gen in range(1, config.generations + 1):
        fitnesses = evaluate_population(population, 4, config.episodes_per_evaluation, config.max_pieces, 6)
        best_f = max(fitnesses)
        avg_f = sum(fitnesses) / len(fitnesses)
        print(f"Gen {gen}: Best Fitness={best_f}, Avg Fitness={avg_f}")

        # Check for fitness improvement
        if best_f_history := best_fitness_history:
            if best_f - max(best_fitness_history) > fitness_threshold:
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1
        best_fitness_history.append(best_f)

        # Calculate population diversity (fitness variance)
        if len(fitnesses) > 1:
            fitness_var = variance(fitnesses)
        else:
            fitness_var = 0.0

        # Adaptive Mutation Rate Logic
        diversity_threshold = 10.0  # Example threshold for variance
        stagnation_threshold = 20  # Number of generations without improvement

        if fitness_var < diversity_threshold or no_improvement_generations >= stagnation_threshold:
            # Increase mutation rate to promote exploration
            mutation_rate = min(mutation_rate * 1.1, 1.0)  # Cap at 1.0
            print(f"Adaptive Mutation Rate Increased to: {mutation_rate:.4f}")
            no_improvement_generations = 0  # Reset stagnation counter
        else:
            # Gradually decrease mutation rate towards min_mutation_rate
            mutation_rate = max(mutation_rate * 0.99, config.min_mutation_rate)
            print(f"Adaptive Mutation Rate Decreased to: {mutation_rate:.4f}")

        # Select elites
        elites = select_elites(population, fitnesses, config.elite_size)

        # Select parents using tournament selection
        parents = select_parents(population, fitnesses, config)

        # Breed next generation with advanced genetic operators
        population = breed_elitism_parents(
            parents,
            pop_size=config.pop_size,
            crossover_points=config.crossover_points,
            mutation_rate=mutation_rate,
            elite_size=config.elite_size
        )

        # Perform hill-climbing on elites
        optimized_elites = []
        for elite in elites:
            optimized = hill_climb(
                elite,
                tetris_pieces={
                    'PIECE_TYPES': ['I', 'O', 'T', 'S', 'Z', 'J', 'L'],
                    **{k: v for k, v in config.TETRIS_PIECES.items()}
                },
                episodes=config.episodes_per_evaluation,
                phase=4,
                max_pieces=config.max_pieces,
                num_weights=6,
                iterations=config.hill_climb_iterations
            )
            optimized_elites.append(optimized)

        # Replace elites in the population with optimized elites
        population = optimized_elites + population[config.elite_size:]

        # Display the best individual every 'visual_spawn_interval' generations
        if gen % visual_spawn_interval == 0:
            best_idx = fitnesses.index(best_f)
            best_weights = population[best_idx]
            print("---- Best Individual (Final State) ----")
            show_best_individuals(best_weights, 4, num_games=1)
            print("Best Weights:", best_weights)
            print("---------------------------------------\n")
            spawn_visualization(best_weights, 4, 6)

    # Final Evaluation
    print("=== Training Completed ===")
    fitnesses = evaluate_population(population, 4, config.episodes_per_evaluation, config.max_pieces, 6)
    best_f = max(fitnesses)
    avg_f = sum(fitnesses) / len(fitnesses)
    best_idx = fitnesses.index(best_f)
    best_weights = population[best_idx]
    print(f"Final Best Fitness: {best_f}")
    print("Final Best Weights:", best_weights)
    print("---- Final Best Individual (Final State) ----")
    show_best_individuals(best_weights, 4, num_games=1)
    print("----------------------------------------------\n")

if __name__ == '__main__':
    main()
