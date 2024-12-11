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

# Configuration Data Class
@dataclass
class Config:
    BOARD_WIDTH: int = 10
    BOARD_HEIGHT: int = 20

    TETRIS_PIECES: dict = field(default_factory=lambda: {
        'I': [ [(0,0),(1,0),(2,0),(3,0)], #Pieces are already stored along their rotations. 
               [(0,0),(0,1),(0,2),(0,3)] ],
        'O': [ [(0,0),(1,0),(0,1),(1,1)] ],
        'T': [ [(0,0),(1,0),(2,0),(1,1)],
               [(0,0),(0,1),(0,2),(1,1)],
               [(0,1),(1,1),(2,1),(1,0)],
               [(1,0),(1,1),(1,2),(0,1)] ],
        'S': [ [(0,1),(1,1),(1,0),(2,0)],
               [(1,0),(1,1),(0,1),(0,2)] ],
        'Z': [ [(0,0),(1,0),(1,1),(2,1)],
               [(1,1),(1,0),(0,0),(0,-1)] ],
        'J': [ [(0,0),(0,1),(1,1),(2,1)],
               [(0,0),(1,0),(0,1),(0,2)],
               [(0,0),(1,0),(2,0),(2,-1)],
               [(1,0),(1,1),(1,2),(0,0)] ],
        'L': [ [(2,0),(2,1),(1,1),(0,1)],
               [(0,0),(0,1),(0,2),(1,0)],
               [(0,0),(1,0),(2,0),(0,1)],
               [(0,2),(1,2),(1,1),(1,0)] ]
    })

    PIECE_TYPES: list = field(default_factory=lambda: list([
        'I', 'O', 'T', 'S', 'Z', 'J', 'L'
    ]))

    POP_SIZE: int = 100
    GENERATIONS: int = 500
    MUTATION_RATE: float = 0.05
    EPISODES_PER_EVALUATION: int = 10 # Each play plays 10 times to remove randomnes due to piece difference 
    # Elitism Configuration
    ELITE_SIZE: int = 5  # Number of top individuals to carry over unchanged
    MAX_PIECES: int = None  # For Phase 4: Full game

def get_config():
    """
    Returns the configuration for the Tetris AI.
    """
    return Config()

# Initialize Configuration
CONFIG = get_config()

# Top-Level Evaluation Function for Multiprocessing
def eval_individual(args):
    """
    Evaluates an individual in the population.
    Args:
        args (tuple): A tuple containing (weights, phase), phase is obsolete in this implementation 
    Returns:
        float: Fitness score of the individual
    """
    weights, phase = args
    return play_game(weights, episodes=CONFIG.EPISODES_PER_EVALUATION, phase=phase)

# Tetris Environment
class TetrisEnv:
    def __init__(self, max_pieces=None, phase=4):
        self.phase = phase
        self.board = self.create_board()
        self.max_pieces = max_pieces
        self.done = False
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.last_piece = None  # To store last placed piece info
        self.setup_board_for_phase(self.phase)
        self.spawn_new_piece()

    def create_board(self):
        return [[0]*CONFIG.BOARD_WIDTH for _ in range(CONFIG.BOARD_HEIGHT)]

    def copy_board(self, board):
        return [row[:] for row in board]

    def is_valid_position(self, board, piece_coords):
        for (x, y) in piece_coords:
            if x < 0 or x >= CONFIG.BOARD_WIDTH:
                return False
            if y < 0 or y >= CONFIG.BOARD_HEIGHT:
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
        while len(new_board) < CONFIG.BOARD_HEIGHT:
            new_board.insert(0, [0]*CONFIG.BOARD_WIDTH)
        return new_board, cleared

    def setup_board_for_phase(self, phase):
        self.board = self.create_board()
        # Phase 4: Empty board, full game
        pass

    def spawn_new_piece(self):
        ptype = random.choice(CONFIG.PIECE_TYPES)
        rotations = CONFIG.TETRIS_PIECES[ptype]
        rotation_idx = random.randint(0, len(rotations)-1)
        self.ptype = ptype
        self.rot_idx = rotation_idx
        self.piece = rotations[rotation_idx]

        # Place at top center
        min_x = min(x for x, y in self.piece)
        max_x = max(x for x, y in self.piece)
        w = max_x - min_x + 1
        start_x = CONFIG.BOARD_WIDTH//2 - w//2
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
        self.spawn_new_piece()

    def clone(self):
        env_copy = TetrisEnv(max_pieces=self.max_pieces, phase=self.phase)
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
def get_all_moves(env: TetrisEnv): # Get all possible moves for the current piece to evaluate 
    moves = []
    ptype = env.ptype
    rotations = CONFIG.TETRIS_PIECES[ptype]

    for r in range(len(rotations)):
        test_piece = rotations[r]
        minx = min(x for x, y in test_piece)
        maxx = max(x for x, y in test_piece)
        width = maxx - minx + 1

        for col in range(CONFIG.BOARD_WIDTH - width + 1):
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
        if x < 0 or x >= CONFIG.BOARD_WIDTH:
            return False
        if y < 0 or y >= CONFIG.BOARD_HEIGHT:
            return False
        if board[y][x] != 0:
            return False
    return True

def get_aggregate_height(board): # Get total hight of the stack 
    heights = []
    for x in range(CONFIG.BOARD_WIDTH):
        h = 0
        for y in range(CONFIG.BOARD_HEIGHT):
            if board[y][x] != 0:
                h = CONFIG.BOARD_HEIGHT - y
                break
        heights.append(h)
    return sum(heights)

def get_number_of_holes(board):
    holes = 0
    for x in range(CONFIG.BOARD_WIDTH):
        block_found = False
        for y in range(CONFIG.BOARD_HEIGHT):
            if board[y][x] != 0:
                block_found = True
            elif block_found and board[y][x] == 0:
                holes += 1
    return holes

def get_bumpiness(board): # calculates how many the stacks differe in height 
    heights = []
    for x in range(CONFIG.BOARD_WIDTH):
        h = 0
        for y in range(CONFIG.BOARD_HEIGHT):
            if board[y][x] != 0:
                h = CONFIG.BOARD_HEIGHT - y
                break
        heights.append(h)
    bumpiness = 0
    for i in range(len(heights)-1):
        bumpiness += abs(heights[i] - heights[i+1])
    return bumpiness

def evaluate_board(board, lines_cleared_in_move, weights):
    a, b, c, d = weights
    agg_height = get_aggregate_height(board)
    holes = get_number_of_holes(board)
    bumpiness = get_bumpiness(board)

    # Continuous loss function ? Maybe not right term 
    score = (a * lines_cleared_in_move) - (b * agg_height) - (c * holes) - (d * bumpiness)
    return score

def play_game(weights, episodes=5, phase=4):
    # Set max_pieces based on phase
    max_pieces = CONFIG.MAX_PIECES  # None for Phase 4

    total_lines = 0
    for _ in range(episodes):
        env = TetrisEnv(max_pieces=max_pieces, phase=phase)
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
    for row in board:
        print(''.join(['X' if x != 0 else '.' for x in row]))
    print("-" * CONFIG.BOARD_WIDTH)

# Genetic Algorithm Functions with Elitism 
def init_population(pop_size=100):
    return [[random.uniform(0,5) for _ in range(4)] for _ in range(pop_size)]

def select_elites(pop, fitnesses, elite_size):
    # Select top 'elite_size' individuals
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [pop[i] for i in sorted_indices[:elite_size]]
    return elites

def select_parents_cross_entropy(pop, fitnesses, elite_size, num_parents):
    # Select elites first
    elites = select_elites(pop, fitnesses, elite_size)
    # Compute mean and standartdeviation of elites for each gene
    mean = [0.0] * 4
    std = [0.0] * 4
    for gene_idx in range(4):
        gene_values = [individual[gene_idx] for individual in elites]
        mean[gene_idx] = sum(gene_values) / len(gene_values)
        variance = sum((x - mean[gene_idx])**2 for x in gene_values) / len(gene_values)
        std[gene_idx] = variance**0.5 if variance > 0 else 1.0
    # Sample new parents based on elites' distribution
    parents = []
    for _ in range(num_parents - elite_size):
        parent = []
        for gene_idx in range(4):
            gene = random.gauss(mean[gene_idx], std[gene_idx] * 0.1)  # 0.1 to control mutation intensity
            gene = max(gene, 0.0)  # Ensure non-negative
            parent.append(gene)
        parents.append(parent)
    # Combine elites and new parents
    return elites + parents

def crossover(p1, p2):
    c = random.randint(1, len(p1)-1)
    return p1[:c] + p2[c:]

def mutate(chrom, rate=0.1):
    for i in range(len(chrom)):
        if random.random() < rate:
            chrom[i] += random.uniform(-0.5, 0.5)  # Increased mutation range
            chrom[i] = max(chrom[i], 0.0)  # Ensure non-negative
    return chrom

def breed_elitism_cross_entropy(parents, pop_size=100, mutation_rate=0.05, elite_size=5):
    next_pop = []
    # Number of new individuals to generate
    num_new = pop_size - elite_size
    for _ in range(num_new):
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child, mutation_rate)
        next_pop.append(child)
    # Combine elites and new offspring
    return parents[:elite_size] + next_pop

# Display Functions
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def visualize_game(weights, phase):
    """
    Runs a single game with the given weights, printing the board after each piece placement.
    Displays the last placed piece, lines cleared, and pieces placed.
    Prompts the user for speed and post-game actions.
    """
    while True:
        # Prompt user for speed (pieces per second)
        while True:
            try:
                speed_input = input("Enter speed (pieces per second, e.g., 1): ").strip()
                speed = float(speed_input)
                if speed <= 0:
                    print("Speed must be a positive number. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value for speed.")

        delay = 1.0 / speed  # Delay in seconds between pieces

        env = TetrisEnv(max_pieces=CONFIG.MAX_PIECES, phase=phase)
        game_id = 1
        print(f"--- Visualization Game {game_id} for Phase {phase} ---")
        print_board(env.board)
        print(f"Pieces Placed: {env.pieces_placed}, Lines Cleared: {env.lines_cleared}")
        time.sleep(delay)

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

            # Clear terminal before printing
            clear_terminal()
            print(f"--- Visualization Game {game_id} for Phase {phase} ---")
            print_board(env.board)
            print(f"Pieces Placed: {env.pieces_placed}, Lines Cleared: {env.lines_cleared}")
            if last_piece_info:
                print(f"Last Placed Piece: Type={last_piece_info['type']}, Rotation={last_piece_info['rotation']}, Coordinates={last_piece_info['coordinates']}")
            time.sleep(delay)

        print(f"--- End of Visualization Game {game_id} for Phase {phase} ---\n")

        # Prompt user to run another game
        while True:
            retry_input = input("Agent has died. Do you want to run a new game with the current best agent? (y/n): ").strip().lower()
            if retry_input == 'y':
                game_id += 1
                break  # Restart the game loop
            elif retry_input == 'n':
                # Give user the option to save the agent
                save_input = input("Do you want to save this agent's weights for later use? (y/n): ").strip().lower()
                if save_input == 'y':
                    filename = input("Enter the custom name for the agent (without extension): ").strip()
                    filename = filename + ".json"
                    try:
                        with open(filename, 'w') as f:
                            json.dump(weights, f)
                        print(f"Agent saved as {filename}.")
                    except Exception as e:
                        print(f"Error saving agent: {e}")
                print("Exiting visualization.")
                sys.exit(0)
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

def show_best_individual(weights, phase, game_id=1):
    """
    Shows one complete game final state for the best individual.
    """
    env = TetrisEnv(max_pieces=CONFIG.MAX_PIECES, phase=phase)
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
    """
    for game_id in range(1, num_games + 1):
        show_best_individual(weights, phase, game_id)

# Parallel Evaluation Function
def evaluate_population(population, phase):
    """
    Evaluates the entire population in parallel.
    Returns a list of fitness scores.
    """
    # Prepare arguments as tuples of (weights, phase)
    args = [(weights, phase) for weights in population]
    
    with ProcessPoolExecutor() as executor:
        # Map returns results in the order of the input
        fitnesses = list(executor.map(eval_individual, args))
    return fitnesses

# Function to spawn a new terminal window for visualization
def spawn_visualization(weights, phase):
    """
    Spawns a new terminal window and runs the visualization game.
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
            do script "python3 \\"{script_path}\\" --visualize --weights_file \\"{tmpfile_path}\\" --phase {phase}"
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
    """
    # Load weights from the JSON file
    try:
        with open(weights_file, 'r') as f:
            weights = json.load(f)
    except Exception as e:
        print(f"Error loading weights file: {e}")
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

    args = parser.parse_args()

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
    population = init_population(pop_size=CONFIG.POP_SIZE)

    # To track generations for visualization spawning
    generations_since_last_visual = 0
    visual_spawn_interval = 2  

    for gen in range(1, CONFIG.GENERATIONS + 1):
        fitnesses = evaluate_population(population, 4)
        best_f = max(fitnesses)
        avg_f = sum(fitnesses) / len(fitnesses)
        print(f"Gen {gen}: Best Fitness={best_f}, Avg Fitness={avg_f}")

        # Select elites
        elites = select_elites(population, fitnesses, CONFIG.ELITE_SIZE)

        # Select parents using Cross-Entropy Method
        parents = select_parents_cross_entropy(population, fitnesses, CONFIG.ELITE_SIZE, CONFIG.POP_SIZE)

        # Breed next generation with Elitism and Cross-Entropy-Based Mutation
        population = breed_elitism_cross_entropy(parents, pop_size=CONFIG.POP_SIZE, mutation_rate=CONFIG.MUTATION_RATE, elite_size=CONFIG.ELITE_SIZE)

        # Display the best individual every 'visual_spawn_interval' generations
        if gen % visual_spawn_interval == 0:
            best_idx = fitnesses.index(best_f)
            best_weights = population[best_idx]
            print("---- Best Individual (Final State) ----")
            show_best_individuals(best_weights, 4, num_games=1)
            print("Best Weights:", best_weights)
            print("---------------------------------------\n")
            spawn_visualization(best_weights, 4)

    # Final Evaluation
    print("=== Training Completed ===")
    fitnesses = evaluate_population(population, 4)
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
