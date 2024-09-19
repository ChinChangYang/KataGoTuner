import datetime
import math
import subprocess
import time
from multiprocessing import Pool, cpu_count
from sgfmill import sgf
from scipy.stats import bernoulli
from typing import List, Tuple
from tqdm import tqdm  # Import tqdm for progress bar


def spawn_process(command: List[str]):
    """
    Execute a command and wait for it to complete.

    Args:
        command (List[str]): The command to execute.
    """
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with exit code {e.returncode}")
    except Exception as e:
        print(f"Error occurred while executing command '{' '.join(command)}': {str(e)}")


def get_command(parameters) -> List[str]:
    """
    Build a GTP command with specified parameters.

    Args:
        parameters (dict): A dictionary containing model and configuration details.

    Returns:
        List[str]: The constructed GTP command as a list of arguments.
    """
    command = [
        parameters["exe"],
        "gtp",
        "-config", parameters["config"],
        "-model", parameters["model"]
    ]
    override_options = [
        "maxVisits",
        "suppressVirtualLossExploreFactor",
        "suppressVirtualLossHindsight",
        "suppressVirtualLossLeakCatchUp",
        "numSearchThreads",
        "useNoisePruning",
        "maxTime",
        "cpuctExploration",
        "cpuctExplorationLog",
        "resignConsecTurns",
        "resignThreshold",
    ]

    for option in override_options:
        if option in parameters:
            command += ["-override-config", f"{option}={parameters[option]}"]

    return command


def match(
    black_parameters: dict,
    white_parameters: dict,
    gogui_classpath: str,
    game_count: int,
    sgffile_prefix: str = "match",
    verbose: bool = True,
) -> int:
    """
    Play a single game between two engines.

    Args:
        black_parameters (dict): Parameters for the black player.
        white_parameters (dict): Parameters for the white player.
        gogui_classpath (str): Path to the Gogui classpath.
        game_count (int): The game count for this match.
        sgffile_prefix (str, optional): Prefix for the SGF file. Defaults to "match".
        verbose (bool, optional): Whether to print game results. Defaults to True.

    Returns:
        int: 1 for black win, -1 for white win, 0 for draw.
    """
    black_command = get_command(black_parameters)
    white_command = get_command(white_parameters)
    board_size = 19
    komi = 7.5

    # Remove the '.sgf' extension from sgffile_base
    sgffile_base = f"{sgffile_prefix}-{game_count}-0"
    twogtp_command = [
        "java",
        "-cp",
        gogui_classpath,
        "net.sf.gogui.tools.twogtp.Main",
        "-black",
        " ".join(black_command),
        "-white",
        " ".join(white_command),
        "-size",
        str(board_size),
        "-komi",
        str(komi),
        "-auto",
        "-sgffile",
        sgffile_base,  # Pass without '.sgf'
    ]

    spawn_process(twogtp_command)

    # Construct the expected filename after twogtp appends '-0.sgf'
    sgffile = f"{sgffile_base}-0.sgf"
    is_won = 0

    try:
        with open(sgffile, "rb") as f:
            sgf_string = f.read()
            game = sgf.Sgf_game.from_bytes(sgf_string)
            winner = game.get_winner()

            if winner == "b":
                # Black won
                is_won = 1
            elif winner == "w":
                # White won
                is_won = -1
            else:
                # Draw
                is_won = 0
    except FileNotFoundError:
        print(f"SGF file {sgffile} not found.")
    except Exception as e:
        print(f"Error processing SGF file {sgffile}: {str(e)}")

    return is_won


def match_helper(args):
    """
    Helper function to unpack arguments and call match.

    Args:
        args (tuple): A tuple containing all arguments for the match function.

    Returns:
        int: Result of the match.
    """
    return match(*args)


def match_games(
    black_parameters: dict,
    white_parameters: dict,
    gogui_classpath: str,
    game_counts: List[int],
    sgffile_prefix: str = "match",
    verbose: bool = True,
    pool: Pool = None, # type: ignore
) -> List[int]:
    """
    Play multiple games between two engines using multiprocessing.

    Args:
        black_parameters (dict): Parameters for the black player.
        white_parameters (dict): Parameters for the white player.
        gogui_classpath (str): Path to the Gogui classpath.
        game_counts (List[int]): List of game counts.
        sgffile_prefix (str, optional): Prefix for the SGF file. Defaults to "match".
        verbose (bool, optional): Whether to print game results. Defaults to True.
        pool (Pool, optional): A multiprocessing Pool. If None, a new Pool is created.

    Returns:
        List[int]: List of game results where 1=Black win, -1=White win, 0=Draw
    """
    is_pool_owner = False
    if pool is None:
        pool_size = cpu_count()
        pool = Pool(pool_size)
        is_pool_owner = True

    try:
        # Prepare arguments for each game
        zipped_parameters = []
        for game_count in game_counts:
            # Alternate black and white assignments
            if game_count % 2 == 0:
                # Even-indexed games: bot A as black, bot B as white
                zipped_parameters.append((
                    black_parameters,
                    white_parameters,
                    gogui_classpath,
                    game_count,
                    sgffile_prefix,
                    verbose
                ))
            else:
                # Odd-indexed games: bot B as black, bot A as white
                zipped_parameters.append((
                    white_parameters,
                    black_parameters,
                    gogui_classpath,
                    game_count,
                    sgffile_prefix,
                    verbose
                ))

        # Use imap_unordered with the helper function
        results = pool.imap_unordered(match_helper, zipped_parameters)
    finally:
        if is_pool_owner:
            pool.close()
            pool.join()

    return results  # Return an iterator for tqdm integration


def elo(M: float, N: float) -> float:
    """
    Calculate expected ELO based on wins and games played.

    Args:
        M (float): Number of wins
        N (float): Number of games played.

    Returns:
        float: expected ELO value.
    """
    if N <= M:
        return float("inf")
    elif M <= 0:
        return float("-inf")
    else:
        return -400 * math.log10(-1 + (N / M))


def elo_range(M: int, N: int, a: float) -> Tuple[float, float]:
    """
    Computes the Elo rating range for a player based on their win record and a
    scaling factor for the standard deviation.

    Args:
        M (int): The number of wins achieved by the player.
        N (int): The total number of games played by the player.
        a (float): A scaling factor for the standard deviation.

    Returns:
        Tuple[float, float]: The lower and upper bounds of the Elo rating range.
    """
    if N == M:
        return (float("inf"), float("inf"))
    elif M == 0:
        return (float("-inf"), float("-inf"))

    p = M / N
    var = bernoulli.var(p) / N
    stdev = math.sqrt(var)
    delta = a * stdev * N

    elo_positive_delta = elo(M + delta, N)
    elo_negative_delta = elo(M - delta, N)

    return (elo_negative_delta, elo_positive_delta)


def print_game_results(bot_a_name: str, bot_b_name: str, bot_b_outcome: float, total_games: int):
    """
    Print the results of the game between two bots.

    Args:
        bot_a_name (str): The name of the first bot.
        bot_b_name (str): The name of the second bot.
        bot_b_outcome (float): The outcome for bot B.
        total_games (int): Total number of games played.
    """
    print(
        f"\nExpected ELO values for the {bot_a_name} and {bot_b_name} parameters, "
        f"based on {total_games} games each, are as follows:"
    )
    print(f"- For {bot_a_name}: 0 (baseline)")
    print(f"- For {bot_b_name}: {elo(bot_b_outcome, total_games):.2f}")
    print(
        f"\nFurthermore, the ELO ranges for the {bot_b_name} parameters, "
        f"considering deviations from the mean by 1, 2, and 3 standard "
        f"deviations, are as follows:"
    )
    print(f"- ELO range (+/- 1.0 SD): {elo_range(bot_b_outcome, total_games, 1.0)}")
    print(f"- ELO range (+/- 2.0 SD): {elo_range(bot_b_outcome, total_games, 2.0)}")
    print(f"- ELO range (+/- 3.0 SD): {elo_range(bot_b_outcome, total_games, 3.0)}")
    print(
        f"\nThe Elo rating range is computed by adjusting the player's win count "
        f"using a scaled measure of the standard deviation of their win ratio, "
        f"thereby reflecting the uncertainty in their performance within a "
        f"specified confidence interval."
    )


def define_bot_opencl_parameters(maxVisits: int) -> dict:
    """
    Define OpenCL parameters for a bot.

    Args:
        maxVisits (int): The value for max visits.

    Returns:
        dict: A dictionary containing the bot parameters.
    """
    return {
        "exe": "/Users/chinchangyang/Code/KataGo/cpp/katago",
        "config": "/Users/chinchangyang/Code/KataGo/cpp/configs/gtp_example.cfg",
        "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b28c512nbt-s7332806912-d4357057652.bin.gz",
        "maxVisits": maxVisits,
        "numSearchThreads": 1,
        "maxTime": 1e20,
        "resignConsecTurns": 6,
        "resignThreshold": -0.99,
    }


def define_bot_coreml_parameters(maxVisits: int, fp32: bool) -> dict:
    """
    Define CoreML parameters for a bot.

    Args:
        maxVisits (int): The value for max visits.
        fp32 (bool): Flag indicating whether to use fp32.

    Returns:
        dict: A dictionary containing the bot parameters.
    """
    postfix = "_fp32" if fp32 else ""

    return {
        "exe": "/Users/chinchangyang/Code/KataGo-CCY/cpp/build/katago",
        "config": f"/Users/chinchangyang/Code/KataGo-CCY/cpp/configs/misc/coreml_gtp{postfix}.cfg",
        "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b28c512nbt-s7332806912-d4357057652.bin.gz",
        "maxVisits": maxVisits,
        "numSearchThreads": 1,
        "maxTime": 1e20,
        "resignConsecTurns": 6,
        "resignThreshold": -0.99,
    }


def main():
    # Record start time
    t0 = time.time()

    # Define maximum visits allowed for a move generation
    maxVisits = 1

    # Define parameters for bot A
    bot_a_name = f"b28c512nbt-opencl-fp16-v{maxVisits}"
    bot_a_parameters = define_bot_opencl_parameters(maxVisits=maxVisits)

    # Define parameters for bot B
    fp32 = False
    fp_name = "fp32" if fp32 else "fp16"
    bot_b_name = f"b28c512nbt-coreml-{fp_name}-v{maxVisits}"
    bot_b_parameters = define_bot_coreml_parameters(
        maxVisits=maxVisits, fp32=fp32
    )

    # Define total number of games
    total_games = 1024
    game_counts = list(range(total_games))

    # Define Gogui classpath
    gogui_classpath = "/Users/chinchangyang/Code/gogui/bin"

    # Initialize a multiprocessing Pool
    pool_size = cpu_count()
    with Pool(pool_size) as pool:
        # Execute all matches in parallel using imap_unordered with the helper function
        results_iterator = match_games(
            black_parameters=bot_a_parameters,
            white_parameters=bot_b_parameters,
            gogui_classpath=gogui_classpath,
            game_counts=game_counts,
            sgffile_prefix="match",
            verbose=False,  # Disable per-game verbosity
            pool=pool
        )

        # Initialize counters
        bot_a_win = 0
        bot_b_win = 0
        draws = 0

        # Initialize the progress bar
        with tqdm(total=total_games, desc="Matching Games", unit="game") as pbar:
            for res in results_iterator:
                if res == 1:
                    bot_a_win += 1
                elif res == -1:
                    bot_b_win += 1
                else:
                    draws += 1

                # Update the progress bar with current scores
                pbar.set_postfix({
                    bot_a_name: bot_a_win,
                    bot_b_name: bot_b_win,
                    "Draws": draws
                })
                pbar.update(1)

    # Calculate outcome for bot B based on wins and draws
    bot_b_outcome = bot_b_win + (draws / 2)

    # Print intermediate results
    print(f"\n{bot_a_name} Won: {bot_a_win}")
    print(f"{bot_b_name} Won: {bot_b_win}")
    print(f"Draw: {draws}")

    # Print final game results
    print_game_results(bot_a_name, bot_b_name, bot_b_outcome, total_games)

    # Record elapsed time and print
    elapsed = time.time() - t0
    print(f"\nElapsed: {str(datetime.timedelta(seconds=round(elapsed)))}")


if __name__ == "__main__":
    main()
