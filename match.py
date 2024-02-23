import datetime
from functools import reduce
import math
from multiprocessing import Pool
import subprocess
import time
from sgfmill import sgf
from scipy.stats import bernoulli


def spawn_process(command: "list[str]"):
    """
    Spawn a process with the command and wait until the process ends.

    Args:
        command (list[str]): The command to execute.
    """
    try:
        process = subprocess.Popen(command)
        process.wait()
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def get_command(parameters) -> str:
    """
    Build a GTP command with specified parameters.

    Args:
        parameters (dict): A dictionary containing model and configuration details.

    Returns:
        str: The constructed GTP command.
    """
    command = f'{parameters["exe"]} gtp -config {parameters["config"]} -model {parameters["model"]}'
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
    ]

    for option in override_options:
        if option in parameters:
            command += f" -override-config {option}={parameters[option]}"

    return command


def match(
    black_parameters,
    white_parameters,
    gogui_classpath,
    game_count,
    sgffile_prefix="match",
    verbose=True,
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
    twogtp = [
        "java",
        "-cp",
        gogui_classpath,
        "net.sf.gogui.tools.twogtp.Main",
        "-black",
        black_command,
        "-white",
        white_command,
        "-size",
        f"{board_size}",
        "-komi",
        f"{komi}",
        "-auto",
        "-sgffile",
        f"{sgffile_prefix}-{game_count}",
    ]

    spawn_process(twogtp)
    sgffile = f"{sgffile_prefix}-{game_count}-0.sgf"
    is_won = 0

    with open(sgffile, "rb") as f:
        sgf_string = f.read()
        game = sgf.Sgf_game.from_bytes(sgf_string)
        winner = game.get_winner()

        if winner == "b":
            # Black won
            is_won = -1
            if verbose:
                print(f"Game {game_count}: Black won")
        elif winner == "w":
            # White won
            is_won = 1
            if verbose:
                print(f"Game {game_count}: White won")
        else:
            # Draw
            is_won = 0
            if verbose:
                print(f"Game {game_count}: Draw")

    return is_won


def match_games(
    black_parameters: dict,
    white_parameters: dict,
    game_count_start: int,
    games: int,
    sgffile_prefix: str = "match",
    verbose: bool = True,
) -> tuple:
    """
    Play multiple games between two engines using multiprocessing.

    Args:
        black_parameters (dict): Parameters for the black player.
        white_parameters (dict): Parameters for the white player.
        game_count_start (int): Starting game count.
        games (int): Number of games to play.
        sgffile_prefix (str, optional): Prefix for the SGF file. Defaults to "match".
        verbose (bool, optional): Whether to print game results. Defaults to True.

    Returns:
        tuple: (black_wins, white_wins, draws)
    """
    results = []
    game_count_stop = game_count_start + games
    game_counts = range(game_count_start, game_count_stop)
    zipped_parameters = [
        (
            black_parameters,
            white_parameters,
            "/Users/chinchangyang/Code/gogui/bin",
            game_count,
            sgffile_prefix,
            verbose,
        )
        for game_count in game_counts
    ]

    if len(game_counts) > 1:
        with Pool(8) as p:
            results = p.starmap(
                match,
                zipped_parameters,
            )
    else:
        results = [match(*zipped_parameters[0])]

    black_wins = sum(result == -1 for result in results)
    white_wins = sum(result == 1 for result in results)
    draws = sum(result == 0 for result in results)

    return (black_wins, white_wins, draws)


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


def elo_range(M: int, N: int, a: float) -> tuple:
    """
    Computes the Elo rating range for a player based on their win record and a
    scaling factor for the standard deviation.

    Parameters:
    - M (int): The number of wins achieved by the player.
    - N (int): The total number of games played by the player.
    - a (float): A scaling factor for the standard deviation, which adjusts the
      confidence interval width, thereby influencing the Elo rating range.

    Returns:
    - tuple: A tuple containing the lower and upper bounds of the Elo rating
      range.

    The function models each game outcome as a Bernoulli trial, with a win
    considered a 'success'. The win ratio serves as the mean of the Bernoulli
    distribution. Utilizing the Central Limit Theorem, the distribution of the
    sample mean is approximated, enabling the computation of a confidence
    interval for the player's true win probability. The coefficient 'a' scales
    the standard deviation of the win ratio to adjust the confidence interval's
    width.
    """

    # Handle edge cases: undefeated record or no wins, returning infinite bounds
    # accordingly.
    if N == M:
        return (float("inf"), float("inf"))
    elif M == 0:
        return (float("-inf"), float("-inf"))

    # Calculate the win ratio, serving as the mean of the Bernoulli distribution
    # for win outcomes.
    p = M / N

    # Compute the variance of the sample mean, applying the Central Limit
    # Theorem to the outcomes' distribution.
    var = bernoulli.var(p) / N

    # Determine the standard deviation of the sample mean, indicating the win
    # ratio's dispersion from the mean.
    stdev = math.sqrt(var)

    # Calculate 'delta', which is the product of the scaling factor 'a', the
    # standard deviation, and the total number of games.
    # This 'delta' adjusts the Elo rating range to reflect the confidence level
    # in the player's performance.
    delta = a * stdev * N

    # Determine the Elo rating bounds by adjusting the win count by 'delta' and
    # calculating the corresponding Elo ratings.
    elo_positive_delta = elo(M + delta, N)  # ELO with an upward adjustment
    elo_negative_delta = elo(M - delta, N)  # ELO with a downward adjustment

    # Return the Elo rating range, encapsulating the estimated bounds of the
    # player's true skill level.
    return (elo_negative_delta, elo_positive_delta)


def print_game_results(bot_a_name, bot_b_name, bot_b_outcome, total_games):
    """
    Print the results of the game between two bots.

    Args:
    - bot_a_name (str): The name of the first bot.
    - bot_b_name (str): The name of the second bot.
    - bot_b_outcome (int): The number of wins by bot B.
    - total_games (int): Total number of games played.
    """
    print(
        f"Expected ELO values for the {bot_a_name} and {bot_b_name} parameters, "
        f"based on {total_games} games each, are as follows:"
    )
    print(f"- For {bot_a_name}: 0 (baseline)")
    print(f"- For {bot_b_name}: {elo(bot_b_outcome, total_games)}")
    print(
        f"Furthermore, the ELO ranges for the {bot_b_name} parameters, "
        f"considering deviations from the mean by 1, 2, and 3 standard "
        f"deviations, are as follows:"
    )
    print(f"- ELO range (+/- 1.0 SD): {elo_range(bot_b_outcome, total_games, 1.0)}")
    print(f"- ELO range (+/- 2.0 SD): {elo_range(bot_b_outcome, total_games, 2.0)}")
    print(f"- ELO range (+/- 3.0 SD): {elo_range(bot_b_outcome, total_games, 3.0)}")
    print(
        f"The Elo rating range is computed by adjusting the player's win count "
        f"using a scaled measure of the standard deviation of their win ratio, "
        f"thereby reflecting the uncertainty in their performance within a "
        f"specified confidence interval."
    )


def define_bot_parameters(name, maxVisits, cpuct_exploration):
    """
    Define parameters for a bot.

    Args:
    name (str): The name of the bot.
    maxVisits (int): The value for max visits.
    cpuct_exploration (float): The value for cpuct exploration.

    Returns:
    dict: A dictionary containing the bot parameters.
    """
    return {
        "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
        "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
        "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b18c384nbt-s9131461376-d4087399203.bin.gz",
        "maxVisits": maxVisits,
        "numSearchThreads": 1,
        "maxTime": 1e20,
        "cpuctExploration": cpuct_exploration,
        "cpuctExplorationLog": 0.45 * cpuct_exploration,
    }


if __name__ == "__main__":
    # Record start time
    t0 = time.time()

    # Define parameters for bot A
    cpuct_exploration_a = "1.0"
    maxVisits = 256
    bot_a_name = f"b18-v{maxVisits}-cpuct-{cpuct_exploration_a}"
    bot_a_parameters = define_bot_parameters(
        bot_a_name, maxVisits=maxVisits, cpuct_exploration=float(cpuct_exploration_a)
    )

    # Define parameters for bot B
    cpuct_exploration_b = "1.1"
    bot_b_name = f"b18-v{maxVisits}-cpuct-{cpuct_exploration_b}"
    bot_b_parameters = define_bot_parameters(
        bot_b_name, maxVisits=maxVisits, cpuct_exploration=float(cpuct_exploration_b)
    )

    # Define total number of games and split into halves
    total_games = 256
    half_games = int(total_games / 2)

    # Play first half of the games
    black_win, white_win, draw = match_games(
        bot_a_parameters, bot_b_parameters, 0, half_games
    )

    # Assign results of first half to corresponding bots
    bot_a_win = black_win
    bot_b_win = white_win
    both_draw = draw

    # Print results of first half
    print(f"{bot_a_name} Won: {bot_a_win}")
    print(f"{bot_b_name} Won: {bot_b_win}")
    print(f"Draw: {both_draw}")

    # Play second half of the games with bot A as black
    black_win, white_win, draw = match_games(
        bot_b_parameters, bot_a_parameters, half_games, half_games
    )

    # Update total wins and draws for both bots
    bot_a_win += white_win
    bot_b_win += black_win
    both_draw += draw

    # Print results of second half
    print(f"{bot_a_name} Won: {bot_a_win}")
    print(f"{bot_b_name} Won: {bot_b_win}")
    print(f"Draw: {both_draw}")

    # Calculate outcome for bot B based on wins and draws
    bot_b_outcome = bot_b_win + (both_draw / 2)

    # Print game results
    print_game_results(bot_a_name, bot_b_name, bot_b_outcome, total_games)

    # Record elapsed time and print
    elapsed = time.time() - t0
    print(f"Elapsed: {str(datetime.timedelta(seconds=round(elapsed)))}")
