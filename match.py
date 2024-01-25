import datetime
from functools import reduce
import math
from multiprocessing import Pool
import subprocess
import time
from sgfmill import sgf
from scipy.stats import bernoulli


def spawn_process(command: "list[str]"):
    """Spawn a process with the command and wait until the process ends.

    Args:
        command (list[str]): the command
    """
    try:
        process = subprocess.Popen(command)
        process.wait()
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def get_command(x) -> str:
    command = f'{x["exe"]} gtp -config {x["config"]} -model {x["model"]}'
    override_options = [
        "maxVisits",
        "suppressVirtualLossExploreFactor",
        "suppressVirtualLossHindsight",
        "suppressVirtualLossLeakCatchUp",
        "numSearchThreads",
        "useNoisePruning",
        "maxTime",
    ]

    for option in override_options:
        if option in x:
            command += f" -override-config {option}={x[option]}"

    return command


def match(
    black_parameters,
    white_parameters,
    gogui_classpath,
    game_count,
    sgffile_prefix="match",
    verbose=True,
) -> int:
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
    black_parameters,
    white_parameters,
    game_count_start,
    games,
    sgffile_prefix="match",
    verbose=True,
) -> (int, int, int):
    results = []
    game_count_stop = game_count_start + games

    with Pool(8) as p:
        game_counts = range(game_count_start, game_count_stop)
        results = p.starmap(
            match,
            [
                (
                    black_parameters,
                    white_parameters,
                    "/Users/chinchangyang/Code/gogui/bin",
                    game_count,
                    sgffile_prefix,
                    verbose,
                )
                for game_count in game_counts
            ],
        )

    black_wins = [1 if result == -1 else 0 for result in results]
    black_win = reduce(lambda x, y: x + y, black_wins)
    white_wins = [1 if result == 1 else 0 for result in results]
    white_win = reduce(lambda x, y: x + y, white_wins)
    draws = [1 if result == 0 else 0 for result in results]
    draw = reduce(lambda x, y: x + y, draws)

    return (black_win, white_win, draw)


def elo(M: float, N: float) -> float:
    """Calculate expected ELO

    Args:
        M (float): Number of wins
        N (float): Number of games

    Returns:
        float: expected ELO
    """
    if N <= M:
        return float("inf")
    elif M <= 0:
        return float("-inf")
    else:
        return -400 * math.log10(-1 + (N / M))


def elo_range(M: int, N: int, a: float) -> float:
    """Calculate ELO standard deviation

    Args:
        M (int): Number of wins
        N (int): Number of games

    Returns:
        float: ELO standard deviation
    """
    if N == M:
        return (float("inf"), float("inf"))
    elif M == 0:
        return (float("-inf"), float("-inf"))
    else:
        p = M / N  # mean
        var = bernoulli.var(p) / N  # variance of sample mean
        stdev = math.sqrt(var)  # standard deviation
        delta = a * stdev * N  # delta for M
        elo_positive_delta = elo(M + delta, N)  # ELO with positive delta
        elo_negative_delta = elo(M - delta, N)  # ELO with negative delta
        return (elo_negative_delta, elo_positive_delta)


def get_bot_names():
    bot_a_name = "b28c512nbt-s2830075392-v64"
    bot_b_name = "b28c512nbt-s2652711936-v64"
    return bot_a_name, bot_b_name


if __name__ == "__main__":
    t0 = time.time()
    bot_a_name, bot_b_name = get_bot_names()

    bot_a_parameters = {
        "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
        "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
        "model": "/Users/chinchangyang/Code/KataGo-Models/b28c512nbt-s2830075392-d3981649212.bin.gz",
        "maxVisits": 64,
        "numSearchThreads": 1,
        "maxTime": 1e20,
    }

    bot_b_parameters = {
        "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
        "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
        "model": "/Users/chinchangyang/Code/KataGo-Models/b28c512nbt-s2652711936-d3972499590.bin.gz",
        "maxVisits": 64,
        "numSearchThreads": 1,
        "maxTime": 1e20,
    }

    total_games = 1024
    half_games = int(total_games / 2)

    black_win, white_win, draw = match_games(
        bot_a_parameters, bot_b_parameters, 0, half_games
    )

    bot_a_win = black_win
    bot_b_win = white_win
    both_draw = draw

    print(f"{bot_a_name} Won: {bot_a_win}")
    print(f"{bot_b_name} Won: {bot_b_win}")
    print(f"Draw: {both_draw}")

    black_win, white_win, draw = match_games(
        bot_b_parameters, bot_a_parameters, half_games, half_games
    )

    bot_a_win = bot_a_win + white_win
    bot_b_win = bot_b_win + black_win
    both_draw = both_draw + draw

    print(f"{bot_a_name} Won: {bot_a_win}")
    print(f"{bot_b_name} Won: {bot_b_win}")
    print(f"Draw: {both_draw}")

    bot_a_outcome = bot_a_win + (both_draw / 2)
    bot_b_outcome = bot_b_win + (both_draw / 2)

    print(
        f"Expected ELO of {bot_a_name} parameters (from {total_games} games) = 0 (baseline)"
    )
    print(
        f"Expected ELO of {bot_b_name} parameters (from {total_games} games) = {elo(bot_b_outcome, total_games)}"
    )
    print(
        f"ELO range (+/- 1.0 standard deviation) of {bot_b_name} parameters = {elo_range(bot_b_outcome, total_games, 1.0)}"
    )
    print(
        f"ELO range (+/- 2.0 standard deviation) of {bot_b_name} parameters = {elo_range(bot_b_outcome, total_games, 2.0)}"
    )
    print(
        f"ELO range (+/- 3.0 standard deviation) of {bot_b_name} parameters = {elo_range(bot_b_outcome, total_games, 3.0)}"
    )

    elapsed = time.time() - t0
    print(f"Elapsed: {str(datetime.timedelta(seconds=round(elapsed)))}")
