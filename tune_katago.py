from functools import reduce, cmp_to_key
import subprocess
from scipy.stats import binom, bernoulli
from random import random
from sgfmill import sgf
import cma
import matplotlib.pyplot as plt
import numpy as np
import pairwiseclop
import pandas as pd
import math

def translate_parameters(x: "list") -> "dict[str, float]":
    """Translate solutions into parameters

    Args:
        x (list): solutions

    Returns:
        list: parameters
    """
    parameters = {
        "cpuctExploration": (x[0] * 4.0),
        "cpuctExplorationLog": (x[0] * 1.80),
        "staticScoreUtilityFactor": (x[1] * 1.0),
        "dynamicScoreUtilityFactor": (x[2] * 1.0),
    }

    return parameters

def translate_solutions(y: "dict[str, float]") -> "list":
    """Translate parameters into solutions

    Args:
        y (dict[str, float]): parameters

    Returns:
        list: solutions
    """
    solutions = [
        y["cpuctExploration"] / 4.0,
        y["staticScoreUtilityFactor"] / 1.0,
        y["dynamicScoreUtilityFactor"] / 1.0,
    ]

    return solutions

def get_katago_parameters(x: "list") -> "dict[str, float]":
    """Get KataGo parameters from a list of values within [0, 1]

    Args:
        x (list): a list of values

    Returns:
        dict[str, float]: KataGo parameters
    """
    sub_parameters = translate_parameters(x)

    parameters = {
        "maxVisits": 256,
        "numSearchThreads": 2,
        "chosenMoveTemperatureEarly": 0.5,
        "chosenMoveTemperatureHalflife": 19,
        "chosenMoveTemperature": 0.1,
        "chosenMoveSubtract": 0,
        "chosenMovePrune": 1,
        "rootNumSymmetriesToSample": 1,
        "lcbStdevs": 5.0,
        "minVisitPropForLCB": 0.15,
        "winLossUtilityFactor": 1.0,
        "staticScoreUtilityFactor": sub_parameters["staticScoreUtilityFactor"],
        "dynamicScoreUtilityFactor": sub_parameters["dynamicScoreUtilityFactor"],
        "dynamicScoreCenterZeroWeight": 0.20,
        "dynamicScoreCenterScale": 0.75,
        "noResultUtilityForWhite": 0.0,
        "drawEquivalentWinsForWhite": 0.5,
        "cpuctExploration": sub_parameters["cpuctExploration"],
        "cpuctExplorationLog": sub_parameters["cpuctExplorationLog"],
        "cpuctUtilityStdevPrior": 0.40,
        "cpuctUtilityStdevPriorWeight": 2.0,
        "cpuctUtilityStdevScale": 0.85,
        "fpuReductionMax": 0.2,
        "rootFpuReductionMax": 0.1,
        "uncertaintyExponent": 1.0,
        "uncertaintyCoeff": 0.25,
        "rootPolicyOptimism": 0.2,
        "policyOptimism": 1.0,
        "valueWeightExponent": 0.25,
        "rootEndingBonusPoints": 0.5,
        "subtreeValueBiasFactor": 0.45,
        "subtreeValueBiasWeightExponent": 0.85,
        "nodeTableShardsPowerOfTwo": 16,
        "numVirtualLossesPerThread": 1,
    }

    return parameters

def get_katago_command(x: "list") -> str:
    """Translate parameters into a KataGo command

    Args:
        x (list): the parameters

    Returns:
        str: KataGo command
    """
    global katago_exe

    # KataGo parameters
    parameters = get_katago_parameters(x)

    # KataGo command
    command = f'{katago_exe} gtp '
    for key, value in parameters.items():
        command += f'-override-config {key}={value} '
    
    return command.strip() # Remove trailing space

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

def get_prefix_sgffile() -> str:
    """Get a new prefix of SGF file name

    Returns:
        str: Prefix of SGF file name
    """
    global match
    prefix_sgffile = f'match-{match}'
    match = match + 1
    return prefix_sgffile

def program_a_play_with_b(a: "list", b: "list") -> int:
    """Spawn a process of `twogtp` that hosts a game to let program A play with program B

    Args:
        a (list): the parameters of program A
        b (list): the parameters of program B

    Returns:
        int: -1 if program A wins; 1 if program B wins; 0 otherwise
    """
    global gogui_classpath
    assert(len(a) == len(b))

    # KataGo command of program A
    black = get_katago_command(a)

    # KataGo command of program B
    white = get_katago_command(b)

    # Prefix of the SGF file
    prefix_sgffile = get_prefix_sgffile()

    # TwoGTP command that host a game
    twogtp = ['java',
              '-cp',
              gogui_classpath,
              'net.sf.gogui.tools.twogtp.Main',
              '-black',
              black,
              '-white',
              white,
              '-size',
              '9',
              '-auto',
              '-sgffile',
              prefix_sgffile
              ]

    # Spawn a process that execute the TwoGTP command
    spawn_process(twogtp)

    # SGF file name
    sgffile = f'{prefix_sgffile}-0.sgf'

    # Default return value
    is_won = 0

    with open(sgffile, "rb") as f:
        # SGF string
        sgf_string = f.read()

        # SGF game
        game = sgf.Sgf_game.from_bytes(sgf_string)

        # Winner of the game
        winner = game.get_winner()

        if (winner == 'b'):
            # Program A won
            is_won = -1
        elif (winner == 'w'):
            # Program B won
            is_won = 1
        else:
            # Draw
            is_won = 0

    return is_won

def sphere(x: "list") -> float:
    """Sphere function. A continuous, convex and unimodal function.

    Args:
        x (list): xi in [-5.12, 5.12] for each xi in x

    Returns:
        float: function value
    """
    y = [(xi ** 2) for xi in x]
    f = reduce(lambda yi, yj: yi + yj, y)

    return f

def elliptic(x: "list") -> float:
    """High conditioned elliptic function. A unimodal function.

    Args:
        x (list): xi in [-100, 100] for each xi in x

    Returns:
        float: function value
    """
    D = len(x)

    if D <= 1:
        c = 1
    else:
        c = [(1e6 ** (i / (D - 1))) for i in range(D)]

    zipped = zip(c, x)
    y = [(ci * (xi ** 2)) for (ci, xi) in zipped]
    f = reduce(lambda yi, yj: yi + yj, y)

    return f

def rotation_matrix_pair(theta: float, dim: int, i: int, j: int):
    """Rotation matrix for a pair

    Args:
        theta (float): rotation theta
        dim (int): dimension
        i (int): first index
        j (int): second index

    Returns:
        ndarray: rotation matrix
    """
    rotation = np.eye(dim)
    rotation[i, i] = np.cos(theta)
    rotation[i, j] = -np.sin(theta)
    rotation[j, i] = np.sin(theta)
    rotation[j, j] = np.cos(theta)

    return rotation

def rotated_elliptic(x: "list", theta=np.radians(30)) -> float:
    """Rotated elliptic function. A unimodal, and non-separable function.

    Args:
        x (list): xi in [-100, 100] for each xi in x
        theta (float, optional): rotation theta. Defaults to np.radians(30).

    Returns:
        float: function value
    """
    # Determine the dimension of x
    D = len(x)
    # Initialize the rotated x as the original x
    x_rot = np.array(x)
    # Apply the rotation matrix for each pair of dimensions
    for i in range(D - 1):
        for j in range(i + 1, D):
            x_rot = np.dot(rotation_matrix_pair(theta, D, i, j), x_rot)
    # Compute the elliptic function value at the rotated x
    return elliptic(x_rot)

def rastrigin(x: "list") -> float:
    """Rastrigin function. A non-linear multimodal function.

    Args:
        x (list): xi in [-5.12, 5.12] for each xi in x

    Returns:
        float: function value
    """
    A = 10
    n = len(x)
    y = [(xi ** 2) - (A * np.cos(2 * np.pi * xi)) for xi in x]
    f = (A * n) + reduce(lambda yi, yj: yi + yj, y)

    return f

def simulate_elo(a: "list") -> float:
    """Simulates ELO rating for given parameters.

    This function computes the ELO rating by applying a penalty on each
    parameter value based on how much it deviates from a predefined optimal
    solution.

    Args:
        a (list): The parameters for which ELO rating is to be computed.

    Returns:
        float: The computed ELO rating for the given parameters.
    """
    global simulated_optimum

    # Combine the parameters and the default solutions
    zipped = zip(a, simulated_optimum)

    # Scale the parameters to the domain space of the test function
    x = [100.0 * (ai - oi) for (ai, oi) in zipped]

    # Define test function
    # test_function = sphere
    # test_function = elliptic
    test_function = rotated_elliptic
    # test_function = rastrigin

    # Get the function value
    f = test_function(x)

    # Define the scaling factor for the function value.
    # A greater scaler makes the parameters more sensitive in the ELO rating.
    if test_function == sphere:
        scaler = 1e-1
    elif test_function == elliptic:
        scaler = 1e-6
    elif test_function == rotated_elliptic:
        scaler = 1e-7
    elif test_function == rastrigin:
        scaler = 1e-1
    else:
        scaler = 1

    # Calculate the ELO rating
    elo = -scaler * f

    return elo

def simulate_program_a_play_with_b(a: "list", b: "list") -> int:
    """Simulate a result of a game where program A plays with program B

    Args:
        a (list): the parameters of program A
        b (list): the parameters of program B

    Returns:
        int: -1 if program A wins; 1 if program B wins; 0 otherwise
    """
    assert(len(a) == len(b))

    # Trigger the side effect of the prefix getter
    get_prefix_sgffile()

    # Elo rating of program A
    fa = simulate_elo(a)

    # Elo rating of program B
    fb = simulate_elo(b)

    # Winrate of program A
    if (fb - fa) / 400 > 10:
        winrate = 0
    elif (fb - fa) / 400 < -10:
        winrate = 1
    else:
        winrate = 1 / (1 + 10 ** ((fb - fa) / 400))

    # Result of a game
    is_won = -1 if random() < winrate else 1 # stochastic
    # is_won = -1 if 0.5 < winrate else 1 # deterministic

    return is_won

def is_program_a_superior_than_b(a: "tuple", b: "tuple") -> int:
    """Get a flag to indicate whether program A is superior than program B

    Args:
        a (tuple): a[0] is ignored, and a[1] contains the parameters of program A
        b (tuple): b[0] is ignored, and b[1] contains the parameters of program B

    Returns:
        int: -1 if program A is superior; 1 if program B is superior; 0 otherwise
    """
    global simulation

    # Number of games
    # A large number of games improves the accuracy of the result
    games = 1

    # A function that returns a result of a game that is played by programs A and B
    result_of = simulate_program_a_play_with_b if simulation else program_a_play_with_b

    # Get results from the games
    results = [result_of(a[1], b[1]) for _ in range(games)]

    # Sum the results
    reduce_sum_results = reduce(lambda x, y: x + y, results)

    if reduce_sum_results <= -1:
        # Program A is superior than B
        is_superior = -1
    elif reduce_sum_results >= 1:
        # Program B is superior than A
        is_superior = 1
    else:
        # Programs A and B are equally good
        is_superior = 0

    return is_superior

def ranking(X: "list") -> "list":
    """Ranking a list of parameters

    Args:
        X (list): a list of parameters

    Returns:
        list: a list of rankings
    """
    global match

    # Sort the list of parameters with the keys of their superiorities
    Y = sorted(enumerate(X), key=cmp_to_key(is_program_a_superior_than_b))

    # Sort the list of the sorted indexes
    Z = sorted(enumerate(Y), key=lambda y: y[1])

    # Ranking is the indexes of the sorted indexes
    # Subtract the counter of match games to prevent a global optimizer from
    # termination on function history tolerance
    F = [zi[0] - match for zi in Z]

    return F

def match_program_a_and_b(a: "list", b: "list", games: int):
    """Match program A and program B

    Args:
        a (list): the parameters of program A
        b (list): the parameters of program B
        games (int): the number of games

    Returns:
        _type_: a tuple of the numbers of games (A wins, draw, B wins)
    """
    global simulation

    # A function that returns a result of a game that is played by programs A and B
    result_of = simulate_program_a_play_with_b if simulation else program_a_play_with_b

    # Get results from the games
    results = [result_of(a, b) for _ in range(games)]

    # Mark those games that program A wins
    a_wins = [1 if result == -1 else 0 for result in results]

    # Mark those games that program B wins
    b_wins = [1 if result == 1 else 0 for result in results]

    # Mark those draw games
    draws = [1 if result == 0 else 0 for result in results]

    # Count the games that program A wins
    a_win = reduce(lambda x, y: x + y, a_wins)

    # Count the games that program B wins
    b_win = reduce(lambda x, y: x + y, b_wins)

    # Count the draw games
    draw = reduce(lambda x, y: x + y, draws)

    return (a_win, draw, b_win)

def print_parameters(parameters: "dict[str, float]"):
    for name in parameters:
        print(f'{name}: {parameters[name]}')

def run_cma_fmin(x0: list, sigma0: float) -> list:
    """Run CMA-ES fmin function

    Args:
        x0 (list): initial guess of minimum solution
        sigma0 (float): initial standard deviation in each coordinate

    Returns:
        list: tuned solutions
    """
    # Modify CMA options
    options = cma.CMAOptions() # initialize CMA options
    options.set('bounds', [0, 1]) # lower and upper boundaries of parameters
    options.set('popsize', 10 * len(x0)) # population size
    options.set('tolx', 1e-2) # tolerance in solution changes
    options.set('maxfevals', 2e4) # maximum number of function evaluations
    options.set('tolconditioncov', 1e12) # tolerance in condition of the covariance matrix

    # Run the stochastic optimizer CMA-ES
    result = cma.fmin(None, x0, sigma0, options=options, parallel_objective=ranking)

    # Plot CMA-ES data from files
    cma.plot()

    # Display the figures
    plt.show()

    return result[5]

def plot_pairwiseclop(optimums, iterations):
    global plotting

    if plotting:
        # Create a DataFrame from the lists
        data = pd.DataFrame(
            optimums,
            index = range(iterations),
            columns = [f'opt_{i}' for i in range(len(optimums[0]))]
        )

        # Create a line plot for each optimum
        for col in data.columns:
            plt.plot(data.index, data[col], label = col)

        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Optimum')
        plt.title('Evolution of Optimums')
        plt.legend()
        plt.grid(True)
        plt.show()

def run_pairwiseclop(x0: list, sigma0: float) -> list:
    """Run pairwise CLOP

    Args:
        x0 (list): initial guess of minimum solution
        sigma0 (float): likely radius around guess

    Returns:
        list: tuned solutions
    """
    global simulation

    # Number of solutions
    D = len(x0)

    # Translate solutions to parameters for CLOP
    parameters = [
        pairwiseclop.Parameter(
            name=str(i),
            guess=x0[i],
            likely_radius_around_guess=sigma0,
            hard_lower_bound=0.0,
            hard_upper_bound=1.0,
        ) for i in range(D)
    ]

    def parameters_to_solutions(parameters) -> list:
        # Translate parameters to solutions for CLOP
        solutions = [parameters[str(i)] for i in range(D)]

        # Check boundaries
        for i in range(len(solutions)):
            assert solutions[i] >= 0.0
            assert solutions[i] <= 1.0

        return solutions
    
    # Initialize CLOP
    clop = pairwiseclop.PairwiseCLOP(parameters)

    # Initialize optimum
    optimum = x0

    # A function that returns a result of a game that is played by programs A and B
    result_of = simulate_program_a_play_with_b if simulation else program_a_play_with_b

    # Initialize the iteration list
    iterations = 500

    # Initialize the optimum list
    optimums = []

    for iteration in range(iterations):
        for _ in range(D):
            # Sample parameters A to evaluate
            a = clop.sample_params_to_evaluate()

            # Sample parameters B to evaluate
            b = clop.sample_params_to_evaluate()

            # Get the result from the games
            is_won = result_of(
                parameters_to_solutions(a),
                parameters_to_solutions(b),
            )

            if is_won == -1:
                # Program A won
                clop.add_win(winner = a, loser = b)
            elif is_won == 1:
                # Program B won
                clop.add_win(winner = b, loser = a)
            else:
                # Draw
                clop.add_draw(a, b)

        # Recompute regression
        clop.recompute()

        # Get the current optimum
        optimum = parameters_to_solutions(clop.get_current_optimum())

        # Append the optimum to the list
        optimums.append(optimum)

        # Print the iteration and the current optimum
        print(f'Iteration: {iteration}, optimum: {optimum}')

    plot_pairwiseclop(optimums, iterations)

    return optimum

def elo(M: float, N: float) -> float:
    """Calculate expected ELO

    Args:
        M (float): Number of wins
        N (float): Number of games

    Returns:
        float: expected ELO
    """
    if N <= M:
        return float('inf')
    elif M <= 0:
        return float('-inf')
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
        return (float('inf'), float('inf'))
    elif M == 0:
        return (float('-inf'), float('-inf'))
    else:
        p = M / N # mean
        var = bernoulli.var(p) / N # variance of sample mean
        stdev = math.sqrt(var) # standard deviation
        delta = a * stdev * N # delta for M
        elo_positive_delta = elo(M + delta, N) # ELO with positive delta
        elo_negative_delta = elo(M - delta, N) # ELO with negative delta
        return (elo_negative_delta, elo_positive_delta)
    
def plot_elo_range(M: int, N: int):
    global plotting

    if plotting:
        expected_elo = elo(M, N)

        ranges = {
            '3.0 SD': elo_range(M, N, 3.0),
            '2.0 SD': elo_range(M, N, 2.0),
            '1.0 SD': elo_range(M, N, 1.0),
        }

        # Plotting
        _, ax = plt.subplots()
        colors = ['blue', 'green', 'red']

        for idx, (label, (low, high)) in enumerate(ranges.items()):
            ax.plot([low, high], [idx, idx], color=colors[idx], label=label, linewidth=10)
            ax.scatter([low, high], [idx, idx], color=colors[idx], s=50)

        ax.scatter(expected_elo, idx + 1, color='black', s=100, marker='X', label="Expected Tuned ELO")
        ax.axvline(x=expected_elo, color='black', linestyle='--')
        ax.set_yticks(list(range(len(ranges) + 1)))
        ax.set_yticklabels(list(ranges.keys()) + ["Expected Tuned ELO"])
        ax.set_title("Tuned ELO Ranges with Standard Deviations")
        ax.set_xlabel("ELO Value")
        ax.set_ylabel("Standard Deviations")
        ax.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

def tune(x0: list, sigma0: float) -> float:
    global match, default_parameters, plotting

    match = 0 # initialize a counter of match games

    # Define the tuner
    # tuner = run_cma_fmin # stochastic optimizer CMA-ES
    tuner = run_pairwiseclop # pairwise CLOP

    # Get tuned solutions
    tuned_solutions = tuner(x0, sigma0)

    # Print the number of match games
    print(f'Match games: {match}')

    # Print default parameters
    print('=== Default KataGo parameters (start) ===')
    print_parameters(default_parameters)
    print('=== Default KataGo parameters (end) ===')

    # Print tuned parameters
    tuned_parameters = translate_parameters(tuned_solutions)
    print('=== Tuned KataGo parameters (start) ===')
    print_parameters(tuned_parameters)
    print('=== Tuned KataGo parameters (end) ===')

    if simulation:
        default_elo = simulate_elo(default_solutions)
        print(f'Elo of default KataGo parameters (simulation): {default_elo}')
        tuned_elo = simulate_elo(tuned_solutions)
        print(f'Elo of Tuned KataGo parameters (simulation): {tuned_elo}')
        print(f'Elo of optimal parameters (simulation): 0')

    if plotting:
        # Parameter names are the same for both dictionaries
        parameter_names = default_parameters.keys()
        default_values = [default_parameters[k] for k in parameter_names]
        cma_values = [tuned_parameters[k] for k in parameter_names]
        x = range(len(parameter_names)) # indices of parameter names

        # Visualize default and Tuned KataGo parameters
        plt.figure()
        plt.barh(x, default_values, height = 0.4, align = 'center', label = 'Default Parameters')
        plt.barh(x, cma_values, height = 0.4, align = 'edge', label = 'Tuned Parameters')
        plt.yticks(x, parameter_names)
        plt.xlabel('Parameter Value')
        plt.legend()
        plt.title('Comparison of Default and Tuned KataGo Parameters')
        plt.tight_layout()
        plt.show()

    games = 200 # number of games to verify goodness of Tuned KataGo command
    print(f'Verifying goodness of Tuned KataGo command with {games} games...')
    half_games = int(games / 2) # half of the number of games

    # Match default KataGo (as black) and Tuned KataGo (as white)
    (default_win, draw, cma_win) = match_program_a_and_b(default_solutions, tuned_solutions, half_games)

    # Record the number of games that black wins
    total_black_win = default_win

    # Record the number of games that white wins
    total_white_win = cma_win

    # Record the numbers of winning and draw games
    (total_default_win, total_draw, total_cma_win) = (default_win, draw, cma_win)

    # Match Tuned KataGo (as black) and default KataGo (as white)
    (cma_win, draw, default_win) = match_program_a_and_b(tuned_solutions, default_solutions, half_games)

    # Record the number of games that black wins
    total_black_win += cma_win

    # Record the number of games that white wins
    total_white_win += default_win

    # Record the number of draw games
    total_draw += draw

    # Record the number of games that default KataGo wins
    total_default_win += default_win

    # Record the number of games that Tuned KataGo wins
    total_cma_win += cma_win

    # Expected ELO
    tuned_elo = elo(total_cma_win, games)

    print(f'Games: {games}')
    print(f'Black:draw:white = {total_black_win}:{total_draw}:{total_white_win}')
    print(f'Default:Tuned = {total_default_win}:{total_cma_win}')
    print(f'ELO of default parameters = 0')
    print(f'Expected ELO of tuned parameters = {tuned_elo}')
    print(f'ELO range (+/- 1.0 standard deviation) = {elo_range(total_cma_win, games, 1.0)}')
    print(f'ELO range (+/- 2.0 standard deviation) = {elo_range(total_cma_win, games, 2.0)}')
    print(f'ELO range (+/- 3.0 standard deviation) = {elo_range(total_cma_win, games, 3.0)}')

    # Plot ELO ranges
    plot_elo_range(total_cma_win, games)

    if plotting:
        # Binomial distribution under the null hypothesis
        n_values = np.arange(games + 1) # possible number of wins for A
        prob_values = binom.pmf(n_values, games, 0.5) # probabilities under the null hypothesis

        # Calculate the cutoffs for the number of wins that correspond to a p-value of 0.05
        lower_cutoff = 0
        upper_cutoff = games
        cumulative_prob = 0

        # Calculate lower cutoff
        for k in range(0, games + 1):
            prob = binom.pmf(k, games, 0.5)
            if cumulative_prob + prob < 0.025:
                cumulative_prob += prob
                lower_cutoff = k
            else:
                break

        cumulative_prob = 0 # reset cumulative probability

        # Calculate upper cutoff
        for k in range(games, -1, -1):
            prob = binom.pmf(k, games, 0.5)
            if cumulative_prob + prob < 0.025:
                cumulative_prob += prob
                upper_cutoff = k
            else:
                break

        # Plot the binomial distribution
        plt.figure()

        # Highlight the region of the distribution that is less extreme than the cutoffs
        mask_not_extreme = np.logical_and(n_values > lower_cutoff, n_values < upper_cutoff) # region where null hypothesis is accepted

        plt.bar(
            n_values[mask_not_extreme],
            prob_values[mask_not_extreme],
            color = 'blue',
            alpha = 0.7,
            label = 'Region where $H_0$ is not rejected'
        )

        # Highlight the region of the distribution that is as extreme as or more extreme than the cutoffs
        mask_extreme = np.logical_or(n_values <= lower_cutoff, n_values >= upper_cutoff) # region where null hypothesis is rejected

        plt.bar(
            n_values[mask_extreme],
            prob_values[mask_extreme],
            color = 'red',
            alpha = 0.7,
            label = 'Region where $H_0$ is rejected'
        )

        # Show where the actual number of wins stands
        plt.axvline(x = total_cma_win, color = 'green', linestyle = '--')

        plt.text(
            total_cma_win + 1,
            max(prob_values) / 2,
            'Actual number of wins for Tuned',
            color = 'green',
            fontsize = 10
        )

        plt.xlabel('Number of wins for Tuned')
        plt.ylabel('Probability')
        plt.title('Binomial distribution under the null hypothesis')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Pause to let users be able to view the diagrams
        print('Press enter to continue...')
        input()

    return tuned_elo

simulation = True # True: simulation; False: real games
plotting = True # draw diagrams
katago_exe = "/Users/chinchangyang/Links/katago-ccy" # Path to KataGo executable file
gogui_classpath = "/Users/chinchangyang/Code/gogui/bin" # Class path of `GoGui`

# Default KataGo parameters
default_parameters = {
    "cpuctExploration": 1.0,
    "cpuctExplorationLog": 0.45,
    "staticScoreUtilityFactor": 0.1,
    "dynamicScoreUtilityFactor": 0.3,
}

# Default KataGo solutions
default_solutions = translate_solutions(default_parameters)

# Sanity check
assert(default_solutions == translate_solutions(translate_parameters(default_solutions)))

match = 0 # initialize a counter of match games
x0 = default_solutions # initial guess of minimum solution
sigma0 = 0.2 # initial standard deviation in each coordinate

# Define simulated optimum
if simulation:
    shift = lambda x, s: (x - s) if (x - s) > 0 else (x + s)
    simulated_optimum = [shift(x0i, sigma0) for x0i in x0]

tuned_num = 1 # number of tuned ELOs
tuned_elos = [] # initialize tuned ELOs

for _ in range(tuned_num):
    # Append tuned ELO
    tuned_elos.append(tune(x0, sigma0))

print('=== Tuned ELOs (start) ===')

for tuned_elo in tuned_elos:
    print(f'{tuned_elo}')

print('=== Tuned ELOs (end) ===')
