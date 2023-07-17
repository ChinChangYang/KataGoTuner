from functools import reduce, cmp_to_key
import math
import subprocess
from scipy.stats import norm
from random import random
from sgfmill import sgf
import cma
import matplotlib.pyplot as plt

def get_katago_command(x: "list") -> str:
    """Translate parameters into a KataGo command

    Args:
        x (list): the parameters

    Returns:
        str: KataGo command
    """
    global katago_exe

    # KataGo parameters
    rootPolicyOptimism = 0.1
    policyOptimism = x[0]
    maxVisits = 4
    numSearchThreads = 1
    staticScoreUtilityFactor = x[1]
    dynamicScoreUtilityFactor = x[2]

    # KataGo command
    command = (f'{katago_exe} '
               f'gtp '
               f'-override-config rootPolicyOptimism={rootPolicyOptimism} '
               f'-override-config policyOptimism={policyOptimism} '
               f'-override-config maxVisits={maxVisits} '
               f'-override-config numSearchThreads={numSearchThreads} '
               f'-override-config staticScoreUtilityFactor={staticScoreUtilityFactor} '
               f'-override-config dynamicScoreUtilityFactor={dynamicScoreUtilityFactor}'
               )
    
    return command

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

def simulate_badness(a: "list") -> float:
    """Simulate the badness of the parameters

    Args:
        a (list): the parameters

    Returns:
        float: the badness of the parameters
    """

    # Optimal parameter
    o = 0.1

    # Scaler of the badness of the parameters
    # A larger scaler results in a more sensitive parameter in term of winrate
    scaler = len(a) / 0.05

    # Define the badness of the parameters
    badness = lambda x: scaler * abs(x - o)

    # Badness of each parameter of program A
    x2 = [badness(xi) for xi in a]

    # Sum the badness of the parameters of program A
    fa = reduce(lambda x2i, x2j: x2i + x2j, x2)

    return fa

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

    # Badness of program A
    fa = simulate_badness(a)

    # Badness of program B
    fb = simulate_badness(b)

    # Winrate of program A
    winrate = 1 - norm.cdf(fa - fb)

    # Result of a game
    is_won = -1 if random() < winrate else 1

    return is_won

def is_program_a_superior_than_b(a: "tuple", b: "tuple") -> int:
    """Get a flag to indicate whether program A is superior than program B

    Args:
        a (tuple): a[0] is ignored, and a[1] contains the parameters of program A
        b (tuple): b[0] is ignored, and b[1] contains the parameters of program B

    Returns:
        int: _description_
    """
    global simulation

    # Number of games
    # A large number of games improves the accuracy of the result
    games = 1

    # A function that returns a result of a game that is played by programs A and B
    resultOf = simulate_program_a_play_with_b if simulation else program_a_play_with_b

    # Get results from the games
    results = [resultOf(a[1], b[1]) for _ in range(games)]

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

simulation = True # True: simulation; False: real games
katago_exe = "/Users/chinchangyang/Code/KataGo/cpp/build/katago" # Path to KataGo executable file
gogui_classpath = "/Users/chinchangyang/Code/gogui/bin" # Class path of `GoGui`

x0 = 3 * [0.5] # initial guess of minimum solution
sigma = 0.25 # initial standard deviation in each coordinate

options = cma.CMAOptions() # initialize CMA options
options.set('bounds', [0, 1]) # lower and upper boundaries of parameters
options.set('popsize', 6) # population size
options.set('tolx', 1e-3) # tolerance in solution changes
options.set('maxfevals', 512) # maximum number of function evaluations

# Run the stochastic optimizer CMA-ES
match = 0 # counter of match games
result = cma.fmin(None, x0, sigma, options=options, parallel_objective=ranking)

# Plot CMA-ES data from files
cma.plot()

# Display the figures
plt.show()

default_parameters = [1.0, 0.10, 0.30]
default_command = get_katago_command(default_parameters)
print(f'Default KataGo command: {default_command}')
print()

cma_parameters = result[5]
cma_command = get_katago_command(cma_parameters)
print(f'CMA KataGo command: {cma_command}')

# Pause to let users be able to view the diagrams
input()

# TODO Validate the program with the best parameters that have been found by CMA
