import datetime
import math
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from match import match_games


def get_bot_names():
    bot_a_name = "b18c384nbt-s8341979392-t1"
    bot_b_name = "b18c384nbt-s8341979392-t32"
    return bot_a_name, bot_b_name


# Define the match function
def match_function(x, y):
    bot_a_name, bot_b_name = get_bot_names()
    bot_b_wins = []
    for xi, yi in zip(x, y):
        xi_int = int(xi)
        yi_int = int(yi)

        bot_a_parameters = {
            "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
            "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
            "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b18c384nbt-s8341979392-d3881113763.bin.gz",
            "maxVisits": f"{xi_int}",
            "numSearchThreads": 1,
            "maxTime": 1e20,
        }

        bot_b_parameters = {
            "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
            "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
            "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b18c384nbt-s8341979392-d3881113763.bin.gz",
            "maxVisits": f"{yi_int}",
            "numSearchThreads": 32,
            "maxTime": 1e20,
        }

        id = np.random.randint(10000)
        sgffile_prefix = f"maxVisits-{xi_int}-{yi_int}-{id}"

        if np.random.randn() < 0.5:
            black_win, white_win, _ = match_games(
                bot_a_parameters,
                bot_b_parameters,
                game_count_start=0,
                games=1,
                sgffile_prefix=sgffile_prefix,
                verbose=False,
            )

            bot_b_win = white_win
        else:
            black_win, white_win, _ = match_games(
                bot_b_parameters,
                bot_a_parameters,
                game_count_start=0,
                games=1,
                sgffile_prefix=sgffile_prefix,
                verbose=False,
            )

            bot_b_win = black_win

        bot_b_wins.append(bot_b_win)

        if bot_b_win > 0:
            print(
                f"{bot_a_name} MaxVisits={xi_int} vs {bot_b_name} MaxVisits={yi_int}: {bot_b_name} won, ID:{id}"
            )
        else:
            print(
                f"{bot_a_name} MaxVisits={xi_int} vs {bot_b_name} MaxVisits={yi_int}: {bot_a_name} won, ID:{id}"
            )

    return bot_b_wins


# Define the simulation function
def simulation_function(x, y):
    x_hat = [xi + 20.0 * np.random.randn() for xi in x]
    y_hat = [yi + 20.0 * np.random.randn() for yi in y]
    z_hat = [(yi > 0.5 * xi).astype(int) for (xi, yi) in zip(x_hat, y_hat)]
    return z_hat


# Generate initial samples
def generate_initial_samples(black_box_function, N_init, x_min, x_max, y_min, y_max):
    N_init = max(1, N_init - 2)
    log_x_min = math.log(x_min)
    log_x_max = math.log(x_max)
    log_x_init = np.random.uniform(log_x_min, log_x_max, N_init)
    log_x_init = np.append(log_x_init, log_x_min)
    log_x_init = np.append(log_x_init, log_x_max)
    x_init = np.exp(log_x_init)
    log_y_min = math.log(y_min)
    log_y_max = math.log(y_max)
    log_y_init = np.random.uniform(log_y_min, log_y_max, N_init)
    log_y_init = np.append(log_y_init, log_y_max)
    log_y_init = np.append(log_y_init, log_y_min)
    y_init = np.exp(log_y_init)
    X_init = np.column_stack((x_init, y_init))
    y_label_init = black_box_function(x_init, y_init)
    return X_init, y_label_init


# Generate candidate points
def generate_candidates(N_cand, x_min, x_max, y_min, y_max):
    log_x_min = math.log(x_min)
    log_x_max = math.log(x_max)
    log_x_cand = np.random.uniform(log_x_min, log_x_max, N_cand)
    x_cand = np.exp(log_x_cand)
    log_y_min = math.log(y_min)
    log_y_max = math.log(y_max)
    log_y_cand = np.random.uniform(log_y_min, log_y_max, N_cand)
    y_cand = np.exp(log_y_cand)
    return np.column_stack((x_cand, y_cand))


# Expected Improvement calculation
def compute_expected_improvement(X, model, temperature=0.0):
    probas = model.predict_proba(np.log(X))
    uncertainty = np.min(probas, axis=1)
    uncertainty = uncertainty + temperature * np.random.rand(np.shape(uncertainty)[0])

    return uncertainty


# Select top K points based on EI
def select_top_k(X, ei, K):
    return X[np.argsort(ei)[-K:]]


# Fit decision boundary
def fit_decision_boundary(black_box_function, x_min, x_max, y_min, y_max, N):
    N_init = max(min(30, N), int(0.05 * N))
    K = 2
    N_iter = int((N - N_init) / K)

    # Initialize Logistic Regression
    model = LogisticRegression()

    # Initial sampling
    X_init, y_label_init = generate_initial_samples(
        black_box_function, N_init, x_min, x_max, y_min, y_max
    )
    model.fit(np.log(X_init), y_label_init)

    # Active Learning Loop
    for iteration in range(N_iter):
        N_cand = 50
        X_cand = generate_candidates(N_cand, x_min, x_max, y_min, y_max)
        temperature = 3.0 * (N_iter - iteration) / N_iter
        ei = compute_expected_improvement(X_cand, model, temperature=temperature)
        X_select = select_top_k(X_cand, ei, K)
        print(f"Iteration: {iteration + 1}/{N_iter} N_cand={N_cand}")
        y_label_select = black_box_function(X_select[:, 0], X_select[:, 1])
        X_init = np.vstack([X_init, X_select])
        y_label_init = np.hstack([y_label_init, y_label_select])
        model.fit(np.log(X_init), y_label_init)

    return X_init, y_label_init, model


# Plot decision boundary
def plot_decision_boundary(model, x_min, x_max, y_min, y_max, X_init, y_label_init):
    # Visualizing the decision boundary on log-log scales
    x_log_space = np.logspace(np.log10(x_min), np.log10(x_max), 50)
    y_log_space = np.logspace(np.log10(y_min), np.log10(y_max), 50)
    x_grid, y_grid = np.meshgrid(x_log_space, y_log_space)
    X_grid_log = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    y_pred_grid_log = model.predict(np.log(X_grid_log)).reshape(x_grid.shape)
    bot_a_name, bot_b_name = get_bot_names()

    plt.figure(figsize=(10, 6))
    plt.contourf(
        x_grid, y_grid, y_pred_grid_log, alpha=0.5, levels=[0, 0.5, 1], cmap="coolwarm"
    )
    plt.scatter(
        X_init[:, 0],
        X_init[:, 1],
        c=y_label_init,
        cmap="coolwarm",
        edgecolor="k",
        label=f"Cool color: {bot_a_name} Won. Warm color: {bot_b_name} Won.",
    )
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.title("Decision Boundary of Max Visits per Move")
    plt.xlabel(f"Max visits per move for {bot_a_name}")
    plt.ylabel(f"Max visits per move for {bot_b_name}")
    plt.legend()
    plt.ion()
    plt.show()
    plt.savefig("boundary.png")


if __name__ == "__main__":
    t0 = time.time()
    x_min, x_max = 32, 1024
    y_min, y_max = x_min, x_max
    N = 512
    # test_function = simulation_function
    test_function = match_function
    X_init, y_label_init, model = fit_decision_boundary(
        test_function, x_min, x_max, y_min, y_max, N
    )
    elapsed = time.time() - t0
    print(f"Elapsed: {str(datetime.timedelta(seconds=round(elapsed)))}")
    plot_decision_boundary(model, x_min, x_max, y_min, y_max, X_init, y_label_init)
