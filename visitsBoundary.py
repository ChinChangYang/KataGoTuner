import datetime
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from match import match_games

# Define the match function
def match_function(x, y):
    bot_b_wins = []
    for (xi, yi) in zip(x, y):
        xi_int = int(xi)
        yi_int = int(yi)
        bot_a_name = 'b18c384nbt-s7709731328'

        bot_a_parameters = {
            "exe": "/Users/chinchangyang/Code/KataGo/cpp/build/katago",
            "config": "/Users/chinchangyang/.katago/default_gtp.cfg",
            "model": "/Users/chinchangyang/Code/KataGo-Models/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz",
            "maxVisits": f'{xi_int}'
        }

        bot_b_name = 'b28c512nbt-s1436726784'

        bot_b_parameters = {
            "exe": "/Users/chinchangyang/Code/KataGo-CCY/cpp/build/katago",
            "config": "/Users/chinchangyang/Code/KataGo-CCY/cpp/configs/misc/coreml_example.cfg",
            "model": "/Users/chinchangyang/Code/KataGo-Models/b28c512nbt-s1436726784-d3907069532.bin.gz",
            "maxVisits": f'{yi_int}'
        }

        id = np.random.randint(10000)
        sgffile_prefix = f'maxVisits-{xi_int}-{yi_int}-{id}'

        if np.random.randn() < 0.5:
            black_win, white_win, _ = match_games(
                bot_a_parameters,
                bot_b_parameters,
                game_count_start=0,
                games=1,
                sgffile_prefix=sgffile_prefix,
                verbose=False)
            
            bot_b_win = white_win
        else:
            black_win, white_win, _ = match_games(
                bot_b_parameters,
                bot_a_parameters,
                game_count_start=0,
                games=1,
                sgffile_prefix=sgffile_prefix,
                verbose=False)

            bot_b_win = black_win

        bot_b_wins.append(bot_b_win)

        if bot_b_win > 0:
            print(f'{bot_a_name} MaxVisits={xi_int} vs {bot_b_name} MaxVisits={yi_int}: {bot_b_name} won, ID:{id}')
        else:
            print(f'{bot_a_name} MaxVisits={xi_int} vs {bot_b_name} MaxVisits={yi_int}: {bot_a_name} won, ID:{id}')

    return bot_b_wins

# Define the simulation function
def simulation_function(x, y):
    x_hat = [xi + 10.0 * np.random.randn() for xi in x]
    y_hat = [yi + 10.0 * np.random.randn() for yi in y]
    z_hat = [(yi > 0.5 * xi).astype(int) for (xi, yi) in zip(x_hat, y_hat)]
    return z_hat

# Generate initial samples
def generate_initial_samples(black_box_function, N_init, x_min, x_max, y_min, y_max):
    x_init = np.random.uniform(x_min, x_max, N_init)
    y_init = np.random.uniform(y_min, y_max, N_init)
    X_init = np.column_stack((x_init, y_init))
    y_label_init = black_box_function(x_init, y_init)
    return X_init, y_label_init

# Generate candidate points
def generate_candidates(N_cand, x_min, x_max, y_min, y_max):
    x_cand = np.random.uniform(x_min, x_max, N_cand)
    y_cand = np.random.uniform(y_min, y_max, N_cand)
    return np.column_stack((x_cand, y_cand))

# Expected Improvement calculation
def compute_expected_improvement(X, model):
    probas = model.predict_proba(X)
    uncertainty = np.min(probas, axis=1)
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
    X_init, y_label_init = generate_initial_samples(black_box_function, N_init, x_min, x_max, y_min, y_max)
    model.fit(X_init, y_label_init)

    # Active Learning Loop
    for iteration in range(N_iter):
        N_cand = int(K + 0.05 * (iteration + 1))
        X_cand = generate_candidates(N_cand, x_min, x_max, y_min, y_max)
        ei = compute_expected_improvement(X_cand, model)
        X_select = select_top_k(X_cand, ei, K)
        y_label_select = black_box_function(X_select[:, 0], X_select[:, 1])
        model.fit(np.vstack([X_init, X_select]), np.hstack([y_label_init, y_label_select]))
        X_init = np.vstack([X_init, X_select])
        y_label_init = np.hstack([y_label_init, y_label_select])

    return X_init, y_label_init, model

# Plot decision boundary
def plot_decision_boundary(model, x_min, x_max, y_min, y_max):
    # Visualizing the decision boundary
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    X_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    y_pred_grid = model.predict(X_grid).reshape(x_grid.shape)
    bot_a_name = 'b18c384nbt-s7709731328'
    bot_b_name = 'b28c512nbt-s1436726784'

    plt.figure(figsize=(10, 6))
    plt.contourf(x_grid, y_grid, y_pred_grid, alpha=0.5, levels=[0,0.5,1], cmap='coolwarm')
    plt.scatter(X_init[:, 0], X_init[:, 1], c=y_label_init, cmap='coolwarm', edgecolor='k', label=f'Warm color: {bot_b_name} Won. Cool color: {bot_b_name} Lost)')
    plt.title('Decision Boundary of Max Visits per Move')
    plt.xlabel(f'Max visits per move for {bot_a_name}')
    plt.ylabel(f'Max visits per move for {bot_b_name}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    x_min, x_max = 1, 32
    y_min, y_max = 1, 32
    N = 32
    test_function = match_function
    X_init, y_label_init, model = fit_decision_boundary(test_function, x_min, x_max, y_min, y_max, N)
    plot_decision_boundary(model, x_min, x_max, y_min, y_max)
    elapsed = time.time() - t0
    print(f'Elapsed: {str(datetime.timedelta(seconds=round(elapsed)))}')
