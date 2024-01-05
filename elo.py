from match import elo, elo_range


if __name__ == "__main__":
    M = 67
    N = 140
    stdev1 = [f"{r:.1f}" for r in elo_range(M, N, 1.0)]
    stdev2 = [f"{r:.1f}" for r in elo_range(M, N, 2.0)]
    stdev3 = [f"{r:.1f}" for r in elo_range(M, N, 3.0)]

    print(f"Expected ELO = {elo(M, N):.1f}")
    print(f"ELO range (+/- 1.0 standard deviation) = {stdev1}")
    print(f"ELO range (+/- 2.0 standard deviation) = {stdev2}")
    print(f"ELO range (+/- 3.0 standard deviation) = {stdev3}")
