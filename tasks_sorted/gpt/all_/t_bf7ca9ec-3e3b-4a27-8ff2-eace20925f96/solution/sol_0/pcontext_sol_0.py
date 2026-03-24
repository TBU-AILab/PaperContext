import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer: (1+1)-Evolution Strategy with
    adaptive step-size (1/5 success rule) + occasional random restarts.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a sequence of length dim.
    dim : int
        Dimensionality of input vector.
    bounds : list of (low, high)
        Box constraints per dimension.
    max_time : int or float
        Time budget in seconds.

    Returns
    -------
    best : float
        Best (minimum) fitness found within the time budget.
    """

    t0 = time.time()

    # Helpers
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        # Project into bounds
        y = list(x)
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Initial solution
    x = rand_point()
    fx = func(x)
    best = fx

    # Initial step size: fraction of average span
    avg_span = sum(spans) / float(dim) if dim > 0 else 1.0
    sigma = 0.2 * avg_span if avg_span > 0 else 1.0
    sigma_min = 1e-12 * (avg_span if avg_span > 0 else 1.0)
    sigma_max = 1e3 * (avg_span if avg_span > 0 else 1.0)

    # 1/5 success rule parameters
    window = 30                  # adaptation window
    successes = 0
    trials = 0
    adapt_up = 1.5
    adapt_down = 1.5

    # Restart logic
    last_improve_time = time.time()
    restart_patience = max(0.05 * max_time, 0.5)  # seconds without improvement
    restart_sigma_factor = 0.2

    # Main loop
    while True:
        if time.time() - t0 >= max_time:
            return best

        # Propose mutation (Gaussian)
        # Use Box-Muller via random.gauss (standard library)
        x_new = [x[i] + random.gauss(0.0, sigma) for i in range(dim)]
        x_new = clamp(x_new)

        f_new = func(x_new)

        trials += 1
        if f_new <= fx:
            x, fx = x_new, f_new
            successes += 1
            if fx < best:
                best = fx
                last_improve_time = time.time()

        # Adapt sigma every 'window' trials
        if trials >= window:
            rate = successes / float(trials)
            if rate > 0.2:
                sigma *= adapt_up
            else:
                sigma /= adapt_down

            # Keep sigma in reasonable range
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            successes = 0
            trials = 0

        # Random restart if stagnating
        if time.time() - last_improve_time >= restart_patience:
            x = rand_point()
            fx = func(x)
            if fx < best:
                best = fx
            # reset sigma to moderate scale
            sigma = max(sigma_min, restart_sigma_factor * avg_span)
            last_improve_time = time.time()
