import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a simple (1+1)-ES with annealed step-size,
    plus occasional random restarts.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/sequence of length dim.
    dim : int
    bounds : list of (low, high)
    max_time : int/float seconds

    Returns
    -------
    best : float
        Best (minimum) fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # Helpers
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [max(1e-12, highs[i] - lows[i]) for i in range(dim)]

    def clip_to_bounds(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Initial solution
    x = rand_point()
    fx = func(x)
    best = fx

    # Step size per dimension (as fraction of range)
    # Start fairly large, cool down over time.
    base_sigma = 0.2  # fraction of range initially

    # Restart control
    no_improve = 0
    restart_after = 200  # iterations without improvement triggers restart
    it = 0

    while time.time() < deadline:
        it += 1

        # Normalized progress in [0,1]
        p = (time.time() - t0) / max(1e-12, (deadline - t0))
        if p > 1.0:
            break

        # Exponential cooling of step size (keeps exploration early, exploitation late)
        cool = math.exp(-3.0 * p)  # from ~1.0 down to ~0.05
        sigma_frac = base_sigma * cool

        # Create candidate by Gaussian perturbation in each dimension
        y = x[:]  # copy
        for i in range(dim):
            step = random.gauss(0.0, sigma_frac * spans[i])
            y[i] += step
        y = clip_to_bounds(y)

        fy = func(y)

        # (1+1) selection: accept if improved
        if fy <= fx:
            x, fx = y, fy
            no_improve = 0
            if fy < best:
                best = fy
        else:
            no_improve += 1

        # Occasional random restart to escape local minima
        # Also inject a small probability of restart regardless of no_improve.
        if no_improve >= restart_after or random.random() < 0.002:
            z = rand_point()
            fz = func(z)
            x, fx = z, fz
            no_improve = 0
            if fz < best:
                best = fz

    return best
