#That syntax error is caused by an invalid list comprehension:
#
highs = [float(b[1]) for_phi=bounds]
#
#It should be:
#
highs = [float(b[1]) for b in bounds]
#
#Below is a fully self-contained, working `run()` implementation (no external libraries like numpy). It uses a simple time-bounded hybrid of random search + local Gaussian perturbation around the current best.
#
import time
import random
import math

def run(func, dim, bounds, max_time):
    start = time.time()

    # Pre-extract bounds (FIXED LIST COMPREHENSION)
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]

    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def random_point():
        return [random.uniform(lows[i], highs[i]) for i in range(dim)]

    # Initial best from random point
    best_x = random_point()
    best = func(best_x)

    # Step sizes relative to bound ranges
    ranges = [highs[i] - lows[i] for i in range(dim)]
    sigma = [0.2 * r if r > 0 else 1.0 for r in ranges]  # start fairly global

    it = 0
    while True:
        if time.time() - start >= max_time:
            return best

        it += 1

        # Mix of global random samples and local improvements
        if it % 5 == 0:
            x = random_point()
        else:
            # Local perturbation around best_x
            x = []
            for i in range(dim):
                # Gaussian step; anneal slowly over time
                step = random.gauss(0.0, sigma[i])
                xi = clamp(best_x[i] + step, lows[i], highs[i])
                x.append(xi)

        f = func(x)

        if f <= best:
            best = f
            best_x = x

            # If improving, slightly expand local search to move faster
            for i in range(dim):
                sigma[i] = min(sigma[i] * 1.05, ranges[i] if ranges[i] > 0 else sigma[i])
        else:
            # If not improving, slowly contract step sizes
            for i in range(dim):
                sigma[i] *= 0.995
                # Keep a minimum sigma to avoid stagnation
                min_sigma = 1e-12 if ranges[i] == 0 else (1e-6 * ranges[i])
                if sigma[i] < min_sigma:
                    sigma[i] = min_sigma
