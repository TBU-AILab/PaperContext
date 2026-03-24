import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower

    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(100 * dim, 2000)
    for i in range(n_init):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        fitness = func(params)
        if fitness < best:
            best = fitness
            best_x = params.copy()

    if best_x is None:
        best_x = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

    # Phase 2: CMA-ES inspired search
    pop_size = max(10, 4 + int(3 * np.log(dim)))
    sigma = 0.3
    mean = best_x.copy()
    cov = np.eye(dim)

    # Evolution loop
    generation = 0
    stagnation = 0
    prev_best = best

    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best

        # Generate population
        population = []
        fitnesses = []
        for j in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best

            z = np.random.randn(dim)
            x = mean + sigma * ranges * z
            x = np.clip(x, lower, upper)
            f = func(x)
            population.append(x)
            fitnesses.append(f)
            if f < best:
                best = f
                best_x = x.copy()

        # Select top half
        indices = np.argsort(fitnesses)
        mu = pop_size // 2
        selected = [population[i] for i in indices[:mu]]

        # Update mean
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        new_mean = np.zeros(dim)
        for k in range(mu):
            new_mean += weights[k] * selected[k]
        mean = new_mean

        # Adapt sigma
        if best < prev_best:
            sigma *= 1.0
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > 10:
                sigma *= 0.8
                if sigma < 1e-8:
                    sigma = 0.3
                    mean = best_x.copy() + 0.1 * ranges * np.random.randn(dim)
                    mean = np.clip(mean, lower, upper)
                    stagnation = 0

        prev_best = best
        generation += 1

        # Phase 3: Local refinement with decreasing perturbation
        if generation % 20 == 0:
            for scale in [0.01, 0.001, 0.0001]:
                for _ in range(10):
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.95):
                        return best
                    x0 = np.clip(
                        best_x + scale * ranges * np.random.randn(dim),
                        lower,
                        upper
                    )
                    f0 = func(x0)
                    if f0 < best:
                        best = f0
                        best_x = x0.copy()

    return best