#Here's the fixed, complete, self-contained code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Latin Hypercube Sampling for initial population
    pop_size = min(20 * dim, 200)
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / pop_size * (upper[i] - lower[i])

    fitness = np.array([evaluate(population[j]) for j in range(pop_size)])

    if elapsed() >= max_time:
        return best

    # Phase 2: Differential Evolution
    F = 0.8
    CR = 0.9

    while elapsed() < max_time * 0.7:
        for i in range(pop_size):
            if elapsed() >= max_time * 0.7:
                break
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = clip(a + F * (b - c))
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = clip(trial)
            f_trial = evaluate(trial)
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial

    # Phase 3: Nelder-Mead simplex from best found solution
    if best_params is None:
        return best

    n = dim
    # Initialize simplex around best point
    simplex = np.zeros((n + 1, n))
    simplex[0] = best_params.copy()
    step = 0.05 * (upper - lower)
    for i in range(n):
        point = best_params.copy()
        point[i] = point[i] + step[i]
        if point[i] > upper[i]:
            point[i] = best_params[i] - step[i]
        simplex[i + 1] = clip(point)

    f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1)])

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    while elapsed() < max_time * 0.98:
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]

        centroid = simplex[:-1].mean(axis=0)

        # Reflection
        xr = clip(centroid + alpha * (centroid - simplex[-1]))
        fr = evaluate(xr)

        if fr < f_simplex[-2] and fr >= f_simplex[0]:
            simplex[-1] = xr
            f_simplex[-1] = fr
            continue

        if fr < f_simplex[0]:
            xe = clip(centroid + gamma * (xr - centroid))
            fe = evaluate(xe)
            if fe < fr:
                simplex[-1] = xe
                f_simplex[-1] = fe
            else:
                simplex[-1] = xr
                f_simplex[-1] = fr
            continue

        # Contraction
        xc = clip(centroid + rho * (simplex[-1] - centroid))
        fc = evaluate(xc)
        if fc < f_simplex[-1]:
            simplex[-1] = xc
            f_simplex[-1] = fc
            continue

        # Shrink
        for i in range(1, n + 1):
            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
            f_simplex[i] = evaluate(simplex[i])

    return best
