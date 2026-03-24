#Here's the fixed, self-contained code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def eval_point(x):
        nonlocal best
        f = func(x)
        if f < best:
            best = f
        return f

    # Phase 1: Latin Hypercube Sampling initialization
    pop_size = min(20 * dim, 200)
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / pop_size * ranges[i]

    fitness = np.array([eval_point(population[j]) for j in range(pop_size)])

    if elapsed() >= max_time:
        return best

    # Phase 2: Differential Evolution with restarts and local search
    F = 0.8
    CR = 0.9

    best_idx = np.argmin(fitness)
    global_best_x = population[best_idx].copy()
    global_best_f = fitness[best_idx]

    generation = 0
    while elapsed() < max_time * 0.85:
        generation += 1
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break

            # Mutation: DE/best/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            mutant = clip(global_best_x + F * (population[a] - population[b]))

            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            trial = clip(trial)
            trial_f = eval_point(trial)

            if trial_f <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_f
                if trial_f < global_best_f:
                    global_best_f = trial_f
                    global_best_x = trial.copy()

    # Phase 3: Nelder-Mead style local search around best
    x_best = global_best_x.copy()
    f_best = global_best_f

    step_size = 0.01
    while elapsed() < max_time * 0.98:
        improved = False
        for di in range(dim):
            if elapsed() >= max_time * 0.98:
                break
            for sign in [1.0, -1.0]:
                x_new = x_best.copy()
                x_new[di] = np.clip(
                    x_best[di] + sign * step_size * ranges[di],
                    lower[di],
                    upper[di]
                )
                f_new = eval_point(x_new)
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
                    improved = True
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-12:
                break

    return best
