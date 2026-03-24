#
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

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    pop_size = min(20 * dim, 200)
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / pop_size * ranges[i]

    fitness = np.array([func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = fitness[best_idx]
    best_x = population[best_idx].copy()

    if elapsed() >= max_time:
        return best

    # --- Phase 2: Differential Evolution with restarts ---
    F = 0.8
    CR = 0.9
    generation = 0

    while elapsed() < max_time * 0.85:
        # Adaptive parameters
        F_val = 0.5 + 0.3 * np.random.random()
        CR_val = 0.8 + 0.2 * np.random.random()

        new_population = np.copy(population)
        new_fitness = np.copy(fitness)

        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break

            # DE/current-to-best/1
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)

            mutant = population[i] + F_val * (best_x - population[i]) + F_val * (population[r1] - population[r2])
            mutant = clip(mutant)

            # Crossover
            cross_points = np.random.random(dim) < CR_val
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = clip(trial)

            trial_fitness = func(trial)
            if trial_fitness <= new_fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()

        population = new_population
        fitness = new_fitness
        generation += 1

        # Occasional population diversity injection
        if generation % 50 == 0:
            worst_indices = np.argsort(fitness)[-max(pop_size // 5, 1):]
            for idx in worst_indices:
                population[idx] = lower + np.random.random(dim) * ranges
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()

    # --- Phase 3: Local refinement with Nelder-Mead-like search ---
    simplex_size = dim + 1
    simplex = np.zeros((simplex_size, dim))
    simplex[0] = best_x.copy()
    step = ranges * 0.01
    for i in range(1, simplex_size):
        simplex[i] = best_x.copy()
        simplex[i][(i - 1) % dim] += step[(i - 1) % dim]
        simplex[i] = clip(simplex[i])

    simplex_fit = np.array([func(s) for s in simplex])
    for i in range(simplex_size):
        if simplex_fit[i] < best:
            best = simplex_fit[i]
            best_x = simplex[i].copy()

    while elapsed() < max_time * 0.99:
        order = np.argsort(simplex_fit)
        simplex = simplex[order]
        simplex_fit = simplex_fit[order]

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        reflected = clip(centroid + 1.0 * (centroid - simplex[-1]))
        ref_fit = func(reflected)
        if ref_fit < best:
            best = ref_fit
            best_x = reflected.copy()

        if ref_fit < simplex_fit[-2]:
            if ref_fit < simplex_fit[0]:
                expanded = clip(centroid + 2.0 * (reflected - centroid))
                exp_fit = func(expanded)
                if exp_fit < best:
                    best = exp_fit
                    best_x = expanded.copy()
                if exp_fit < ref_fit:
                    simplex[-1] = expanded
                    simplex_fit[-1] = exp_fit
                else:
                    simplex[-1] = reflected
                    simplex_fit[-1] = ref_fit
            else:
                simplex[-1] = reflected
                simplex_fit[-1] = ref_fit
        else:
            contracted = clip(centroid + 0.5 * (simplex[-1] - centroid))
            con_fit = func(contracted)
            if con_fit < best:
                best = con_fit
                best_x = contracted.copy()
            if con_fit < simplex_fit[-1]:
                simplex[-1] = contracted
                simplex_fit[-1] = con_fit
            else:
                for i in range(1, simplex_size):
                    simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    simplex_fit[i] = func(simplex[i])
                    if simplex_fit[i] < best:
                        best = simplex_fit[i]
                        best_x = simplex[i].copy()

    return best
