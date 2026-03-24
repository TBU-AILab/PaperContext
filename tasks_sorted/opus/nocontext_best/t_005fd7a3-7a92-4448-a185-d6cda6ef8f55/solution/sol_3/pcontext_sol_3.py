#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    def clip_to_bounds(x):
        return np.clip(x, lb, ub)

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def remaining():
        return max_time - elapsed()

    # Phase 1: L-SHADE algorithm
    NP_init = min(18 * dim, 200)
    NP_min = 4
    H = 6  # memory size
    NP = NP_init

    # Initialize population
    pop = np.random.uniform(lb, ub, (NP, dim))
    fitness = np.array([func(pop[i]) for i in range(NP)])

    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = pop[best_idx].copy()

    # Memory for CR and F
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k = 0  # memory index

    max_evals_estimate = NP_init * 100
    evals_used = NP
    generation = 0

    while remaining() > max_time * 0.15:
        if NP < NP_min:
            break

        S_CR = []
        S_F = []
        delta_f = []

        # Generate CR and F for each individual
        r_i = np.random.randint(0, H, NP)
        CR = np.zeros(NP)
        F = np.zeros(NP)

        for i in range(NP):
            mu_cr = M_CR[r_i[i]]
            if mu_cr < 0:
                CR[i] = 0
            else:
                CR[i] = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)

            mu_f = M_F[r_i[i]]
            fi = -1
            while fi <= 0:
                fi = min(np.random.standard_cauchy() * 0.1 + mu_f, 1.0)
            F[i] = fi

        # Sort population by fitness for p-best
        sorted_idx = np.argsort(fitness)

        trial_pop = np.empty_like(pop)
        trial_fitness = np.full(NP, float('inf'))

        # Archive (simplified: use current population as archive)
        archive = pop.copy()
        archive_fit = fitness.copy()

        for i in range(NP):
            if remaining() < max_time * 0.1:
                break

            # current-to-pbest/1
            p = max(2, int(np.round(0.11 * NP)))
            p_best_idx = sorted_idx[np.random.randint(0, p)]

            # Select r1 != i
            candidates = list(range(NP))
            candidates.remove(i)
            r1 = candidates[np.random.randint(0, len(candidates))]
            candidates.remove(r1)

            # r2 from pop + archive
            r2_pool_size = NP + len(archive)
            r2 = np.random.randint(0, r2_pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, r2_pool_size)

            if r2 < NP:
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - NP]

            mutant = pop[i] + F[i] * (pop[p_best_idx] - pop[i]) + F[i] * (pop[r1] - x_r2)
            mutant = clip_to_bounds(mutant)

            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.random() < CR[i] or j == j_rand:
                    trial[j] = mutant[j]

            trial_fitness[i] = func(trial)
            trial_pop[i] = trial
            evals_used += 1

            if trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    S_CR.append(CR[i])
                    S_F.append(F[i])
                    delta_f.append(abs(fitness[i] - trial_fitness[i]))

                if trial_fitness[i] < best:
                    best = trial_fitness[i]
                    best_x = trial.copy()

        # Selection
        for i in range(NP):
            if trial_fitness[i] <= fitness[i]:
                pop[i] = trial_pop[i]
                fitness[i] = trial_fitness[i]

        # Update memory
        if len(S_CR) > 0 and len(S_F) > 0:
            weights = np.array(delta_f)
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(len(delta_f)) / len(delta_f)

            s_cr = np.array(S_CR)
            s_f = np.array(S_F)

            mean_cr = np.sum(weights * s_cr)
            mean_f = np.sum(weights * s_f ** 2) / max(np.sum(weights * s_f), 1e-30)

            M_CR[k] = mean_cr
            M_F[k] = mean_f
            k = (k + 1) % H

        # Linear population size reduction
        generation += 1
        ratio = evals_used / max_evals_estimate
        new_NP = int(np.round(NP_init + (NP_min - NP_init) * min(ratio, 1.0)))
        new_NP = max(new_NP, NP_min)

        if new_NP < NP:
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:new_NP]]
            fitness = fitness[sorted_idx[:new_NP]]
            NP = new_NP

    # Phase 2: Local search with Nelder-Mead style simplex
    if best_x is not None and remaining() > 0.5:
        # Simple coordinate descent / pattern search
        x_curr = best_x.copy()
        f_curr = best

        step_sizes = (ub - lb) * 0.01

        while remaining() > 0.3:
            improved = False
            for d in range(dim):
                if remaining() < 0.2:
                    break

                for sign in [1.0, -1.0]:
                    x_trial = x_curr.copy()
                    x_trial[d] = np.clip(x_trial[d] + sign * step_sizes[d], lb[d], ub[d])
                    f_trial = func(x_trial)
                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        improved = True
                        break

            if f_curr < best:
                best = f_curr
                best_x = x_curr.copy()

            if not improved:
                step_sizes *= 0.5
                if np.max(step_sizes / (ub - lb + 1e-30)) < 1e-12:
                    break

    # Phase 3: Use remaining time for random restarts with local search
    while remaining() > 0.5:
        # Random point near best
        scale = 0.1 * (ub - lb)
        x_new = best_x + np.random.normal(0, 1, dim) * scale
        x_new = clip_to_bounds(x_new)
        f_new = func(x_new)

        if f_new < best:
            best = f_new
            best_x = x_new.copy()

        # Quick local search from this point
        x_curr = x_new.copy()
        f_curr = f_new
        step_sizes = (ub - lb) * 0.005

        for _ in range(3):
            if remaining() < 0.3:
                break
            for d in range(dim):
                if remaining() < 0.2:
                    break
                for sign in [1.0, -1.0]:
                    x_trial = x_curr.copy()
                    x_trial[d] = np.clip(x_trial[d] + sign * step_sizes[d], lb[d], ub[d])
                    f_trial = func(x_trial)
                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        break
            step_sizes *= 0.5

        if f_curr < best:
            best = f_curr
            best_x = x_curr.copy()

    return best
