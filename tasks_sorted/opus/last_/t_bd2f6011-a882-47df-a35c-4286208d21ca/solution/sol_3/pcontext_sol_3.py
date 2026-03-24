import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok():
        return elapsed() < max_time * 0.95
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def lhs_sample(n):
        result = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, d] = (perm[i] + np.random.random()) / n
        return lower + result * ranges
    
    def evaluate(x):
        nonlocal evals, best, best_sol
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    # ---- L-SHADE with restarts and local search ----
    pop_size_init = min(max(50, 15 * dim), 500)
    pop_size_min = 4
    pop_size = pop_size_init
    
    population = lhs_sample(pop_size)
    fitness = np.array([evaluate(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if best_sol is None:
        best_sol = population[best_idx].copy()
    
    # Success history
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size_init
    
    generation = 0
    
    stagnation_counter = 0
    prev_best = best
    
    time_per_eval = elapsed() / max(evals, 1)
    
    def nelder_mead_local(x0, budget_evals, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """Simple Nelder-Mead local search."""
        nonlocal best, best_sol
        n = dim
        if budget_evals < n + 2:
            return
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_vals = np.zeros(n + 1)
        f_vals[0] = evaluate(x0)
        
        step = ranges * 0.05
        for i in range(n):
            if not time_ok():
                return
            p = x0.copy()
            p[i] += step[i]
            p = clip(p)
            simplex[i + 1] = p
            f_vals[i + 1] = evaluate(p)
        
        used = n + 1
        
        while used < budget_evals and time_ok():
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            used += 1
            
            if fr < f_vals[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                used += 1
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            elif fr < f_vals[-2]:
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                # Contract
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    used += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if used >= budget_evals or not time_ok():
                                return
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_vals[i] = evaluate(simplex[i])
                            used += 1
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc)
                    used += 1
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if used >= budget_evals or not time_ok():
                                return
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_vals[i] = evaluate(simplex[i])
                            used += 1
            
            # Convergence check
            if np.max(np.abs(f_vals - f_vals[0])) < 1e-14:
                break
    
    restart_count = 0
    
    while time_ok():
        generation += 1
        
        remaining_time = max_time * 0.95 - elapsed()
        if evals > pop_size_init:
            time_per_eval = elapsed() / evals
        est_remaining_evals = remaining_time / max(time_per_eval, 1e-9)
        total_est_evals = evals + est_remaining_evals
        
        ratio = min(1.0, evals / max(total_est_evals, 1))
        new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        sorted_idx = np.argsort(fitness)
        
        trial_pop = np.empty_like(population)
        trial_fit = np.empty(pop_size)
        
        for i in range(pop_size):
            if not time_ok():
                return best
            
            ri = np.random.randint(0, H)
            
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 1.5:
                    Fi = 1.5
            Fi = min(Fi, 1.5)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(max(0.05, 0.25 - 0.20 * ratio) * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            for d2 in range(dim):
                if mutant[d2] < lower[d2]:
                    mutant[d2] = (lower[d2] + population[i][d2]) / 2
                elif mutant[d2] > upper[d2]:
                    mutant[d2] = (upper[d2] + population[i][d2]) / 2
            
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            tf = evaluate(trial)
            trial_pop[i] = trial
            trial_fit[i] = tf
            
            if tf <= fitness[i]:
                delta = fitness[i] - tf
                if delta > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive_max > 0:
                        archive[np.random.randint(len(archive))] = population[i].copy()
        
        improved = trial_fit <= fitness
        population[improved] = trial_pop[improved]
        fitness[improved] = trial_fit[improved]
        
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        if new_pop_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_pop_size]]
            fitness = fitness[si[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            archive_max = pop_size
        
        if abs(prev_best - best) < 1e-15:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best = best
        
        # Local search periodically
        if stagnation_counter > 0 and stagnation_counter % 20 == 0 and time_ok() and best_sol is not None:
            ls_budget = min(max(50, 5 * dim), int(est_remaining_evals * 0.1))
            nelder_mead_local(best_sol.copy(), ls_budget)
        
        # Full restart on heavy stagnation
        if stagnation_counter > max(80, 5 * dim) and time_ok():
            restart_count += 1
            # Do a local search before restart
            if best_sol is not None:
                ls_budget = min(max(100, 10 * dim), int(est_remaining_evals * 0.15))
                nelder_mead_local(best_sol.copy(), ls_budget)
            
            keep = max(1, min(3, pop_size // 5))
            pop_size = min(pop_size_init // (1 + restart_count // 3), max(20, 8 * dim))
            pop_size = max(pop_size, pop_size_min + keep)
            si = np.argsort(fitness)
            kept_pop = population[si[:keep]].copy()
            kept_fit = fitness[si[:keep]].copy()
            
            new_pop = lhs_sample(pop_size - keep)
            new_fit = []
            for x in new_pop:
                if not time_ok():
                    return best
                new_fit.append(evaluate(x))
            new_fit = np.array(new_fit)
            
            population = np.vstack([kept_pop, new_pop])
            fitness = np.concatenate([kept_fit, new_fit])
            stagnation_counter = 0
            M_F[:] = 0.5
            M_CR[:] = 0.5
            archive.clear()
            archive_max = pop_size
    
    # Final local search with remaining time
    if best_sol is not None and time_ok():
        remaining = max_time * 0.95 - elapsed()
        if remaining > 0.1 and time_per_eval > 0:
            ls_budget = int(remaining / max(time_per_eval, 1e-9))
            ls_budget = min(ls_budget, max(200, 20 * dim))
            if ls_budget > dim + 2:
                nelder_mead_local(best_sol.copy(), ls_budget)
    
    return best