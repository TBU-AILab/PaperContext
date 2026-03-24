import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
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

    # ---- SHADE with Linear Population Size Reduction (L-SHADE) ----
    pop_size_init = min(max(40, 12 * dim), 400)
    pop_size_min = 4
    pop_size = pop_size_init
    
    population = lhs_sample(pop_size)
    fitness = np.array([evaluate(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_sol = population[best_idx].copy()
    best = fitness[best_idx]
    
    # Success history
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size_init
    
    generation = 0
    max_evals_estimate = 100000  # rough estimate, adjusted dynamically
    total_evals_at_start = evals
    
    stagnation_counter = 0
    prev_best = best
    
    # Track time per eval for budget estimation
    time_per_eval = elapsed() / max(evals, 1)
    
    while time_ok():
        generation += 1
        
        # Estimate remaining budget for LPSR
        remaining_time = max_time * 0.95 - elapsed()
        if evals > pop_size_init:
            time_per_eval = elapsed() / evals
        est_remaining_evals = remaining_time / max(time_per_eval, 1e-9)
        total_est_evals = evals + est_remaining_evals
        
        # Linear population size reduction
        ratio = evals / max(total_est_evals, 1)
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
            
            # Cauchy for F
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 1.5:
                    Fi = 1.5
            Fi = min(Fi, 1.5)
            
            # Normal for CR
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best
            p = max(2, int(max(0.05, 0.2 - 0.15 * ratio) * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p)]
            
            # r1
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Bounce-back
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
        
        # Selection
        improved = trial_fit <= fitness
        population[improved] = trial_pop[improved]
        fitness[improved] = trial_fit[improved]
        
        # Update history
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        # Reduce population
        if new_pop_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_pop_size]]
            fitness = fitness[si[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            archive_max = pop_size
        
        # Stagnation restart
        if abs(prev_best - best) < 1e-15:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best = best
        
        if stagnation_counter > max(50, 3 * dim) and time_ok():
            keep = max(1, pop_size // 5)
            si = np.argsort(fitness)
            pop_size = min(pop_size_init // 2, max(20, 6 * dim))
            new_pop = lhs_sample(pop_size)
            new_fit = np.array([evaluate(x) for x in new_pop[:pop_size - keep] if time_ok()])
            if len(new_fit) < pop_size - keep:
                return best
            population = np.vstack([population[si[:keep]], new_pop[:pop_size - keep]])
            fitness = np.concatenate([fitness[si[:keep]], new_fit])
            stagnation_counter = 0
            M_F[:] = 0.5
            M_CR[:] = 0.5
            archive.clear()
    
    return best