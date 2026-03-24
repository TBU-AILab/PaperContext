#Looking at the results, the best algorithm (26.66) uses SHADE with coordinate descent local search. I'll improve it with:
#
#1. **Better r2 selection** (the current code has a bug-prone r2 selection)
#2. **Multiple diverse restarts** in local search phase using top population members
#3. **Golden section line search** along promising directions
#4. **Faster boundary handling** using vectorized operations
#5. **Better time allocation** and a hybrid local search combining coordinate descent with random direction search
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
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

    # --- Phase 1: SHADE ---
    pop_size = min(max(30, 8 * dim), 300)
    H = 100

    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.array([evaluate(population[i]) for i in range(pop_size)])
    if elapsed() >= max_time * 0.95:
        return best

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    # Store top candidates for later local search
    top_candidates = []
    
    while elapsed() < max_time * 0.60:
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness_vals)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.60:
                break
            
            ri = np.random.randint(H)
            
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 2.0:
                    Fi = -1
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            # Select r1 != i
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i:
                r1 += 1
            
            # Select r2 from pop + archive, != i, != r1
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size - 2)
            if r2 >= min(i, r1):
                r2 += 1
            if r2 >= max(i, r1):
                r2 += 1
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                if f_trial < fitness_vals[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness_vals[i] - f_trial))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive_max > 0:
                        archive[np.random.randint(archive_max)] = population[i].copy()
                population[i] = trial
                fitness_vals[i] = f_trial
        
        if S_F:
            weights = np.array(delta_f)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(len(delta_f)) / len(delta_f)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k % H] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[k % H] = np.sum(weights * scr)
            k += 1
    
    # Collect top candidates
    sidx = np.argsort(fitness_vals)
    n_top = min(5, pop_size)
    top_candidates = [(population[sidx[i]].copy(), fitness_vals[sidx[i]]) for i in range(n_top)]
    
    # --- Phase 2: Multi-start local search ---
    def local_search(x0, fx0, time_frac):
        x = x0.copy()
        fx = fx0
        step = 0.05 * ranges.copy()
        
        while elapsed() < max_time * time_frac:
            improved = False
            # Coordinate descent
            perm = np.random.permutation(dim)
            for j in perm:
                if elapsed() >= max_time * time_frac:
                    break
                for sign in [1, -1]:
                    x_new = x.copy()
                    x_new[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                    f_new = evaluate(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        step[j] *= 1.3
                        improved = True
                        break
            
            if not improved:
                # Try random direction
                d = np.random.randn(dim)
                d /= (np.linalg.norm(d) + 1e-30)
                alpha_s = np.mean(step) * 2
                x_new = clip(x + alpha_s * d)
                f_new = evaluate(x_new)
                if f_new < fx:
                    x = x_new
                    fx = f_new
                else:
                    step *= 0.5
                    if np.max(step / ranges) < 1e-13:
                        break
        return x, fx
    
    time_per_candidate = 0.35 / max(n_top, 1)
    for idx, (cx, cf) in enumerate(top_candidates):
        t_end = min(0.60 + (idx + 1) * time_per_candidate, 0.95)
        local_search(cx, cf, t_end)
        if elapsed() >= max_time * 0.95:
            return best
    
    # Final polish on absolute best
    if best_params is not None:
        local_search(best_params.copy(), best, 0.98)
    
    return best
