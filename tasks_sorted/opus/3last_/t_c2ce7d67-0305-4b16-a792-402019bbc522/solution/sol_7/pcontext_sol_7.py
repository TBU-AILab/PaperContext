#Looking at the results, algorithm 2 (simple DE with coordinate descent) got 103.8 which was best, while the complex SHADE+CMA-ES approaches got ~200. This suggests the problem may benefit from a more straightforward but well-tuned approach. The complex algorithms likely spent too much time on overhead or got stuck in suboptimal configurations.
#
#Key insights:
#1. Simpler DE with good exploitation worked better
#2. Need better balance between exploration and exploitation
#3. Multiple restart strategies with different methods
#4. Add Nelder-Mead simplex which is good for local refinement
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    evals = 0
    def eval_f(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Latin Hypercube Sampling
    n_init = min(500, max(80, dim * 10))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    init_samples = (perms + np.random.rand(n_init, dim)) / n_init
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.05:
            break
        x = lower + init_samples[i] * ranges
        eval_f(x)

    # Phase 2: Multiple DE populations with different strategies
    def run_de(time_budget_frac, strategy='best'):
        nonlocal best, best_x
        pop_size = min(max(6 * dim, 40), 200)
        pop = lower + np.random.rand(pop_size, dim) * ranges
        fit = np.zeros(pop_size)
        
        # Seed with best known
        if best_x is not None:
            pop[0] = best_x.copy()
            # Add perturbations of best
            for j in range(1, min(pop_size // 4, 10)):
                pop[j] = np.clip(best_x + 0.1 * ranges * np.random.randn(dim), lower, upper)
        
        for i in range(pop_size):
            if remaining() < max_time * 0.1:
                return
            fit[i] = eval_f(pop[i])
        
        end_time = elapsed() + max_time * time_budget_frac
        
        # Adaptive parameters (jDE style)
        F_arr = np.full(pop_size, 0.5)
        CR_arr = np.full(pop_size, 0.9)
        
        gen = 0
        stagnation = 0
        prev_best = best
        
        while elapsed() < end_time and remaining() > max_time * 0.08:
            gen += 1
            improved_gen = False
            
            # Sort population
            sort_idx = np.argsort(fit)
            
            for i in range(pop_size):
                if elapsed() >= end_time or remaining() < max_time * 0.08:
                    return
                
                # Self-adaptive F and CR (jDE)
                if np.random.rand() < 0.1:
                    F_i = 0.1 + 0.9 * np.random.rand()
                else:
                    F_i = F_arr[i]
                if np.random.rand() < 0.1:
                    CR_i = np.random.rand()
                else:
                    CR_i = CR_arr[i]
                
                indices = list(range(pop_size))
                indices.remove(i)
                
                if strategy == 'best':
                    # DE/current-to-best/1
                    a, b = np.random.choice(indices, 2, replace=False)
                    best_idx = sort_idx[0]
                    mutant = pop[i] + F_i * (pop[best_idx] - pop[i]) + F_i * (pop[a] - pop[b])
                elif strategy == 'pbest':
                    p = max(2, int(0.1 * pop_size))
                    pb = sort_idx[np.random.randint(p)]
                    a, b = np.random.choice(indices, 2, replace=False)
                    mutant = pop[i] + F_i * (pop[pb] - pop[i]) + F_i * (pop[a] - pop[b])
                else:
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = pop[a] + F_i * (pop[b] - pop[c])
                
                # Bounce-back boundary
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2.0
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2.0
                
                cross_points = np.random.rand(dim) < CR_i
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_f(trial)
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    F_arr[i] = F_i
                    CR_arr[i] = CR_i
                    if f_trial < prev_best:
                        improved_gen = True
            
            if improved_gen:
                stagnation = 0
                prev_best = best
            else:
                stagnation += 1
            
            if stagnation > 30 + 10 * dim // pop_size:
                # Restart half population around best
                half = pop_size // 2
                for j in range(half, pop_size):
                    if best_x is not None:
                        scale = 0.05 * (1 + stagnation * 0.02)
                        pop[j] = np.clip(best_x + scale * ranges * np.random.randn(dim), lower, upper)
                    else:
                        pop[j] = lower + np.random.rand(dim) * ranges
                    fit[j] = eval_f(pop[j])
                stagnation = 0

    # Run DE with different strategies
    run_de(0.30, 'pbest')
    run_de(0.25, 'best')
    run_de(0.20, 'rand')

    # Phase 3: Coordinate descent refinement
    if best_x is not None:
        for scale_power in range(8):
            if remaining() < 0.05:
                break
            step = ranges * (0.02 * (0.4 ** scale_power))
            improved = True
            while improved and remaining() > 0.03:
                improved = False
                perm = np.random.permutation(dim)
                for i in perm:
                    if remaining() < 0.02:
                        break
                    for d_val in [step[i], -step[i]]:
                        xt = best_x.copy()
                        xt[i] = np.clip(best_x[i] + d_val, lower[i], upper[i])
                        if eval_f(xt) < best:
                            improved = True
                            break

    return best
