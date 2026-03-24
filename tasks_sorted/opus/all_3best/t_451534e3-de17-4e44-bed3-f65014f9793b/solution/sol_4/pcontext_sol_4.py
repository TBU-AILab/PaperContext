#Looking at the results, the best algorithm (24.58) uses L-SHADE with coordinate descent local search. The algorithm scoring 212.17 tried adding CMA-ES but was too complex and slow. I need to focus on:
#
#1. **Better time management** - spend more time on productive search
#2. **Improved L-SHADE** with smaller initial population for faster generations
#3. **More effective local search** - Nelder-Mead style + coordinate descent hybrid
#4. **Smarter restarts** - vary exploration/exploitation balance
#5. **Avoid overhead** - streamline inner loops
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_sol, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    def coordinate_descent(x0, f0, init_step=0.05, min_step=1e-14, max_evals=None, time_limit=None):
        if max_evals is None:
            max_evals = dim * 30
        if time_limit is None:
            time_limit = remaining() * 0.3
        t_start = elapsed()
        x_cur = x0.copy()
        f_cur = f0
        step = init_step * ranges.copy()
        n_evals = 0
        
        while n_evals < max_evals and (elapsed() - t_start) < time_limit and remaining() > 0.1:
            improved = False
            for d in range(dim):
                if remaining() <= 0.05 or n_evals >= max_evals:
                    return x_cur, f_cur
                
                # Positive step
                x_try = x_cur.copy()
                x_try[d] += step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    # Accelerate
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] += step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                        else:
                            step[d] *= 0.5
                            break
                    improved = True
                    continue
                
                # Negative step
                x_try = x_cur.copy()
                x_try[d] -= step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] -= step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                        else:
                            step[d] *= 0.5
                            break
                    improved = True
                else:
                    step[d] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < min_step:
                    break
        
        return x_cur, f_cur

    def gradient_free_descent(x0, f0, max_evals=None):
        """Simplex gradient approximation descent"""
        if max_evals is None:
            max_evals = dim * 15
        x_cur = x0.copy()
        f_cur = f0
        n_evals = 0
        alpha = 0.01 * np.mean(ranges)
        
        for _ in range(max(1, max_evals // (dim + 2))):
            if remaining() <= 0.1:
                return x_cur, f_cur
            # Estimate gradient via finite differences
            grad = np.zeros(dim)
            h = alpha * 0.1
            for d in range(dim):
                if n_evals >= max_evals or remaining() <= 0.05:
                    return x_cur, f_cur
                x_p = x_cur.copy()
                x_p[d] += h
                x_p = clip(x_p)
                fp = eval_func(x_p)
                n_evals += 1
                grad[d] = (fp - f_cur) / (h + 1e-30)
            
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-20:
                break
            direction = -grad / gnorm
            
            # Line search
            step = alpha
            for _ in range(5):
                if n_evals >= max_evals or remaining() <= 0.05:
                    return x_cur, f_cur
                x_try = clip(x_cur + step * direction)
                f_try = eval_func(x_try)
                n_evals += 1
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    alpha *= 1.2
                    break
                step *= 0.5
            else:
                alpha *= 0.5
                if alpha < 1e-15:
                    break
        
        return x_cur, f_cur

    # ============= Main L-SHADE loop with restarts =============
    restart_count = 0
    total_budget_fraction = 0.7  # fraction for DE, rest for local search
    
    while remaining() > 0.5:
        restart_count += 1
        time_for_de = remaining() * total_budget_fraction / max(1, 4 - min(restart_count, 3))
        
        N_init = min(max(18, 6 * dim), 200)
        N_min = max(4, dim // 2 + 1)
        pop_size = N_init
        max_nfe_estimate = max(1, int(time_for_de * 600))
        nfe_at_start = evals
        
        H = 60
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        if restart_count > 1:
            memory_F = np.full(H, 0.15 + 0.7 * np.random.rand())
            memory_CR = np.full(H, 0.15 + 0.7 * np.random.rand())
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        # Initialize population with opposition-based learning
        half = N_init // 2
        pop1 = np.random.uniform(lower, upper, (half, dim))
        pop2 = lower + upper - pop1
        population = np.vstack([pop1, pop2])[:N_init]
        
        if restart_count > 1 and best_sol is not None:
            n_local = max(1, pop_size // 4)
            scale = 0.3 / restart_count
            for j in range(n_local):
                population[j] = clip(best_sol + scale * ranges * np.random.randn(dim))
        
        fitness = np.array([eval_func(ind) for ind in population])
        
        if remaining() <= 0.5:
            break
        
        generation = 0
        stagnation = 0
        prev_best = best
        de_start_time = elapsed()
        
        while remaining() > 0.5 and (elapsed() - de_start_time) < time_for_de:
            generation += 1
            
            nfe_since_start = evals - nfe_at_start
            ratio = min(1.0, nfe_since_start / max(1, max_nfe_estimate))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                si = np.argsort(fitness)
                population = population[si[:new_pop_size]]
                fitness = fitness[si[:new_pop_size]]
                pop_size = new_pop_size
            
            p_min = 2.0 / pop_size
            p_max = 0.25
            p = p_max - (p_max - p_min) * ratio
            p_best_size = max(2, int(p * pop_size))
            
            ri = np.random.randint(0, H, pop_size)
            
            Fs = np.empty(pop_size)
            for idx in range(pop_size):
                for _ in range(20):
                    f_val = memory_F[ri[idx]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        Fs[idx] = min(f_val, 1.0)
                        break
                else:
                    Fs[idx] = 0.5
            
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
            
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fitness)
            
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0.3:
                    break
                
                pi = sorted_idx[np.random.randint(0, p_best_size)]
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                combined_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(combined_size)
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
                
                jrand = np.random.randint(dim)
                mask = np.random.rand(dim) < CRs[i]
                mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                
                below = trial < lower
                above = trial > upper
                trial[below] = (lower[below] + population[i][below]) / 2.0
                trial[above] = (upper[above] + population[i][above]) / 2.0
                
                trial_f = eval_func(trial)
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    if trial_f < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fs[i])
                        S_CR.append(CRs[i])
                        S_delta.append(delta + 1e-30)
                    new_population[i] = trial
                    new_fitness[i] = trial_f
            
            population = new_population
            fitness = new_fitness
            
            if S_F:
                weights = np.array(S_delta)
                weights /= weights.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                mean_F = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
                mean_CR = np.sum(weights * scr)
                memory_F[k_idx % H] = mean_F
                memory_CR[k_idx % H] = mean_CR
                k_idx += 1
            
            if abs(prev_best - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best
            
            if stagnation > 40 or pop_size <= N_min:
                break
        
        # Local search after each DE phase
        if best_sol is not None and remaining() > 0.5:
            best_sol, _ = coordinate_descent(best_sol, best, init_step=0.05, max_evals=dim*20, time_limit=remaining()*0.25)
        if best_sol is not None and remaining() > 0.5:
            best_sol, _ = gradient_free_descent(best_sol, best, max_evals=dim*10)
        if best_sol is not None and remaining() > 0.5:
            best_sol, _ = coordinate_descent(best_sol, best, init_step=0.002, max_evals=dim*15, time_limit=remaining()*0.2)
    
    # Final local search with remaining time
    if best_sol is not None and remaining() > 0.2:
        best_sol, _ = coordinate_descent(best_sol, best, init_step=0.001, min_step=1e-16, max_evals=dim*100, time_limit=remaining()*0.9)
    
    return best
