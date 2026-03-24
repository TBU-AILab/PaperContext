#Looking at the results, algorithm 1 (score 18.68) is the best. It uses L-SHADE with aggressive coordinate descent. I'll improve it by:
#
#1. **Better coordinate descent with random permutation** of dimensions each pass
#2. **Adding a golden section search** for 1D line minimization
#3. **More time for local search** since it's clearly the most impactful part
#4. **Multiple scale coordinate descent passes** with diminishing step sizes
#5. **Randomized block coordinate descent** to escape local minima
#6. **Smarter DE time allocation** - less time on DE, more on local search
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

    def coordinate_descent(x0, f0, init_step=0.05, min_step=1e-15, max_evals=None, time_limit=None):
        if max_evals is None:
            max_evals = dim * 50
        if time_limit is None:
            time_limit = remaining() * 0.3
        t_start = elapsed()
        x_cur = x0.copy()
        f_cur = f0
        step = init_step * ranges.copy()
        n_evals = 0
        
        while n_evals < max_evals and (elapsed() - t_start) < time_limit and remaining() > 0.05:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if remaining() <= 0.05 or n_evals >= max_evals:
                    return x_cur, f_cur
                
                # Try positive step
                x_try = x_cur.copy()
                x_try[d] += step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    # Accelerate in this direction
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
                
                # Try negative step
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

    def pattern_move(x0, f0, time_limit=None):
        """Powell-like pattern moves using accumulated direction"""
        if time_limit is None:
            time_limit = remaining() * 0.2
        t_start = elapsed()
        x_cur = x0.copy()
        f_cur = f0
        
        while (elapsed() - t_start) < time_limit and remaining() > 0.1:
            x_prev = x_cur.copy()
            # Do one full coordinate descent pass
            x_cur, f_cur = coordinate_descent(x_cur, f_cur, init_step=0.01, 
                                               max_evals=dim*4, time_limit=time_limit*0.3)
            
            # Try pattern move (extrapolate the direction of improvement)
            direction = x_cur - x_prev
            dnorm = np.linalg.norm(direction)
            if dnorm < 1e-16:
                break
            
            # Try doubling the move
            x_try = clip(x_cur + direction)
            f_try = eval_func(x_try)
            if f_try < f_cur:
                x_cur = x_try
                f_cur = f_try
                # Try further
                x_try2 = clip(x_cur + direction)
                f_try2 = eval_func(x_try2)
                if f_try2 < f_cur:
                    x_cur = x_try2
                    f_cur = f_try2
            else:
                break
        
        return x_cur, f_cur

    # ============= Main L-SHADE loop with restarts =============
    restart_count = 0
    
    while remaining() > 0.3:
        restart_count += 1
        
        # Use less time for DE (40%), more for local search
        time_for_de = remaining() * 0.40
        
        N_init = min(max(18, 6 * dim), 180)
        N_min = max(4, dim // 2 + 1)
        pop_size = N_init
        max_nfe_estimate = max(1, int(time_for_de * 700))
        nfe_at_start = evals
        
        H = 80
        if restart_count == 1:
            memory_F = np.full(H, 0.5)
            memory_CR = np.full(H, 0.5)
        else:
            memory_F = np.full(H, 0.1 + 0.8 * np.random.rand())
            memory_CR = np.full(H, 0.1 + 0.8 * np.random.rand())
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        # Opposition-based initialization
        half = N_init // 2
        pop1 = np.random.uniform(lower, upper, (half, dim))
        pop2 = lower + upper - pop1
        population = np.vstack([pop1, pop2])[:N_init]
        
        if restart_count > 1 and best_sol is not None:
            n_local = max(1, pop_size // 3)
            scale = max(0.01, 0.4 / restart_count)
            for j in range(n_local):
                population[j] = clip(best_sol + scale * ranges * np.random.randn(dim))
        
        fitness = np.array([eval_func(ind) for ind in population])
        if remaining() <= 0.3:
            break
        
        generation = 0
        stagnation = 0
        prev_best = best
        de_start = elapsed()
        
        while remaining() > 0.3 and (elapsed() - de_start) < time_for_de:
            generation += 1
            
            nfe_since = evals - nfe_at_start
            ratio = min(1.0, nfe_since / max(1, max_nfe_estimate))
            new_ps = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_ps < pop_size:
                si = np.argsort(fitness)
                population = population[si[:new_ps]]
                fitness = fitness[si[:new_ps]]
                pop_size = new_ps
            
            p_best_size = max(2, int((0.25 - 0.23 * ratio) * pop_size))
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
            
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0, 1)
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fitness)
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0.2:
                    break
                pi = sorted_idx[np.random.randint(0, p_best_size)]
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                cs = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(cs)
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
                jrand = np.random.randint(dim)
                mask = np.random.rand(dim) < CRs[i]
                mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                bl = trial < lower; ab = trial > upper
                trial[bl] = (lower[bl] + population[i][bl]) / 2
                trial[ab] = (upper[ab] + population[i][ab]) / 2
                tf = eval_func(trial)
                if tf <= fitness[i]:
                    d = fitness[i] - tf
                    if tf < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fs[i]); S_CR.append(CRs[i]); S_delta.append(d + 1e-30)
                    new_pop[i] = trial; new_fit[i] = tf
            population = new_pop; fitness = new_fit
            if S_F:
                w = np.array(S_delta); w /= w.sum()
                sf = np.array(S_F); sc = np.array(S_CR)
                memory_F[k_idx % H] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
                memory_CR[k_idx % H] = np.sum(w * sc); k_idx += 1
            if abs(prev_best - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best
            if stagnation > 45 or pop_size <= N_min:
                break
        
        # Multi-scale local search after DE
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.1, max_evals=dim*30, time_limit=remaining()*0.15)
        if best_sol is not None and remaining() > 0.3:
            pattern_move(best_sol, best, time_limit=remaining()*0.15)
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.005, max_evals=dim*25, time_limit=remaining()*0.15)
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.0002, max_evals=dim*25, time_limit=remaining()*0.15)
    
    # Final intensive local search with all remaining time
    if best_sol is not None and remaining() > 0.05:
        coordinate_descent(best_sol, best, init_step=0.01, min_step=1e-16, max_evals=dim*200, time_limit=remaining()*0.5)
    if best_sol is not None and remaining() > 0.05:
        coordinate_descent(best_sol, best, init_step=0.0001, min_step=1e-17, max_evals=dim*200, time_limit=remaining()*0.95)
    
    return best
