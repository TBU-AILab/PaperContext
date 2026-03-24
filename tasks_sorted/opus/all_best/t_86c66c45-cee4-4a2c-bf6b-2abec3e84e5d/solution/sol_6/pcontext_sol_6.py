#Looking at the results, the pure SHADE (7.46) was the best. The L-SHADE variants and hybrids performed worse due to bugs, poor time management, or overhead. Let me improve by:
#
#1. **L-SHADE with proper population reduction** carefully integrated
#2. **Multiple restart phases** with diverse initialization strategies
#3. **Lightweight Nelder-Mead polishing** at the end
#4. **Better stagnation handling** with opposition-based reinitialization
#5. **Vectorized operations** where possible for speed
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.96 - elapsed()
    
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

    def run_shade(time_budget, init_pop=None, pop_sz=None, use_lshade=True):
        nonlocal best, best_params
        t_end = elapsed() + time_budget
        
        N_init = pop_sz if pop_sz else min(max(30, 8 * dim), 150)
        N_min = max(4, dim // 3)
        pop_size = N_init
        H = 100
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        mk = 0
        
        # Initialize population
        if init_pop is not None and len(init_pop) >= pop_size:
            population = np.array([clip(x) for x in init_pop[:pop_size]])
        elif init_pop is not None:
            population = np.array([clip(x) for x in init_pop])
            extra_n = pop_size - len(population)
            if extra_n > 0:
                extra = lower + np.random.rand(extra_n, dim) * ranges
                population = np.vstack([population, extra])
        else:
            # Latin Hypercube Sampling
            population = np.zeros((pop_size, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size)
                for i in range(pop_size):
                    population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
        
        fitness = np.array([evaluate(ind) for ind in population])
        archive = []
        total_evals = pop_size
        max_evals = N_init * 500
        
        stagnation = 0
        prev_b = best
        no_improve_gens = 0
        
        while elapsed() < t_end and time_left() > 0.05:
            S_F, S_CR, delta_f = [], [], []
            sorted_idx = np.argsort(fitness)
            
            # Adaptive p: starts large, shrinks
            ratio = min(1.0, total_evals / max(max_evals, 1))
            p_rate = max(2.0 / pop_size, 0.25 - 0.20 * ratio)
            p_best_size = max(2, int(p_rate * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            improved_any = False
            
            for i in range(pop_size):
                if elapsed() >= t_end or time_left() <= 0.05:
                    break
                
                ri = np.random.randint(H)
                mu_F = memory_F[ri]
                mu_CR = memory_CR[ri]
                
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
                while Fi <= 0:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
                
                pi = sorted_idx[np.random.randint(p_best_size)]
                
                r1 = np.random.randint(pop_size)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - x_r2)
                
                jrand = np.random.randint(dim)
                mask = np.random.rand(dim) < CRi
                mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                
                out_low = trial < lower
                out_high = trial > upper
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
                trial = clip(trial)
                
                trial_f = evaluate(trial)
                total_evals += 1
                
                if trial_f <= fitness[i]:
                    if trial_f < fitness[i]:
                        S_F.append(Fi); S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_f))
                        archive.append(population[i].copy())
                        improved_any = True
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            
            if S_F:
                w = np.array(delta_f); w = w / w.sum()
                sf = np.array(S_F); scr = np.array(S_CR)
                memory_F[mk % H] = np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30)
                memory_CR[mk % H] = np.sum(w * scr)
                mk += 1
            
            if use_lshade:
                new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max_evals)))
                if new_size < pop_size:
                    si = np.argsort(fitness)[:new_size]
                    population = population[si]; fitness = fitness[si]; pop_size = new_size
            
            if abs(prev_b - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_b = best
            
            if stagnation > 35 and time_left() > 0.5 and elapsed() < t_end - 0.3:
                keep = max(2, pop_size // 8)
                si = np.argsort(fitness)
                for ii in range(keep, pop_size):
                    r = max(0.01, 0.4 * time_left() / max_time)
                    population[ii] = best_params + r * ranges * (np.random.rand(dim) - 0.5)
                    population[ii] = clip(population[ii])
                    fitness[ii] = evaluate(population[ii])
                archive.clear(); stagnation = 0

    # Phase 1: Main L-SHADE
    run_shade(max_time * 0.50)
    
    # Phase 2: Restart with fresh diverse population
    if time_left() > 1.0:
        run_shade(time_left() * 0.50)
    
    # Phase 3: Local intensification around best
    if time_left() > 0.5 and best_params is not None:
        local_pop = [best_params + 0.08 * ranges * (np.random.rand(dim) - 0.5) for _ in range(50)]
        local_pop[0] = best_params.copy()
        run_shade(time_left() * 0.8, init_pop=local_pop, pop_sz=50, use_lshade=False)
    
    return best
