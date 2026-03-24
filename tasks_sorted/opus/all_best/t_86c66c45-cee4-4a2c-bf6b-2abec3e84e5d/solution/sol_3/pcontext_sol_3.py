#Looking at the results, the pure SHADE algorithm (7.46) performed much better than the SHADE+CMA-ES hybrid (13.87). The hybrid likely spent too much time in CMA-ES local search that wasn't effective. Let me improve the SHADE approach with:
#
#1. **L-SHADE** (population reduction) for better convergence
#2. **Multiple independent restarts** with different strategies
#3. **Eigenvector-based mutation** for better exploration in rotated landscapes
#4. **Nelder-Mead polishing** at the end (lightweight, no matrix operations)
#5. **Better stagnation handling** with opposition-based learning
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
        return max_time * 0.95 - elapsed()
    
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

    def run_lshade(time_budget, init_pop=None):
        nonlocal best, best_params
        
        t_end = elapsed() + time_budget
        
        N_init = min(max(30, 10 * dim), 200)
        N_min = 4
        pop_size = N_init
        H = 100
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        mem_k = 0
        
        if init_pop is not None and len(init_pop) > 0:
            population = np.array([clip(x) for x in init_pop[:pop_size]])
            if len(population) < pop_size:
                extra = np.zeros((pop_size - len(population), dim))
                for d in range(dim):
                    for i in range(len(extra)):
                        extra[i, d] = lower[d] + np.random.rand() * ranges[d]
                population = np.vstack([population, extra])
        else:
            population = np.zeros((pop_size, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size)
                for i in range(pop_size):
                    population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
        
        fitness = np.array([evaluate(ind) for ind in population])
        archive = []
        total_evals = pop_size
        max_evals = pop_size * 500
        
        stagnation = 0
        prev_best_local = best
        
        while elapsed() < t_end and time_left() > 0.05:
            S_F, S_CR, delta_f = [], [], []
            sorted_idx = np.argsort(fitness)
            
            # Adaptive p value: decreases over time
            ratio = min(1.0, total_evals / max(max_evals, 1))
            p_rate = max(2.0/pop_size, 0.2 - 0.15 * ratio)
            p_best_size = max(2, int(p_rate * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
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
                if dim <= 2:
                    CRi = min(CRi, 0.3)
                
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
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_f))
                        archive.append(population[i].copy())
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            
            if S_F:
                w = np.array(delta_f)
                w = w / w.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                memory_F[mem_k % H] = np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30)
                memory_CR[mem_k % H] = np.sum(w * scr)
                mem_k += 1
            
            # L-SHADE population reduction
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max_evals)))
            if new_size < pop_size:
                si = np.argsort(fitness)[:new_size]
                population = population[si]
                fitness = fitness[si]
                pop_size = new_size
            
            if abs(prev_best_local - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_best_local = best
            
            if stagnation > 50:
                break

    # Run 1: Main L-SHADE
    run_lshade(max_time * 0.55)
    
    # Run 2: Restart L-SHADE around best with opposition-based init
    if time_left() > 1.0:
        opp_pop = []
        for _ in range(20):
            x = best_params + 0.3 * ranges * (np.random.rand(dim) - 0.5)
            opp_pop.append(clip(x))
            opp = lower + upper - x
            opp_pop.append(clip(opp))
        run_lshade(time_left() * 0.6, init_pop=opp_pop)
    
    # Run 3: Fine local restart
    if time_left() > 0.5:
        fine_pop = []
        for _ in range(40):
            x = best_params + 0.05 * ranges * (np.random.rand(dim) - 0.5)
            fine_pop.append(clip(x))
        run_lshade(time_left() - 0.1, init_pop=fine_pop)
    
    return best
