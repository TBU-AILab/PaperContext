#Looking at the results, the best algorithm (7.46) is a pure SHADE. The L-SHADE variant (226.4) performed terribly, likely due to bugs in the restart logic. The SHADE+CMA-ES hybrid (13.87) was worse than pure SHADE. Let me improve by:
#
#1. **L-SHADE with proper population reduction**
#2. **Multiple independent restarts** with different initializations
#3. **Nelder-Mead simplex polishing** (lightweight local search)
#4. **Better restart strategy** - full reinitializations, not just local perturbations
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

    def nelder_mead(x0, initial_step=0.1, max_evals=500):
        n = len(x0)
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = evaluate(x0)
        evals = 1
        for i in range(n):
            p = x0.copy()
            p[i] += initial_step * ranges[i]
            p = clip(p)
            simplex[i + 1] = p
            f_simplex[i + 1] = evaluate(p)
            evals += 1
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        while evals < max_evals and time_left() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr); evals += 1
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe); evals += 1
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc); evals += 1
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i]); evals += 1
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc); evals += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i]); evals += 1
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break

    def run_shade(time_budget, init_pop=None, pop_sz=None):
        nonlocal best, best_params
        t_end = elapsed() + time_budget
        
        pop_size = pop_sz if pop_sz else min(max(30, 8 * dim), 150)
        N_init = pop_size
        N_min = 4
        H = 100
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        mk = 0
        
        if init_pop is not None and len(init_pop) >= pop_size:
            population = np.array([clip(x) for x in init_pop[:pop_size]])
        else:
            population = np.zeros((pop_size, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size)
                for i in range(pop_size):
                    population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
        
        fitness = np.array([evaluate(ind) for ind in population])
        archive = []
        total_evals = pop_size
        max_evals = pop_size * 400
        stagnation = 0
        prev_b = best
        
        while elapsed() < t_end and time_left() > 0.05:
            S_F, S_CR, delta_f = [], [], []
            sorted_idx = np.argsort(fitness)
            ratio = min(1.0, total_evals / max(max_evals, 1))
            p_rate = max(2.0 / pop_size, 0.2 - 0.15 * ratio)
            p_best_size = max(2, int(p_rate * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if elapsed() >= t_end or time_left() <= 0.05:
                    break
                ri = np.random.randint(H)
                Fi = memory_F[ri] + 0.1 * np.random.standard_cauchy()
                while Fi <= 0:
                    Fi = memory_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                CRi = np.clip(memory_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                pi = sorted_idx[np.random.randint(p_best_size)]
                r1 = np.random.randint(pop_size)
                while r1 == i: r1 = np.random.randint(pop_size)
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1: r2 = np.random.randint(pool_size)
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - x_r2)
                jrand = np.random.randint(dim)
                mask = np.random.rand(dim) < CRi; mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                out_low = trial < lower; out_high = trial > upper
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
                trial = clip(trial)
                trial_f = evaluate(trial); total_evals += 1
                if trial_f <= fitness[i]:
                    if trial_f < fitness[i]:
                        S_F.append(Fi); S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_f))
                        archive.append(population[i].copy())
                    new_pop[i] = trial; new_fit[i] = trial_f
            
            population = new_pop; fitness = new_fit
            while len(archive) > pop_size: archive.pop(np.random.randint(len(archive)))
            if S_F:
                w = np.array(delta_f); w = w / w.sum()
                memory_F[mk % H] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
                memory_CR[mk % H] = np.sum(w * np.array(S_CR)); mk += 1
            
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max_evals)))
            if new_size < pop_size:
                si = np.argsort(fitness)[:new_size]
                population = population[si]; fitness = fitness[si]; pop_size = new_size
            
            if abs(prev_b - best) < 1e-15: stagnation += 1
            else: stagnation = 0
            prev_b = best
            if stagnation > 50: break

    # Phase 1: Main L-SHADE run
    run_shade(max_time * 0.45)
    
    # Phase 2: Nelder-Mead polish
    if time_left() > 0.5 and best_params is not None:
        nelder_mead(best_params.copy(), initial_step=0.05, max_evals=dim * 30)
    
    # Phase 3: Restart L-SHADE with fresh population
    if time_left() > 1.0:
        run_shade(time_left() * 0.5)
    
    # Phase 4: Local L-SHADE around best
    if time_left() > 0.5 and best_params is not None:
        local_pop = []
        for _ in range(60):
            x = best_params + 0.1 * ranges * (np.random.rand(dim) - 0.5)
            local_pop.append(clip(x))
        run_shade(time_left() * 0.7, init_pop=local_pop, pop_sz=min(60, len(local_pop)))
    
    # Final polish
    if time_left() > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), initial_step=0.01, max_evals=dim * 20)
    
    return best
