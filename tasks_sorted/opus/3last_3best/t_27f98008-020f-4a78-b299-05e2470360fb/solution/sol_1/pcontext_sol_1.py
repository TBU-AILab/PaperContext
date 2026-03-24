#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Use SHADE (Success-History based Adaptive DE)** - a more sophisticated adaptive DE variant
#2. **Better population sizing and time management**
#3. **Multiple restarts with opposition-based learning**
#4. **CMA-ES-like local search** instead of simple pattern search
#5. **Vectorized operations where possible**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # SHADE parameters
    pop_size = min(max(10 * dim, 40), 150)
    H = pop_size  # history size
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    memory_idx = 0
    archive = []
    archive_max = pop_size
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, d] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fitness) < pop_size:
        fitness = np.append(fitness, np.full(pop_size - len(fitness), float('inf')))
    
    # Opposition-based population
    if elapsed() < max_time * 0.3:
        opp_pop = lower + upper - population
        opp_fit = np.array([evaluate(opp_pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.3])
        if len(opp_fit) == pop_size:
            combined_pop = np.vstack([population, opp_pop])
            combined_fit = np.concatenate([fitness, opp_fit])
            idx = np.argsort(combined_fit)[:pop_size]
            population = combined_pop[idx]
            fitness = combined_fit[idx]
    
    p_min = 2.0 / pop_size
    p_max = 0.2
    
    generation = 0
    
    while elapsed() < max_time * 0.85:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        progress = min(elapsed() / (max_time * 0.85), 1.0)
        p = p_max - (p_max - p_min) * progress
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            ri = np.random.randint(0, H)
            mu_F = memory_F[ri]
            mu_CR = memory_CR[ri]
            
            # Generate F from Cauchy
            Fi = -1
            while Fi <= 0:
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)
            
            # current-to-pbest/1
            p_num = max(int(np.ceil(p * pop_size)), 2)
            pbest_idx = np.random.randint(0, p_num)
            sorted_idx = np.argsort(fitness)
            x_pbest = population[sorted_idx[pbest_idx]]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            pool = list(range(pop_size)) + list(range(len(archive)))
            pool = [j for j in pool if j != i and j != r1]
            r2 = np.random.choice(pool) if pool else r1
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            
            cross = np.random.rand(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = clip(trial)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(abs(fitness[i] - trial_f))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive:
                        archive[np.random.randint(len(archive))] = population[i].copy()
                population[i] = trial
                fitness[i] = trial_f
        
        if S_F:
            weights = np.array(S_delta)
            weights /= weights.sum() + 1e-30
            memory_F[memory_idx] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            memory_CR[memory_idx] = np.sum(weights * np.array(S_CR))
            memory_idx = (memory_idx + 1) % H
    
    # Local search: Nelder-Mead simplex
    if best_x is not None and elapsed() < max_time * 0.98:
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x.copy()
        step = ranges * 0.05
        for j in range(dim):
            simplex[j+1] = best_x.copy()
            simplex[j+1][j] += step[j] if best_x[j] + step[j] <= upper[j] else -step[j]
        sim_f = np.array([evaluate(simplex[k]) for k in range(dim+1) if elapsed() < max_time*0.98])
        if len(sim_f) == dim + 1:
            while elapsed() < max_time * 0.98:
                order = np.argsort(sim_f)
                simplex = simplex[order]
                sim_f = sim_f[order]
                centroid = simplex[:-1].mean(axis=0)
                # Reflect
                xr = clip(centroid + (centroid - simplex[-1]))
                fr = evaluate(xr)
                if fr < sim_f[-2] and fr >= sim_f[0]:
                    simplex[-1] = xr; sim_f[-1] = fr; continue
                if fr < sim_f[0]:
                    xe = clip(centroid + 2*(xr - centroid))
                    fe = evaluate(xe)
                    if fe < fr: simplex[-1]=xe; sim_f[-1]=fe
                    else: simplex[-1]=xr; sim_f[-1]=fr
                    continue
                xc = clip(centroid + 0.5*(simplex[-1]-centroid))
                fc = evaluate(xc)
                if fc < sim_f[-1]:
                    simplex[-1]=xc; sim_f[-1]=fc; continue
                for j in range(1, dim+1):
                    simplex[j] = clip(simplex[0] + 0.5*(simplex[j]-simplex[0]))
                    sim_f[j] = evaluate(simplex[j])
                    if elapsed() >= max_time*0.98: break
    
    return best
