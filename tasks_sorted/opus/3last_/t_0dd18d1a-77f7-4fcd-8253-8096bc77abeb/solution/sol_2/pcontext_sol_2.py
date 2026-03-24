#Looking at the results, algorithm 2 (33.88) performed much better than algorithm 1 (209.64). I'll improve by combining the best elements: SHADE-style adaptive DE with better local search using a proper CMA-ES implementation, better time allocation, and population size reduction over time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    best_params = None
    
    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(30, 10 * dim), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    archive = []
    k = 0
    
    while elapsed() < max_time * 0.55:
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(0, H)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.05, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [x for x in combined if x != i and x != r1]
            r2 = np.random.choice(combined)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = clip(np.where(cross_points, mutant, population[i]))
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                delta = fitness[i] - trial_fitness
                if delta > 0:
                    archive.append(population[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        if S_F:
            w = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k % H] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k % H] = np.sum(w * np.array(S_CR))
            k += 1
    
    # --- Phase 3: CMA-ES-like local search with restarts ---
    sigma = 0.1
    mean = best_params.copy()
    n_restart = 0
    
    while elapsed() < max_time * 0.95:
        lam = max(8, 4 + int(3 * np.log(dim)))
        sig = sigma * (0.5 ** min(n_restart, 4))
        current_best_fit = best
        no_imp = 0
        
        while elapsed() < max_time * 0.95 and no_imp < 30 * dim:
            samples = np.array([clip(mean + sig * np.random.randn(dim) * ranges) for _ in range(lam)])
            fits = np.array([func(s) for s in samples])
            
            idx_sort = np.argsort(fits)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            mean = np.sum(weights[:, None] * samples[idx_sort[:mu]], axis=0)
            mean = clip(mean)
            
            if fits[idx_sort[0]] < best:
                best = fits[idx_sort[0]]
                best_params = samples[idx_sort[0]].copy()
                no_imp = 0
            else:
                no_imp += lam
            
            sig *= 0.995
        
        n_restart += 1
        mean = best_params + np.random.randn(dim) * ranges * 0.05 * (0.5 ** min(n_restart, 5))
        mean = clip(mean)
    
    return best
