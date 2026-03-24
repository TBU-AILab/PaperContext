#Looking at the previous algorithm, I see it uses CMA-ES with Nelder-Mead refinement. I'll improve it with several enhancements:
#
#1. Better time management
#2. Differential Evolution as the main search (more robust for diverse landscapes)
#3. Multiple CMA-ES restarts with adaptive sigma
#4. More aggressive local search refinement
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
    
    def remaining():
        return max_time - elapsed()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        f = func(clip(x))
        if f < best:
            best = f
            best_params = clip(x).copy()
        return f
    
    # --- Phase 1: Differential Evolution ---
    pop_size = min(max(10 * dim, 40), 200)
    
    # Latin Hypercube init
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok(0.95):
            return best
        fitness[i] = evaluate(pop[i])
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    # Use ~50% of time for DE
    while time_ok(0.50):
        generation += 1
        improved = False
        
        # Adaptive F and CR
        F_use = 0.5 + 0.3 * np.random.random()
        CR_use = 0.8 + 0.2 * np.random.random()
        
        best_idx = np.argmin(fitness)
        
        for i in range(pop_size):
            if not time_ok(0.50):
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mix strategies
            if np.random.random() < 0.5:
                # DE/best/1
                mutant = pop[best_idx] + F_use * (pop[r1] - pop[r2])
            else:
                # DE/current-to-best/1
                r3 = np.random.choice(idxs)
                mutant = pop[i] + F_use * (pop[best_idx] - pop[i]) + F_use * (pop[r1] - pop[r3])
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_use or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (pop[i][j] - lower[j])
                elif trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - pop[i][j])
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # Restart worst half if stagnant
        if stagnation > 5 + dim:
            order = np.argsort(fitness)
            half = pop_size // 2
            for i in order[half:]:
                pop[i] = clip(lower + np.random.random(dim) * ranges)
                if time_ok(0.50):
                    fitness[i] = evaluate(pop[i])
            stagnation = 0
    
    # --- Phase 2: CMA-ES from best solutions ---
    top_k = min(5, pop_size)
    top_indices = np.argsort(fitness)[:top_k]
    
    for restart in range(top_k):
        if not time_ok(0.80):
            break
        
        x0 = pop[top_indices[restart]].copy()
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = np.mean(ranges) * 0.2 / (restart + 1)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        counteval = 0
        
        for gen in range(500):
            if not time_ok(0.92):
                break
            
            try:
                C = np.triu(C) + np.triu(C, 1).T
                D_vals, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_vals, 1e-20))
            except:
                break
            
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for k in range(lam):
                if not time_ok(0.92):
                    break
                z = np.random.randn(n)
                arx[k] = clip(mean + sigma * (B @ (D * z)))
                arf[k] = evaluate(arx[k])
                counteval += 1
            
            idx = np.argsort(arf)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            ymean = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (1.0/D * (B.T @ ymean)))
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-15)
            sigma = min(sigma, np.max(ranges))
            
            if sigma < 1e-12:
                break
    
    # --- Phase 3: Local Powell-like search ---
    if best_params is not None:
        x = best_params.copy()
        step = ranges * 0.001
        
        while time_ok(0.97):
            improved_any = False
            for d in range(dim):
                if not time_ok(0.97):
                    break
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[d] += sign * step[d]
                    f = evaluate(clip(trial))
                    if f < evaluate(x):
                        x = clip(trial)
                        improved_any = True
                        step[d] *= 1.5
                        break
                else:
                    step[d] *= 0.5
            if not improved_any:
                if np.max(step) < 1e-14:
                    break
    
    return best
