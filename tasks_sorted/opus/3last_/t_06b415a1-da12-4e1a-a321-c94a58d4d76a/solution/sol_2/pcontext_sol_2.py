#Looking at the two algorithms, algorithm 1 (score 83.74) performs better with its DE + CMA-ES + local search approach. I'll improve upon it with:
#
#1. **Better initialization** with Sobol-like quasi-random sampling
#2. **Improved DE** with SHADE-like adaptive F/CR
#3. **Multiple CMA-ES restarts** with better budget management
#4. **Pattern search** for final refinement
#5. **Better time management** throughout
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
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size = min(max(15 * dim, 60), 300)
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
    
    # --- Phase 2: SHADE-like Differential Evolution ---
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    
    generation = 0
    stagnation = 0
    
    while time_ok(0.40):
        generation += 1
        S_F, S_CR, S_df = [], [], []
        
        best_idx = np.argmin(fitness)
        
        # Sort population for p-best
        sorted_idx = np.argsort(fitness)
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if not time_ok(0.40):
                break
            
            # Pick from memory
            ri = np.random.randint(memory_size)
            F_i = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.0)
            CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best
            p = max(2, int(0.1 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(p)]
            
            # DE/current-to-pbest/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2_idx = np.random.randint(pool_size)
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(pool_size)
            if r2_idx < pop_size:
                xr2 = pop[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - xr2)
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CR_i)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = (lower[j] + pop[i][j]) / 2
                elif trial[j] > upper[j]:
                    trial[j] = (upper[j] + pop[i][j]) / 2
            
            f_trial = evaluate(clip(trial))
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_df.append(fitness[i] - f_trial)
                    archive.append(pop[i].copy())
                new_pop[i] = clip(trial)
                new_fitness[i] = f_trial
        
        pop = new_pop
        fitness = new_fitness
        
        # Update memory
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % memory_size
        
        # Trim archive
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        # Restart worst if stagnant
        if generation % (10 + dim) == 0:
            order = np.argsort(fitness)
            for idx in order[pop_size//2:]:
                pop[idx] = clip(lower + np.random.random(dim) * ranges)
                if time_ok(0.40):
                    fitness[idx] = evaluate(pop[idx])
    
    # --- Phase 3: CMA-ES from best solutions ---
    top_k = min(5, pop_size)
    top_indices = np.argsort(fitness)[:top_k]
    
    for restart in range(top_k):
        if not time_ok(0.75):
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
        mean = x0.copy(); sigma = np.mean(ranges) * 0.15 / (restart + 1)
        pc = np.zeros(n); ps = np.zeros(n); C = np.eye(n); counteval = 0
        
        for gen in range(1000):
            if not time_ok(0.90): break
            try:
                C = np.triu(C) + np.triu(C, 1).T
                D_vals, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(D_vals, 1e-20))
            except: break
            arx = np.zeros((lam, n)); arf = np.zeros(lam)
            for kk in range(lam):
                if not time_ok(0.90): break
                arx[kk] = clip(mean + sigma * (B @ (D * np.random.randn(n))))
                arf[kk] = evaluate(arx[kk]); counteval += 1
            idx = np.argsort(arf); old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            ymean = (mean - old_mean) / sigma
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (B @ (1.0/D * (B.T @ ymean)))
            hsig = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*ymean
            artmp = (arx[idx[:mu]] - old_mean)/sigma
            C = (1-c1-cmu_v)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu_v*(artmp.T@np.diag(weights)@artmp)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-15, np.max(ranges))
            if sigma < 1e-13: break
    
    # --- Phase 4: Coordinate-wise refinement ---
    if best_params is not None:
        x = best_params.copy()
        step = ranges * 0.01
        while time_ok(0.98):
            improved = False
            for d in range(dim):
                if not time_ok(0.98): break
                for sign in [1, -1]:
                    trial = x.copy(); trial[d] += sign * step[d]
                    f = evaluate(trial)
                    if f < func(x):
                        x = clip(trial); improved = True; step[d] *= 1.5; break
                else:
                    step[d] *= 0.5
            if not improved and np.max(step/ranges) < 1e-14: break
    
    return best
