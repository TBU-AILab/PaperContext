#Looking at the previous algorithm, I'll make several improvements:
#
#1. **Use CMA-ES-inspired approach** - more effective for continuous optimization than basic DE
#2. **Better adaptive mechanisms** - SHADE-style parameter adaptation
#3. **Multiple restarts with decreasing search radius**
#4. **More efficient boundary handling**
#5. **Better local search integration**
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

    # SHADE-like DE with multiple strategies
    pop_size = min(max(30, 8 * dim), 300)
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fitness) < pop_size:
        fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))
    
    # SHADE memory
    mem_size = 20
    M_F = np.full(mem_size, 0.5)
    M_CR = np.full(mem_size, 0.5)
    k = 0
    
    archive = []
    max_archive = pop_size
    
    while elapsed() < max_time * 0.80:
        S_F, S_CR, S_w = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            ri = np.random.randint(mem_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.5)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            idxs = [j for j in range(pop_size) if j != i]
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.argsort(fitness)[:p]
            xpbest = population[np.random.choice(pbest_idx)]
            
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Use archive for xb if available
            if archive and np.random.random() < 0.5:
                xb_candidate = archive[np.random.randint(len(archive))]
            else:
                xb_candidate = population[b]
            
            mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[a] - xb_candidate)
            
            cross = np.random.rand(dim) < CRi
            if not np.any(cross):
                cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            # Midpoint boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            tf = evaluate(trial)
            
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    S_w.append(abs(fitness[i] - tf))
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = tf
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % mem_size

    # Local search: Nelder-Mead around best
    if best_params is not None and elapsed() < max_time * 0.98:
        n = dim
        scale = 0.02 * ranges
        simplex = np.zeros((n+1, n)); fs = np.zeros(n+1)
        simplex[0] = best_params.copy(); fs[0] = best
        for i in range(1, n+1):
            if elapsed() >= max_time * 0.97: break
            simplex[i] = best_params.copy()
            simplex[i][(i-1) % n] += scale[(i-1) % n] * (1 if np.random.random() > 0.5 else -1)
            simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
        
        for _ in range(2000):
            if elapsed() >= max_time * 0.98: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + (c - simplex[-1])); fr = evaluate(xr)
            if fr < fs[0]:
                xe = clip(c + 2*(xr - c)); fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr
            else:
                xc = clip(c + 0.5*(simplex[-1] - c)); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1] = xc; fs[-1] = fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= max_time*0.98: break
                        simplex[i] = simplex[0] + 0.5*(simplex[i]-simplex[0])
                        simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-15: break
    
    return best
