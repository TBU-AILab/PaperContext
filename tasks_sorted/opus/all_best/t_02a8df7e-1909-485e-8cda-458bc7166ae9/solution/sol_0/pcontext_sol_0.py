#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    # Generate LHS samples
    init_samples = []
    init_fitnesses = []
    
    for i in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        params = lower + np.random.rand(dim) * (upper - lower)
        fitness = func(params)
        init_samples.append(params)
        init_fitnesses.append(fitness)
        if fitness < best:
            best = fitness
            best_params = params.copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Sort and pick top candidates
    sorted_indices = np.argsort(init_fitnesses)
    
    # Run multiple restarts of a Nelder-Mead / CMA-ES hybrid
    def cma_es_search(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        
        n = dim
        lam = 4 + int(3 * np.log(n))  # population size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        counteval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        deadline = timedelta(seconds=max_time * budget_fraction)
        
        generation = 0
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
            
            # Eigendecomposition
            if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            else:
                if generation == 0:
                    D = np.ones(n)
                    B = np.eye(n)
            
            # Sample population
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return
                
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = func(arx[k])
                counteval += 1
                
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (1.0/D * (B.T @ (mean - old_mean)))) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2/(n + 1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Step size update
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            generation += 1
            
            # Restart condition
            if sigma < 1e-12:
                return
            if generation > 1000:
                return
    
    # Run CMA-ES from top candidates with restarts
    n_restarts = max(3, min(10, int(max_time / 2)))
    
    for restart in range(n_restarts):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            break
        
        # Pick starting point
        if restart < len(sorted_indices):
            x0 = init_samples[sorted_indices[restart]].copy()
        else:
            # Random restart near best or fully random
            if np.random.rand() < 0.5 and best_params is not None:
                x0 = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.rand(dim) * (upper - lower)
        
        sigma0 = 0.3 * np.mean(upper - lower)
        
        remaining = (timedelta(seconds=max_time * 0.98) - (datetime.now() - start)).total_seconds()
        if remaining <= 0:
            break
        
        budget_frac = min(0.98, ((datetime.now() - start).total_seconds() + remaining * 0.8) / max_time)
        
        cma_es_search(x0, sigma0, 0.98)
    
    # --- Phase 3: Local refinement with Nelder-Mead around best ---
    if best_params is not None and (datetime.now() - start) < timedelta(seconds=max_time * 0.95):
        # Simple coordinate-wise local search
        step = 0.001 * (upper - lower)
        current = best_params.copy()
        current_fit = best
        
        improved = True
        while improved:
            improved = False
            for i in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return best
                
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[i] = np.clip(trial[i] + direction * step[i], lower[i], upper[i])
                    f = func(trial)
                    if f < current_fit:
                        current = trial
                        current_fit = f
                        improved = True
                        if f < best:
                            best = f
                            best_params = trial.copy()
    
    return best
