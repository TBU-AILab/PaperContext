#Looking at the previous algorithm, I see a CMA-ES approach with Nelder-Mead refinement. Let me improve it with:
#
#1. Better initialization (more diverse sampling)
#2. More robust CMA-ES with proper restart strategy (IPOP-CMA-ES style)
#3. Better time management
#4. Differential Evolution as a complementary global search
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
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 1000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok():
            return best
        init_fitness[i] = eval_func(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: DE + CMA-ES hybrid with restarts ---
    
    # Differential Evolution phase
    def de_phase(time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        pop_size = min(max(10 * dim, 40), 200)
        # Initialize from best LHS samples + random
        n_elite = min(pop_size // 4, len(sorted_idx))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
        
        # Evaluate remaining
        for i in range(n_elite, pop_size):
            if not time_ok() or elapsed() > deadline:
                return
            fit[i] = eval_func(pop[i])
        
        F = 0.8
        CR = 0.9
        
        gen = 0
        while time_ok() and elapsed() < deadline:
            gen += 1
            # Adaptive parameters
            for i in range(pop_size):
                if not time_ok() or elapsed() > deadline:
                    return
                
                # current-to-pbest/1
                p = max(2, pop_size // 5)
                pbest_idx = np.argsort(fit)[:p]
                
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.0)
                CRi = CR + 0.1 * np.random.randn()
                CRi = np.clip(CRi, 0.0, 1.0)
                
                xpbest = pop[pbest_idx[np.random.randint(p)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - pop[r2])
                mutant = clip(mutant)
                
                # Binomial crossover
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = eval_func(trial)
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # --- Phase 3: CMA-ES ---
    def cma_es_run(x0, sigma0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        
        mean = x0.copy()
        sigma = sigma0
        
        gen = 0
        no_improve_count = 0
        prev_best = best
        
        while time_ok() and elapsed() < deadline:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if not time_ok() or elapsed() > deadline:
                    return
                fitnesses[k] = eval_func(arx[k])
            
            idx = np.argsort(fitnesses)
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            
            diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[idx[k]] - old_mean) / sigma
            
            C = ((1 - c1 - cmu_val) * C +
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu_val * sum(weights[k] * np.outer(artmp[k], artmp[k]) for k in range(mu)))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            if gen % max(1, int(1/(c1 + cmu_val + 1e-20)/n/10)) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if best < prev_best - 1e-10:
                no_improve_count = 0
                prev_best = best
            else:
                no_improve_count += 1
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges) or no_improve_count > 50 + 10*n:
                break
    
    # Allocate time: 40% DE, 55% CMA-ES restarts, 5% buffer
    de_budget = remaining() * 0.35
    de_phase(de_budget)
    
    # CMA-ES with increasing population (IPOP style)
    restart = 0
    while time_ok(0.95):
        rem = remaining()
        if rem < 0.1:
            break
        
        n_restarts_left = max(1, min(5, int(rem / 1.0)))
        t_budget = rem / n_restarts_left * 0.9
        
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim) * ranges
            sig = 0.3 * np.max(ranges)
        else:
            # Mix: sometimes from best, sometimes random
            if np.random.random() < 0.5 and best_params is not None:
                x0 = best_params + 0.1 * ranges * np.random.randn(dim)
                x0 = clip(x0)
                sig = 0.1 * np.max(ranges) * (1 + restart * 0.5)
            else:
                x0 = lower + np.random.random(dim) * ranges
                sig = 0.5 * np.max(ranges)
        
        cma_es_run(x0, sig, t_budget)
        restart += 1
    
    return best
