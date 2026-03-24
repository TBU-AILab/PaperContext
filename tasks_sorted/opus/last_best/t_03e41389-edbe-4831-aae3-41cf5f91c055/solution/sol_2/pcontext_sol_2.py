#Looking at the previous algorithm, I see several areas for improvement:
#
#1. The CMA-ES implementation has bugs (e.g., `old_mean` computed but unused, covariance update using wrong variables)
#2. Need better restart strategy with larger population diversity
#3. Should use a proper CMA-ES with sep-CMA for high dimensions
#4. Better time budget allocation
#
#Let me rewrite with a cleaner, more robust approach:
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Diverse initial sampling ---
    n_init = min(max(50, 20 * dim), 500)
    
    # Latin Hypercube Sampling
    population = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        population[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    init_fitness = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            break
        init_fitness.append((evaluate(population[i]), i))
    
    init_fitness.sort()
    
    # --- Phase 2: CMA-ES with restarts ---
    def run_cmaes(x0, sigma0, time_budget):
        nonlocal best, best_x
        t_start = elapsed()
        t_end = t_start + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if n <= 50:
            C = np.eye(n)
            use_full = True
        else:
            diagC = np.ones(n)
            use_full = False
        
        gen = 0
        stagnation = 0
        prev_best = best
        
        while elapsed() < min(t_end, max_time * 0.95):
            gen += 1
            
            if use_full:
                try:
                    eigenvalues, B = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            if use_full:
                arx = mean[None,:] + sigma * (arz @ (B * D[None,:]).T)
            else:
                arx = mean[None,:] + sigma * arz * np.sqrt(diagC)[None,:]
            arx = np.clip(arx, lower, upper)
            
            fits = np.array([evaluate(arx[k]) for k in range(lam)])
            idx = np.argsort(fits)
            
            xold = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ (mean-xold)/sigma if use_full else (mean-xold)/(sigma*np.sqrt(diagC)))
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/chiN < 1.4+2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(mean-xold)/sigma
            
            if use_full:
                artmp = (arx[idx[:mu]] - xold) / sigma
                C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(weights) @ artmp)
            else:
                diagC = (1-c1-cmu)*diagC + c1*(pc**2 + (1-hsig)*cc*(2-cc)*diagC) + cmu*np.sum(weights[:,None]*((arx[idx[:mu]]-xold)/sigma)**2, axis=0)
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges)*2)
            
            if best == prev_best: stagnation += 1
            else: stagnation = 0; prev_best = best
            if stagnation > 10 + 30*n/lam: break
            if sigma < 1e-16: break
    
    remaining = max_time * 0.95 - elapsed()
    if remaining > 0:
        n_restarts = max(1, min(8, int(remaining / max(0.5, elapsed() - 0.01))))
        for r in range(n_restarts):
            rem = max_time * 0.95 - elapsed()
            if rem <= 0.1: break
            budget = rem / (n_restarts - r)
            if r == 0:
                x0 = best_x.copy()
                sig = 0.2 * np.mean(ranges)
            elif r % 3 == 0:
                x0 = lower + np.random.rand(dim) * ranges
                sig = 0.3 * np.mean(ranges)
            else:
                x0 = best_x + (0.05 + 0.15*r/n_restarts) * ranges * (np.random.rand(dim) - 0.5)
                x0 = np.clip(x0, lower, upper)
                sig = 0.1 * np.mean(ranges)
            run_cmaes(x0, sig, budget)
    
    # Phase 3: Local refinement
    if best_x is not None:
        scale = 0.0005 * ranges
        for _ in range(30):
            if elapsed() >= max_time * 0.99: break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.99: break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] += direction * scale[d]
                    if evaluate(trial) < best: improved = True
            if not improved:
                scale *= 0.5
                if np.max(scale) < 1e-16: break
    
    return best
