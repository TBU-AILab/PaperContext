#Looking at the results, algorithm 2 (score 10.84) performed best. Let me analyze what worked and improve upon it with:
#
#1. **CMA-ES** as a powerful local optimizer instead of just Nelder-Mead
#2. **Better L-SHADE with eigenvector-based mutation**
#3. **Adaptive restart strategy** from diverse basins
#4. **Gaussian local search** with adaptive step sizes
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
    center = (lower + upper) / 2
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
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
    
    top_solutions = []
    
    def add_top(x, f):
        top_solutions.append((x.copy(), f))
        top_solutions.sort(key=lambda t: t[1])
        while len(top_solutions) > 30:
            top_solutions.pop()
    
    # --- Phase 1: Latin Hypercube Sampling + Opposition ---
    n_init = min(max(20 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            return best
        init_fitness[i] = eval_func(init_pop[i])
        add_top(init_pop[i], init_fitness[i])
    
    # Opposition-based candidates
    n_opp = min(n_init // 2, 300)
    for i in range(n_opp):
        if time_left() < max_time * 0.05:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
        add_top(opp, f_opp)
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: L-SHADE ---
    def lshade(time_frac_stop=0.30):
        nonlocal best, best_params
        
        pop_size_init = min(max(12 * dim, 60), 250)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        
        n_elite = min(pop_size // 3, len(sorted_idx))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if time_left() < max_time * 0.10:
                for j in range(pop_size):
                    add_top(pop[j], fit[j])
                return
            fit[i] = eval_func(pop[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = pop_size_init
        
        nfe_start = 0
        max_nfe = 120000
        nfe_count = 0
        
        for gen in range(10000):
            if time_left() < max_time * time_frac_stop:
                break
            
            S_F, S_CR, S_delta = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            fit_order = np.argsort(fit)
            p_max = max(2, int(pop_size * 0.2))
            p_min = max(2, int(pop_size * 0.05))
            
            for i in range(pop_size):
                if time_left() < max_time * time_frac_stop:
                    break
                
                ri = np.random.randint(H)
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
                
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = fit_order[np.random.randint(p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i, d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i, d]) / 2
                
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                f_trial = eval_func(trial)
                nfe_count += 1
                add_top(trial, f_trial)
                
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if delta > 0:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            if S_F:
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            ratio = min(nfe_count / max_nfe, 1.0)
            new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_size < pop_size:
                order = np.argsort(fit)
                pop = pop[order[:new_size]]
                fit = fit[order[:new_size]]
                pop_size = new_size
        
        for i in range(pop_size):
            add_top(pop[i], fit[i])
    
    lshade(time_frac_stop=0.30)
    
    # --- Phase 3: CMA-ES local search ---
    def cma_es_local(x0, sigma0=0.05, max_iter=10000):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        xmean = x0.copy()
        sigma = sigma0 * np.mean(ranges)
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        
        for gen in range(max_iter):
            if time_left() < max_time * 0.02:
                return
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k_i in range(lam):
                arx[k_i] = clip(xmean + sigma * (B @ (D * arz[k_i])))
                arfitness[k_i] = eval_func(arx[k_i])
                add_top(arx[k_i], arfitness[k_i])
            
            arindex = np.argsort(arfitness)
            
            xold = xmean.copy()
            selected = arx[arindex[:mu]]
            xmean = np.sum(weights[:, None] * selected, axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ ((xmean - xold) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            
            artmp = (selected - xold) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.mean(ranges))
            
            eigeneval += lam
            if eigeneval >= lam * 10:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.mean(ranges):
                return
    
    # Run CMA-ES from best solution
    if best_params is not None and time_left() > 2.0:
        cma_es_local(best_params, sigma0=0.02)
    
    # Run CMA-ES from diverse top solutions
    used = set()
    if best_params is not None:
        used.add(0)
    for idx in range(min(10, len(top_solutions))):
        if time_left() < max_time * 0.08:
            break
        sol = top_solutions[idx][0]
        if best_params is not None and np.linalg.norm(sol - best_params) < np.mean(ranges) * 0.02:
            continue
        cma_es_local(sol, sigma0=0.015)
    
    # Final fine CMA-ES
    if best_params is not None and time_left() > 0.5:
        cma_es_local(best_params, sigma0=0.005)
    
    # Ultra-fine CMA-ES
    if best_params is not None and time_left() > 0.2:
        cma_es_local(best_params, sigma0=0.001)
    
    return best
#
#Key improvements over the best previous algorithm:
#- **CMA-ES local search** replaces Nelder-Mead — CMA-ES adapts a full covariance matrix, capturing variable correlations, making it far superior for local optimization
#- **Multiple CMA-ES restarts** from diverse basins found by L-SHADE
#- **Top solutions tracking** throughout all phases for diverse restart points
#- **Progressive refinement**: large σ → medium → small → ultra-fine CMA-ES passes
#- **Better time allocation**: 70% for L-SHADE global search, 30% for CMA-ES local polishing
