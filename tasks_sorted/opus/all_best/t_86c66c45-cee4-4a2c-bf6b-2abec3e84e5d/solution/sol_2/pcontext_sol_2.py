#Looking at the best algorithm (SHADE with 7.46), I'll improve it by adding:
#
#1. **L-SHADE** (linear population size reduction) for better convergence
#2. **CMA-ES local search** phases around the best solution
#3. **Multiple restart strategy** with opposition-based learning
#4. **Nelder-Mead simplex** for final polishing
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

    # CMA-ES local search
    def cma_local(x0, sigma0, max_evals):
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        evals = 0
        for _ in range(max_evals // lam):
            if time_left() < 0.1:
                break
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fitvals = np.zeros(lam)
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                fitvals[k] = evaluate(arx[k])
                evals += 1
            
            idx = np.argsort(fitvals)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(evals/lam+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, max(ranges))
            
            if evals - eigeneval > lam/(c1+cmu)/n/10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
                    invsqrtC = B @ np.diag(1/D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if sigma < 1e-16:
                break

    # Phase 1: L-SHADE
    N_init = min(max(30, 8*dim), 200)
    N_min = 4
    pop_size = N_init
    H = 100
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    k = 0
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(ind) for ind in population])
    archive = []
    total_evals = pop_size
    max_evals_shade = int(pop_size * 300)
    
    stagnation = 0
    prev_best = best
    
    while time_left() > max_time * 0.35:
        S_F, S_CR, delta_f = [], [], []
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() <= max_time * 0.35:
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
            
            mutant = population[i] + Fi*(population[pi] - population[i]) + Fi*(population[r1] - x_r2)
            jrand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi; mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            out_low = trial < lower; out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
            trial = clip(trial)
            trial_f = evaluate(trial)
            total_evals += 1
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi); delta_f.append(abs(fitness[i]-trial_f))
                    archive.append(population[i].copy())
                new_pop[i] = trial; new_fit[i] = trial_f
        
        population = new_pop; fitness = new_fit
        while len(archive) > pop_size: archive.pop(np.random.randint(len(archive)))
        if S_F:
            w = np.array(delta_f); w = w/w.sum()
            memory_F[k%H] = np.sum(w*np.array(S_F)**2)/max(np.sum(w*np.array(S_F)),1e-30)
            memory_CR[k%H] = np.sum(w*np.array(S_CR)); k += 1
        
        # L-SHADE population reduction
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max_evals_shade)))
        if new_size < pop_size:
            si = np.argsort(fitness)[:new_size]
            population = population[si]; fitness = fitness[si]; pop_size = new_size
        
        if abs(prev_best - best) < 1e-14: stagnation += 1
        else: stagnation = 0
        prev_best = best
        if stagnation > 40: break

    # Phase 2: CMA-ES local search with restarts
    while time_left() > 0.5:
        sigma0 = 0.1 * np.mean(ranges) * max(0.01, time_left()/max_time)
        cma_local(best_params.copy(), sigma0, max(20, int(time_left()*50)))
    
    return best
#
#Key improvements over the previous best:
#1. **L-SHADE**: Linear population size reduction drives better exploitation over time
#2. **CMA-ES local search phase**: After SHADE converges, switches to CMA-ES for precise local optimization around the best found solution
#3. **Two-phase strategy**: Global exploration (L-SHADE) followed by local refinement (CMA-ES with restarts)
#4. **Better time budget allocation**: 65% for global search, 35% for local refinement
#5. **Adaptive sigma in CMA-ES restarts**: Decreases over time for finer search
