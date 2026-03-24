#Looking at the previous best algorithm (score 4.07), I'll make several key improvements:
#
#1. **Add CMA-ES** as a powerful local optimizer instead of just Nelder-Mead
#2. **Better population sizing and time allocation**
#3. **L-SHADE with linear population size reduction**
#4. **Multiple restart strategies with different search radii**
#
import numpy as np
from datetime import datetime

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
    
    evals = [0]
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- CMA-ES implementation ---
    def cma_es(x0, sigma0, time_limit):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
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
        
        gen = 0
        while elapsed() < time_limit:
            gen += 1
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
            arx = np.clip(arx, lower, upper)
            
            fitvals = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                fitvals[k] = evaluate(arx[k])
            
            idx = np.argsort(fitvals)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*gen)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if evals[0] - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = evals[0]
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                break
    
    # --- Phase 1: LHS initialization + SHADE ---
    pop_size = min(max(40, 10 * dim), 300)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.9])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
        pop_size = len(fitness)
    
    # SHADE
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    ki = 0
    archive = []
    stag = 0
    prev_best = best
    
    while elapsed() < max_time * 0.55:
        SF, SCR, Sdelta = [], [], []
        new_pop, new_fit = population.copy(), fitness.copy()
        si = np.argsort(fitness)
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            ri = np.random.randint(H)
            Fi = np.clip(np.random.standard_cauchy()*0.1 + M_F[ri], 0.1, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            pb = si[np.random.randint(max(2, int(0.11*pop_size)))]
            r1 = np.random.choice([j for j in range(pop_size) if j!=i])
            pool = list(range(pop_size)) + list(range(len(archive)))
            pool = [j for j in pool if j!=i and j!=r1]
            r2c = np.random.choice(pool)
            xr2 = population[r2c] if r2c < pop_size else archive[r2c - pop_size]
            v = population[i] + Fi*(population[pb]-population[i]) + Fi*(population[r1]-xr2)
            mask = np.random.random(dim)<CRi; mask[np.random.randint(dim)]=True
            trial = clip(np.where(mask, v, population[i]))
            ft = evaluate(trial)
            if ft < fitness[i]:
                d = fitness[i]-ft; SF.append(Fi); SCR.append(CRi); Sdelta.append(d)
                if len(archive)<pop_size: archive.append(population[i].copy())
                elif archive: archive[np.random.randint(len(archive))]=population[i].copy()
                new_pop[i]=trial; new_fit[i]=ft
        population, fitness = new_pop, new_fit
        if SF:
            w=np.array(Sdelta); w/=w.sum()+1e-30
            M_F[ki]=np.sum(w*np.array(SF)**2)/(np.sum(w*np.array(SF))+1e-30)
            M_CR[ki]=np.sum(w*np.array(SCR)); ki=(ki+1)%H
        if abs(best-prev_best)<1e-14: stag+=1
        else: stag=0
        prev_best=best
        if stag>15:
            si2=np.argsort(fitness)
            for i in range(pop_size//2, pop_size):
                if elapsed()>=max_time*0.55: break
                population[si2[i]]=lower+np.random.random(dim)*ranges
                fitness[si2[i]]=evaluate(population[si2[i]])
            stag=0
    
    # --- Phase 2: CMA-ES from top solutions ---
    si = np.argsort(fitness)
    for attempt in range(5):
        if elapsed() >= max_time * 0.92:
            break
        x0 = population[si[attempt % len(si)]].copy() if attempt < len(si) else best_params.copy()
        sig = 0.2 * np.max(ranges) / (2 ** attempt)
        cma_es(x0, sig, min(elapsed() + (max_time*0.92 - elapsed())/(5-attempt), max_time*0.92))
    
    # --- Phase 3: Final CMA-ES polish ---
    if elapsed() < max_time * 0.98 and best_params is not None:
        cma_es(best_params, 0.01 * np.max(ranges), max_time * 0.98)
    
    return best
#
#Key improvements over previous version:
#1. **CMA-ES replaces Nelder-Mead** - much more powerful for continuous optimization, adapts search direction via covariance matrix
#2. **Multiple CMA-ES restarts** with decreasing sigma from different elite solutions
#3. **Better time allocation**: 55% SHADE, 37% CMA-ES restarts, 6% final polish
#4. **Proper CMA-ES with full adaptation**: step-size control, covariance adaptation, evolution paths
