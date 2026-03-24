#Looking at the previous best algorithm (output 10.28), I can identify several improvements:
#
#1. **Add CMA-ES local search** - much better than Nelder-Mead for exploitation
#2. **Better vectorized DE operations** to save time per generation
#3. **Multiple restart strategy** with different search modes
#4. **Improved time allocation** - spend more time on DE, use CMA-ES for local refinement
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def evaluate_batch(xs):
        return np.array([evaluate(x) for x in xs])

    # ---- CMA-ES local search ----
    def cma_es(x0, sigma0, time_limit):
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
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
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
            
            fitvals = evaluate_batch(arx)
            if elapsed() >= time_limit:
                break
            
            idx = np.argsort(fitvals)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            # CSA
            zmean = np.sum(weights[:, None] * arz[idx[:mu]], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN) < (1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((mean - old_mean) / sigma)
            
            # Covariance update
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            gen += 1
            eigeneval += 1
            if eigeneval >= lam / (c1 + cmu) / n / 10 or gen <= 2:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    eigeneval = 0
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                break

    # ---- Phase 1: L-SHADE ----
    pop_size_init = min(max(14 * dim, 60), 250)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.9])
    if len(fitness) < pop_size:
        fitness = np.append(fitness, np.full(pop_size - len(fitness), float('inf')))
    
    mem_size = 6; M_F = np.full(mem_size, 0.5); M_CR = np.full(mem_size, 0.5); k = 0
    archive = []; stag = 0; prev_b = best
    
    while elapsed() < max_time * 0.55:
        SF, SCR, SD = [], [], []
        new_pop, new_fit = population.copy(), fitness.copy()
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55: break
            ri = np.random.randint(mem_size)
            Fi = min(max(np.random.standard_cauchy()*0.1+M_F[ri], 0.01), 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            p_cnt = max(2, int(0.15*pop_size))
            pb = np.random.choice(np.argsort(fitness)[:p_cnt])
            cs = [j for j in range(pop_size) if j!=i]; r1 = np.random.choice(cs)
            cs2 = pop_size+len(archive); r2 = np.random.randint(cs2)
            while r2==i or r2==r1: r2 = np.random.randint(cs2)
            xr2 = archive[r2-pop_size] if r2>=pop_size else population[r2]
            mut = population[i]+Fi*(population[pb]-population[i])+Fi*(population[r1]-xr2)
            mut = np.where(mut<lower, (lower+population[i])/2, np.where(mut>upper, (upper+population[i])/2, mut))
            mask = (np.random.random(dim)<CRi)|(np.arange(dim)==np.random.randint(dim))
            trial = np.where(mask, mut, population[i])
            ft = evaluate(trial)
            if ft < fitness[i]:
                SF.append(Fi); SCR.append(CRi); SD.append(fitness[i]-ft)
                if len(archive)<pop_size_init: archive.append(population[i].copy())
                elif archive: archive[np.random.randint(len(archive))]=population[i].copy()
                new_pop[i]=trial; new_fit[i]=ft
        population, fitness = new_pop, new_fit
        if SF:
            w=np.array(SD); w=w/(w.sum()+1e-30); sf=np.array(SF); sc=np.array(SCR)
            M_F[k]=np.sum(w*sf*sf)/(np.sum(w*sf)+1e-30); M_CR[k]=np.sum(w*sc); k=(k+1)%mem_size
        nps=max(min_pop_size,int(round(pop_size_init-(pop_size_init-min_pop_size)*elapsed()/(max_time*0.55))))
        if nps<pop_size:
            si=np.argsort(fitness); population=population[si[:nps]]; fitness=fitness[si[:nps]]; pop_size=nps
        stag = stag+1 if abs(prev_b-best)<1e-14 else 0; prev_b=best
        if stag>20:
            o=np.argsort(fitness); kp=max(pop_size//4,2)
            for i in range(kp,pop_size):
                if elapsed()>=max_time*0.55: break
                population[o[i]]=best_params+np.random.randn(dim)*0.1*ranges if np.random.random()<0.5 else lower+np.random.random(dim)*ranges
                population[o[i]]=np.clip(population[o[i]],lower,upper); fitness[o[i]]=evaluate(population[o[i]])
            stag=0

    # ---- Phase 2: CMA-ES refinement ----
    if best_params is not None:
        for sig in [0.2, 0.05, 0.01]:
            if elapsed() >= max_time * 0.95: break
            cma_es(best_params.copy(), sig * np.max(ranges), min(elapsed() + (max_time*0.95 - elapsed())/3, max_time*0.95))

    return best
