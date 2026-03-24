#Looking at the results, the best algorithm (9.56) uses SHADE + CMA-ES-like local search. I'll improve it with:
#
#1. **Proper CMA-ES** with covariance matrix adaptation instead of isotropic sampling
#2. **Multiple restarts** of local search from top DE solutions
#3. **Better population sizing and time allocation**
#4. **Improved SHADE with linear population size reduction (L-SHADE)**
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
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS Initialization ---
    init_pop_size = min(max(40, 10 * dim), 400)
    
    population = np.zeros((init_pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(init_pop_size)
        for j in range(init_pop_size):
            population[perm[j], i] = (j + np.random.random()) / init_pop_size
    population = lower + population * ranges
    
    fitness = np.full(init_pop_size, float('inf'))
    for i in range(init_pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_func(population[i])

    pop_size = init_pop_size
    
    # --- Phase 2: L-SHADE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = init_pop_size
    min_pop_size = max(4, dim)
    
    gen = 0
    while elapsed() < max_time * 0.55:
        S_F, S_CR, S_delta = [], [], []
        
        sorted_idx = np.argsort(fitness[:pop_size])
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(memory_size)
            if M_F[ri] <= 0:
                Fi = 0.5
            else:
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi >= 2.0:
                        Fi = M_F[ri]  # resample
                Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            p = max(2, int(np.random.uniform(2.0/pop_size, 0.2) * pop_size))
            pbest = sorted_idx[np.random.randint(0, p)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            trial_fit = eval_func(trial)
            
            if trial_fit <= fitness[i]:
                delta = fitness[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = trial_fit
        
        if S_F:
            w = np.array(S_delta) / (sum(S_delta) + 1e-30)
            M_F[k] = float(np.sum(w * np.array(S_F)**2)) / (float(np.sum(w * np.array(S_F))) + 1e-30)
            M_CR[k] = float(np.sum(w * np.array(S_CR)))
            k = (k + 1) % memory_size
        
        gen += 1
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / (max_time * 0.55))))
        if new_pop_size < pop_size:
            idx = np.argsort(fitness[:pop_size])[:new_pop_size]
            population = population[idx].copy()
            fitness = fitness[idx].copy()
            pop_size = new_pop_size

    # --- Phase 3: CMA-ES local search with restarts ---
    sorted_fit = np.argsort(fitness[:pop_size])
    restart_points = [best_params.copy()] + [population[sorted_fit[i]].copy() for i in range(min(3, pop_size))]
    
    for rp_idx, start_point in enumerate(restart_points):
        if elapsed() >= max_time * 0.95:
            break
        
        mean = start_point.copy()
        sigma = 0.05 * (1 + rp_idx * 0.5)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        lam = max(6, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        time_frac = 0.55 + (0.95 - 0.55) * (rp_idx + 1) / len(restart_points)
        
        while elapsed() < max_time * time_frac:
            try:
                sqC = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                sqC = np.eye(dim)
            
            arz = np.random.randn(lam, dim)
            arx = np.array([clip(mean + sigma * sqC @ z) for z in arz])
            fits = np.array([eval_func(x) for x in arx if elapsed() < max_time * 0.95])
            if len(fits) < mu:
                break
            
            idx = np.argsort(fits)
            zmean = np.sum(weights[:, None] * arz[idx[:mu]], axis=0)
            mean = clip(mean + sigma * sqC @ zmean)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(evals[0]//lam+1))) < (1.4 + 2/(dim+1)) * chiN)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqC @ zmean)
            
            artmp = sqC @ arz[idx[:mu]].T
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc * (2-cc) * C) + cmu * (artmp * weights) @ artmp.T
            
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 1.0)
            
            if sigma < 1e-15:
                break

    # --- Phase 4: Final fine-tuning ---
    scale = 0.005
    while elapsed() < max_time * 0.99:
        trial = best_params + np.random.randn(dim) * ranges * scale
        f = eval_func(clip(trial))
        if f >= best:
            scale *= 0.995
            if scale < 1e-15:
                scale = 0.01
    
    return best
