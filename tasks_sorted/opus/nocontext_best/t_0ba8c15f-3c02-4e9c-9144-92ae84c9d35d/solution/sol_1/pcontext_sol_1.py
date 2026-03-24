#Looking at the current algorithm, I'll improve it with several key changes:
#
#1. **Better CMA-ES implementation** with proper restart strategy (IPOP-CMA-ES)
#2. **Differential Evolution** as a complementary global search
#3. **More sophisticated local search** (pattern search + golden section refinement)
#4. **Better time management** across phases
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
    
    def time_ok(frac=0.97):
        return elapsed() < max_time * frac
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Sobol-like initialization with LHS ---
    n_init = min(max(30 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.15):
            break
        init_fitness[i] = eval_f(init_pop[i])
    
    valid = init_fitness < float('inf')
    if np.any(valid):
        sorted_idx = np.argsort(init_fitness[valid])
        pop = init_pop[valid][sorted_idx]
        fit = init_fitness[valid][sorted_idx]
    else:
        return best

    # --- Phase 2: Differential Evolution with adaptive parameters ---
    pop_size = min(max(10 * dim, 40), 100, len(pop))
    de_pop = pop[:pop_size].copy()
    de_fit = fit[:pop_size].copy()
    
    F = 0.8
    CR = 0.9
    
    generation = 0
    while time_ok(0.55):
        for i in range(pop_size):
            if not time_ok(0.55):
                break
            
            # Mutation: current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            best_idx = np.argmin(de_fit)
            
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = de_pop[i] + Fi * (de_pop[best_idx] - de_pop[i]) + Fi * (de_pop[a] - de_pop[b])
            
            # Crossover
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.0, 1.0)
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            
            trial = np.where(cross_points, mutant, de_pop[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (de_pop[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - de_pop[i][d])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            if f_trial <= de_fit[i]:
                de_pop[i] = trial.copy()
                de_fit[i] = f_trial
        
        generation += 1
        
        # Inject diversity occasionally
        if generation % 20 == 0:
            worst_idx = np.argsort(de_fit)[-pop_size//5:]
            for wi in worst_idx:
                de_pop[wi] = np.array([np.random.uniform(l, u) for l, u in bounds])
                de_fit[wi] = eval_f(de_pop[wi])

    # --- Phase 3: CMA-ES from best found ---
    def run_cmaes(x0, sigma0, time_frac_end):
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
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        
        gen = 0
        while time_ok(time_frac_end):
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if not time_ok(time_frac_end):
                    return
                fitnesses[k] = eval_f(arx[k])
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            mean = clip(mean)
            
            md = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ md) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * md / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges))
            
            if (gen + 1) % max(1, int(1 / (c1 + cmu_val) / n / 10)) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    Dsq, B = np.linalg.eigh(C)
                    Dsq = np.maximum(Dsq, 1e-20)
                    D = np.sqrt(Dsq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            gen += 1
            if sigma < 1e-16:
                break
    
    # Multiple CMA-ES restarts with increasing population
    sigma_base = 0.25 * np.mean(ranges)
    restart = 0
    while time_ok(0.88):
        if restart == 0 and best_params is not None:
            x0 = best_params.copy()
        elif restart < 3:
            # Perturbed best
            x0 = clip(best_params + np.random.randn(dim) * sigma_base * (0.5 ** restart) * ranges)
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        time_per_restart = min(0.12, (0.88 - elapsed()/max_time))
        end_frac = elapsed()/max_time + time_per_restart
        run_cmaes(x0, sigma_base * (0.7 ** min(restart, 5)), min(end_frac, 0.88))
        restart += 1

    # --- Phase 4: Fine local search ---
    if best_params is not None:
        step = 0.01 * ranges
        while time_ok(0.97):
            improved = False
            for d in range(dim):
                if not time_ok(0.97):
                    break
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = eval_f(trial)
                    if f < best:
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12:
                    break
    
    return best
