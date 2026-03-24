#Looking at the results, the best algorithm (output 11.70) uses SHADE + Nelder-Mead + coordinate search. I'll improve it by:
#
#1. **Adding CMA-ES** as the primary local search (much better than Nelder-Mead for continuous optimization)
#2. **Multiple restarts** with different strategies
#3. **Better time allocation** between exploration and exploitation
#4. **Opposition-based learning** for diversity
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_f(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Initial sampling with LHS ---
    pop_size = min(max(30, 10 * dim), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    # --- Phase 2: SHADE-like Adaptive DE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    de_time_frac = 0.45
    
    while elapsed() < max_time * de_time_frac:
        generation += 1
        
        S_F = []
        S_CR = []
        S_df = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        sorted_indices = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * (de_time_frac - 0.02):
                break
            
            ri = np.random.randint(0, H)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 1:
                    Fi = 1.0
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(sorted_indices[:p])
            
            indices = list(range(pop_size))
            indices.remove(i)
            r1 = np.random.choice(indices)
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(0, pool_size - 1)
            if r2 >= i:
                r2 += 1
            if r2 == r1:
                r2 = (r2 + 1) % pool_size
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(fitness[i] - f_trial)
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive:
                        archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            weights = np.array(S_df)
            weights = weights / (weights.sum() + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            stagnation = 0
            n_replace = pop_size // 3
            worst_indices = np.argsort(fitness)[-n_replace:]
            for idx in worst_indices:
                if np.random.random() < 0.6 and best_x is not None:
                    sigma = ranges * 0.05 * np.random.random()
                    population[idx] = best_x + np.random.randn(dim) * sigma
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = eval_f(population[idx])
    
    # --- Phase 3: CMA-ES local search ---
    def run_cmaes(x0, sigma0, time_limit):
        nonlocal best, best_x
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
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        while elapsed() < time_limit:
            try:
                if eigeneval > lam/(c1+cmu)/n/10:
                    eigeneval = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                else:
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
            except:
                C = np.eye(n)
                D = np.ones(n)
                B = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ np.diag(D) @ B.T)
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([eval_f(arx[i]) for i in range(lam)])
            if elapsed() >= time_limit:
                break
            
            idx = np.argsort(arfitness)
            mean_old = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (B @ np.diag(1/D) @ B.T) @ (mean - mean_old) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*evals/lam)) / chiN < 1.4 + 2/(n+1)
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - mean_old) / sigma
            
            artmp = (arx[idx[:mu]] - mean_old) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            eigeneval += lam
            
            if sigma < 1e-12 * np.max(ranges):
                break
    
    if best_x is not None:
        run_cmaes(best_x, 0.1 * np.mean(ranges), max_time * 0.85)
    
    # Multiple restarts with smaller sigma
    top_indices = np.argsort(fitness)[:5]
    for ti in top_indices:
        if elapsed() >= max_time * 0.95:
            break
        run_cmaes(population[ti], 0.02 * np.mean(ranges), min(elapsed() + (max_time * 0.95 - elapsed()) / max(1, len(top_indices)), max_time * 0.95))
    
    # --- Phase 4: Fine coordinate search ---
    if best_x is not None and elapsed() < max_time * 0.99:
        x_cur = best_x.copy()
        f_cur = best
        step = 0.001 * ranges
        for _ in range(200):
            if elapsed() >= max_time * 0.99:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.99:
                    break
                for sign in [1, -1]:
                    xt = x_cur.copy()
                    xt[d] += sign * step[d]
                    xt = np.clip(xt, lower, upper)
                    ft = eval_f(xt)
                    if ft < f_cur:
                        x_cur, f_cur = xt, ft
                        step[d] *= 1.5
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    break
    
    return best
