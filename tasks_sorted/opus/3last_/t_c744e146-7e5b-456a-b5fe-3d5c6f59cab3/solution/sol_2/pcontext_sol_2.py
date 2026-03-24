#Looking at the previous algorithms, I'll improve by implementing a more sophisticated approach combining L-SHADE (adaptive DE) with CMA-ES local search, better initialization, and a multi-restart strategy.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (upper + lower) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # LHS initialization
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = lower[d] + (perm[i] + np.random.rand()) / n * ranges[d]
        return samples

    # ---- CMA-ES local search ----
    def cma_es(x0, sigma0, max_evals=None, max_t=None):
        nonlocal best, best_x
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        evals = 0
        t_end = elapsed() + max_t if max_t else float('inf')
        me = max_evals if max_evals else float('inf')
        
        while evals < me and elapsed() < t_end and time_left() > 0.1:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = xmean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
            
            # Evaluate
            fit = np.array([eval_f(arx[k]) for k in range(lam)])
            evals += lam
            
            # Sort
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            # Recombination
            xold = xmean.copy()
            xmean = arx[:mu].T @ weights
            
            zmean = arz[:mu].T @ weights
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*evals/lam)) / chiN) < (1.4 + 2/(n+1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            
            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, max(ranges) * 0.5)
            
            # Decomposition
            if evals - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * max(ranges):
                break
        
        return best

    # ---- L-SHADE ----
    pop_size_init = min(max(40, 12 * dim), 400)
    pop_size = pop_size_init
    min_pop = 4
    
    population = lhs_sample(pop_size)
    fitness = np.array([eval_f(ind) for ind in population])
    
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size_init
    
    total_budget_frac = 0.65  # fraction of time for SHADE
    shade_deadline = elapsed() + max_time * total_budget_frac
    
    generation = 0
    max_gen_estimate = max(1, int(max_time * 50))  # rough estimate
    
    while elapsed() < shade_deadline and time_left() > 0.5:
        generation += 1
        
        S_F, S_CR, delta_f = [], [], []
        
        p_best_rate = max(2.0/pop_size, 0.05 + 0.15 * (1 - generation/max_gen_estimate))
        
        for i in range(pop_size):
            if elapsed() >= shade_deadline or time_left() < 0.3:
                break
            
            ri = np.random.randint(0, memory_size)
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = max(2, int(p_best_rate * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest = sorted_idx[np.random.randint(0, p)]
            
            cands = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(cands)
            
            pool = [j for j in range(pop_size) if j != i and j != r1]
            arc_len = len(archive)
            r2_choice = np.random.randint(len(pool) + arc_len)
            if r2_choice < len(pool):
                xr2 = population[pool[r2_choice]]
            else:
                xr2 = archive[r2_choice - len(pool)]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            
            cross = np.random.rand(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            tf = eval_f(trial)
            
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - tf))
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = tf
        
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Linear population size reduction
        new_pop_size = max(min_pop, int(round(pop_size_init - (pop_size_init - min_pop) * (elapsed() / (max_time * total_budget_frac)))))
        if new_pop_size < pop_size:
            sidx = np.argsort(fitness)
            population = population[sidx[:new_pop_size]]
            fitness = fitness[sidx[:new_pop_size]]
            pop_size = new_pop_size

    # ---- CMA-ES phase on best solution ----
    if best_x is not None and time_left() > 0.5:
        sigma0 = np.mean(ranges) * 0.1
        cma_es(best_x.copy(), sigma0, max_t=time_left() * 0.7)
    
    # ---- Final fine-grained CMA-ES ----
    if best_x is not None and time_left() > 0.3:
        sigma0 = np.mean(ranges) * 0.01
        cma_es(best_x.copy(), sigma0, max_t=time_left() * 0.8)
    
    # ---- Final coordinate descent polish ----
    if best_x is not None and time_left() > 0.1:
        x_local = best_x.copy()
        f_local = best
        step = ranges * 0.001
        for _ in range(10):
            if time_left() < 0.05:
                break
            improved = False
            for d in range(dim):
                if time_left() < 0.05:
                    break
                for sign in [1, -1]:
                    xt = x_local.copy()
                    xt[d] = np.clip(xt[d] + sign * step[d], lower[d], upper[d])
                    ft = eval_f(xt)
                    if ft < f_local:
                        f_local = ft
                        x_local = xt
                        improved = True
            if not improved:
                step *= 0.5
    
    return best
