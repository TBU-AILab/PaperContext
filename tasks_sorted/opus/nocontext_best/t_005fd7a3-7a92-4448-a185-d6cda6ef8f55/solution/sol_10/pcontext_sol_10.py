#Looking at the current algorithm, I see it's a SHADE variant scoring ~1.91. I'll improve it with:
#
#1. **Better initialization** (opposition-based learning)
#2. **L-SHADE** (linear population size reduction)
#3. **Aggressive Nelder-Mead local search** on best solutions
#4. **CMA-ES-inspired covariance adaptation** for local search
#5. **Multiple restarts** with progressively focused search
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
        return elapsed() < max_time * 0.97
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    # --- Nelder-Mead local search ---
    def nelder_mead(x0, initial_step=0.1, max_evals=500):
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n+1)
        f_simplex[0] = eval_func(x0)
        
        for i in range(n):
            if not time_left():
                return
            p = x0.copy()
            step = initial_step * ranges[i]
            p[i] += step
            if p[i] > upper[i]:
                p[i] = x0[i] - step
            simplex[i+1] = clip_to_bounds(p)
            f_simplex[i+1] = eval_func(simplex[i+1])
        
        ev = 0
        while ev < max_evals and time_left():
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip_to_bounds(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_func(xr); ev += 1
            
            if fr < f_simplex[0]:
                xe = clip_to_bounds(centroid + gamma * (xr - centroid))
                fe = eval_func(xe); ev += 1
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip_to_bounds(centroid + rho * (xr - centroid))
                    fc = eval_func(xc); ev += 1
                    if fc <= fr:
                        simplex[-1], f_simplex[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            if not time_left(): return
                            simplex[i] = clip_to_bounds(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i]); ev += 1
                else:
                    xc = clip_to_bounds(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_func(xc); ev += 1
                    if fc < f_simplex[-1]:
                        simplex[-1], f_simplex[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            if not time_left(): return
                            simplex[i] = clip_to_bounds(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i]); ev += 1
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    # --- CMA-ES ---
    def cma_es(x0, sigma0=0.3, max_evals=2000):
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
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0 * np.mean(ranges)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        ev = 0
        
        while ev < max_evals and time_left():
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fit = np.zeros(lam)
            for k in range(lam):
                if not time_left(): return
                arx[k] = clip_to_bounds(mean + sigma * (B @ (D * arz[k])))
                fit[k] = eval_func(arx[k])
                ev += 1
            
            idx = np.argsort(fit)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            mean = clip_to_bounds(mean)
            
            # CSA
            zmean = np.zeros(n)
            for k in range(mu):
                zmean += weights[k] * arz[idx[k]]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(ev/lam+1))) / chiN) < 1.4 + 2/(n+1)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance update
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[idx[k]] - old_mean) / sigma
            
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc)*C)
            for k in range(mu):
                C += cmu_val * weights[k] * np.outer(artmp[k], artmp[k])
            
            # Sigma update
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.mean(ranges))
            
            # Eigen decomposition
            if ev - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = ev
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            # Check convergence
            if sigma * np.max(D) < 1e-12 * np.mean(ranges):
                break
    
    # --- L-SHADE ---
    def lshade(time_budget_frac=0.5):
        nonlocal best, best_params
        budget_end = elapsed() + (max_time * 0.97 - elapsed()) * time_budget_frac
        
        init_pop_size = min(18 * dim, 200)
        min_pop_size = 4
        
        pop_size = init_pop_size
        population = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = np.random.permutation(pop_size)
            population[:, i] = lower[i] + (perm + np.random.rand(pop_size)) / pop_size * ranges[i]
        
        # Opposition-based initialization
        opp_pop = lower + upper - population
        all_pop = np.vstack([population, opp_pop])
        all_fit = np.zeros(len(all_pop))
        for i in range(len(all_pop)):
            if not time_left() or elapsed() > budget_end:
                if i < pop_size:
                    for j in range(i, pop_size):
                        all_fit[j] = eval_func(all_pop[j])
                    return
                break
            all_fit[i] = eval_func(all_pop[i])
        
        if len(all_fit) >= 2 * pop_size:
            idx = np.argsort(all_fit)[:pop_size]
            population = all_pop[idx]
            fitness = all_fit[idx]
        else:
            fitness = all_fit[:pop_size]
        
        mem_size = 6
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.5)
        mem_idx = 0
        archive = []
        max_archive = pop_size
        
        gen = 0
        max_gen_est = 300
        
        while time_left() and elapsed() < budget_end and pop_size >= min_pop_size:
            gen += 1
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            S_F, S_CR, S_w = [], [], []
            
            p_min = max(2, int(0.05 * pop_size))
            p_max = max(2, int(0.2 * pop_size))
            
            trials = np.empty_like(population)
            trial_fits = np.full(pop_size, np.inf)
            
            for i in range(pop_size):
                if not time_left() or elapsed() > budget_end:
                    break
                
                ri = np.random.randint(mem_size)
                if M_F[ri] > 0:
                    Fi = -1
                    while Fi <= 0:
                        Fi = min(M_F[ri] + 0.1 * np.random.standard_cauchy(), 1.5)
                else:
                    Fi = 0.5
                
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
                
                p_best = np.random.randint(p_min, p_max + 1)
                pbest_idx = np.random.randint(p_best)
                
                # Mutation: current-to-pbest/1
                candidates = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(candidates)
                
                pool_size = len(candidates) + len(archive)
                r2_idx = np.random.randint(pool_size)
                if r2_idx < len(candidates):
                    xr2 = population[candidates[r2_idx]]
                else:
                    xr2 = archive[r2_idx - len(candidates)]
                
                mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
                
                # Binomial crossover
                cross = np.random.rand(dim) < CRi
                jrand = np.random.randint(dim)
                cross[jrand] = True
                trial = np.where(cross, mutant, population[i])
                
                # Bounce-back boundary handling
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + population[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + population[i][d]) / 2
                
                t_fit = eval_func(trial)
                trials[i] = trial
                trial_fits[i] = t_fit
                
                if t_fit <= fitness[i]:
                    if t_fit < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > max_archive:
                            archive.pop(np.random.randint(len(archive)))
                        w = max(abs(fitness[i] - t_fit), 1e-30)
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_w.append(w)
                    population[i] = trial
                    fitness[i] = t_fit
            
            # Update memory
            if S_F:
                w = np.array(S_w)
                w /= w.sum() + 1e-30
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[mem_idx] = np.sum(w * scr)
                mem_idx = (mem_idx + 1) % mem_size
            
            # Linear population size reduction
            new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * gen / max_gen_est)))
            if new_pop_size < pop_size:
                sort_idx = np.argsort(fitness)
                population = population[sort_idx[:new_pop_size]]
                fitness = fitness[sort_idx
