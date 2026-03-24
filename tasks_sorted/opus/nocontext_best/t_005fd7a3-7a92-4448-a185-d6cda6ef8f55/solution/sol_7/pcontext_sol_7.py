#Looking at the current algorithm, I need to improve the optimization strategy. Key issues I see:
#
#1. **L-SHADE population reduction** estimate is poor - using fixed `max_gen_estimate=5000`
#2. **CMA-ES restarts** could be more strategic with IPOP/BIPOP
#3. **No Nelder-Mead or Powell local search** for final refinement
#4. **Memory parameters** could be tuned better
#5. **Time allocation** might not be optimal
#
#Let me build a more robust hybrid optimizer with better CMA-ES, improved L-SHADE, and a proper local search phase.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def time_left():
        return max_time * 0.99 - elapsed()
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    best = float('inf')
    best_params = None
    
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
    
    def bounce_back(mutant, parent):
        x = mutant.copy()
        lo_mask = x < lower
        hi_mask = x > upper
        x[lo_mask] = (lower[lo_mask] + parent[lo_mask]) / 2.0
        x[hi_mask] = (upper[hi_mask] + parent[hi_mask]) / 2.0
        return x

    # ==================== L-SHADE ====================
    def run_lshade(time_budget):
        nonlocal best, best_params
        budget_end = elapsed() + time_budget
        
        N_init = min(max(18 * dim, 50), 400)
        N_min = 4
        pop_size = N_init
        
        # Latin Hypercube Sampling
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        
        # Opposition-based learning
        if time_left() > 0.5 and elapsed() < budget_end:
            opp_pop = lower + upper - population
            opp_fit = np.array([eval_func(opp_pop[i]) for i in range(pop_size) if elapsed() < budget_end])
            if len(opp_fit) == pop_size:
                all_pop = np.vstack([population, opp_pop])
                all_fit = np.concatenate([fitness, opp_fit])
                order = np.argsort(all_fit)[:pop_size]
                population = all_pop[order].copy()
                fitness = all_fit[order].copy()
        
        H = 100
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.3)
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        total_evals_estimate = 0
        evals_used = 0
        gen = 0
        
        while elapsed() < budget_end and time_left() > 0.05 and pop_size >= N_min:
            gen += 1
            S_CR, S_F, S_delta = [], [], []
            
            sorted_idx = np.argsort(fitness)
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            # Vectorized generation of CR and F
            ri_all = np.random.randint(0, H, size=pop_size)
            
            for i in range(pop_size):
                if elapsed() >= budget_end or time_left() <= 0.05:
                    break
                
                ri = ri_all[i]
                
                if M_CR[ri] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                F_i = -1
                while F_i <= 0:
                    F_i = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if F_i >= 1.0:
                        F_i = 1.0
                        break
                
                p_i = np.random.uniform(max(2.0 / pop_size, 0.05), 0.2)
                n_pbest = max(1, int(np.ceil(p_i * pop_size)))
                pbest_idx = sorted_idx[np.random.randint(n_pbest)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                combined_size = pop_size + len(archive)
                r2 = np.random.randint(combined_size)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 25:
                    r2 = np.random.randint(combined_size)
                    attempts += 1
                
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - xr2)
                mutant = bounce_back(mutant, population[i])
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                trial_f = eval_func(trial)
                evals_used += 1
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    if delta > 0:
                        S_CR.append(CR_i)
                        S_F.append(F_i)
                        S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            if S_CR and S_F:
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                
                M_F[k_idx] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                
                if max(scr) == 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = np.sum(w * scr ** 2) / (np.sum(w * scr) + 1e-30)
                
                k_idx = (k_idx + 1) % H
            
            # LPSR: linear population size reduction based on time fraction
            time_frac = min(1.0, (elapsed() - (budget_end - time_budget)) / time_budget)
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * time_frac)))
            if new_pop_size < pop_size:
                order = np.argsort(fitness)[:new_pop_size]
                population = population[order]
                fitness = fitness[order]
                pop_size = new_pop_size
                archive_max = max(N_min, new_pop_size)
                while len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))

    # ==================== CMA-ES with restarts ====================
    def run_cmaes(time_budget):
        nonlocal best, best_params
        budget_end = elapsed() + time_budget
        
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        restart_count = 0
        lam_mult = 1
        
        while elapsed() < budget_end and time_left() > 0.1:
            restart_count += 1
            
            # IPOP: double population on restarts
            if restart_count <= 1:
                lam = base_lam
            else:
                lam_mult *= 2
                lam = min(base_lam * lam_mult, 500)
            
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
            
            if best_params is not None and restart_count == 1:
                xmean = best_params.copy()
                sigma = 0.15 * np.mean(ranges)
            elif best_params is not None:
                # Perturb from best
                xmean = best_params + np.random.randn(n) * 0.1 * ranges
                xmean = clip(xmean)
                sigma = (0.05 + 0.2 * np.random.random()) * np.mean(ranges)
            else:
                xmean = np.array([np.random.uniform(l, u) for l, u in bounds])
                sigma = 0.3 * np.mean(ranges)
            
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
            counteval = 0
            stag_count = 0
            prev_best_f = best
            
            while elapsed() < budget_end and time_left() > 0.05:
                arx = np.zeros((lam, n))
                arf = np.zeros(lam)
                
                for k in range(lam):
                    if elapsed() >= budget_end or time_left() <= 0.05:
                        return
                    z = np.random.randn(n)
                    arx[k] = xmean + sigma * (B @ (D * z))
                    arx[k] = clip(arx[k])
                    arf[k] = eval_func(arx[k])
                    counteval += 1
                
                arindex = np.argsort(arf)
                xold = xmean.copy()
                
                xmean = np.zeros(n)
                for j in range(mu):
                    xmean += weights[j] * arx[arindex[j]]
                
                diff = (xmean - xold) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
                
                ps_norm = np.linalg.norm(ps)
                hsig = float(ps_norm / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2.0 / (n + 1))
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = np.zeros((mu, n))
                for j in range(mu):
                    artmp[j] = (arx[arindex[j]] - xold) / sigma
                
                C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                for j in range(mu):
                    C += cmu_val * weights[j] * np.outer(artmp[j], artmp[j])
                
                sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
                sigma = min(sigma, np.mean(ranges))
                
                if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        break
                
                # Stagnation check
                if best < prev_best_f - 1e-10:
                    stag_count = 0
                    prev_best_f = best
                else:
                    stag_count += 1
                
                if sigma < 1e-14 * np.mean(ranges) or stag_count > 50 + 10 * n:
                    break
                
                if np.max(D) / np.min(D) > 1e7:
                    break

    # ==================== Nelder-Mead ====================
    def run_nelder_mead(time_budget):
        nonlocal best, best_params
        if best_params is None:
            return
        budget_end = elapsed() + time_budget
        
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex around best
        scale = 0.02 * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_vals = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        
        while elapsed()
