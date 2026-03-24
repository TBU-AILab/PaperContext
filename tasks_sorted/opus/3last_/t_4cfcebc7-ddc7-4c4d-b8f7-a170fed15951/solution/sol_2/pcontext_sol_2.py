#Looking at the results, algorithm 2 (score 20.49) significantly outperformed algorithm 1 (score 206.95). Let me analyze what worked and build an improved version combining the best elements:
#
#Key improvements:
#1. Better SHADE with L-SHADE (linear population size reduction)
#2. Multiple restart Nelder-Mead from diverse good points
#3. Better time allocation
#4. Opposition-based learning for initialization
#5. More aggressive local search refinement
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
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
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
    
    # Opposition-based candidates
    n_opp = min(n_init, 200)
    for i in range(n_opp):
        if time_left() < max_time * 0.05:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: L-SHADE ---
    def lshade():
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
                return
            fit[i] = eval_func(pop[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = pop_size_init
        
        max_gen = 10000
        gen_count = 0
        nfe_start = eval_count[0]
        max_nfe = 100000
        
        for gen in range(max_gen):
            gen_count += 1
            if time_left() < max_time * 0.15:
                return
            
            S_F, S_CR, S_delta = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            fit_order = np.argsort(fit)
            p_max = max(2, int(pop_size * 0.2))
            p_min = max(2, int(pop_size * 0.05))
            
            for i in range(pop_size):
                if time_left() < max_time * 0.12:
                    return
                
                ri = np.random.randint(H)
                mu_F = M_F[ri]
                mu_CR = M_CR[ri]
                
                Fi = -1
                while Fi <= 0:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
                
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = fit_order[np.random.randint(p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
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
                w = w / w.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            # L-SHADE: reduce population
            nfe_used = eval_count[0] - nfe_start
            ratio = min(nfe_used / max_nfe, 1.0)
            new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_size < pop_size:
                order = np.argsort(fit)
                pop = pop[order[:new_size]]
                fit = fit[order[:new_size]]
                pop_size = new_size
    
    lshade()
    
    # --- Phase 3: Nelder-Mead polish from multiple starts ---
    def nelder_mead(x0, scale=0.03, max_iter=3000):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += ranges[i] * scale * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = clip(p)
        f_s = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        for it in range(max_iter):
            if time_left() < max_time * 0.01: return
            o = np.argsort(f_s); simplex = simplex[o]; f_s = f_s[o]
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15: return
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_func(xr)
            if fr < f_s[0]:
                xe = clip(c + 2*(xr - c)); fe = eval_func(xe)
                simplex[-1], f_s[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_s[-2]: simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + 0.5*(xr - c)); fc = eval_func(xc)
                    if fc <= fr: simplex[-1], f_s[-1] = xc, fc; continue
                else:
                    xc = clip(c - 0.5*(c - simplex[-1])); fc = eval_func(xc)
                    if fc < f_s[-1]: simplex[-1], f_s[-1] = xc, fc; continue
                for i in range(1, n+1):
                    if time_left() < 0.01: return
                    simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                    f_s[i] = eval_func(simplex[i])
    
    if best_params is not None and time_left() > 0.5:
        nelder_mead(best_params, scale=0.02)
    if best_params is not None and time_left() > 0.2:
        nelder_mead(best_params, scale=0.005)
    if best_params is not None and time_left() > 0.1:
        nelder_mead(best_params, scale=0.001)
    
    return best
