#Looking at the results, algorithm 1 (score 10.22) performed best with DE + multi-start Nelder-Mead. Key improvements I'll make:
#
#1. **SHADE with proper success-history adaptation** (cleaner than algorithm 2)
#2. **Smaller population for faster generations**
#3. **More time for local search** (40% instead of 30%)
#4. **Add coordinate-wise local search** for fine-tuning
#5. **Better restart diversity** with opposition-based learning
#6. **Vectorized operations** where possible
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- LHS Initialization ---
    pop_size = min(max(15, 6 * dim), 150)
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.rand(pop_size)) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([eval_f(pop[i]) for i in range(pop_size)])
    
    idx = np.argsort(fit)
    pop = pop[idx]
    fit = fit[idx]
    
    # SHADE memory
    H = 30
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    mem_k = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    
    # --- Phase 1: SHADE ---
    de_deadline = max_time * 0.55
    
    while elapsed() < de_deadline:
        new_pop = pop.copy()
        new_fit = fit.copy()
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= de_deadline:
                break
            
            ri = np.random.randint(0, H)
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p_best = max(1, int(np.random.uniform(0.05, 0.2) * pop_size))
            pb = np.random.randint(0, p_best)
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(0, pool_size - 1)
            if r2 >= i: r2 += 1
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
            
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            
            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                delta = fit[i] - f_trial
                if delta > 0:
                    archive.append(pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                new_pop[i] = trial; new_fit[i] = f_trial
        
        pop, fit = new_pop, new_fit
        idx = np.argsort(fit); pop = pop[idx]; fit = fit[idx]
        
        if S_F:
            w = np.array(S_delta); w /= w.sum()
            M_F[mem_k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[mem_k] = np.sum(w * np.array(S_CR))
            mem_k = (mem_k + 1) % H
        
        if abs(prev_best - best) < 1e-14: stagnation += 1
        else: stagnation = 0
        prev_best = best
        
        if stagnation > 5:
            for j in range(pop_size//2, pop_size):
                if np.random.rand() < 0.4 and best_x is not None:
                    pop[j] = best_x + 0.05 * ranges * np.random.randn(dim)
                elif np.random.rand() < 0.5 and best_x is not None:
                    pop[j] = lower + upper - best_x + 0.1 * ranges * np.random.randn(dim)
                else:
                    pop[j] = lower + np.random.rand(dim) * ranges
                pop[j] = np.clip(pop[j], lower, upper); fit[j] = eval_f(pop[j])
            idx = np.argsort(fit); pop = pop[idx]; fit = fit[idx]; stagnation = 0
    
    # --- Phase 2: Multi-start Nelder-Mead ---
    n_starts = min(5, pop_size)
    for si in range(n_starts):
        if remaining() < max_time * 0.04: break
        x0 = pop[si].copy()
        n_s = dim + 1
        simplex = np.zeros((n_s, dim)); sf = np.zeros(n_s)
        simplex[0] = x0; sf[0] = fit[si]
        scale = 0.02 * ranges * (1 + si * 0.3)
        for j in range(1, n_s):
            simplex[j] = x0.copy()
            d_idx = (j - 1) % dim
            simplex[j][d_idx] += scale[d_idx] * (1 if np.random.rand() > 0.5 else -1)
            simplex[j] = np.clip(simplex[j], lower, upper); sf[j] = eval_f(simplex[j])
        
        deadline_nm = elapsed() + remaining() / (n_starts - si + 0.5)
        while elapsed() < min(deadline_nm, max_time * 0.97):
            oi = np.argsort(sf); simplex, sf = simplex[oi], sf[oi]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)
            if fr < sf[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper)
                fe = eval_f(xe)
                simplex[-1], sf[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < sf[-2]:
                simplex[-1], sf[-1] = xr, fr
            else:
                xc = np.clip(centroid + 0.5*((xr if fr < sf[-1] else simplex[-1]) - centroid), lower, upper)
                fc = eval_f(xc)
                if fc < sf[-1]: simplex[-1], sf[-1] = xc, fc
                else:
                    for j in range(1, n_s):
                        simplex[j] = simplex[0] + 0.5*(simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], lower, upper); sf[j] = eval_f(simplex[j])
    
    return best
