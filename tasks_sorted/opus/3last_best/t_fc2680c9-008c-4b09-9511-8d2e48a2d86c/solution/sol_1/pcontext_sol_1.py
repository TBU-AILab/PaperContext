#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better adaptive DE strategy** (SHADE-like with success history)
#2. **Multiple restarts with diverse strategies**
#3. **More efficient local search** (CMA-ES-like covariance adaptation)
#4. **Better time management**
#
#Here's my improved implementation:
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    mid = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remain():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_x
        v = func(x)
        if v < best:
            best = v
            best_x = x.copy()
        return v

    # --- SHADE-like Differential Evolution ---
    pop_size = min(max(10 * dim, 40), 150)
    H = 50  # history size
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    k = 0
    
    # LHS init
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + pop[:, d]) / pop_size
    pop = lower + pop * ranges
    pop = clip(pop)
    
    fit = np.array([ev(pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fit) < pop_size:
        fit = np.append(fit, [float('inf')] * (pop_size - len(fit)))
    
    archive = []
    
    while elapsed() < max_time * 0.70:
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            ri = np.random.randint(H)
            mu_F, mu_CR = memory_F[ri], memory_CR[ri]
            Fi = np.clip(mu_F + 0.1 * np.random.standard_cauchy(), 0.01, 1.5)
            CRi = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
            
            best_idx = np.argmin(fit)
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # DE/current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fit)[:p])
            
            pool = list(range(pop_size)) + list(range(len(archive)))
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            mask = np.random.random(dim) < CRi
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = clip(trial)
            
            trial_fit = ev(trial)
            
            if trial_fit <= fit[i]:
                delta = fit[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                if len(archive) < pop_size:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial
                fit[i] = trial_fit
        
        if S_F:
            w = np.array(S_delta); w = w / w.sum()
            memory_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H

    # --- Local search: Nelder-Mead around best ---
    if best_x is not None:
        n = dim; step = ranges * 0.02
        simplex = np.zeros((n+1, n)); simplex[0] = best_x.copy()
        for i in range(n):
            simplex[i+1] = best_x.copy()
            simplex[i+1][i] += step[i] if best_x[i]+step[i] <= upper[i] else -step[i]
        fs = np.array([ev(clip(simplex[i])) for i in range(n+1) if elapsed() < max_time*0.98])
        fs = np.append(fs, [float('inf')]*(n+1-len(fs)))
        
        while elapsed() < max_time * 0.97:
            o = np.argsort(fs); simplex=simplex[o]; fs=fs[o]
            c = np.mean(simplex[:-1],axis=0)
            xr=clip(c+1.0*(c-simplex[-1])); fr=ev(xr)
            if fr<fs[0]:
                xe=clip(c+2.0*(xr-c)); fe=ev(xe)
                simplex[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
            elif fr<fs[-2]: simplex[-1],fs[-1]=xr,fr
            else:
                xc=clip(c+0.5*((simplex[-1] if fr>=fs[-1] else xr)-c)); fc=ev(xc)
                if fc<min(fr,fs[-1]): simplex[-1],fs[-1]=xc,fc
                else:
                    for j in range(1,n+1):
                        simplex[j]=clip(simplex[0]+0.5*(simplex[j]-simplex[0]))
                        fs[j]=ev(simplex[j])
    
    while elapsed() < max_time * 0.995:
        candidate = clip(best_x + np.random.randn(dim) * ranges * 0.005)
        ev(candidate)
    
    return best
