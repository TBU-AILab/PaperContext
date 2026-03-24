#Looking at the two algorithms, #1 (score 19.89) uses a simpler DE with restarts, while #2 (score 173.57) uses SHADE+CMA-ES but is more complex and slower. I'll build on #1's approach but incorporate SHADE-style parameter adaptation, better local search, and a more efficient structure.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    rng = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.96):
        return elapsed() < max_time * fraction
    
    def ev(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # Phase 1: LHS initialization
    pop_size = min(max(10 * dim, 40), 150)
    
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * rng
    
    fit = np.array([ev(pop[i]) for i in range(pop_size) if time_ok()])
    if len(fit) < pop_size:
        pop = pop[:len(fit)]
    
    # SHADE memory
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    mi = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    
    while time_ok():
        S_F, S_CR, S_d = [], [], []
        
        sorted_idx = np.argsort(fit)
        
        for i in range(len(pop)):
            if not time_ok():
                return best
            
            ri = np.random.randint(H)
            while True:
                Fi = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0, 1)
                if Fi > 0: break
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1 with archive
            p = max(2, int(0.15 * len(pop)))
            pb = sorted_idx[np.random.randint(p)]
            
            idxs = [j for j in range(len(pop)) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = len(pop) + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < len(pop) else archive[r2 - len(pop)]
            
            mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            jr = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi)
            mask[jr] = True
            trial[mask] = mutant[mask]
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            
            tf = ev(trial)
            if tf <= fit[i]:
                delta = fit[i] - tf
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_d.append(delta)
                if len(archive) < pop_size:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial
                fit[i] = tf
        
        if S_F:
            w = np.array(S_d); w /= w.sum()
            M_F[mi] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
            M_CR[mi] = np.sum(w * np.array(S_CR))
            mi = (mi + 1) % H
        
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 12 + dim:
            nk = max(3, len(pop) // 5)
            si = np.argsort(fit)
            pop[:] = pop[si]; fit[:] = fit[si]
            for i in range(nk, len(pop)):
                if not time_ok(): return best
                if np.random.random() < 0.5:
                    pop[i] = best_solution + 0.1 * rng * np.random.randn(dim) * np.random.random()
                else:
                    pop[i] = lower + np.random.random(dim) * rng
                pop[i] = np.clip(pop[i], lower, upper)
                fit[i] = ev(pop[i])
            M_F[:] = 0.5; M_CR[:] = 0.5; archive.clear(); stagnation = 0
    
    return best
