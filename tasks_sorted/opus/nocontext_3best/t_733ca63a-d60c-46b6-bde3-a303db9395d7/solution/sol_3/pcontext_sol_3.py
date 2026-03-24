#Looking at the results, algorithm #1 (score 19.89) with simple DE + restarts performed best, while #2 (SHADE, score 26.06) was decent. I'll build an improved version combining SHADE-style adaptation with better local search (Nelder-Mead simplex), more aggressive exploitation, and multiple restart strategies.
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
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.96):
        return elapsed() < max_time * fraction
    
    def ev(x):
        nonlocal best, best_solution, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # --- Phase 1: LHS initialization ---
    pop_size = min(max(12 * dim, 40), 150)
    
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * rng
    
    fit = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok():
            return best
        fit[i] = ev(pop[i])
    
    # SHADE memory
    H = 8
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    mi = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # Nelder-Mead local search
    def nelder_mead(x0, max_evals_nm, initial_scale=0.05):
        nonlocal best
        n = len(x0)
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += initial_scale * rng[i] * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.array([ev(simplex[i]) for i in range(n + 1) if time_ok(0.94)])
        if len(f_simplex) < n + 1:
            return
        
        used = n + 1
        
        while used < max_evals_nm and time_ok(0.94):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = ev(xr); used += 1
            
            if fr < f_simplex[0]:
                # Expand
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = ev(xe); used += 1
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                # Contract
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lower, upper)
                fc = ev(xc); used += 1
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc; f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        if not time_ok(0.94): return
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = ev(simplex[i]); used += 1
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-14:
                break
    
    while time_ok():
        generation += 1
        S_F, S_CR, S_d = [], [], []
        
        sorted_idx = np.argsort(fit)
        
        for i in range(pop_size):
            if not time_ok():
                return best
            
            ri = np.random.randint(H)
            while True:
                Fi = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0, 1.0)
                if Fi > 0: break
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1 with archive
            p = max(2, int(0.11 * pop_size))
            pb = sorted_idx[np.random.randint(p)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            jr = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[jr] = True
            trial[mask] = mutant[mask]
            
            below = trial < lower; above = trial > upper
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
                pop[i] = trial; fit[i] = tf
        
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
        
        # Periodic local search
        if stagnation > 0 and stagnation % 5 == 0 and time_ok(0.85) and best_solution is not None:
            scale = 0.05 * max(0.1, 1.0 / (1 + stagnation * 0.1))
            nelder_mead(best_solution.copy(), max(dim * 3, 50), initial_scale=scale)
        
        if stagnation > 10 + dim:
            nk = max(3, pop_size // 5)
            si = np.argsort(fit)
            pop[:] = pop[si]; fit[:] = fit[si]
            for i in range(nk, pop_size):
                if not time_ok(): return best
                if np.random.random() < 0.5:
                    pop[i] = best_solution + 0.1 * rng * np.random.randn(dim) * np.random.random()
                else:
                    pop[i] = lower + np.random.random(dim) * rng
                pop[i] = np.clip(pop[i], lower, upper)
                fit[i] = ev(pop[i])
            M_F[:] = 0.5; M_CR[:] = 0.5; archive.clear(); stagnation = 0
    
    # Final Nelder-Mead refinement
    if time_ok(0.90) and best_solution is not None:
        nelder_mead(best_solution.copy(), max(dim * 5, 200), initial_scale=0.01)
    
    return best
