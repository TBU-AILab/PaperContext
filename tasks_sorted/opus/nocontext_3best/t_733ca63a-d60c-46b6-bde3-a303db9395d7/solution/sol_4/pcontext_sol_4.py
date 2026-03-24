#Looking at the results, algorithm #1 (score 9.415) with SHADE + Nelder-Mead performed best. I'll improve it by:
#
#1. Better population sizing and initialization
#2. L-SHADE style population reduction
#3. Improved local search with multiple strategies (Nelder-Mead + pattern search)
#4. More aggressive exploitation near the best solution
#5. Better restart strategy with opposition-based learning
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
    
    # Opposition-based point
    def opposite(x):
        return np.clip(lower + upper - x, lower, upper)
    
    # LHS initialization
    init_pop_size = min(max(15 * dim, 50), 200)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
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
        # Also evaluate opposite
        if time_ok() and i < pop_size // 2:
            opp = opposite(pop[i])
            of = ev(opp)
            if of < fit[i]:
                pop[i] = opp
                fit[i] = of
    
    # SHADE memory
    H = 10
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    mi = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # Nelder-Mead local search
    def nelder_mead(x0, max_evals_nm, initial_scale=0.05):
        n = len(x0)
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            delta = initial_scale * rng[i]
            simplex[i + 1][i] += delta if np.random.random() > 0.5 else -delta
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if not time_ok(0.93):
                return
            f_simplex[i] = ev(simplex[i])
        
        used = n + 1
        
        while used < max_evals_nm and time_ok(0.93):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = ev(xr); used += 1
            
            if fr < f_simplex[0]:
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
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = ev(xc); used += 1
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.93): return
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = ev(simplex[i]); used += 1
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = ev(xc); used += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.93): return
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = ev(simplex[i]); used += 1
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / np.maximum(rng, 1e-30))
            if spread < 1e-15:
                break
    
    # Coordinate-wise local search
    def coord_search(x0, max_evals_cs, step_scale=0.01):
        x = x0.copy()
        fx = ev(x)
        used = 1
        step = step_scale * rng
        
        improved = True
        while improved and used < max_evals_cs and time_ok(0.93):
            improved = False
            for j in range(dim):
                if not time_ok(0.93) or used >= max_evals_cs:
                    return
                # Try positive step
                xp = x.copy(); xp[j] = np.clip(x[j] + step[j], lower[j], upper[j])
                fp = ev(xp); used += 1
                if fp < fx:
                    x = xp; fx = fp; improved = True; continue
                # Try negative step
                xn = x.copy(); xn[j] = np.clip(x[j] - step[j], lower[j], upper[j])
                fn = ev(xn); used += 1
                if fn < fx:
                    x = xn; fx = fn; improved = True
            step *= 0.5
            if np.max(step / rng) < 1e-15:
                break
    
    while time_ok():
        generation += 1
        S_F, S_CR, S_d = [], [], []
        
        sorted_idx = np.argsort(fit)
        current_pop_size = len(pop)
        
        for i in range(current_pop_size):
            if not time_ok():
                return best
            
            ri = np.random.randint(H)
            # Cauchy for F
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0: break
            Fi = min(Fi, 1.0)
            # Normal for CR
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1 with archive
            p = max(2, int(0.1 * current_pop_size))
            pb = sorted_idx[np.random.randint(p)]
            
            idxs = [j for j in range(current_pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = current_pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < current_pop_size else archive[r2 - current_pop_size]
            
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
                if len(archive) < init_pop_size:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fit[i] = tf
        
        # Update SHADE memory
        if S_F:
            w = np.array(S_d); w /= (w.sum() + 1e-30)
            M_F[mi] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
            M_CR[mi] = np.sum(w * np.array(S_CR))
            mi = (mi + 1) % H
        
        # L-SHADE: reduce population
        time_ratio = elapsed() / max_time
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * time_ratio)))
        if new_pop_size < len(pop):
            si = np.argsort(fit)
            pop = pop[si[:new_pop_size]]
            fit = fit[si[:new_pop_size]]
            sorted_idx = np.argsort(fit)
            while len(archive) > new_pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Periodic local search on best
        if stagnation > 0 and stagnation % 4 == 0 and time_ok(0.82) and best_solution is not None:
            scale = 0.05 * max(0.01, 1.0 / (1 + stagnation * 0.15))
            nelder_mead(best_solution.copy(), max(dim * 4, 80), initial_scale=scale)
        
        # Coordinate search periodically
        if stagnation > 0 and stagnation % 7 == 0 and time_ok(0.82) and best_solution is not None:
            coord_search(best_solution.copy(), max(dim * 3, 60), step_scale=0.02 / (1 + stagnation * 0.1))
        
        # Full restart if deeply stagnated
        if stagnation > 8 + dim:
            nk = max(2, len(pop) // 4)
            si = np.argsort(fit)
            pop[:] = pop[si]; fit[:] = fit[si]
            for i in range(nk, len(pop)):
                if not time_ok(): return best
                r = np.random.random()
                if r < 0.4:
                    pop[i] = best_solution + 0.05 * rng * np.random.randn(dim) * np.random.random()
                elif r < 0.7:
                    pop[i] = opposite(best_solution) + 0.1 * rng * np.random.randn(dim)
                else:
                    pop[i] = lower + np.random.random(dim) * rng
                pop[i] = np.clip(pop[i], lower, upper)
                fit[i] = ev(pop[i])
            M_F[:] = 0.5; M_CR[:] = 0.8; archive.clear(); stagnation = 0
    
    # Final intensive local search
    if time_ok(0.88) and best_solution is not None:
        nelder_mead(best_solution.copy(), max(dim * 8, 300), initial_scale=0.01)
    if time_ok(0.92) and best_solution is not None:
        coord_search(best_solution.copy(), max(dim * 5, 150), step_scale=0.005)
    if time_ok(0.94) and best_solution is not None:
        nelder_mead(best_solution.copy(), max(dim * 5, 200), initial_scale=0.001)
    
    return best
