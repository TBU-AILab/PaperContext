#Looking at the results, algorithm #1 (score 1.74) performed best with L-SHADE + Nelder-Mead + coordinate search. I'll improve it with:
#
#1. Better CMA-ES-like local search for exploitation
#2. Improved parameter adaptation (jSO-style)
#3. Multi-strategy mutation (rand/1, current-to-pbest/1)
#4. More aggressive population reduction
#5. Better time allocation between exploration and exploitation
#6. Gaussian process-inspired surrogate for local search direction
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
    
    def opposite(x):
        return np.clip(lower + upper - x, lower, upper)
    
    # LHS initialization
    init_pop_size = min(max(12 * dim, 50), 200)
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
        if time_ok() and i < pop_size // 3:
            opp = opposite(pop[i])
            of = ev(opp)
            if of < fit[i]:
                pop[i] = opp
                fit[i] = of
    
    # SHADE memory
    H = 12
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
        no_improve = 0
        
        while used < max_evals_nm and time_ok(0.93):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            old_best_f = f_simplex[0]
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
            
            if f_simplex.min() >= old_best_f - 1e-15:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve > n + 5:
                break
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / np.maximum(rng, 1e-30))
            if spread < 1e-15:
                break
    
    # Coordinate-wise local search with golden section
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
                xp = x.copy(); xp[j] = np.clip(x[j] + step[j], lower[j], upper[j])
                fp = ev(xp); used += 1
                if fp < fx:
                    x = xp; fx = fp; improved = True
                    # Accelerate in this direction
                    while used < max_evals_cs and time_ok(0.93):
                        step[j] *= 1.5
                        xp2 = x.copy(); xp2[j] = np.clip(x[j] + step[j], lower[j], upper[j])
                        fp2 = ev(xp2); used += 1
                        if fp2 < fx:
                            x = xp2; fx = fp2
                        else:
                            step[j] /= 1.5
                            break
                    continue
                xn = x.copy(); xn[j] = np.clip(x[j] - step[j], lower[j], upper[j])
                fn = ev(xn); used += 1
                if fn < fx:
                    x = xn; fx = fn; improved = True
                    while used < max_evals_cs and time_ok(0.93):
                        step[j] *= 1.5
                        xn2 = x.copy(); xn2[j] = np.clip(x[j] - step[j], lower[j], upper[j])
                        fn2 = ev(xn2); used += 1
                        if fn2 < fx:
                            x = xn2; fx = fn2
                        else:
                            step[j] /= 1.5
                            break
            step *= 0.5
            if np.max(step / rng) < 1e-15:
                break

    # Simple CMA-like local search
    def cma_local(x0, max_evals, sigma0=0.05):
        n = dim
        mu_lam = max(4, n)
        lam = 2 * mu_lam
        sigma = sigma0
        mean = x0.copy()
        C = np.eye(n)
        used = 0
        
        for _ in range(max_evals // lam + 1):
            if not time_ok(0.93) or used >= max_evals:
                return
            
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                L = np.eye(n)
            
            samples = np.zeros((lam, n))
            f_samples = np.zeros(lam)
            for k in range(lam):
                if not time_ok(0.93) or used >= max_evals:
                    return
                z = np.random.randn(n)
                samples[k] = np.clip(mean + sigma * L @ z, lower, upper)
                f_samples[k] = ev(samples[k])
                used += 1
            
            idx = np.argsort(f_samples)
            selected = samples[idx[:mu_lam]]
            
            new_mean = np.mean(selected, axis=0)
            diff = selected - mean
            C = 0.8 * C + 0.2 * (diff.T @ diff) / mu_lam
            # Regularize
            C = 0.9 * C + 0.1 * np.eye(n)
            mean = new_mean
            
            if f_samples[idx[0]] < f_samples[idx[-1]] * 0.999:
                sigma *= 1.0
            else:
                sigma *= 0.8
            
            if sigma < 1e-15:
                break
    
    while time_ok(0.75):
        generation += 1
        S_F, S_CR, S_d = [], [], []
        
        sorted_idx = np.argsort(fit)
        current_pop_size = len(pop)
        
        for i in range(current_pop_size):
            if not time_ok(0.75):
                break
            
            ri = np.random.randint(H)
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0: break
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # jSO-style: adjust F and CR based on generation progress
            time_ratio = elapsed() / max_time
            if time_ratio < 0.2:
                Fi = max(Fi, 0.7)
                CRi = max(CRi, 0.9)
            elif time_ratio < 0.4:
                Fi = max(Fi, 0.5)
            
            p = max(2, int(max(0.05, 0.2 - 0.15 * time_ratio) * current_pop_size))
            pb = sorted_idx[np.random.randint(p)]
            
            # Multi-strategy mutation
            strat = np.random.random()
            idxs = [j for j in range(current_pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = current_pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < current_pop_size else archive[r2 - current_pop_size]
            
            if strat < 0.8:
                # current-to-pbest/1
                mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
            else:
                # rand/1 for diversity
                r3_idxs = [j for j in idxs if j != r1]
                if r3_idxs:
                    r3 = np.random.choice(r3_idxs)
                else:
                    r3 = r1
                mutant = pop[r1] + Fi * (pop[pb] - xr2) + 0.5 * Fi * (pop[r3] - pop[i])
            
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
        
        if S_F:
            w = np.array(S_d); w /= (w.sum() + 1e-30)
            M_F[mi] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
            M_CR[mi] = np.sum(w * np.array(S_CR))
            mi = (mi + 1) % H
        
        # L-SHADE population reduction
        time_ratio = elapsed() / max_time
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * min(time_ratio / 0.75, 1.0))))
        if new_pop_size < len(pop):
            si = np.argsort(fit)
            pop = pop[si[:new_pop_size]]
            fit = fit[si[:new_pop_size]]
            sorted_idx = np.argsort(fit)
            
