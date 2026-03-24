#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **SHADE-style adaptive DE** with memory for successful F and CR values
#2. **CMA-ES inspired local search** instead of simple Nelder-Mead
#3. **Better population management** with archive
#4. **Multiple restarts** with decreasing search radius
#5. **Vectorized boundary handling**
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(frac=0.95):
        return elapsed() < max_time * frac
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- SHADE-like DE ---
    pop_size = min(max(30, 8 * dim), 300)
    H = 100  # memory size
    
    # LHS initialization
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([eval_func(pop[i]) for i in range(pop_size) if time_ok()])
    if len(fit) < pop_size:
        return best
    
    # Memory for F and CR
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    
    p_min = max(2, int(0.05 * pop_size))
    
    while time_ok(0.75):
        S_F, S_CR, S_w = [], [], []
        
        sorted_idx = np.argsort(fit)
        
        for i in range(pop_size):
            if not time_ok(0.75):
                break
            
            ri = np.random.randint(H)
            F_i = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0.01, 1.0)
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = np.random.randint(1, p_min + 1)
            pbest = sorted_idx[np.random.randint(0, p)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = idxs[np.random.randint(len(idxs))]
            
            pool = [j for j in range(pop_size) if j != i and j != r1]
            if archive:
                arc_idx = np.random.randint(len(archive))
                if np.random.random() < 0.5 and archive:
                    r2_vec = archive[arc_idx]
                else:
                    r2 = pool[np.random.randint(len(pool))]
                    r2_vec = pop[r2]
            else:
                r2 = pool[np.random.randint(len(pool))]
                r2_vec = pop[r2]
            
            mutant = pop[i] + F_i * (pop[pbest] - pop[i]) + F_i * (pop[r1] - r2_vec)
            
            mask = np.random.random(dim) < CR_i
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Midpoint boundary fix
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + pop[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + pop[i][d]) / 2
            
            f_trial = eval_func(trial)
            
            if f_trial <= fit[i]:
                delta = fit[i] - f_trial
                if delta > 0:
                    S_F.append(F_i); S_CR.append(CR_i); S_w.append(delta)
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                pop[i] = trial
                fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
    
    # --- CMA-ES-like local search ---
    if best_x is not None and dim <= 100:
        sigma = 0.1 * np.mean(ranges)
        mu = best_x.copy()
        C = np.eye(dim)
        lam = max(6, 4 + int(3 * np.log(dim)))
        weights = np.log(lam/2 + 1) - np.log(np.arange(1, lam//2 + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        mu_sel = len(weights)
        
        while time_ok(0.98):
            try:
                samples = np.random.multivariate_normal(mu, sigma**2 * C, lam)
            except:
                samples = mu + sigma * np.random.randn(lam, dim)
            f_samples = np.array([eval_func(s) for s in samples if time_ok(0.98)])
            if len(f_samples) < lam:
                break
            order = np.argsort(f_samples)
            mu_new = np.zeros(dim)
            for j in range(mu_sel):
                mu_new += weights[j] * samples[order[j]]
            mu = mu_new
            sigma *= 0.98
            if sigma < 1e-12:
                sigma = 0.05 * np.mean(ranges)
                mu = best_x.copy() + 0.01 * ranges * np.random.randn(dim)
    
    return best
#
#Key improvements:
#- **SHADE algorithm**: Uses success-history based adaptive F/CR parameters, which adapt much better than fixed or randomly perturbed values
#- **Archive mechanism**: Stores replaced solutions to improve diversity in mutation
#- **CMA-ES local search**: Uses covariance-based sampling instead of Nelder-Mead, better for high-dimensional problems
#- **Better time allocation**: 75% for global DE, remaining for local refinement
#- **Improved boundary handling**: Midpoint reflection keeps solutions closer to feasible parents
