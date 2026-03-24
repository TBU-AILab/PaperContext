#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Better adaptive DE strategy** (SHADE/L-SHADE style with success history)
#2. **More restarts with different strategies**
#3. **Better time management** - spend more time in the productive phases
#4. **CMA-ES for local search** instead of Nelder-Mead (more effective in higher dimensions)
#5. **Multiple restart strategy** to escape local optima
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
    
    def remaining():
        return max_time - elapsed()
    
    def eval_clip(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f, x
    
    # ---- SHADE-like Differential Evolution with restarts ----
    def run_shade(time_fraction):
        nonlocal best, best_x
        deadline = elapsed() + max_time * time_fraction
        
        pop_size = max(30, 5 * dim)
        pop_size = min(pop_size, 300)
        
        # Initialize population with LHS
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        pop = lower + pop * ranges
        
        fit = np.array([eval_clip(pop[i])[0] for i in range(pop_size)])
        if elapsed() > deadline:
            return
        
        # SHADE memory
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        # Archive
        archive = []
        archive_max = pop_size
        
        p_min = 2.0 / pop_size
        p_max = 0.2
        
        gen = 0
        stag_count = 0
        prev_best_gen = best
        
        while elapsed() < deadline:
            gen += 1
            S_F = []
            S_CR = []
            S_delta = []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            # Sort for pbest
            sorted_idx = np.argsort(fit)
            
            for i in range(pop_size):
                if elapsed() >= deadline:
                    return
                
                # Pick memory index
                ri = np.random.randint(0, H)
                
                # Generate F
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                # Generate CR
                CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                # pbest
                p = np.random.uniform(p_min, p_max)
                n_pbest = max(1, int(p * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, n_pbest)]
                
                # Mutation: current-to-pbest/1 with archive
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = list(range(pop_size)) + list(range(len(archive)))
                combined = [c for c in combined if c != i]
                r2_idx = np.random.choice(len(combined))
                r2_val = combined[r2_idx]
                if r2_val < pop_size:
                    xr2 = pop[r2_val]
                else:
                    xr2 = archive[r2_val - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Binomial crossover
                j_rand = np.random.randint(0, dim)
                mask = (np.random.rand(dim) < CRi)
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back bounds
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                
                f_trial, trial = eval_clip(trial)
                
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if f_trial < fit[i]:
                        archive.append(pop[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_delta.append(delta)
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            # Update memory
            if len(S_F) > 0:
                weights = np.array(S_delta)
                weights = weights / (weights.sum() + 1e-30)
                
                mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
                mean_CR = np.sum(weights * np.array(S_CR))
                
                M_F[k] = mean_F
                M_CR[k] = mean_CR
                k = (k + 1) % H
            
            # Stagnation check
            if abs(prev_best_gen - best) < 1e-14:
                stag_count += 1
            else:
                stag_count = 0
            prev_best_gen = best
            
            if stag_count > 15:
                # Restart half the population
                n_restart = pop_size // 2
                worst_idx = np.argsort(fit)[-n_restart:]
                for idx in worst_idx:
                    pop[idx] = lower + np.random.rand(dim) * ranges
                    fit[idx], pop[idx] = eval_clip(pop[idx])
                stag_count = 0
    
    # Run multiple SHADE restarts
    n_restarts = 3
    time_per_restart = 0.55 / n_restarts
    for _ in range(n_restarts):
        if remaining() < max_time * 0.1:
            break
        run_shade(time_per_restart)
    
    # ---- Local search: CMA-ES style around best ----
    if best_x is not None:
        sigma = 0.1 * np.mean(ranges)
        x_mean = best_x.copy()
        lam = max(8, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = 4.0 / (dim + 4)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while remaining() > max_time * 0.03:
            try:
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
            except:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
            
            samples = np.zeros((lam, dim))
            f_samples = np.zeros(lam)
            for j in range(lam):
                if remaining() < max_time * 0.02:
                    return best
                z = np.random.randn(dim)
                samples[j] = x_mean + sigma * (B @ (D * z))
                f_samples[j], samples[j] = eval_clip(samples[j])
            
            idx = np.argsort(f_samples)
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * samples[idx[:mu]], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.solve(B @ np.diag(D) @ B.T + 1e-20*np.eye(dim), x_mean - x_old) / sigma
            hs = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2)) < (1.4 + 2/(dim+1)) * chi_n)
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean - x_old) / sigma
            
            artmp = (samples[idx[:mu]] - x_old) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hs)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, np.mean(ranges))
            sigma = max(sigma, 1e-15 * np.mean(ranges))
    
    return best
