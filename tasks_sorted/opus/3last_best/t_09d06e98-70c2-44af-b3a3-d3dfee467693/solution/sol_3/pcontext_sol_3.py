#Looking at the best algorithm (score 11.73), I'll analyze the key issues and improve:
#
#1. **SHADE memory indexing bug**: `ri` was sampled once per generation instead of per individual
#2. **Population size reduction (L-SHADE)** for better convergence
#3. **Better local search**: combine Nelder-Mead with pattern search
#4. **Multiple restarts** to avoid local optima
#5. **Improved mutation strategy** with proper per-individual memory sampling
#
import numpy as np
from datetime import datetime

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
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def run_lshade(time_frac):
        nonlocal best, best_params
        deadline = elapsed() + max_time * time_frac
        
        N_init = min(max(25, 8 * dim), 180)
        N_min = 4
        pop_size = N_init
        H = 80
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        
        # LHS initialization
        pop = np.random.uniform(0, 1, (pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + pop[:, d]) / pop_size
        pop = lower + pop * ranges
        
        # Inject best known if available
        if best_params is not None:
            pop[0] = best_params.copy()
        
        fit = np.array([ev(pop[i]) for i in range(pop_size) if elapsed() < deadline])
        if len(fit) < pop_size:
            pop = pop[:len(fit)]
            pop_size = len(fit)
        
        archive = []
        total_evals_est = max(1, int((deadline - elapsed()) * pop_size / max(0.01, elapsed() - (deadline - max_time * time_frac) + 0.001)))
        max_evals = max(pop_size * 5, total_evals_est)
        evals_used = pop_size
        
        while elapsed() < deadline and pop_size >= N_min:
            S_F, S_CR, S_df = [], [], []
            sorted_idx = np.argsort(fit)
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if elapsed() >= deadline:
                    break
                
                ri = np.random.randint(0, H)
                mu_f = M_F[ri]
                mu_cr = M_CR[ri]
                
                F = -1
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + mu_f
                    if F >= 1.0:
                        F = 1.0
                        break
                CR = min(1.0, max(0.0, np.random.normal(mu_cr, 0.1)))
                
                p = max(2, int(max(0.05, 0.25 - 0.20 * evals_used / max(1, max_evals)) * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                candidates = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(candidates)
                
                union_size = pop_size + len(archive)
                r2_pool = [j for j in range(union_size) if j != i and j != r1]
                r2 = np.random.choice(r2_pool) if r2_pool else r1
                x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - x_r2)
                
                jrand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[jrand] = True
                trial = clip(np.where(mask, mutant, pop[i]))
                
                f_trial = ev(trial)
                evals_used += 1
                
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(F); S_CR.append(CR); S_df.append(fit[i] - f_trial)
                        archive.append(pop[i].copy())
                        if len(archive) > N_init:
                            archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial; new_fit[i] = f_trial
            
            pop, fit = new_pop, new_fit
            
            if S_F:
                w = np.array(S_df); w /= w.sum() + 1e-30
                sf = np.array(S_F)
                M_F[k_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k_idx] = np.sum(w * np.array(S_CR))
                k_idx = (k_idx + 1) % H
            
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * evals_used / max(1, max_evals))))
            if new_size < pop_size:
                keep = np.argsort(fit)[:new_size]
                pop = pop[keep]; fit = fit[keep]; pop_size = new_size

    run_lshade(0.55)
    if time_left() > max_time * 0.20:
        run_lshade(time_left() / max_time * 0.75)
    if time_left() > max_time * 0.10:
        run_lshade(time_left() / max_time * 0.75)

    # Local search: pattern search + coordinate descent
    if best_params is not None and time_left() > 0.05:
        cur = best_params.copy(); cur_f = best
        step = ranges * 0.03
        while time_left() > 0.02:
            improved = False
            for d in range(dim):
                if time_left() <= 0.02: break
                for s in [1, -1]:
                    t = cur.copy()
                    t[d] = np.clip(cur[d] + s * step[d], lower[d], upper[d])
                    ft = ev(t)
                    if ft < cur_f:
                        cur, cur_f = t, ft; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-13: break

    return best
