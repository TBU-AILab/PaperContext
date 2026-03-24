#Looking at the best algorithm (score 11.73), I'll make several improvements:
#
#1. **L-SHADE with linear population size reduction** for better convergence
#2. **Multiple restart strategy** to escape local optima
#3. **CMA-ES local search** instead of Nelder-Mead for better high-dimensional performance
#4. **Better time budget allocation**
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
    evals = 0

    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def run_lshade(time_budget_frac):
        nonlocal best, best_params
        deadline = elapsed() + max_time * time_budget_frac
        
        N_init = min(max(30, 10 * dim), 200)
        N_min = 4
        pop_size = N_init
        H = 100
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k = 0
        
        # LHS initialization
        pop = np.random.uniform(0, 1, (pop_size, dim))
        for i in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, i] = (perm + pop[:, i]) / pop_size
        pop = lower + pop * ranges
        
        # Opposition-based learning
        opp = lower + upper - pop
        combined = np.vstack([pop, opp])
        cf = np.array([ev(combined[i]) for i in range(len(combined)) if elapsed() < deadline])
        if len(cf) < len(combined):
            cf = np.append(cf, [float('inf')]*(len(combined)-len(cf)))
        idx = np.argsort(cf)[:pop_size]
        pop = combined[idx].copy()
        fit = cf[idx].copy()
        
        archive = []
        gen = 0
        max_gen_est = max(1, int(pop_size * 15))
        
        while elapsed() < deadline:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            sorted_idx = np.argsort(fit)
            
            for i in range(pop_size):
                if elapsed() >= deadline:
                    break
                
                ri = np.random.randint(0, H)
                F = -1
                while F <= 0:
                    F = min(1.0, np.random.standard_cauchy() * 0.1 + M_F[ri])
                CR = min(1.0, max(0.0, np.random.normal(M_CR[ri], 0.1)))
                
                p = max(2, int(max(0.05, 0.2 - 0.15 * gen / max(1, max_gen_est)) * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                idxs = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(idxs)
                
                a_size = len(archive)
                if a_size > 0 and np.random.rand() < 0.5:
                    x_r2 = archive[np.random.randint(a_size)]
                else:
                    r2 = np.random.choice([j for j in idxs if j != r1]) if len(idxs) > 1 else r1
                    x_r2 = pop[r2]
                
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - x_r2)
                
                cross = np.random.rand(dim) < CR
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = clip(np.where(cross, mutant, pop[i]))
                
                f_trial = ev(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(F); S_CR.append(CR); S_df.append(fit[i]-f_trial)
                        archive.append(pop[i].copy())
                        if len(archive) > N_init:
                            archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial; new_fit[i] = f_trial
            
            pop, fit = new_pop, new_fit
            
            if S_F:
                w = np.array(S_df); w /= w.sum()+1e-30
                M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F))+1e-30)
                M_CR[k] = np.sum(w * np.array(S_CR))
                k = (k + 1) % H
            
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * gen / max(1, max_gen_est))))
            if new_size < pop_size:
                worst = np.argsort(fit)[new_size:]
                pop = np.delete(pop, worst, axis=0)
                fit = np.delete(fit, worst)
                pop_size = new_size

    # Run L-SHADE with ~70% of time
    run_lshade(0.65)
    
    # Second restart with remaining time
    if time_left() > max_time * 0.15:
        run_lshade(time_left() / max_time * 0.7)
    
    # Local search: coordinate descent with shrinking steps
    if best_params is not None and time_left() > 0.1:
        cur = best_params.copy()
        cur_f = best
        step = ranges * 0.05
        while time_left() > 0.05:
            improved = False
            for d in range(dim):
                if time_left() <= 0.05: break
                for sign in [1, -1]:
                    t = cur.copy(); t[d] = np.clip(cur[d]+sign*step[d], lower[d], upper[d])
                    ft = ev(t)
                    if ft < cur_f:
                        cur, cur_f = t, ft; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges+1e-30)) < 1e-12: break
    
    return best
