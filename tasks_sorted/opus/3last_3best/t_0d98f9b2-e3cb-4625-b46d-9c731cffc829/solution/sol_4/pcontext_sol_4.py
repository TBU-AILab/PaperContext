#Looking at the results, the best algorithm (score 22.09) uses SHADE-DE with coordinate descent. Let me analyze what can be improved:
#
#1. **The SHADE-DE mutation is using archive inefficiently** - r2 should come from pop+archive union
#2. **Need better time allocation** - the L-SHADE population reduction wasn't properly implemented in the best version
#3. **CMA-ES with full covariance** for moderate dims, and better restarts
#4. **Coordinate descent needs golden section / quadratic interpolation** for faster convergence
#5. **Multiple independent restarts** to avoid getting stuck
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # === L-SHADE ===
    def run_lshade(time_frac_end, pop_init=None):
        nonlocal best, best_x
        time_limit = max_time * time_frac_end
        
        pop_size_init = min(max(30, 8 * dim), 250)
        pop_size_min = 4
        
        if pop_init is not None:
            pop = pop_init.copy()
            pop_size_init = len(pop)
        else:
            pop = np.zeros((pop_size_init, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size_init)
                pop[:, d] = (perm + np.random.uniform(0, 1, pop_size_init)) / pop_size_init
            pop = lower + pop * ranges
        
        pop_size = len(pop)
        fit = np.array([evaluate(pop[i]) for i in range(pop_size) if elapsed() < time_limit])
        if len(fit) < pop_size:
            fit = np.append(fit, [float('inf')] * (pop_size - len(fit)))
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        mem_k = 0
        archive = []
        max_archive = pop_size_init
        nfe = pop_size
        max_nfe = pop_size_init * 300
        
        while elapsed() < time_limit:
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fit[:pop_size])
            
            trials = np.empty((pop_size, dim))
            trial_from = np.zeros(pop_size, dtype=int)
            
            for i in range(pop_size):
                if elapsed() >= time_limit:
                    break
                
                ri = np.random.randint(0, H)
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                while F_i <= 0:
                    F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)
                CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                p_rate = max(0.05, 0.25 - 0.20 * nfe / max_nfe)
                p = max(2, int(p_rate * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                r1 = np.random.randint(0, pop_size - 1)
                if r1 >= i: r1 += 1
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size - 1)
                if r2 == i: r2 = (r2 + 1) % union_size
                if r2 == r1: r2 = (r2 + 1) % union_size
                x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - x_r2)
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                out_low = trial < lower; out_high = trial > upper
                trial[out_low] = (lower[out_low] + pop[i][out_low]) / 2
                trial[out_high] = (upper[out_high] + pop[i][out_high]) / 2
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                nfe += 1
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if delta > 0:
                        archive.append(pop[i].copy())
                        if len(archive) > max_archive:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(F_i); S_CR.append(CR_i); S_delta.append(delta)
                    pop[i] = trial; fit[i] = f_trial
            
            if S_F:
                w = np.array(S_delta); w = w / w.sum()
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[mem_k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[mem_k % H] = np.sum(w * scr)
                mem_k += 1
            
            new_ps = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * nfe / max_nfe)))
            if new_ps < pop_size:
                keep = np.argsort(fit[:pop_size])[:new_ps]
                pop = pop[keep].copy(); fit = fit[keep].copy(); pop_size = new_ps

    run_lshade(0.55)
    
    if best_x is not None and elapsed() < max_time * 0.75:
        rp = min(max(15, 4*dim), 100)
        rpop = np.array([best_x + 0.03*ranges*np.random.randn(dim) if np.random.random()<0.8 else lower+np.random.random(dim)*ranges for _ in range(rp)])
        run_lshade(0.78, clip(rpop))

    if best_x is not None:
        x, fx = best_x.copy(), best
        step = 0.005 * ranges
        while elapsed() < max_time * 0.97:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.97: return best
                for sign in [1, -1]:
                    xt = x.copy(); xt[d] = np.clip(x[d]+sign*step[d], lower[d], upper[d])
                    ft = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft; step[d] *= 1.5; improved = True; break
                else:
                    step[d] *= 0.5
            if not improved or np.max(step) < 1e-16:
                step = 0.002 * ranges * np.random.random(dim)

    return best
