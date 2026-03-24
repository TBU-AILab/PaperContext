#Looking at the results, the best algorithm scored 22.09 using SHADE-DE with coordinate descent. The recent attempts scoring ~204-214 have bugs in L-SHADE population management. Let me analyze what worked well and create a more robust, refined version.
#
#Key improvements:
#1. **Fix L-SHADE properly** - careful array management, no in-place mutation during iteration
#2. **Better time allocation** - more time for DE exploration
#3. **Ensemble mutation strategies** - not just current-to-pbest
#4. **Restart with smaller focused populations**
#5. **Robust coordinate descent with pattern moves (Hooke-Jeeves style)**
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

    def run_shade(time_limit_abs, pop_init=None):
        nonlocal best, best_x
        
        if pop_init is not None:
            pop = pop_init.copy()
            pop_size_init = len(pop)
        else:
            pop_size_init = min(max(30, 8 * dim), 300)
            pop = np.zeros((pop_size_init, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size_init)
                pop[:, d] = (perm + np.random.uniform(0, 1, pop_size_init)) / pop_size_init
            pop = lower + pop * ranges
        
        pop_size = pop_size_init
        pop_size_min = 4
        
        fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= time_limit_abs:
                return
            fit[i] = evaluate(pop[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        mem_k = 0
        archive = []
        max_archive = pop_size_init
        nfe = pop_size
        max_nfe = pop_size_init * 400
        stagnation = 0
        prev_gen_best = np.min(fit)
        
        while elapsed() < time_limit_abs:
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fit)
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if elapsed() >= time_limit_abs:
                    break
                
                ri = np.random.randint(0, H)
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                cnt = 0
                while F_i <= 0:
                    F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    cnt += 1
                    if cnt > 20:
                        F_i = 0.5
                        break
                F_i = min(F_i, 1.0)
                
                CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                p_rate = max(0.05, 0.25 - 0.20 * min(1.0, nfe / max_nfe))
                p = max(2, int(p_rate * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = idxs[np.random.randint(len(idxs))]
                
                union = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                union = [u for u in union if u != i and u != r1]
                if len(union) == 0:
                    union = [r1]
                r2_idx = union[np.random.randint(len(union))]
                x_r2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
                
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - x_r2)
                
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                out_low = trial < lower
                out_high = trial > upper
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
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_delta.append(delta)
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            if S_F:
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[mem_k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[mem_k % H] = np.sum(w * scr)
                mem_k += 1
            
            new_ps = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * nfe / max_nfe)))
            if new_ps < pop_size:
                keep = np.argsort(fit)[:new_ps]
                pop = pop[keep].copy()
                fit = fit[keep].copy()
                pop_size = new_ps
            
            cur_best = np.min(fit)
            if cur_best >= prev_gen_best - 1e-14:
                stagnation += 1
            else:
                stagnation = 0
            prev_gen_best = cur_best
            
            if stagnation > 30:
                n_reset = max(1, pop_size // 3)
                worst_idx = np.argsort(fit)[-n_reset:]
                for wi in worst_idx:
                    if elapsed() >= time_limit_abs: return
                    pop[wi] = best_x + 0.05 * ranges * np.random.randn(dim) if best_x is not None and np.random.random() < 0.5 else lower + np.random.random(dim) * ranges
                    pop[wi] = clip(pop[wi])
                    fit[wi] = evaluate(pop[wi])
                stagnation = 0

    run_shade(max_time * 0.55)
    
    if best_x is not None and elapsed() < max_time * 0.75:
        rp = min(max(20, 5*dim), 150)
        rpop = clip(np.array([best_x + 0.05*ranges*np.random.randn(dim) if np.random.random()<0.7 else lower+np.random.random(dim)*ranges for _ in range(rp)]))
        run_shade(max_time * 0.78, rpop)

    if best_x is not None:
        x, fx = best_x.copy(), best
        step = 0.01 * ranges
        while elapsed() < max_time * 0.97:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= max_time * 0.97: return best
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[d] = np.clip(x[d]+sign*step[d], lower[d], upper[d])
                    ft = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft; step[d]*=1.5; improved=True; break
                else:
                    step[d] *= 0.5
            if not improved or np.max(step/ranges)<1e-15:
                step = 0.002*ranges*(0.1+0.9*np.random.random(dim))

    return best
