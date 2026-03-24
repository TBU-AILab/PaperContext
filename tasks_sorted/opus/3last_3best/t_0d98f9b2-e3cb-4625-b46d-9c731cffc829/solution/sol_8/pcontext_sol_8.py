#Looking at the results, the best algorithm scored 22.09 using SHADE-DE with coordinate descent. The algorithms scoring ~197-214 had issues with L-SHADE implementation (bugs in population management, time allocation). Let me create a carefully improved version that combines the best elements:
#
#1. **Robust L-SHADE** with proper generational updates and Lehmer mean
#2. **Multiple restarts** with progressive focusing
#3. **Diagonal CMA-ES** for exploitation near best
#4. **Adaptive coordinate descent** as final polish
#5. **Better time budgeting**
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

    # === SHADE-DE ===
    def run_shade(time_end, pop_init=None):
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
            if elapsed() >= time_end:
                return
            fit[i] = evaluate(pop[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        mem_k = 0
        archive = []
        max_archive = pop_size_init
        nfe = pop_size
        max_nfe = pop_size_init * 300
        stagnation = 0
        prev_gen_best = np.min(fit)
        
        while elapsed() < time_end:
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fit[:pop_size])
            
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if elapsed() >= time_end:
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
                
                p_rate = max(0.05, 0.20 - 0.15 * min(1.0, nfe / max_nfe))
                p = max(2, int(p_rate * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                r1 = np.random.randint(0, pop_size - 1)
                if r1 >= i:
                    r1 += 1
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 25:
                    r2 = np.random.randint(0, union_size)
                    attempts += 1
                
                x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
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
            
            pop[:pop_size] = new_pop
            fit[:pop_size] = new_fit
            
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
                keep = np.argsort(fit[:pop_size])[:new_ps]
                pop = pop[keep].copy()
                fit = fit[keep].copy()
                pop_size = new_ps
            
            cur_best = np.min(fit[:pop_size])
            if cur_best >= prev_gen_best - 1e-14:
                stagnation += 1
            else:
                stagnation = 0
            prev_gen_best = cur_best
            
            if stagnation > 25:
                n_reset = max(1, pop_size // 3)
                worst_idx = np.argsort(fit[:pop_size])[-n_reset:]
                for wi in worst_idx:
                    if elapsed() >= time_end:
                        return
                    if best_x is not None and np.random.random() < 0.5:
                        pop[wi] = best_x + 0.05 * ranges * np.random.randn(dim)
                    else:
                        pop[wi] = lower + np.random.random(dim) * ranges
                    pop[wi] = clip(pop[wi])
                    fit[wi] = evaluate(pop[wi])
                stagnation = 0

    # Phase 1: Main SHADE
    run_shade(elapsed() + max_time * 0.50)
    
    # Phase 2: Focused restart near best
    if best_x is not None and elapsed() < max_time * 0.73:
        rp = min(max(20, 5 * dim), 150)
        rpop = np.array([best_x + 0.05 * ranges * np.random.randn(dim) if np.random.random() < 0.75 else lower + np.random.random(dim) * ranges for _ in range(rp)])
        rpop = clip(rpop)
        run_shade(elapsed() + max_time * 0.23, rpop)

    # Phase 3: Coordinate descent
    if best_x is not None:
        x, fx = best_x.copy(), best
        step = 0.01 * ranges
        momentum = np.zeros(dim)
        while elapsed() < max_time * 0.97:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= max_time * 0.97:
                    return best
                signs = [1, -1] if momentum[d] == 0 else [int(np.sign(momentum[d])), -int(np.sign(momentum[d]))]
                found = False
                for sign in signs:
                    xt = x.copy()
                    xt[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    ft = evaluate(xt)
                    if ft < fx:
                        momentum[d] = sign * step[d]
                        x, fx = xt, ft
                        step[d] *= 1.5
                        improved = True
                        found = True
                        break
                if not found:
                    momentum[d] = 0
                    step[d] *= 0.5
            if not improved or np.max(step / ranges) < 1e-15:
                step = 0.002 * ranges * (0.1 + 0.9 * np.random.random(dim))
                momentum = np.zeros(dim)

    return best
