#Looking at the results, the best algorithm scored 22.09 using SHADE-DE with coordinate descent. The algorithms scoring ~197-217 had bugs in L-SHADE (population array management, r2 selection issues). Let me create a carefully improved version that:
#
#1. Fixes L-SHADE with proper generational update (not in-place mutation during iteration)
#2. Uses opposition-based learning for initialization
#3. Better adaptive parameters with proper Lehmer mean
#4. Multi-phase: L-SHADE → focused restart → CMA-ES-like diagonal search → coordinate descent
#5. More careful time budgeting
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

    def run_shade(time_frac_end, pop_init=None, focused=False):
        nonlocal best, best_x
        time_limit = max_time * time_frac_end
        
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
            # Opposition-based: replace half with opposition
            n_opp = pop_size_init // 4
            for idx in range(n_opp):
                opp = lower + upper - pop[pop_size_init - 1 - idx]
                pop[pop_size_init - 1 - idx] = clip(opp)
        
        pop_size = len(pop)
        pop_size_min = 4
        
        fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= time_limit:
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
        
        while elapsed() < time_limit:
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fit)
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            # Generate all parameters first
            for i in range(pop_size):
                if elapsed() >= time_limit:
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
                
                # Adaptive p value
                p_rate = max(0.05, 0.20 - 0.15 * min(1.0, nfe / max_nfe))
                p = max(2, int(p_rate * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                # r1 from current population
                r1 = np.random.randint(0, pop_size - 1)
                if r1 >= i:
                    r1 += 1
                
                # r2 from population + archive
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 25:
                    r2 = np.random.randint(0, union_size)
                    attempts += 1
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                # current-to-pbest/1
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - x_r2)
                
                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Midpoint bounce-back
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
            
            # Update SHADE memory with weighted Lehmer mean
            if S_F:
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[mem_k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[mem_k % H] = np.sum(w * scr)
                mem_k += 1
            
            # L-SHADE population reduction
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
            
            if stagnation > 25 and not focused:
                n_reset = max(1, pop_size // 3)
                worst_idx = np.argsort(fit)[-n_reset:]
                for wi in worst_idx:
                    if elapsed() >= time_limit:
                        return
                    if best_x is not None and np.random.random() < 0.5:
                        pop[wi] = best_x + 0.05 * ranges * np.random.randn(dim)
                    else:
                        pop[wi] = lower + np.random.random(dim) * ranges
                    pop[wi] = clip(pop[wi])
                    fit[wi] = evaluate(pop[wi])
                stagnation = 0

    # Phase 1: Main L-SHADE
    run_shade(0.50)
    
    # Phase 2: Focused L-SHADE near best
    if best_x is not None and elapsed() < max_time * 0.72:
        rp = min(max(20, 5 * dim), 150)
        rpop = np.zeros((rp, dim))
        for i in range(rp):
            if np.random.random() < 0.75:
                rpop[i] = best_x + 0.05 * ranges * np.random.randn(dim)
            else:
                rpop[i] = lower + np.random.random(dim) * ranges
        rpop = clip(rpop)
        run_shade(0.75, rpop, focused=True)
    
    # Phase 3: Coordinate descent with momentum and acceleration
    if best_x is not None:
        x = best_x.copy()
        fx = best
        step = 0.01 * ranges
        momentum = np.zeros(dim)
        
        while elapsed() < max_time * 0.97:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= max_time * 0.97:
                    return best
                
                signs = [1, -1]
                if momentum[d] != 0:
                    signs = [int(np.sign(momentum[d])), -int(np.sign(momentum[d]))]
                
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
