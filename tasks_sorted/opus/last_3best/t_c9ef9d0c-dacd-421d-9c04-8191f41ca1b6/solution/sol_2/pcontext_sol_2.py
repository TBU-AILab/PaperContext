#Looking at the best algorithm (output 16.43), I can see it uses DE with multiple strategies + Nelder-Mead local search. The second algorithm (SHADE) scored worse, likely due to bugs and overhead. Let me combine the best elements: clean SHADE with L-SHADE population reduction, better local search with multiple restarts, and more efficient implementation.
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    init_pop_size = min(max(15 * dim, 50), 200)
    min_pop_size = max(4, dim)
    H = 50
    
    def run_lshade(time_frac_end):
        nonlocal best, best_params
        
        pop_size = init_pop_size
        pop = np.random.uniform(0, 1, (pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + pop[:, d]) / pop_size
        pop = lower + pop * ranges
        
        # Seed some individuals near best if available
        if best_params is not None:
            n_seed = min(pop_size // 5, 10)
            for i in range(n_seed):
                scale = 0.1 * ranges * (i + 1) / n_seed
                pop[i] = best_params + np.random.normal(0, 1, dim) * scale
                pop[i] = np.clip(pop[i], lower, upper)
        
        fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= max_time * time_frac_end:
                return
            fit[i] = evaluate(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        max_evals_approx = 100000
        evals_done = pop_size
        
        while elapsed() < max_time * time_frac_end:
            S_F, S_CR, S_w = [], [], []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if elapsed() >= max_time * time_frac_end:
                    return
                
                ri = np.random.randint(H)
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
                
                # current-to-pbest/1
                p_val = max(2, int(max(0.05, 0.2 - 0.15 * evals_done / max_evals_approx) * pop_size))
                top_p = np.argpartition(fit, min(p_val, pop_size-1))[:p_val]
                xp = pop[top_p[np.random.randint(len(top_p))]]
                
                r1 = np.random.randint(pop_size - 1)
                if r1 >= i: r1 += 1
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(union_size - 1)
                if r2 >= i: r2 += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
                
                mask = np.random.random(dim) < CRi
                mask[np.random.randint(dim)] = True
                trial = np.where(mask, mutant, pop[i])
                
                below = trial < lower
                above = trial > upper
                trial[below] = (lower[below] + pop[i][below]) / 2
                trial[above] = (upper[above] + pop[i][above]) / 2
                
                f_trial = evaluate(trial)
                evals_done += 1
                
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi); S_CR.append(CRi)
                        S_w.append(fit[i] - f_trial)
                        if len(archive) < init_pop_size:
                            archive.append(pop[i].copy())
                        elif archive:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    new_pop[i] = trial; new_fit[i] = f_trial
            
            pop = new_pop; fit = new_fit
            
            if S_F:
                w = np.array(S_w); w /= w.sum() + 1e-30
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            # L-SHADE: reduce population
            new_size = max(min_pop_size, int(init_pop_size - (init_pop_size - min_pop_size) * evals_done / max_evals_approx))
            if new_size < pop_size:
                best_idx = np.argsort(fit)[:new_size]
                pop = pop[best_idx]; fit = fit[best_idx]
                pop_size = new_size
    
    run_lshade(0.75)
    
    # Nelder-Mead local search
    if best_params is not None:
        n = dim
        scale = 0.05 * ranges
        simplex = np.zeros((n+1, n))
        sf = np.zeros(n+1)
        simplex[0] = best_params.copy(); sf[0] = best
        for i in range(n):
            if elapsed() >= max_time * 0.98: return best
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
            sf[i+1] = evaluate(simplex[i+1])
        
        while elapsed() < max_time * 0.98:
            o = np.argsort(sf); simplex = simplex[o]; sf = sf[o]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = evaluate(xr)
            if elapsed() >= max_time * 0.98: break
            if sf[0] <= fr < sf[-2]:
                simplex[-1] = xr; sf[-1] = fr
            elif fr < sf[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper)
                fe = evaluate(xe)
                if fe < fr: simplex[-1] = xe; sf[-1] = fe
                else: simplex[-1] = xr; sf[-1] = fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1] - centroid), lower, upper)
                fc = evaluate(xc)
                if fc < sf[-1]: simplex[-1] = xc; sf[-1] = fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= max_time*0.98: break
                        simplex[i] = simplex[0] + 0.5*(simplex[i]-simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        sf[i] = evaluate(simplex[i])
    
    return best
