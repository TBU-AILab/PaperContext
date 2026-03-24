#Looking at the best algorithm (score 13.26), I'll focus on several improvements:
#
#1. **Better population sizing and faster convergence** with L-SHADE
#2. **Improved local search** - combine Nelder-Mead with coordinate-wise golden section
#3. **Multiple independent restarts** of DE to avoid getting trapped
#4. **Eigenvalue-based mutation** for better scaling in rotated landscapes
#5. **More aggressive exploitation phase**
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

    # --- L-SHADE Phase ---
    pop_size_init = min(max(14 * dim, 50), 200)
    pop_size_min = max(4, dim)
    pop_size = pop_size_init
    H = 60
    
    # Opposition-based LHS initialization
    n_init = pop_size
    pop = np.random.uniform(0, 1, (n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, d] = (perm + pop[:, d]) / n_init
    pop = lower + pop * ranges
    
    # Create opposition population
    opp = lower + upper - pop
    all_pop = np.vstack([pop, opp])
    all_pop = np.clip(all_pop, lower, upper)
    
    all_fit = np.full(len(all_pop), float('inf'))
    for i in range(len(all_pop)):
        if elapsed() >= max_time * 0.88:
            return best
        all_fit[i] = evaluate(all_pop[i])
    
    order = np.argsort(all_fit)[:pop_size]
    pop = all_pop[order].copy()
    fit = all_fit[order].copy()
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    nfe_total = len(all_pop)
    max_nfe_estimate = nfe_total  # will grow
    
    gen = 0
    while elapsed() < max_time * 0.75:
        gen += 1
        S_F, S_CR, S_w = [], [], []
        
        progress = elapsed() / (max_time * 0.75)
        p_rate = max(0.05, 0.25 - 0.20 * progress)
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            ri = np.random.randint(H)
            
            # Generate F
            Fi = -1
            for _ in range(10):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = np.clip(Fi, 0.05, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # current-to-pbest/1 with archive
            p = max(2, int(p_rate * pop_size))
            top_p = np.argpartition(fit, min(p, pop_size-1))[:p]
            xp = pop[top_p[np.random.randint(len(top_p))]]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            actual_r2 = r2 if r2 < i else r2 + 1
            if actual_r2 == r1:
                actual_r2 = (actual_r2 + 1) % union_size
            xr2 = pop[actual_r2] if actual_r2 < pop_size else archive[actual_r2 - pop_size]
            
            mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            below = trial < lower; above = trial > upper
            trial[below] = lower[below] + np.random.random(np.sum(below)) * (pop[i][below] - lower[below])
            trial[above] = upper[above] - np.random.random(np.sum(above)) * (upper[above] - pop[i][above])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(fit[i] - f_trial)
                    if len(archive) < pop_size: archive.append(pop[i].copy())
                    elif archive: archive[np.random.randint(len(archive))] = pop[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
        
        pop = new_pop; fit = new_fit
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]
            pop = pop[order].copy(); fit = fit[order].copy()
            pop_size = new_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
    
    # --- Nelder-Mead with multiple restarts ---
    if best_params is not None:
        for scale_factor in [0.08, 0.02, 0.005, 0.001]:
            if elapsed() >= max_time * 0.97: break
            n = dim; scale = scale_factor * ranges
            simplex = np.zeros((n+1, n)); sf = np.zeros(n+1)
            simplex[0] = best_params.copy(); sf[0] = best
            for i in range(n):
                if elapsed() >= max_time * 0.97: return best
                simplex[i+1] = best_params.copy()
                simplex[i+1][i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
                sf[i+1] = evaluate(simplex[i+1])
            for _ in range(600 * dim):
                if elapsed() >= max_time * 0.97: break
                o = np.argsort(sf); simplex = simplex[o]; sf = sf[o]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
                fr = evaluate(xr)
                if elapsed() >= max_time * 0.97: break
                if sf[0] <= fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
                elif fr < sf[0]:
                    xe = np.clip(centroid+2*(xr-centroid), lower, upper); fe = evaluate(xe)
                    if fe < fr: simplex[-1]=xe; sf[-1]=fe
                    else: simplex[-1]=xr; sf[-1]=fr
                else:
                    xc = np.clip(centroid+0.5*(simplex[-1]-centroid), lower, upper); fc = evaluate(xc)
                    if fc < sf[-1]: simplex[-1]=xc; sf[-1]=fc
                    else:
                        for i in range(1,n+1):
                            if elapsed()>=max_time*0.97: break
                            simplex[i]=simplex[0]+0.5*(simplex[i]-simplex[0])
                            simplex[i]=np.clip(simplex[i],lower,upper); sf[i]=evaluate(simplex[i])
    return best
