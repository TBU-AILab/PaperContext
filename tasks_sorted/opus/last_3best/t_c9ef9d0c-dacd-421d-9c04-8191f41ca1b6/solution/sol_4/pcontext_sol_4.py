#Looking at the results, algorithm 1 (score 16.43) is the best. Let me analyze what made it work and improve upon it:
#
#Key improvements I'll make:
#1. **Better SHADE with weighted Lehmer mean** for F adaptation
#2. **Smaller, more efficient population** to get more generations
#3. **Multiple Nelder-Mead restarts** with different scales
#4. **Opposition-based learning** for initialization
#5. **Linear population size reduction** (L-SHADE style but controlled)
#6. **Better time allocation** between exploration and exploitation
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

    pop_size_init = min(max(12 * dim, 40), 180)
    pop_size_min = max(4, dim)
    pop_size = pop_size_init
    H = 50
    
    # LHS + opposition initialization
    pop = np.random.uniform(0, 1, (pop_size * 2, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size * 2)
        pop[:, d] = (perm + pop[:, d]) / (pop_size * 2)
    pop = lower + pop * ranges
    # Add opposition
    pop[pop_size:] = lower + upper - pop[:pop_size]
    pop = np.clip(pop, lower, upper)
    
    fit = np.full(pop_size * 2, float('inf'))
    for i in range(pop_size * 2):
        if elapsed() >= max_time * 0.90:
            return best
        fit[i] = evaluate(pop[i])
    
    # Keep best pop_size
    order = np.argsort(fit)[:pop_size]
    pop = pop[order].copy()
    fit = fit[order].copy()
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    gen_count = 0
    
    while elapsed() < max_time * 0.82:
        S_F, S_CR, S_w = [], [], []
        gen_count += 1
        
        # Adaptive p value
        p_rate = max(0.05, 0.2 - 0.15 * (elapsed() / (max_time * 0.82)))
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.82:
                break
            
            ri = np.random.randint(H)
            # Generate F via Cauchy
            Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            attempts = 0
            while Fi <= 0 and attempts < 10:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # Weighted strategy selection: favor current-to-pbest
            strategy = np.random.choice([0, 0, 0, 1, 2])
            
            if strategy == 0:
                # current-to-pbest/1
                p = max(2, int(p_rate * pop_size))
                top_p = np.argpartition(fit, p)[:p]
                xp = pop[top_p[np.random.randint(len(top_p))]]
                
                r1 = np.random.randint(pop_size - 1)
                if r1 >= i: r1 += 1
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(union_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(union_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            
            elif strategy == 1:
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = best_params + Fi * (pop[idxs[0]] - pop[idxs[1]])
            else:
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = pop[idxs[0]] + Fi * (pop[idxs[1]] - pop[idxs[2]])
            
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
                pop[i] = trial; fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        # L-SHADE population reduction
        new_size = max(pop_size_min, int(pop_size_init - (pop_size_init - pop_size_min) * elapsed() / (max_time * 0.82)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]
            pop = pop[order].copy(); fit = fit[order].copy()
            pop_size = new_size
    
    # --- Multiple Nelder-Mead restarts with different scales ---
    if best_params is not None:
        for scale_factor in [0.05, 0.01, 0.002]:
            if elapsed() >= max_time * 0.98: break
            n = dim; scale = scale_factor * ranges
            simplex = np.zeros((n+1, n)); sf = np.zeros(n+1)
            simplex[0] = best_params.copy(); sf[0] = best
            for i in range(n):
                if elapsed() >= max_time * 0.98: return best
                simplex[i+1] = best_params.copy()
                simplex[i+1][i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
                sf[i+1] = evaluate(simplex[i+1])
            for _ in range(500 * dim):
                if elapsed() >= max_time * 0.98: break
                o = np.argsort(sf); simplex = simplex[o]; sf = sf[o]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
                fr = evaluate(xr)
                if elapsed() >= max_time * 0.98: break
                if sf[0] <= fr < sf[-2]: simplex[-1] = xr; sf[-1] = fr
                elif fr < sf[0]:
                    xe = np.clip(centroid + 2*(xr - centroid), lower, upper); fe = evaluate(xe)
                    if fe < fr: simplex[-1] = xe; sf[-1] = fe
                    else: simplex[-1] = xr; sf[-1] = fr
                else:
                    xc = np.clip(centroid + 0.5*(simplex[-1] - centroid), lower, upper); fc = evaluate(xc)
                    if fc < sf[-1]: simplex[-1] = xc; sf[-1] = fc
                    else:
                        for i in range(1, n+1):
                            if elapsed() >= max_time*0.98: break
                            simplex[i] = simplex[0] + 0.5*(simplex[i]-simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper); sf[i] = evaluate(simplex[i])
    return best
