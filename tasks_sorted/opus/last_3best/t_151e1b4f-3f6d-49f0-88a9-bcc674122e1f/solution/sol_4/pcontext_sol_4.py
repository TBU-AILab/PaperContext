#Looking at the results, the SHADE-like DE (algorithm 1) performed best at 27.23. Let me analyze what can be improved:
#
#1. **SHADE worked well** - keep the adaptive parameter mechanism
#2. **L-SHADE with linear population size reduction** could help focus search over time
#3. **Better local search** - add coordinate descent before Nelder-Mead
#4. **Multiple restarts with smaller populations** after convergence
#5. **Improve archive usage** for better diversity
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
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

    # --- Phase 1: LHS initialization ---
    pop_init = min(max(30, 12 * dim), 200)
    pop_min = max(4, dim)
    pop_size = pop_init
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.08:
            break
        fitness[i] = ev(population[i])
    
    # Opposition-based initialization
    for i in range(pop_size):
        if elapsed() >= max_time * 0.12:
            break
        opp = lower + upper - population[i]
        of = ev(opp)
        if of < fitness[i]:
            population[i] = clip(opp)
            fitness[i] = of

    # --- Phase 2: L-SHADE ---
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    nfe = 2 * pop_size  # function evals so far approximately
    max_nfe_estimate = pop_size * 300  # rough estimate
    
    stagnation = 0
    prev_best = best
    generation = 0

    de_time_limit = max_time * 0.72

    while elapsed() < de_time_limit:
        generation += 1
        sorted_idx = np.argsort(fitness)
        
        # Adaptive p value for p-best
        p_min = 2.0 / pop_size
        p_max = 0.2
        
        S_F = []
        S_CR = []
        delta_f = []
        
        trial_pop = []
        trial_fit = []
        
        for i in range(pop_size):
            if elapsed() >= de_time_limit:
                break
            
            ri = np.random.randint(H)
            # Cauchy for F
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            # Normal for CR
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # p-best index
            pi = max(2, int(np.random.uniform(p_min, p_max) * pop_size))
            pb = sorted_idx[np.random.randint(pi)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            trial_f = ev(trial)
            nfe += 1
            
            if trial_f < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                delta_f.append(fitness[i] - trial_f)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                population[i] = trial
                fitness[i] = trial_f
            elif trial_f == fitness[i]:
                population[i] = trial
                fitness[i] = trial_f
        
        # Update memory
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        # L-SHADE: linear population size reduction
        progress = elapsed() / de_time_limit
        new_pop_size = max(pop_min, int(round(pop_init + (pop_min - pop_init) * progress)))
        
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            keep = sorted_idx[:new_pop_size]
            population = population[keep]
            fitness = fitness[keep]
            pop_size = new_pop_size
            max_archive = pop_size
            if len(archive) > max_archive:
                archive = archive[:max_archive]
        
        # Stagnation handling
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 30 and pop_size > pop_min + 2:
            n_replace = max(1, pop_size // 4)
            sorted_idx = np.argsort(fitness)
            for ii in range(n_replace):
                idx = sorted_idx[-(ii+1)]
                population[idx] = best_params + 0.1 * ranges * np.random.randn(dim)
                population[idx] = clip(population[idx])
                if elapsed() >= de_time_limit:
                    break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: Coordinate descent from best ---
    if best_params is not None and elapsed() < max_time * 0.85:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        for _ in range(5):
            if elapsed() >= max_time * 0.85:
                break
            for d in range(dim):
                if elapsed() >= max_time * 0.85:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = ev(xn)
                    if fn < fx:
                        x = xn; fx = fn
            step *= 0.5

    # --- Phase 4: Nelder-Mead ---
    if best_params is not None and elapsed() < max_time * 0.97:
        x0 = best_params.copy(); n = dim
        scale = 0.01 * ranges
        simplex = np.vstack([x0, x0 + np.diag(scale)])
        fs = np.array([ev(simplex[j]) for j in range(n+1) if elapsed()<max_time*0.97])
        if len(fs) == n+1:
            while elapsed() < max_time * 0.97:
                o=np.argsort(fs); simplex=simplex[o]; fs=fs[o]
                c=np.mean(simplex[:-1],axis=0)
                xr=clip(c+(c-simplex[-1])); fr=ev(xr)
                if fs[0]<=fr<fs[-2]: simplex[-1]=xr;fs[-1]=fr
                elif fr<fs[0]:
                    xe=clip(c+2*(xr-c));fe=ev(xe)
                    simplex[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
                else:
                    xc=clip(c+0.5*(simplex[-1]-c));fc=ev(xc)
                    if fc<fs[-1]: simplex[-1]=xc;fs[-1]=fc
                    else:
                        for j in range(1,n+1):
                            if elapsed()>=max_time*0.97: return best
                            simplex[j]=clip(simplex[0]+0.5*(simplex[j]-simplex[0]));fs[j]=ev(simplex[j])
    return best
