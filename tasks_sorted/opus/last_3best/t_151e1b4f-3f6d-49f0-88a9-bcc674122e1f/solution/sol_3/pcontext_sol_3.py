#Looking at the results, algorithm 1 (DE with current-to-best + Nelder-Mead) performed best at 43.83. Let me analyze what worked and improve:
#
#1. **DE/current-to-best/1** was effective - keep it
#2. The simpler, more focused approach won over the complex adaptive one
#3. I'll add: SHADE-like parameter adaptation, opposition-based learning, better local search with coordinate descent, and multiple restart strategies
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
    pop_size = min(max(20, 10 * dim), 150)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        fitness[i] = ev(population[i])
    
    # Opposition-based initialization
    for i in range(pop_size):
        if elapsed() >= max_time * 0.15:
            break
        opp = lower + upper - population[i]
        of = ev(opp)
        if of < fitness[i]:
            population[i] = clip(opp)
            fitness[i] = of

    # --- Phase 2: SHADE-like DE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best

    while elapsed() < max_time * 0.78:
        sorted_idx = np.argsort(fitness)
        bi = sorted_idx[0]
        p_best_size = max(2, int(0.1 * pop_size))
        
        S_F = []
        S_CR = []
        delta_f = []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.78:
                break
            
            ri = np.random.randint(H)
            # Cauchy for F
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 1: Fi = 1.0
            # Normal for CR
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            pb = sorted_idx[np.random.randint(p_best_size)]
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool = idxs + list(range(len(archive)))
            pool = [j for j in pool if j != i]
            r2_idx = np.random.randint(len(pool))
            r2v = pool[r2_idx]
            x_r2 = population[r2v] if r2v < pop_size else archive[r2v - pop_size]
            
            mutant = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial_f = ev(trial)
            if trial_f < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); delta_f.append(fitness[i] - trial_f)
                if len(archive) < max_archive: archive.append(population[i].copy())
                elif archive: archive[np.random.randint(len(archive))] = population[i].copy()
                population[i] = clip(trial); fitness[i] = trial_f
        
        if S_F:
            w = np.array(delta_f); w /= w.sum() + 1e-30
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        if abs(prev_best - best) < 1e-12: stagnation += 1
        else: stagnation = 0
        prev_best = best
        if stagnation > 25:
            for i in range(pop_size//2, pop_size):
                idx = sorted_idx[i]; population[idx] = lower + np.random.random(dim) * ranges
                if elapsed() >= max_time * 0.78: break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: Nelder-Mead from top solutions ---
    top_k = min(3, pop_size)
    top_idx = np.argsort(fitness)[:top_k]
    for ti in top_idx:
        if elapsed() >= max_time * 0.97: break
        x0 = population[ti].copy(); n = dim
        scale = 0.03 * ranges
        simplex = np.vstack([x0, x0 + np.diag(scale)])
        fs = np.array([ev(simplex[j]) for j in range(n+1) if elapsed()<max_time*0.97])
        if len(fs)<n+1: break
        while elapsed()<max_time*0.97:
            o=np.argsort(fs); simplex=simplex[o]; fs=fs[o]
            c=np.mean(simplex[:-1],axis=0)
            xr=clip(c+1.0*(c-simplex[-1])); fr=ev(xr)
            if fs[0]<=fr<fs[-2]: simplex[-1]=xr;fs[-1]=fr
            elif fr<fs[0]:
                xe=clip(c+2.0*(xr-c));fe=ev(xe)
                simplex[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
            else:
                xc=clip(c+0.5*(simplex[-1]-c));fc=ev(xc)
                if fc<fs[-1]: simplex[-1]=xc;fs[-1]=fc
                else:
                    for j in range(1,n+1):
                        if elapsed()>=max_time*0.97: return best
                        simplex[j]=clip(simplex[0]+0.5*(simplex[j]-simplex[0]));fs[j]=ev(simplex[j])
    return best
