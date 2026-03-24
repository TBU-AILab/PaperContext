#Looking at the results, the last algorithm scored 187.98 which is terrible, while the best algorithm scored 10.28. The best performer uses L-SHADE + Nelder-Mead + coordinate descent. Let me analyze what went wrong with the 187.98 algorithm and improve upon the 10.28 one.
#
#Key improvements:
#1. **More robust L-SHADE** with better parameter handling
#2. **Better time allocation** - give DE more time
#3. **Improved Nelder-Mead** with proper time budgeting per scale
#4. **Add a simple CMA-ES-like local search** after NM
#5. **More aggressive restart on stagnation**
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

    # ---- Phase 1: LHS initialization ----
    pop_size_init = min(max(12 * dim, 50), 200)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # ---- Phase 2: L-SHADE ----
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k_idx = 0
    archive = []
    max_archive = pop_size_init
    
    stagnation = 0
    prev_best = best
    de_time_frac = 0.65
    
    while elapsed() < max_time * de_time_frac:
        S_F, S_CR, S_delta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p_count = max(2, int(np.ceil(np.random.uniform(2.0/pop_size, 0.2) * pop_size)))
            pbest_idx = np.random.choice(sorted_idx[:p_count])
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else population[r2]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            for dd in range(dim):
                if mutant[dd] < lower[dd]:
                    mutant[dd] = (lower[dd] + population[i][dd]) / 2.0
                elif mutant[dd] > upper[dd]:
                    mutant[dd] = (upper[dd] + population[i][dd]) / 2.0
            
            jrand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == jrand)
            trial = np.where(mask, mutant, population[i])
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            sc = np.array(S_CR)
            M_F[k_idx] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k_idx] = np.sum(w * sc)
            k_idx = (k_idx + 1) % memory_size
        
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * de_time_frac))))
        if new_ps < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_ps]]
            fitness = fitness[si[:new_ps]]
            pop_size = new_ps
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            keep = max(pop_size // 4, 2)
            order = np.argsort(fitness)
            for ii in range(keep, pop_size):
                if elapsed() >= max_time * de_time_frac:
                    break
                if np.random.random() < 0.5:
                    population[order[ii]] = best_params + np.random.randn(dim) * 0.1 * ranges
                else:
                    population[order[ii]] = lower + np.random.random(dim) * ranges
                population[order[ii]] = np.clip(population[order[ii]], lower, upper)
                fitness[order[ii]] = evaluate(population[order[ii]])
            stagnation = 0

    # ---- Phase 3: Nelder-Mead at multiple scales ----
    def nelder_mead(x0, sf, tl):
        n = dim
        scale = sf * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += scale[i] * (1 if np.random.random() > 0.5 else -1)
        simplex = np.clip(simplex, lower, upper)
        fs = []
        for i in range(n + 1):
            if elapsed() >= tl:
                return
            fs.append(evaluate(simplex[i]))
        fs = np.array(fs)
        for _ in range(10000):
            if elapsed() >= tl:
                break
            o = np.argsort(fs)
            simplex = simplex[o]
            fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = np.clip(2 * c - simplex[-1], lower, upper)
            fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr; continue
            if fr < fs[0]:
                xe = np.clip(c + 2 * (xr - c), lower, upper)
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe; fs[-1] = fe
                else:
                    simplex[-1] = xr; fs[-1] = fr
                continue
            if fr < fs[-1]:
                xc = np.clip(c + 0.5 * (xr - c), lower, upper)
                fc = evaluate(xc)
                if fc <= fr:
                    simplex[-1] = xc; fs[-1] = fc; continue
            else:
                xc = np.clip(c + 0.5 * (simplex[-1] - c), lower, upper)
                fc = evaluate(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc; fs[-1] = fc; continue
            for i in range(1, n + 1):
                if elapsed() >= tl:
                    break
                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lower, upper)
                fs[i] = evaluate(simplex[i])
            if np.max(np.abs(fs - fs[0])) < 1e-16:
                break

    if best_params is not None:
        scales = [0.1, 0.02, 0.005, 0.001]
        nm_end = 0.92
        for idx_s, sf in enumerate(scales):
            if elapsed() >= max_time * nm_end:
                break
            remaining = max_time * nm_end - elapsed()
            time_budget = remaining / (len(scales) - idx_s)
            nelder_mead(best_params.copy(), sf, elapsed() + time_budget)

    # ---- Phase 4: Coordinate descent ----
    if best_params is not None:
        x = best_params.copy()
        fx = best
        for ss in [0.01, 0.002, 0.0005]:
            if elapsed() >= max_time * 0.98:
                break
            step = ss * ranges
            improved = True
            while improved and elapsed() < max_time * 0.98:
                improved = False
                for d in range(dim):
                    if elapsed() >= max_time * 0.98:
                        break
                    for dr in [1, -1]:
                        xn = x.copy()
                        xn[d] = np.clip(x[d] + dr * step[d], lower[d], upper[d])
                        fn = evaluate(xn)
                        if fn < fx:
                            x = xn
                            fx = fn
                            improved = True
                            break
    return best
