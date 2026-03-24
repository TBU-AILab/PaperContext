#Looking at the results, algorithm 1 (output 10.28) performed best with L-SHADE + Nelder-Mead + coordinate descent. The CMA-ES variant (algorithm 3) performed worse, likely due to overhead. Let me improve by:
#
#1. **Better time allocation** - more time for DE exploration
#2. **Improved DE with multiple mutation strategies** (rand/1 + current-to-pbest/1)
#3. **Restart mechanism with diverse initialization**
#4. **Faster Nelder-Mead with adaptive restarts from multiple elite solutions**
#5. **Pattern search** as final polishing (more robust than coordinate descent)
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

    # ---- Phase 1: LHS + Opposition initialization ----
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
    
    # Opposition-based: evaluate opposites of worst quarter
    order = np.argsort(fitness)
    quarter = pop_size * 3 // 4
    for i in range(quarter, pop_size):
        if elapsed() >= max_time * 0.85:
            break
        idx = order[i]
        opp = lower + upper - population[idx]
        opp = np.clip(opp, lower, upper)
        f_opp = evaluate(opp)
        if f_opp < fitness[idx]:
            population[idx] = opp
            fitness[idx] = f_opp

    # ---- Phase 2: L-SHADE with improved strategies ----
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size_init
    
    stagnation = 0
    prev_best = best
    de_time_frac = 0.62
    
    # Keep track of top-k diverse elites for later local search
    elite_solutions = []
    
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
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # current-to-pbest/1 with archive
            p_count = max(2, int(np.ceil(np.random.uniform(2.0/pop_size, 0.2) * pop_size)))
            pbest_idx = np.random.choice(sorted_idx[:p_count])
            
            candidates = list(range(pop_size)); candidates.remove(i)
            r1 = np.random.choice(candidates)
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else population[r2]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            mutant = np.where(mutant < lower, (lower + population[i]) / 2, np.where(mutant > upper, (upper + population[i]) / 2, mutant))
            
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == np.random.randint(dim))
            trial = np.where(mask, mutant, population[i])
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive: archive.append(population[i].copy())
                elif archive: archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness = new_fit
        if S_F:
            w = np.array(S_delta); w /= (w.sum() + 1e-30)
            sf = np.array(S_F); sc = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * sc); k = (k + 1) % memory_size
        
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * de_time_frac))))
        if new_ps < pop_size:
            si = np.argsort(fitness); population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
        
        stagnation = stagnation + 1 if abs(prev_best - best) < 1e-14 else 0; prev_best = best
        if stagnation > 20:
            keep = max(pop_size // 4, 2); order = np.argsort(fitness)
            for ii in range(keep, pop_size):
                if elapsed() >= max_time * de_time_frac: break
                if np.random.random() < 0.5:
                    population[order[ii]] = best_params + np.random.randn(dim) * 0.1 * ranges
                else:
                    population[order[ii]] = lower + np.random.random(dim) * ranges
                population[order[ii]] = np.clip(population[order[ii]], lower, upper)
                fitness[order[ii]] = evaluate(population[order[ii]])
            stagnation = 0
    
    # Collect diverse elites
    si = np.argsort(fitness)
    for idx in si[:min(5, pop_size)]:
        elite_solutions.append((population[idx].copy(), fitness[idx]))

    # ---- Phase 3: Multi-start Nelder-Mead ----
    def nelder_mead(x0, sf, tl):
        n = dim; scale = sf * ranges
        simplex = np.zeros((n+1, n)); simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] * (1 if np.random.random() > 0.5 else -1)
        simplex = np.clip(simplex, lower, upper)
        fs = []
        for i in range(n+1):
            if elapsed() >= tl: return
            fs.append(evaluate(simplex[i]))
        fs = np.array(fs)
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        for _ in range(10000):
            if elapsed() >= tl: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = np.clip(c + alpha * (c - simplex[-1]), lower, upper); fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr; continue
            if fr < fs[0]:
                xe = np.clip(c + gamma * (xr - c), lower, upper); fe = evaluate(xe)
                if fe < fr: simplex[-1] = xe; fs[-1] = fe
                else: simplex[-1] = xr; fs[-1] = fr
                continue
            if fr < fs[-1]:
                xc = np.clip(c + rho * (xr - c), lower, upper); fc = evaluate(xc)
                if fc <= fr: simplex[-1] = xc; fs[-1] = fc; continue
            else:
                xc = np.clip(c - rho * (c - simplex[-1]), lower, upper); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1] = xc; fs[-1] = fc; continue
            for i in range(1, n+1):
                if elapsed() >= tl: break
                simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), lower, upper)
                fs[i] = evaluate(simplex[i])
            if np.max(np.abs(fs - fs[0])) < 1e-16: break

    if best_params is not None:
        starts = [best_params.copy()] + [e[0] for e in elite_solutions[:4]]
        seen = set()
        unique_starts = []
        for s in starts:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        
        nm_end = 0.90
        for sf in [0.08, 0.02, 0.005, 0.001]:
            for sx in unique_starts:
                if elapsed() >= max_time * nm_end: break
                remaining = max_time * nm_end - elapsed()
                time_per = remaining / max(1, (4 - [0.08,0.02,0.005,0.001].index(sf)) * len(unique_starts))
                nelder_mead(sx, sf, min(elapsed() + time_per, max_time * nm_end))
            if elapsed() >= max_time * nm_end: break

    # ---- Phase 4: Pattern search (Hooke-Jeeves) ----
    if best_params is not None:
        x = best_params.copy(); fx = best
        for step_mult in [0.02, 0.005, 0.001, 0.0002]:
            if elapsed() >= max_time * 0.98: break
            step = step_mult * ranges
            improved = True
            while improved and elapsed() < max_time * 0.98:
                improved = False
                for d in range(dim):
                    if elapsed() >= max_time * 0.98: break
                    current_f = fx
                    # Try positive direction
                    xp = x.copy(); xp[d] = min(x[d] + step[d], upper[d])
                    fp = evaluate(xp)
                    if fp < current_f:
                        x = xp; fx = fp; improved = True; continue
                    # Try negative direction
                    xn = x.copy(); xn[d] = max(x[d] - step[d], lower[d])
                    fn = evaluate(xn)
                    if fn < current_f:
                        x = xn; fx = fn; improved = True; continue
    
    # ---- Phase 5: Final golden section per dimension ----
    if best_params is not None and elapsed() < max_time * 0.995:
        x = best_params.copy()
        gr = (np.sqrt(5) + 1) / 2
        for ss in [0.003, 0.0005]:
            if elapsed() >= max_time * 0.995: break
            for d in range(dim):
                if elapsed() >= max_time * 0.995: break
                a = max(lower[d], x[d] - ss * ranges[d])
                b = min(upper[d], x[d] + ss * ranges[d])
                if b - a < 1e-15: continue
                c = b - (b - a) / gr; d2 = a + (b - a) / gr
                xc = x.copy(); xc[d] = c; fc = evaluate(xc)
                xd = x.copy(); xd[d] = d2; fd = evaluate(xd)
                for _ in range(25):
                    if elapsed() >= max_time * 0.995 or b - a < 1e-15: break
                    if fc < fd:
                        b = d2; d2 = c; fd = fc; c = b - (b - a) / gr
                        xc = x.copy(); xc[d] = c; fc = evaluate(xc)
                    else:
                        a = c; c = d2; fc = fd; d2 = a + (b - a) / gr
                        xd = x.copy(); xd[d] = d2; fd = evaluate(xd)
                x[d] = (a + b) / 2
                xt = x.copy(); evaluate(xt)
    
    return best
