#Looking at the current algorithm, I see it's a SHADE variant that scores 1.906. Let me analyze potential improvements:
#
#1. **Add a proper local search phase** (Nelder-Mead or CMA-ES style) around the best solution
#2. **Better restart strategy** when stagnating
#3. **Population size reduction** (L-SHADE style)
#4. **Multi-strategy mutation** with strategy adaptation
#5. **Dedicated final local refinement phase**
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
    
    def time_left():
        return elapsed() < max_time * 0.95
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def nelder_mead(x0, budget_frac=0.15, initial_scale=0.05):
        nonlocal best, best_params
        max_t = min(elapsed() + max_time * budget_frac, max_time * 0.95)
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            step = initial_scale * ranges[i]
            if step < 1e-12:
                step = 1e-6
            p[i] += step
            p = np.clip(p, lower, upper)
            simplex[i + 1] = p
        
        f_vals = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        
        while elapsed() < max_t:
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_func(xr)
            
            if fr < f_vals[0]:
                # Expand
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            elif fr < f_vals[-2]:
                simplex[-1], f_vals[-1] = xr, fr
            else:
                if fr < f_vals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, f_vals[-1]):
                    simplex[-1], f_vals[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_vals[i] = eval_func(simplex[i])
            
            # Convergence check
            if np.max(np.abs(f_vals - f_vals[0])) < 1e-15:
                break
    
    # === Phase 1: L-SHADE ===
    init_pop_size = min(18 * dim, 200)
    pop_size = init_pop_size
    min_pop_size = max(4, dim // 2)
    
    # Latin hypercube sampling
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = lower[i] + (perm + np.random.rand(pop_size)) / pop_size * ranges[i]
    
    fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
    if not time_left():
        return best
    
    mem_size = 6
    M_F = np.full(mem_size, 0.5)
    M_CR = np.full(mem_size, 0.5)
    mem_idx = 0
    archive = []
    gen = 0
    total_evals = pop_size
    max_evals_shade = 10000 * dim  # for LPSR calculation
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        if not time_left():
            return best
        
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        S_F, S_CR, S_w = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        p_best_size = max(2, int(0.11 * pop_size))
        
        for i in range(pop_size):
            if not time_left():
                return best
            
            ri = np.random.randint(mem_size)
            mu_f = M_F[ri]
            mu_cr = M_CR[ri]
            
            Fi = -1
            while Fi <= 0:
                Fi = mu_f + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.5)
            
            CRi = np.clip(np.random.randn() * 0.1 + mu_cr, 0.0, 1.0)
            
            p_best = np.random.randint(p_best_size)
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            a = np.random.choice(candidates)
            
            pool_size = pop_size + len(archive)
            b = np.random.randint(pool_size - 1)  # exclude i
            if b >= i:
                b += 1
            if b < pop_size:
                xb = population[b]
            else:
                xb = archive[b - pop_size]
            
            mutant = population[i] + Fi * (population[p_best] - population[i]) + Fi * (population[a] - xb)
            
            jrand = np.random.randint(dim)
            trial = np.copy(population[i])
            for d in range(dim):
                if np.random.rand() < CRi or d == jrand:
                    trial[d] = mutant[d]
            
            # Bounce-back
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2.0
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2.0
            
            t_fit = eval_func(trial)
            total_evals += 1
            
            if t_fit <= fitness[i]:
                if t_fit < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    w = abs(fitness[i] - t_fit)
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_w.append(w)
                new_pop[i] = trial
                new_fit[i] = t_fit
        
        population, fitness = new_pop, new_fit
        
        if S_F:
            weights = np.array(S_w)
            weights /= weights.sum() + 1e-30
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % mem_size
        
        # L-SHADE population reduction
        new_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * total_evals / max_evals_shade)))
        if new_size < pop_size:
            sort_idx = np.argsort(fitness)
            population = population[sort_idx[:new_size]]
            fitness = fitness[sort_idx[:new_size]]
            pop_size = new_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(best - prev_best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best
        
        # Restart worst half on stagnation
        if stagnation > 50:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = lower + np.random.rand(dim) * ranges
                if not time_left():
                    return best
                fitness[i] = eval_func(population[i])
            stagnation = 0
            archive.clear()
            M_F[:] = 0.5
            M_CR[:] = 0.5
        
        gen += 1
    
    # === Phase 2: Nelder-Mead local search from best ===
    if best_params is not None and time_left():
        nelder_mead(best_params, budget_frac=0.12, initial_scale=0.05)
    
    # === Phase 3: Multiple restarts of NM with different scales ===
    for scale in [0.2, 0.01, 0.002, 0.1, 0.005]:
        if not time_left():
            break
        if best_params is not None:
            nelder_mead(best_params, budget_frac=0.04, initial_scale=scale)
    
    # === Phase 4: Coordinate-wise golden section refinement ===
    if best_params is not None and time_left():
        x = best_params.copy()
        gr = (np.sqrt(5) + 1) / 2
        for _ in range(3):
            if not time_left():
                break
            for d in range(dim):
                if not time_left():
                    break
                a_d = max(lower[d], x[d] - 0.05 * ranges[d])
                b_d = min(upper[d], x[d] + 0.05 * ranges[d])
                for _ in range(20):
                    if not time_left():
                        break
                    if b_d - a_d < 1e-14:
                        break
                    c = b_d - (b_d - a_d) / gr
                    dd = a_d + (b_d - a_d) / gr
                    xc = x.copy(); xc[d] = c
                    xd = x.copy(); xd[d] = dd
                    fc = eval_func(xc)
                    fd = eval_func(xd)
                    if fc < fd:
                        b_d = dd
                    else:
                        a_d = c
                x[d] = (a_d + b_d) / 2.0
                eval_func(x)
    
    return best
