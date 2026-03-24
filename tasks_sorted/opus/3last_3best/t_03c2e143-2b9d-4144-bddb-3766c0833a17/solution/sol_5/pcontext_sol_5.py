#Looking at the results, the best algorithm (21.83) uses L-SHADE with coordinate descent local search. I'll improve it with:
#
#1. **Better local search**: Add Nelder-Mead-style moves and gradient approximation
#2. **Accelerated coordinate descent** with momentum/pattern moves
#3. **More aggressive population reduction**
#4. **Better time splitting** - give more time to local search
#5. **Random direction search** to escape local optima
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: L-SHADE ---
    init_pop_size = min(max(40, 10 * dim), 400)
    min_pop_size = 4
    pop_size = init_pop_size
    H = 100

    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.array([evaluate(population[i]) for i in range(pop_size)])
    if elapsed() >= max_time * 0.95:
        return best

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size

    shade_time = max_time * 0.55

    while elapsed() < shade_time:
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness_vals)
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= shade_time:
                break
            
            ri = np.random.randint(H)
            
            Fi = -1
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if 0 < Fi < 2.0:
                    break
                Fi = -1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(0.11 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            pool = pop_size + len(archive)
            r2 = np.random.randint(pool - 1)
            if r2 >= min(i, r1): r2 += 1
            if r2 >= max(i, r1): r2 += 1
            if r2 >= pool: r2 = 0
            
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                if f_trial < fitness_vals[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    delta_f.append(fitness_vals[i] - f_trial)
                    if len(archive) < archive_max: archive.append(population[i].copy())
                    elif archive_max > 0: archive[np.random.randint(archive_max)] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness_vals = new_fit
        
        if S_F:
            w = np.array(delta_f); w = w / (w.sum() + 1e-30)
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k % H] = np.sum(w * scr); k += 1
        
        ratio = min(1.0, elapsed() / shade_time)
        new_ps = max(min_pop_size, int(init_pop_size - (init_pop_size - min_pop_size) * ratio))
        if new_ps < pop_size:
            bi = np.argsort(fitness_vals)[:new_ps]
            population = population[bi]; fitness_vals = fitness_vals[bi]
            pop_size = new_ps; archive_max = pop_size

    # --- Phase 2: Enhanced local search ---
    if best_params is not None:
        x = best_params.copy(); fx = best
        step = 0.05 * ranges
        prev_move = np.zeros(dim)
        
        while elapsed() < max_time * 0.97:
            improved = False
            # Pattern move
            if np.any(prev_move != 0):
                xp = clip(x + prev_move)
                fp = evaluate(xp)
                if fp < fx:
                    prev_move = xp - x
                    x = xp; fx = fp; improved = True
            
            perm = np.random.permutation(dim)
            old_x = x.copy()
            for j in perm:
                if elapsed() >= max_time * 0.97: break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn; fx = fn; step[j] *= 1.3; improved = True; break
            prev_move = x - old_x
            
            if not improved:
                # Try random direction
                d = np.random.randn(dim)
                d /= (np.linalg.norm(d) + 1e-30)
                for sc in [np.mean(step)*2, np.mean(step)*0.5]:
                    xn = clip(x + sc * d * ranges)
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn; fx = fn; improved = True; break
                if not improved:
                    step *= 0.5
                    if np.max(step / ranges) < 1e-13:
                        x = best_params + np.random.randn(dim) * 0.003 * ranges
                        x = clip(x); fx = evaluate(x); step = 0.01 * ranges
                        prev_move = np.zeros(dim)
    
    return best
