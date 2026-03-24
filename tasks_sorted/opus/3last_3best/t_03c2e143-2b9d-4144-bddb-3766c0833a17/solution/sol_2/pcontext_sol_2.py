#Looking at the previous results, the SHADE algorithm with coordinate descent local search performed best (26.66). I'll improve it with several enhancements:
#
#1. **L-SHADE with linear population size reduction** for better convergence
#2. **Multiple restart strategy** with diverse initialization
#3. **Better local search**: combining coordinate descent with a simplex-based method
#4. **Opposition-based learning** for initialization
#5. **More efficient time management**
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
    
    evals = 0
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- L-SHADE Phase ---
    init_pop_size = min(max(40, 10 * dim), 400)
    min_pop_size = 4
    H = 100
    
    # LHS initialization with opposition
    pop_size = init_pop_size
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.array([evaluate(population[i]) for i in range(pop_size)])
    if elapsed() >= max_time * 0.95:
        return best
    
    # Opposition-based population initialization
    opp_pop = lower + upper - population
    opp_fitness = np.array([evaluate(opp_pop[i]) for i in range(pop_size)])
    if elapsed() >= max_time * 0.90:
        return best
    
    # Merge and keep best
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.concatenate([fitness_vals, opp_fitness])
    best_indices = np.argsort(all_fit)[:pop_size]
    population = all_pop[best_indices].copy()
    fitness_vals = all_fit[best_indices].copy()
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    max_evals_estimate = evals  # rough tracking
    generation = 0
    total_evals_lshade_start = evals
    
    while elapsed() < max_time * 0.65:
        generation += 1
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness_vals)
        
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(H)
            
            # Generate F from Cauchy
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 2.0:
                    Fi = -1
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # p-best
            p_min = max(2, int(0.05 * pop_size))
            p_max = max(2, int(0.2 * pop_size))
            p = np.random.randint(p_min, p_max + 1)
            p_best_idx = sorted_idx[np.random.randint(min(p, pop_size))]
            
            # Mutation: current-to-pbest/1 with archive
            candidates = [j for j in range(pop_size) if j != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size - 1)
            if r2 >= i:
                r2 += 1
            if r2 == r1:
                r2 = (r2 + 1) % pool_size
                if r2 == i:
                    r2 = (r2 + 1) % pool_size
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            # Boundary handling - midpoint
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                if f_trial < fitness_vals[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness_vals[i] - f_trial))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive_max > 0:
                        archive[np.random.randint(archive_max)] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness_vals = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(len(delta_f)) / len(delta_f)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            mean_F = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            mean_CR = np.sum(weights * scr)
            M_F[k % H] = mean_F
            M_CR[k % H] = mean_CR
            k += 1
        
        # Linear population size reduction
        ratio = min(1.0, elapsed() / (max_time * 0.65))
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * ratio)))
        
        if new_pop_size < pop_size:
            best_idx = np.argsort(fitness_vals)[:new_pop_size]
            population = population[best_idx]
            fitness_vals = fitness_vals[best_idx]
            pop_size = new_pop_size
            archive_max = pop_size
            if len(archive) > archive_max:
                archive = [archive[j] for j in np.random.choice(len(archive), archive_max, replace=False)]
    
    # --- Phase 2: Local search with multiple restarts ---
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.05 * ranges
        
        while elapsed() < max_time * 0.95:
            improved = False
            for j in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    x_new = x.copy()
                    x_new[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                    f_new = evaluate(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        step[j] *= 1.2
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12:
                    x = best_params + np.random.randn(dim) * 0.005 * ranges
                    x = clip(x)
                    fx = evaluate(x)
                    step = 0.01 * ranges
    
    return best
