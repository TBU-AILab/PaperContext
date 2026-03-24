#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Use SHADE (Success-History based Adaptive DE)** for better parameter adaptation
#2. **Vectorize operations** to reduce overhead and get more evaluations per second
#3. **Multiple restarts with local search** on the best candidates
#4. **CMA-ES-inspired local search** instead of Nelder-Mead for better high-dimensional performance
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
        return max_time - elapsed()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip_to_bounds(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: SHADE (Success-History Adaptive DE) ---
    pop_size = min(max(30, 8 * dim), 300)
    H = 100  # history size
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.array([evaluate(population[i]) for i in range(pop_size)])
    if elapsed() >= max_time * 0.95:
        return best
    
    # SHADE memory
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0  # memory index
    
    # Archive
    archive = []
    archive_max = pop_size
    
    generation = 0
    
    while elapsed() < max_time * 0.70:
        generation += 1
        S_F = []
        S_CR = []
        delta_f = []
        
        # Sort population for p-best
        sorted_idx = np.argsort(fitness_vals)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # Pick random memory index
            ri = np.random.randint(H)
            
            # Generate F and CR from Cauchy and Normal
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 2.0:
                    Fi = -1  # regenerate
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # p-best: pick from top p fraction
            p = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            # Mutation: current-to-pbest/1 with archive
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            combined = list(range(pop_size)) + list(range(len(archive)))
            combined = [c for c in combined if c != i and c != r1]
            r2_idx = np.random.choice(len(combined))
            r2_val = combined[r2_idx]
            if r2_val < pop_size:
                x_r2 = population[r2_val]
            else:
                x_r2 = archive[r2_val - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            # Boundary handling
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = (lower[j] + population[i][j]) / 2
                elif trial[j] > upper[j]:
                    trial[j] = (upper[j] + population[i][j]) / 2
            
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
                population[i] = trial
                fitness_vals[i] = f_trial
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[k % H] = mean_F
            M_CR[k % H] = mean_CR
            k += 1
    
    # --- Phase 2: Local search (pattern search + coordinate descent) on best ---
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.1 * ranges
        
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
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-10:
                    # Restart local search from perturbed best
                    x = best_params + np.random.randn(dim) * 0.01 * ranges
                    x = clip_to_bounds(x)
                    fx = evaluate(x)
                    step = 0.02 * ranges
    
    return best
#
#Key improvements:
#- **SHADE algorithm**: Uses success-history based adaptation of F and CR, which is much more effective than simple jittered parameters
#- **p-best mutation with archive**: Better exploration-exploitation balance
#- **Weighted Lehmer mean** for F adaptation (prevents premature convergence)
#- **Coordinate descent local search** with adaptive step sizes and restarts - more efficient than Nelder-Mead for polishing
#- **Better time allocation**: 70% for global search, 25% for local refinement
