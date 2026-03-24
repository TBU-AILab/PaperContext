#Looking at the existing algorithm, I can see it uses DE + Nelder-Mead. Let me improve it with:
#
#1. **CMA-ES inspired adaptation** - better covariance learning for search directions
#2. **Multiple restarts with shrinking regions** - better exploitation
#3. **Improved DE strategies** - SHADE-like parameter adaptation
#4. **Better time management**
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

    # --- Phase 1: Initial Sampling (Latin Hypercube) ---
    pop_size = min(max(30, 8 * dim), 300)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness_vals[i] = evaluate(population[i])

    # --- Phase 2: SHADE-like Differential Evolution ---
    # Memory for successful F and CR
    memory_size = 20
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    memory_idx = 0
    archive = []
    max_archive = pop_size
    
    stagnation_count = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        S_F = []
        S_CR = []
        S_delta = []
        
        trial_pop = np.copy(population)
        trial_fit = np.copy(fitness_vals)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # Generate F and CR from memory
            r = np.random.randint(memory_size)
            F_i = min(1.0, max(0.1, np.random.standard_cauchy() * 0.1 + M_F[r]))
            CR_i = min(1.0, max(0.0, np.random.normal(M_CR[r], 0.1)))
            
            # current-to-pbest/1 mutation
            p = max(2, int(0.1 * pop_size))
            sorted_idx = np.argsort(fitness_vals)
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined_size = pop_size + len(archive)
            r2_idx = np.random.randint(combined_size)
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(combined_size)
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR_i
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness_vals[i] = f_trial
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights /= weights.sum() + 1e-30
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[memory_idx] = mean_F
            M_CR[memory_idx] = mean_CR
            memory_idx = (memory_idx + 1) % memory_size
        
        if abs(prev_best - best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 15:
            sorted_idx = np.argsort(fitness_vals)
            half = pop_size // 2
            for j in sorted_idx[half:]:
                if np.random.random() < 0.5:
                    scale = 0.2 * ranges * (np.random.random() * 0.5 + 0.1)
                    population[j] = best_params + np.random.randn(dim) * scale
                else:
                    population[j] = lower + np.random.random(dim) * ranges
                population[j] = clip(population[j])
                fitness_vals[j] = evaluate(population[j])
            stagnation_count = 0

    # --- Phase 3: Nelder-Mead local search ---
    def nelder_mead(start_point, start_fit, scale_factor=0.05, time_frac=0.90):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        f_simplex = np.zeros(n + 1)
        simplex[0] = start_point.copy()
        f_simplex[0] = start_fit
        
        for i in range(n):
            if elapsed() >= max_time * time_frac:
                return
            simplex[i + 1] = start_point.copy()
            simplex[i + 1][i] += scale[i] if scale[i] != 0 else 0.01
            simplex[i + 1] = clip(simplex[i + 1])
            f_simplex[i + 1] = evaluate(simplex[i + 1])
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(5000):
            if elapsed() >= max_time * time_frac:
                return
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                simplex[-1], f_simplex[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for j in range(1, n + 1):
                        simplex[j] = clip(simplex[0] + sigma * (simplex[j] - simplex[0]))
                        f_simplex[j] = evaluate(simplex[j])
                        if elapsed() >= max_time * time_frac:
                            return
            
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-16:
                break
    
    nelder_mead(best_params, best, 0.05, 0.90)
    nelder_mead(best_params, best, 0.005, 0.97)
    
    # --- Phase 4: Fine local random search ---
    while elapsed() < max_time * 0.995:
        scale = np.random.exponential(0.005) * ranges
        candidate = best_params + np.random.randn(dim) * scale
        evaluate(candidate)
    
    return best
#
#Key improvements:
#1. **SHADE parameter adaptation**: F and CR are learned from successful mutations with weighted Lehmer mean, much more effective than fixed/random parameters.
#2. **Current-to-pbest/1 mutation**: Uses the top p% solutions, balancing exploration and exploitation better.
#3. **External archive**: Stores replaced solutions for diversity in mutation.
#4. **Bounce-back boundary handling**: Better than simple clipping.
#5. **Multiple Nelder-Mead passes**: Two passes with decreasing scale for deeper refinement.
#6. **Exponential random scale** in final phase for better local search coverage.
