#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    init_samples = []
    for i in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        params = lower + (upper - lower) * np.random.rand(dim)
        fitness = func(params)
        init_samples.append((fitness, params.copy()))
        if fitness < best:
            best = fitness
            best_params = params.copy()
    
    # Sort by fitness
    init_samples.sort(key=lambda x: x[0])
    
    # --- Phase 2: CMA-ES inspired search ---
    # Use top samples to initialize
    mu = max(10, dim)
    pop_size = max(20, 2 * dim)
    
    # Initialize mean from best solution
    mean = best_params.copy()
    sigma = 0.3 * (upper - lower)
    
    # Keep track of elite population
    elite_size = min(mu, len(init_samples))
    elite = [s[1] for s in init_samples[:elite_size]]
    elite_fitness = [s[0] for s in init_samples[:elite_size]]
    
    # Adaptive covariance - start with diagonal
    C = np.diag(((upper - lower) / 6.0) ** 2)
    use_full_cov = dim <= 50
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    # Differential evolution weight and crossover
    F = 0.8
    CR = 0.9
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.95):
            return best
        
        remaining = max_time - elapsed.total_seconds()
        if remaining <= 0.05:
            return best
        
        generation += 1
        
        # Adaptive strategy selection based on progress
        new_pop = []
        new_fitness = []
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.93):
                return best
            
            r = np.random.rand()
            
            if r < 0.4:
                # CMA-ES style: sample around mean with covariance
                if use_full_cov and dim <= 50:
                    try:
                        trial = np.random.multivariate_normal(mean, sigma_scale_factor(sigma, generation) * C)
                    except:
                        trial = mean + sigma * np.random.randn(dim) / np.sqrt(generation * 0.01 + 1)
                else:
                    trial = mean + sigma * np.random.randn(dim)
                    
            elif r < 0.7:
                # DE/best/1 style mutation
                if len(elite) >= 3:
                    idxs = np.random.choice(len(elite), 3, replace=True)
                    a, b, c = elite[idxs[0]], elite[idxs[1]], elite[idxs[2]]
                    mutant = best_params + F * (b - c)
                    # Binomial crossover
                    trial = np.copy(best_params)
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                else:
                    trial = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                    
            elif r < 0.85:
                # Local search around best
                scale = 0.05 * (upper - lower) / (1 + generation * 0.05)
                trial = best_params + scale * np.random.randn(dim)
                
            else:
                # Random restart with bias toward best
                trial = lower + (upper - lower) * np.random.rand(dim)
                if np.random.rand() < 0.5:
                    trial = 0.5 * trial + 0.5 * best_params
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            fitness = func(trial)
            new_pop.append(trial.copy())
            new_fitness.append(fitness)
            
            if fitness < best:
                best = fitness
                best_params = trial.copy()
        
        # Update elite
        combined = list(zip(elite_fitness + new_fitness, elite + new_pop))
        combined.sort(key=lambda x: x[0])
        combined = combined[:elite_size]
        elite_fitness = [x[0] for x in combined]
        elite = [x[1] for x in combined]
        
        # Update mean from elite
        weights = np.array([np.log(elite_size + 0.5) - np.log(i + 1) for i in range(len(elite))])
        weights = weights / weights.sum()
        mean = np.zeros(dim)
        for w, e in zip(weights, elite):
            mean += w * e
        
        # Update covariance estimate
        if use_full_cov and len(elite) > 1 and dim <= 50:
            try:
                diff = np.array(elite) - mean
                C = np.zeros((dim, dim))
                for w, d in zip(weights, diff):
                    C += w * np.outer(d, d)
                C += 1e-8 * np.eye(dim)
            except:
                pass
        
        # Adaptive sigma
        if best < prev_best:
            sigma *= 1.0  # keep sigma
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count > 5:
                sigma *= 0.85
            if stagnation_count > 15:
                # Reset sigma and diversify
                sigma = 0.3 * (upper - lower)
                stagnation_count = 0
        
        prev_best = best
    
    return best


def sigma_scale_factor(sigma, generation):
    """Create a scaling matrix from sigma vector."""
    factor = 1.0 / (1 + generation * 0.01)
    return factor * np.diag(sigma ** 2)
#
#Wait, I have `sigma_scale_factor` defined outside the `run` function but it's used inside. Let me also reconsider the approach - let me make it cleaner and more robust:
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
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
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
    
    # --- Phase 1: Initial sampling ---
    n_init = min(max(30 * dim, 200), 1000)
    population = []
    fitnesses = []
    
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            return best
        x = lower + ranges * np.random.rand(dim)
        f = evaluate(x)
        population.append(x.copy())
        fitnesses.append(f)
    
    population = np.array(population)
    fitnesses = np.array(fitnesses)
    
    # --- Phase 2: Hybrid DE + local search ---
    pop_size = min(max(10 * dim, 50), 200)
    
    # Select top individuals as initial population
    sorted_idx = np.argsort(fitnesses)
    pop = population[sorted_idx[:pop_size]].copy()
    pop_fit = fitnesses[sorted_idx[:pop_size]].copy()
    
    F = 0.8
    CR = 0.9
    generation = 0
    
    while True:
        if time_left() < max_time * 0.02:
            return best
        
        generation += 1
        
        # Adaptive parameters
        F_gen = 0.5 + 0.3 * np.random.rand()
        CR_gen = 0.8 + 0.2 * np.random.rand()
        
        for i in range(pop_size):
            if time_left() < max_time * 0.02:
                return best
            
            strategy = np.random.rand()
            
            if strategy < 0.5:
                # DE/best/1/bin
                idxs = np.random.choice(pop_size, 2, replace=False)
                while i in idxs:
                    idxs = np.random.choice(pop_size, 2, replace=False)
                mutant = best_params + F_gen * (pop[idxs[0]] - pop[idxs[1]])
            elif strategy < 0.8:
                # DE/rand/1/bin
                idxs = np.random.choice(pop_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(pop_size, 3, replace=False)
                mutant = pop[idxs[0]] + F_gen * (pop[idxs[1]] - pop[idxs[2]])
            else:
                # DE/current-to-best/1
                idxs = np.random.choice(pop_size, 2, replace=False)
                while i in idxs:
                    idxs = np.random.choice(pop_size, 2, replace=False)
                mutant = pop[i] + F_gen * (best_params - pop[i]) + F_gen * (pop[idxs[0]] - pop[idxs[1]])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR_gen
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial <= pop_fit[i]:
                pop[i] = trial
                pop_fit[i] = f_trial
        
        # Periodically do local search around best
        if generation % 3 == 0 and time_left() > max_time * 0.1:
            # Nelder-Mead-like simplex or coordinate descent
            local_x = best_params.copy()
            local_f = best
            step = 0.01 * ranges
            
            for coord_iter in range(2):
                for d in range(dim):
                    if time_left() < max_time * 0.05:
                        return best
                    
                    # Try positive step
                    trial_p = local_x.copy()
                    trial_p[d] += step[d]
                    f_p = evaluate(clip(trial_p))
                    
                    # Try negative step
                    trial_n = local_x.copy()
                    trial_n[d] -= step[d]
                    f_n = evaluate(clip(trial_n))
                    
                    if f_p < local_f and f_p <= f_n:
                        local_x = clip(trial_p)
                        local_f = f_p
                    elif f_n < local_f:
                        local_x = clip(trial_n)
                        local_f = f_n
                
                step *= 0.5
        
        # Replace worst individuals with random ones to maintain diversity
        if generation % 10 == 0:
            n_replace = max(1, pop_size // 10)
            worst_idx = np.argsort(pop_fit)[-n_replace:]
            for idx in worst_idx:
                if time_left() < max_time * 0.02:
                    return best
                # Bias toward best region
                new_x = best_params + 0.3 * ranges * (2 * np.random.rand(dim) - 1)
                new_x = clip(new_x)
                new_f = evaluate(new_x)
                pop[idx] = new_x
                pop_fit[idx] = new_f
    
    return best
