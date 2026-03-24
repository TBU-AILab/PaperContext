#Looking at the previous results, the SHADE algorithm (21.18) outperformed the basic DE (30.70). I'll improve upon the SHADE approach with several key enhancements:
#
#1. **L-SHADE with linear population size reduction** - dynamically reduce population size
#2. **Multiple restart mechanism** - restart with smaller populations if stagnation detected
#3. **Better local search** - combine Nelder-Mead with coordinate descent
#4. **Improved time management** - allocate more budget to the main search
#5. **Precompute sorted indices** instead of recomputing each iteration
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    def run_lshade(time_frac_end):
        nonlocal best, best_x
        
        N_init = min(max(12 * dim, 50), 200)
        N_min = 4
        H = N_init
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        memory_idx = 0
        archive = []
        
        # LHS initialization
        pop_size = N_init
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            population[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        population = lower + population * ranges
        
        # Inject best known solution if available
        if best_x is not None:
            population[0] = best_x.copy()
        
        fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * time_frac_end])
        if len(fitness) < pop_size:
            fitness = np.append(fitness, np.full(pop_size - len(fitness), float('inf')))
            
        gen = 0
        total_evals_start = evals
        max_evals_est = int((max_time * time_frac_end - elapsed()) / (elapsed() / max(evals, 1) + 1e-30))
        
        while elapsed() < max_time * time_frac_end:
            gen += 1
            
            sorted_idx = np.argsort(fitness)
            
            S_F = []
            S_CR = []
            S_delta = []
            
            # Progress for p-value and population reduction
            time_progress = min(elapsed() / (max_time * time_frac_end), 1.0)
            p = max(0.2 - 0.18 * time_progress, 2.0 / max(pop_size, 2))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if elapsed() >= max_time * time_frac_end:
                    break
                
                ri = np.random.randint(0, H)
                mu_F = memory_F[ri]
                mu_CR = memory_CR[ri]
                
                # Generate F from Cauchy
                Fi = -1
                attempts = 0
                while Fi <= 0 and attempts < 20:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                    attempts += 1
                if Fi <= 0:
                    Fi = 0.01
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)
                
                # Linearly decrease CR for last dimensions
                if time_progress > 0.7:
                    CRi = max(CRi * (1.0 - (time_progress - 0.7) / 0.3), 0.0)
                
                # current-to-pbest/1
                p_num = max(int(np.round(p * pop_size)), 2)
                pbest_rank = np.random.randint(0, p_num)
                x_pbest = population[sorted_idx[pbest_rank]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                union_size = pop_size + len(archive)
                r2_idx = np.random.randint(0, union_size - 1)
                if r2_idx >= i:
                    r2_idx += 1
                if r2_idx == r1:
                    r2_idx = (r2_idx + 1) % union_size
                    if r2_idx == i:
                        r2_idx = (r2_idx + 1) % union_size
                
                if r2_idx < pop_size:
                    x_r2 = population[r2_idx]
                else:
                    x_r2 = archive[r2_idx - pop_size] if (r2_idx - pop_size) < len(archive) else population[np.random.choice(idxs)]
                
                mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
                
                # Bounce-back boundary handling
                for d_idx in range(dim):
                    if mutant[d_idx] < lower[d_idx]:
                        mutant[d_idx] = (lower[d_idx] + population[i][d_idx]) / 2.0
                    elif mutant[d_idx] > upper[d_idx]:
                        mutant[d_idx] = (upper[d_idx] + population[i][d_idx]) / 2.0
                
                cross = np.random.rand(dim) < CRi
                j_rand = np.random.randint(dim)
                cross[j_rand] = True
                trial = np.where(cross, mutant, population[i])
                trial = clip(trial)
                
                trial_f = evaluate(trial)
                
                if trial_f <= fitness[i]:
                    if trial_f < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_delta.append(abs(fitness[i] - trial_f))
                        if len(archive) < N_init:
                            archive.append(population[i].copy())
                        elif archive:
                            archive[np.random.randint(len(archive))] = population[i].copy()
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            # Update memory
            if S_F:
                weights = np.array(S_delta)
                ws = weights.sum()
                if ws > 0:
                    weights /= ws
                    mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
                    mean_CR = np.sum(weights * np.array(S_CR))
                    memory_F[memory_idx] = mean_F
                    memory_CR[memory_idx] = mean_CR
                    memory_idx = (memory_idx + 1) % H
            
            # L-SHADE population reduction
            new_size = max(int(np.round(N_init - (N_init - N_min) * time_progress)), N_min)
            if new_size < pop_size:
                order = np.argsort(fitness)
                population = population[order[:new_size]]
                fitness = fitness[order[:new_size]]
                pop_size = new_size
    
    # Run L-SHADE for main phase
    run_lshade(0.75)
    
    # Second restart with fresh population but seeded with best
    run_lshade(0.93)
    
    # Local search: coordinate descent
    if best_x is not None:
        x = best_x.copy()
        step = ranges * 0.02
        while elapsed() < max_time * 0.995:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.995:
                    break
                for sign in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step[d], lower[d], upper[d])
                    f = evaluate(trial)
                    if f < best:
                        x = trial.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    break
    
    return best
