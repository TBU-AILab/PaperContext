#Looking at the results, the best algorithm (6.40) uses L-SHADE with 2 restarts (0.75 and 0.93 time fractions) plus coordinate descent. The key issues I see:
#
#1. **CR reduction is harmful** - the best code has aggressive CR reduction after 70% progress, which hurts
#2. **Time allocation matters** - 2 restarts (75%/93%) beats 3 restarts (55%/80%/92%)
#3. **Need better local search** - coordinate descent is limited; adding pattern search / simplex could help
#4. **Population initialization** - can improve seeding strategy
#
#I'll refine with: remove CR reduction, keep 2 L-SHADE restarts with better time splits, add a more effective local search (golden section + pattern search), and improve the overall robustness.
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    def run_lshade(time_frac_end, N_init=None):
        nonlocal best, best_x
        
        if N_init is None:
            N_init = min(max(12 * dim, 50), 200)
        N_min = 4
        H = min(N_init, 100)
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        memory_idx = 0
        archive = []
        
        pop_size = N_init
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            population[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        population = lower + population * ranges
        
        if best_x is not None:
            population[0] = best_x.copy()
            n_perturb = min(pop_size // 4, 10)
            for k in range(1, n_perturb + 1):
                scale = 0.1 * (k / n_perturb)
                population[k] = clip(best_x + np.random.randn(dim) * ranges * scale)
        
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            if elapsed() >= max_time * time_frac_end:
                fitness[i:] = float('inf')
                break
            fitness[i] = evaluate(population[i])
        
        phase_start = elapsed()
        phase_end = max_time * time_frac_end
        phase_duration = phase_end - phase_start
        if phase_duration <= 0:
            return
        
        while elapsed() < phase_end:
            sorted_idx = np.argsort(fitness)
            
            S_F = []
            S_CR = []
            S_delta = []
            
            time_progress = min((elapsed() - phase_start) / max(phase_duration, 1e-10), 1.0)
            p = max(0.2 - 0.15 * time_progress, 2.0 / max(pop_size, 2))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if elapsed() >= phase_end:
                    break
                
                ri = np.random.randint(0, H)
                mu_F = memory_F[ri]
                mu_CR = memory_CR[ri]
                
                Fi = -1
                for _ in range(20):
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                    if Fi > 0:
                        break
                Fi = np.clip(Fi, 0.01, 1.0)
                
                CRi = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)
                
                p_num = max(int(np.round(p * pop_size)), 2)
                x_pbest = population[sorted_idx[np.random.randint(0, p_num)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(0, pool_size - 1)
                if r2 >= i: r2 += 1
                if r2 == r1: r2 = (r2 + 1) % pool_size
                if r2 == i: r2 = (r2 + 1) % pool_size
                
                if r2 < pop_size:
                    x_r2 = population[r2]
                elif (r2 - pop_size) < len(archive):
                    x_r2 = archive[r2 - pop_size]
                else:
                    x_r2 = population[np.random.choice(idxs)]
                
                mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
                mutant = np.where(mutant < lower, (lower + population[i]) / 2, mutant)
                mutant = np.where(mutant > upper, (upper + population[i]) / 2, mutant)
                
                cross = np.random.rand(dim) < CRi
                cross[np.random.randint(dim)] = True
                trial = clip(np.where(cross, mutant, population[i]))
                
                trial_f = evaluate(trial)
                
                if trial_f <= fitness[i]:
                    if trial_f < fitness[i]:
                        S_F.append(Fi); S_CR.append(CRi)
                        S_delta.append(abs(fitness[i] - trial_f))
                        if len(archive) < N_init:
                            archive.append(population[i].copy())
                        elif archive:
                            archive[np.random.randint(len(archive))] = population[i].copy()
                    new_pop[i] = trial; new_fit[i] = trial_f
            
            population = new_pop; fitness = new_fit
            
            if S_F:
                w = np.array(S_delta); ws = w.sum()
                if ws > 0:
                    w /= ws
                    memory_F[memory_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                    memory_CR[memory_idx] = np.sum(w * np.array(S_CR))
                    memory_idx = (memory_idx + 1) % H
            
            new_size = max(int(np.round(N_init - (N_init - N_min) * time_progress)), N_min)
            if new_size < pop_size:
                order = np.argsort(fitness)
                population = population[order[:new_size]]; fitness = fitness[order[:new_size]]
                pop_size = new_size
    
    run_lshade(0.70, N_init=min(max(12*dim, 50), 200))
    run_lshade(0.92, N_init=min(max(8*dim, 30), 120))
    
    # Pattern search + coordinate descent local search
    if best_x is not None:
        x = best_x.copy(); f_x = best
        step = ranges * 0.02
        while elapsed() < max_time * 0.997:
            improved = False
            order = np.random.permutation(dim)
            for d in order:
                if elapsed() >= max_time * 0.997: break
                for sign in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step[d], lower[d], upper[d])
                    f = evaluate(trial)
                    if f < f_x:
                        x = trial.copy(); f_x = f; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-15: break
    
    return best
