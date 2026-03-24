#Looking at the algorithms, #1 performs best (13.38). I'll improve it with:
#
#1. Better L-SHADE with proper population reduction
#2. A real CMA-ES with covariance adaptation (not just isotropic sampling)
#3. Multiple restart strategies
#4. Better time allocation and avoiding wasted evaluations (like the extra `func(mean)` call in #3)
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
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size_init = min(max(40, 18 * dim), 400)
    pop_size = pop_size_init
    min_pop_size = max(6, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: L-SHADE ---
    memory_size = 25
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # Estimate generations based on time
    t_before_de = elapsed()
    
    while elapsed() < max_time * 0.55:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.2)))
            p_best_idx = np.random.randint(0, p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = evaluate(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(abs(fitness[i] - trial_fitness))
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = mean_F
            M_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % memory_size
        
        # Population size reduction
        ratio = min(1.0, (elapsed() - t_before_de) / (max_time * 0.55 - t_before_de + 1e-10))
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 35:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0

    # --- Phase 3: CMA-ES local search ---
    if best_params is not None and dim <= 100:
        sigma = 0.05
        mean = best_params.copy()
        lam = max(10, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff-1)/(dim+1))-1) + c_sigma
        cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c1 = 2 / ((dim+1.3)**2 + mu_eff)
        cmu = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((dim+2)**2+mu_eff))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while elapsed() < max_time * 0.94:
            try:
                A = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                A = np.eye(dim)
            
            samples, f_s, zs = [], [], []
            for _ in range(lam):
                if elapsed() >= max_time * 0.94:
                    break
                z = np.random.randn(dim)
                x = clip(mean + sigma * (A @ z) * ranges)
                f = evaluate(x)
                samples.append(x); f_s.append(f); zs.append(z)
            if len(samples) < mu:
                break
            idx_s = np.argsort(f_s)
            sel = np.array([samples[idx_s[j]] for j in range(mu)])
            z_sel = np.array([zs[idx_s[j]] for j in range(mu)])
            
            old_mean = mean.copy()
            mean = clip(np.dot(w, sel))
            
            zm = np.dot(w, z_sel)
            ps = (1-c_sigma)*ps + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*zm
            h_sig = 1 if np.linalg.norm(ps)/np.sqrt(1-(1-c_sigma)**(2*(generation+1))) < (1.4+2/(dim+1))*chiN else 0
            pc = (1-cc)*pc + h_sig*np.sqrt(cc*(2-cc)*mu_eff)*(mean-old_mean)/(sigma*ranges+1e-30)
            
            artmp = (sel - old_mean) / (sigma * ranges + 1e-30)
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-h_sig)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(w) @ artmp)
            
            sigma *= np.exp((c_sigma/d_sigma) * (np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-12, 1.0)
    
    # --- Phase 4: Fine local search ---
    if best_params is not None:
        scale = 0.001
        improvements = 0
        total = 0
        while elapsed() < max_time * 0.995:
            candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            old_best = best
            evaluate(candidate)
            total += 1
            if best < old_best:
                improvements += 1
            if total % 50 == 0:
                if improvements < 2:
                    scale *= 0.5
                elif improvements > 10:
                    scale *= 1.5
                scale = np.clip(scale, 1e-8, 0.01)
                improvements = 0
                total = 0
    
    return best
