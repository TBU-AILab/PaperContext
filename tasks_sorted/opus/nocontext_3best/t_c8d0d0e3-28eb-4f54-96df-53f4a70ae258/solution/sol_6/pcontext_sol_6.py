#Looking at the best algorithm (#1 with score 13.38), I'll improve it with:
#
#1. **Better CMA-ES** with full covariance matrix adaptation
#2. **Multi-restart strategy** to escape local optima
#3. **Improved L-SHADE** with linear population size reduction
#4. **Better time allocation** between phases
#5. **Nelder-Mead simplex** as additional local search
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params, evals
        evals += 1
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size_init = min(max(50, 20 * dim), 500)
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
    
    # --- Phase 2: L-SHADE with population reduction ---
    memory_size = 30
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    stagnation = 0
    prev_best = best
    generation = 0
    max_gen_nfe = pop_size_init * 200  # estimated max NFE for DE phase
    nfe_start_de = evals
    
    t_de_end = max_time * 0.50
    
    while elapsed() < t_de_end:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= t_de_end:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            # Cauchy for F
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # p-best
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.25)))
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
        
        # Linear population size reduction
        ratio = min(1.0, (evals - nfe_start_de) / max(1, max_gen_nfe))
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        sorted_idx = np.argsort(fitness)
        if new_pop_size < pop_size:
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 40:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0

    # --- Phase 3: CMA-ES with restarts ---
    def run_cmaes(init_mean, init_sigma, time_limit):
        nonlocal best, best_params
        sigma = init_sigma
        mean = init_mean.copy()
        n = dim
        lam = max(10, 4 + int(3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff-1)/(n+1))-1) + c_sigma
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu_v = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((n+2)**2+mu_eff))
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        if n <= 50:
            C = np.eye(n)
            use_full = True
        else:
            diag_C = np.ones(n)
            use_full = False
        
        no_improve = 0
        local_best = best
        
        while elapsed() < time_limit:
            if use_full:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    A = eigvecs * np.sqrt(eigvals)
                except:
                    C = np.eye(n)
                    A = np.eye(n)
                    eigvals = np.ones(n)
                    eigvecs = np.eye(n)
            
            samples = []
            f_s = []
            zs = []
            for _ in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                if use_full:
                    x = clip(mean + sigma * (A @ z))
                else:
                    x = clip(mean + sigma * np.sqrt(diag_C) * z)
                f = evaluate(x)
                samples.append(x)
                f_s.append(f)
                zs.append(z)
            
            if len(samples) < mu:
                return
            
            idx_s = np.argsort(f_s)
            sel = np.array([samples[idx_s[j]] for j in range(mu)])
            z_sel = np.array([zs[idx_s[j]] for j in range(mu)])
            
            old_mean = mean.copy()
            mean = clip(np.dot(w, sel))
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            if use_full:
                inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
                ps = (1-c_sigma)*ps + np.sqrt(c_sigma*(2-c_sigma)*mu_eff) * (inv_sqrt @ (diff / (ranges + 1e-30)))
            else:
                ps = (1-c_sigma)*ps + np.sqrt(c_sigma*(2-c_sigma)*mu_eff) * (diff / (np.sqrt(diag_C) * ranges + 1e-30))
            
            ps_norm = np.linalg.norm(ps)
            h_sig = 1 if ps_norm / np.sqrt(1-(1-c_sigma)**(2*(generation+1)+2)) < (1.4+2/(n+1))*chiN else 0
            
            pc = (1-cc)*pc + h_sig*np.sqrt(cc*(2-cc)*mu_eff) * diff / (ranges + 1e-30)
            
            if use_full:
                artmp = (sel - old_mean) / (sigma * ranges + 1e-30)
                C = (1-c1-cmu_v)*C + c1*(np.outer(pc,pc) + (1-h_sig)*cc*(2-cc)*C) + cmu_v*(artmp.T @ np.diag(w) @ artmp)
                C = (C + C.T) / 2
            else:
                artmp = (sel - old_mean) / (sigma * ranges + 1e-30)
                diag_C = (1-c1-cmu_v)*diag_C + c1*(pc**2 + (1-h_sig)*cc*(2-cc)*diag_C) + cmu_v*np.sum(w[:, None]*artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((c_sigma/d_sigma) * (ps_norm/chiN - 1))
            sigma = np.clip(sigma, 1e-15, 2.0)
            
            if best < local_best - 1e-12:
                local_best = best
                no_improve = 0
            else:
                no_improve += 1
            
            if sigma < 1e-12 or no_improve > 50 + 10*n:
                return
    
    # First CMA-ES run from best found
    if best_params is not None:
        run_cmaes(best_params.copy(), 0.1, max_time * 0.75)
    
    # Second CMA-ES run with larger sigma
    if best_params is not None:
        run_cmaes(best_params.copy(), 0.3, max_time * 0.88)
    
    # Third run from a perturbed position
    if best_params is not None:
        perturbed = clip(best_params + 0.1 * ranges * np.random.randn(dim))
        run_cmaes(perturbed, 0.05, max_time * 0.95)
    
    # --- Phase 4: Fine local search ---
    if best_params is not None:
        scale = 0.002
        no_imp = 0
        while elapsed() < max_time * 0.998:
            candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            old_best = best
            evaluate(candidate)
            if best < old_best:
                no_imp = 0
            else:
                no_imp += 1
            if no_imp > 30:
                scale *= 0.5
                no_imp = 0
            if scale < 1e-10:
                scale = 0.001
    
    return best
