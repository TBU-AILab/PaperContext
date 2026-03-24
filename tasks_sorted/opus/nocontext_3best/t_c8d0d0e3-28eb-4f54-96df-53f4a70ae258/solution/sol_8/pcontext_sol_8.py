#Looking at the results, algorithm #1 (score 13.38) is the best. I'll improve it with:
#
#1. **Better SHADE** with weighted Lehmer mean and proper archive handling
#2. **Proper CMA-ES** with sep-CMA for high dimensions
#3. **Multi-restart CMA-ES** with different starting points from DE population
#4. **Adaptive Nelder-Mead** as final polishing
#5. **Better time allocation**
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
    pop_size = min(max(40, 15 * dim), 300)
    pop_size_init = pop_size
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
    t_de_start = elapsed()
    t_de_end = max_time * 0.50
    
    while elapsed() < t_de_end:
        generation += 1
        S_F, S_CR, S_delta = [], [], []
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= t_de_end:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F, mu_CR = M_F[ri], M_CR[ri]
            
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
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Population reduction
        ratio = min(1.0, (elapsed() - t_de_start) / (t_de_end - t_de_start + 1e-10))
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx[:min(new_ps, pop_size)]]
        fitness = fitness[sorted_idx[:min(new_ps, pop_size)]]
        pop_size = len(population)
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        if stagnation > 35:
            nr = max(1, pop_size // 3)
            for j in range(pop_size - nr, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0

    # Collect top candidates for restarts
    top_k = min(5, pop_size)
    candidates = [population[i].copy() for i in range(top_k)]

    # --- Phase 3: Sep-CMA-ES restarts ---
    def run_sep_cma(init_mean, init_sigma, t_end):
        nonlocal best, best_params
        n = dim
        sigma = init_sigma
        mean = init_mean.copy()
        lam = max(10, 4 + int(3 * np.log(n)))
        mu = lam // 2
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts /= wts.sum()
        mu_eff = 1.0 / np.sum(wts**2)
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1))-1) + c_s
        cc = (4+mu_eff/n)/(n+4+2*mu_eff/n)
        c1 = 2/((n+1.3)**2+mu_eff)
        cmu_val = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((n+2)**2+mu_eff))
        chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        ps = np.zeros(n); pc = np.zeros(n)
        diag_C = np.ones(n)
        gen = 0; no_imp = 0; lb = best
        while elapsed() < t_end:
            gen += 1
            sqC = np.sqrt(diag_C)
            samps, fs, zs = [], [], []
            for _ in range(lam):
                if elapsed() >= t_end: break
                z = np.random.randn(n)
                x = clip(mean + sigma * sqC * z * ranges)
                f = evaluate(x)
                samps.append(x); fs.append(f); zs.append(z)
            if len(samps) < mu: break
            idx = np.argsort(fs)
            sel = np.array([samps[idx[j]] for j in range(mu)])
            zsel = np.array([zs[idx[j]] for j in range(mu)])
            old_mean = mean.copy()
            mean = clip(np.dot(wts, sel))
            zm = np.dot(wts, zsel)
            ps = (1-c_s)*ps + np.sqrt(c_s*(2-c_s)*mu_eff)*zm
            pn = np.linalg.norm(ps)
            hsig = 1 if pn/np.sqrt(1-(1-c_s)**(2*gen)) < (1.4+2/(n+1))*chiN else 0
            dy = (mean - old_mean)/(sigma*ranges+1e-30)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mu_eff)*dy
            artmp = (sel - old_mean)/(sigma*ranges+1e-30)
            diag_C = (1-c1-cmu_val)*diag_C + c1*(pc**2+(1-hsig)*cc*(2-cc)*diag_C) + cmu_val*np.sum(wts[:,None]*artmp**2, axis=0)
            diag_C = np.maximum(diag_C, 1e-20)
            sigma *= np.exp((c_s/d_s)*(pn/chiN-1))
            sigma = np.clip(sigma, 1e-14, 1.0)
            if best < lb - 1e-12: lb = best; no_imp = 0
            else: no_imp += 1
            if sigma < 1e-12 or no_imp > 40+5*n: break

    # Multiple restarts
    sigmas = [0.05, 0.15, 0.3]
    time_fracs = [0.70, 0.82, 0.92]
    for restart in range(3):
        if elapsed() >= max_time * 0.92:
            break
        ci = restart % len(candidates)
        run_sep_cma(candidates[ci] if restart < len(candidates) else best_params.copy(), sigmas[restart], max_time * time_fracs[restart])

    # --- Phase 4: Fine local Nelder-Mead style + random perturbation ---
    if best_params is not None:
        scale = 0.002
        no_imp = 0
        while elapsed() < max_time * 0.998:
            # Coordinate-wise search mixed with random
            if no_imp < 20:
                candidate = best_params.copy()
                k = np.random.randint(0, dim)
                candidate[k] += scale * ranges[k] * np.random.randn()
                candidate = clip(candidate)
            else:
                candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            old_b = best
            evaluate(candidate)
            if best < old_b:
                no_imp = 0
            else:
                no_imp += 1
            if no_imp > 60:
                scale *= 0.5
                no_imp = 0
            if scale < 1e-10:
                scale = 0.001

    return best
