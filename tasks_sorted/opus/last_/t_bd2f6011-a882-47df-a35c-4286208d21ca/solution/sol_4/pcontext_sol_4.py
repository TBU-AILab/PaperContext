#Looking at the previous algorithm, I see it's using L-SHADE with Nelder-Mead local search. Let me improve it with several enhancements:
#
#1. Better initial sampling with more diverse strategies
#2. Improved local search using a combination of Nelder-Mead and coordinate-wise line search
#3. More aggressive exploitation around the best solution
#4. Better parameter adaptation and restart strategy
#5. CMA-ES-inspired local search for better convergence
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok():
        return elapsed() < max_time * 0.95
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal evals, best, best_sol
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f
    
    def lhs_sample(n):
        result = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, d] = (perm[i] + np.random.random()) / n
        return lower + result * ranges
    
    # ---- CMA-ES inspired local search ----
    def cmaes_local(x0, sigma0, budget):
        nonlocal best, best_sol
        n = dim
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        used = 0
        eigeneval = 0
        
        while used < budget and time_ok():
            # Eigen decomposition
            if used - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = used
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            else:
                if used == 0:
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k_i in range(lam):
                arx[k_i] = clip(mean + sigma * (B @ (D * arz[k_i])))
            
            # Evaluate
            arfitness = np.zeros(lam)
            for k_i in range(lam):
                if not time_ok() or used >= budget:
                    return
                arfitness[k_i] = evaluate(arx[k_i])
                used += 1
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            # CSA
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(used/lam+1))) / chiN < 1.4 + 2/(n+1))
            
            # CMA
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            artmp = (selected - old_mean) / sigma
            C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Step size
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if sigma < 1e-16 * sigma0:
                break
            
            # Check condition
            if np.max(D) > 1e7 * np.min(D):
                break
    
    # ---- Golden section line search ----
    def line_search(x0, direction, max_step, ls_budget):
        nonlocal best, best_sol
        gr = (np.sqrt(5) + 1) / 2
        a = 0.0
        b = max_step
        used = 0
        
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = evaluate(clip(x0 + c * direction))
        fd = evaluate(clip(x0 + d * direction))
        used += 2
        
        while used < ls_budget and (b - a) > 1e-10 and time_ok():
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = evaluate(clip(x0 + c * direction))
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = evaluate(clip(x0 + d * direction))
            used += 1
        
        best_t = c if fc < fd else d
        return clip(x0 + best_t * direction)
    
    def coordinate_descent(x0, budget):
        nonlocal best, best_sol
        x = x0.copy()
        fx = evaluate(x)
        used = 1
        step_sizes = ranges * 0.1
        
        while used < budget and time_ok():
            improved_any = False
            for d in range(dim):
                if used + 3 >= budget or not time_ok():
                    return
                direction = np.zeros(dim)
                direction[d] = 1.0
                
                # Try positive
                x_plus = clip(x + step_sizes[d] * direction)
                f_plus = evaluate(x_plus)
                used += 1
                
                # Try negative
                x_minus = clip(x - step_sizes[d] * direction)
                f_minus = evaluate(x_minus)
                used += 1
                
                if f_plus < fx and f_plus <= f_minus:
                    # Line search in positive direction
                    x_new = line_search(x, direction, step_sizes[d]*3, min(8, budget-used))
                    f_new = evaluate(x_new)
                    used += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved_any = True
                elif f_minus < fx:
                    x_new = line_search(x, -direction, step_sizes[d]*3, min(8, budget-used))
                    f_new = evaluate(x_new)
                    used += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved_any = True
            
            if not improved_any:
                step_sizes *= 0.5
                if np.max(step_sizes / ranges) < 1e-12:
                    break
    
    # ---- L-SHADE ----
    pop_size_init = min(max(40, 12 * dim), 400)
    pop_size_min = 4
    pop_size = pop_size_init
    
    population = lhs_sample(pop_size)
    fitness = np.array([evaluate(ind) for ind in population])
    
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    
    archive = []
    archive_max = pop_size_init
    
    stagnation_counter = 0
    prev_best = best
    generation = 0
    restart_count = 0
    
    while time_ok():
        generation += 1
        
        time_per_eval = elapsed() / max(evals, 1)
        remaining_time = max_time * 0.95 - elapsed()
        est_remaining_evals = remaining_time / max(time_per_eval, 1e-9)
        total_est_evals = evals + est_remaining_evals
        ratio = min(1.0, evals / max(total_est_evals, 1))
        
        new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
        
        S_F, S_CR, S_delta = [], [], []
        sorted_idx = np.argsort(fitness)
        
        trial_pop = np.empty_like(population)
        trial_fit = np.empty(pop_size)
        
        for i in range(pop_size):
            if not time_ok():
                return best
            
            ri = np.random.randint(0, H)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 1.5:
                    Fi = 1.5
            Fi = min(Fi, 1.5)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(max(0.05, 0.2 - 0.15 * ratio) * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            for d2 in range(dim):
                if mutant[d2] < lower[d2]:
                    mutant[d2] = (lower[d2] + population[i][d2]) / 2
                elif mutant[d2] > upper[d2]:
                    mutant[d2] = (upper[d2] + population[i][d2]) / 2
            
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            tf = evaluate(trial)
            trial_pop[i] = trial
            trial_fit[i] = tf
            
            if tf <= fitness[i]:
                delta = fitness[i] - tf
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive_max > 0:
                        archive[np.random.randint(len(archive))] = population[i].copy()
        
        improved = trial_fit <= fitness
        population[improved] = trial_pop[improved]
        fitness[improved] = trial_fit[improved]
        
        if S_F:
            w = np.array(S_delta); w = w / (w.sum() + 1e-30)
            sf, scr = np.array(S_F), np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        if new_pop_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_pop_size]]
            fitness = fitness[si[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            archive_max = pop_size
        
        stagnation_counter = stagnation_counter + 1 if abs(prev_best - best) < 1e-15 else 0
        prev_best = best
        
        if stagnation_counter > 0 and stagnation_counter % 15 == 0 and time_ok() and best_sol is not None:
            budget = min(max(80, 8*dim), int(est_remaining_evals * 0.08))
            if ratio < 0.5:
                coordinate_descent(best_sol.copy(), budget)
            else:
                sigma0 = np.mean(ranges) * max(0.01, 0.1 * (1 - ratio))
                cmaes_local(best_sol.copy(), sigma0, budget)
        
        if stagnation_counter > max(60, 4*dim) and time_ok():
            restart_count += 1
            budget = min(max(150, 15*dim), int(est_remaining_evals * 0.12))
            cmaes_local(best_sol.copy(), np.mean(ranges)*0.05, budget)
            coordinate_descent(best_
