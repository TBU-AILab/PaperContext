#Looking at the results, algorithm #2 performed best (5.92). I'll analyze what made it successful and improve upon it:
#
#1. L-SHADE with good parameter initialization (M_F=0.5, M_CR=0.5)
#2. Nelder-Mead local search at multiple scales
#3. Fine coordinate descent at the end
#
#Key improvements I'll make:
#- Add a simple CMA-ES local search phase (lightweight implementation)
#- Better multi-start strategy using top solutions from DE
#- Improved time allocation
#- Add random perturbation restarts when local search stagnates
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # --- Phase 1: LHS initial population ---
    pop_size_init = min(max(14 * dim, 40), 140)
    pop_size = pop_size_init
    min_pop_size = max(4, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # Store top solutions for later multi-start
    top_solutions = []
    
    # --- Phase 2: L-SHADE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    h_idx = 0
    archive = []
    archive_max = pop_size_init
    
    stagnation_counter = 0
    last_best = best
    
    while elapsed() < max_time * 0.45:
        sorted_idx = np.argsort(fitness)
        
        S_F, S_CR, S_w = [], [], []
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        p_min = max(2.0 / pop_size, 0.05)
        p_max = 0.2
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.45:
                break
            
            ri = np.random.randint(0, H)
            
            Fi = -1
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0) if M_CR[ri] >= 0 else 0.0
            
            p = np.random.uniform(p_min, p_max)
            p_num = max(2, int(np.ceil(p * pop_size)))
            pbest_idx = sorted_idx[np.random.randint(0, p_num)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            combined_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.random(dim) < CRi
            cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(abs(fitness[i] - trial_f))
                new_population[i] = trial; new_fitness[i] = trial_f
        
        population = new_population; fitness = new_fitness
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[h_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[h_idx] = np.sum(w * np.array(S_CR))
            h_idx = (h_idx + 1) % H
        
        ratio = elapsed() / (max_time * 0.45)
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        if new_ps < pop_size:
            si = np.argsort(fitness); population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
        
        if best < last_best - 1e-12: stagnation_counter = 0; last_best = best
        else: stagnation_counter += 1
        if stagnation_counter > 20 + dim:
            si = np.argsort(fitness); keep = max(2, pop_size // 4)
            for idx in si[keep:]:
                if elapsed() >= max_time * 0.45: break
                population[idx] = lower + np.random.random(dim) * ranges; fitness[idx] = evaluate(population[idx])
            stagnation_counter = 0; M_F[:] = 0.5; M_CR[:] = 0.5
    
    # Collect top solutions
    si = np.argsort(fitness)
    for i in range(min(5, pop_size)):
        top_solutions.append((fitness[si[i]], population[si[i]].copy()))
    if best_solution is not None:
        top_solutions.append((best, best_solution.copy()))
    top_solutions.sort(key=lambda x: x[0])
    
    # --- Phase 3: Simplified CMA-ES local search ---
    def cma_local(x0, sigma0, deadline):
        n = dim
        lam = max(4 + int(3 * np.log(n)), 8)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        counteval = 0
        while elapsed() < deadline:
            # Eigendecomposition
            if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            else:
                if counteval == 0:
                    D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.empty((lam, n))
            arf = np.empty(lam)
            for k in range(lam):
                if elapsed() >= deadline:
                    return
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
                arf[k] = evaluate(arx[k])
                counteval += 1
            
            # Sort
            arindex = np.argsort(arf)
            
            # Recombination
            old_mean = mean.copy()
            sel = arx[arindex[:mu]]
            mean = weights @ sel
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (sel - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if sigma < 1e-16 * np.max(ranges):
                break
    
    if best_solution is not None:
        # CMA-ES from best with medium sigma
        cma_local(best_solution, 0.1 * np.mean(ranges), max_time * 0.65)
        # CMA-ES from best with small sigma
        if best_solution is not None:
            cma_local(best_solution, 0.02 * np.mean(ranges), max_time * 0.78)
    
    # Try from other top solutions
    for i in range(min(2, len(top_solutions))):
        if elapsed() >= max_time * 0.82:
            break
        _, sol = top_solutions[i]
        if np.linalg.norm(sol - best_solution) > 0.01 * np.mean(ranges):
            cma_local(sol, 0.05 * np.mean(ranges), max_time * 0.82)
    
    # --- Phase 4: Nelder-Mead ---
    if best_solution is None: return best
    
    def nelder_mead(x0, scale, deadline):
        n = dim
        simplex = np.empty((n+1, n)); fs = np.empty(n+1)
        simplex[0] = x0.copy(); fs[0] = evaluate(x0)
        for i in range(n):
            p = x0.copy(); p[i] += scale * ranges[i]; p = np.clip(p, lower, upper)
            simplex[i+1] = p; fs[i+1] = evaluate(p)
        while elapsed() < deadline:
            order = np.argsort(fs); simplex = simplex[order]; fs = fs[order]
            centroid = simplex[:-1].mean(axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper); fr = evaluate(xr)
            if fr < fs[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper); fe = evaluate(xe)
                if fe < fr: simplex[-1]=xe; fs[-1]=fe
                else: simplex[-1]=xr; fs[-1]=fr
            elif fr < fs[-2]: simplex[-1]=xr; fs[-1]=fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1]-centroid), lower, upper); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1]=xc; fs[-1]=fc
                else:
                    for i in range(1,n+1): simplex[i]=np.clip(simplex[0]+0.5*(simplex[i]-simplex[0]),lower,upper); fs[i]=evaluate(simplex[i])
    
    nelder_mead(best_solution, 0.05, max_time * 0.88)
    nelder_mead(best_solution, 0.01, max_time * 0.93)
    
    # Phase 5: Fine coordinate search
    current = best_solution.copy(); current_f = best; step = ranges * 0.005
    while elapsed() < max_time * 0.98:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.98: return best
            for s in [1, -1]:
                t = current.copy(); t[d] = np.clip(t[d]+s*step[d], lower[d], upper[d]); ft = evaluate(t)
                if ft < current_f: current=t; current_f=ft; improved=True; break
        if not improved:
            step *= 0.5
            if np.max(step/(ranges+1e-30)) < 1e-15: break
    return best
