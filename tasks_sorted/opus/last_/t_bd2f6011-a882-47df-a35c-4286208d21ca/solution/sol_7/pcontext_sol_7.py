#
#
#Looking at the previous algorithm, I see it scores 34.97. Let me analyze the weaknesses:
#
#1. The DE phase is limited to 60 individuals and may not explore well enough
#2. Nelder-Mead is slow in high dimensions (O(dim) evals per iteration)
#3. Need better local search (e.g., pattern search or CMA-ES with small population)
#4. Budget allocation could be improved
#
#Let me implement a more sophisticated approach with CMA-ES restarts (using sep-CMA for high dimensions), better DE, and smarter time management.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    center = (lower + upper) / 2.0

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

    # --- Sep-CMA-ES implementation (diagonal covariance) ---
    def sep_cma_es(x0, sigma0, max_evals=None, time_budget=None):
        nonlocal best, best_params
        t0 = datetime.now()
        n = len(x0)
        
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters for sep-CMA
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy().astype(float)
        sigma = sigma0
        diagD = np.ones(n) * sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        evals_done = 0
        gen = 0
        stagnation = 0
        prev_best_local = float('inf')
        
        while True:
            if time_budget and (datetime.now() - t0).total_seconds() > time_budget:
                break
            if time_left() < 0.3:
                break
            if max_evals and evals_done >= max_evals:
                break
            
            # Sample
            solutions = []
            fitnesses = []
            for k in range(lam):
                if time_left() < 0.2:
                    return
                z = np.random.randn(n)
                x = mean + diagD * z
                x = clip(x)
                f = evaluate(x)
                solutions.append((x, z))
                fitnesses.append(f)
                evals_done += 1
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            
            # Check stagnation
            local_best = fitnesses[order[0]]
            if local_best >= prev_best_local - 1e-12:
                stagnation += 1
            else:
                stagnation = 0
            prev_best_local = min(prev_best_local, local_best)
            
            if stagnation > 10 + 30 * n / lam:
                break
            
            # Update mean
            old_mean = mean.copy()
            selected_x = np.array([solutions[order[i]][0] for i in range(mu)])
            selected_z = np.array([solutions[order[i]][1] for i in range(mu)])
            mean = np.dot(weights, selected_x)
            
            # Update evolution paths
            zmean = np.dot(weights, selected_z)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps)**2 / (1 - (1-cs)**(2*(gen+1))) / n) < 2 + 4.0/(n+1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / diagD
            
            # Update diagonal covariance
            diagC = diagD**2 / sigma**2
            diagC = (1 - c1 - cmu_val) * diagC + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*diagC)
            for i in range(mu):
                diagC += cmu_val * weights[i] * selected_z[i]**2
            
            diagD = sigma * np.sqrt(np.maximum(diagC, 1e-20))
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            
            # Break if sigma too small
            if sigma < 1e-12 * np.max(ranges):
                break

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    n_init = min(300, max(50, dim * 10))
    init_points = []
    init_fits = []
    
    for i in range(n_init):
        if time_left() < 0.5:
            return best
        x = lower + np.random.uniform(0, 1, dim) * ranges
        f = evaluate(x)
        init_points.append(x)
        init_fits.append(f)
    
    init_points = np.array(init_points)
    init_fits = np.array(init_fits)

    # --- Phase 2: DE/current-to-pbest/1 ---
    pop_size = min(80, max(20, dim * 3))
    idx_sorted = np.argsort(init_fits)[:pop_size]
    pop = init_points[idx_sorted].copy()
    pop_f = init_fits[idx_sorted].copy()
    
    archive = []
    F_vals = np.full(pop_size, 0.5)
    CR_vals = np.full(pop_size, 0.5)
    MF = [0.5]
    MCR = [0.5]
    memory_size = 5
    k = 0

    de_budget = 0.35  # fraction of remaining time
    de_deadline = time_left() * de_budget
    de_start = datetime.now()
    
    while True:
        if (datetime.now() - de_start).total_seconds() > de_deadline or time_left() < 1.0:
            break
        
        S_F = []
        S_CR = []
        S_df = []
        
        for i in range(pop_size):
            if time_left() < 0.8:
                break
            
            # SHADE-like parameter adaptation
            ri = np.random.randint(len(MF))
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + MF[ri], 0.01, 1.5)
            CRi = np.clip(np.random.randn() * 0.1 + MCR[ri], 0.0, 1.0)
            
            # p-best
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(pop_f)[:p])
            
            # Mutation: current-to-pbest/1 with archive
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            r2_idx = np.random.choice(combined)
            if r2_idx < pop_size:
                xr2 = pop[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Crossover
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            
            if f_trial < pop_f[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(pop_f[i] - f_trial)
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                pop_f[i] = f_trial
                pop[i] = trial.copy()
            elif f_trial == pop_f[i]:
                pop_f[i] = f_trial
                pop[i] = trial.copy()
        
        # Update memory
        if S_F:
            w = np.array(S_df)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            new_MF = np.sum(w * np.array(S_F)**2) / np.sum(w * np.array(S_F))
            new_MCR = np.sum(w * np.array(S_CR))
            if len(MF) < memory_size:
                MF.append(new_MF)
                MCR.append(new_MCR)
            else:
                MF[k % memory_size] = new_MF
                MCR[k % memory_size] = new_MCR
            k += 1

    # --- Phase 3: CMA-ES from best solutions, with restarts ---
    # Get top solutions for restarts
    top_idx = np.argsort(pop_f)[:5]
    
    restart_count = 0
    while time_left() > 0.5:
        if restart_count < len(top_idx):
            x0 = pop[top_idx[restart_count]].copy()
            sigma0 = 0.1 * np.max(ranges)
        elif restart_count < len(top_idx) + 3:
            # Restart from best with different scales
            scale = 0.01 * (3 ** (restart_count - len(top_idx)))
            x0 = best_params + np.random.randn(dim) * ranges * scale
            x0 = clip(x0)
            sigma0 = scale * np.max(ranges)
        else:
            # Random restart
            x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sigma0 = 0.3 * np.max(ranges)
        
        t_budget = min(time_left() - 0.3, max(time_left() * 0.3, 1.0))
        if t_budget < 0.3:
            break
        
        sep_cma_es(x0, sigma0, time_budget=t_budget)
        restart_count += 1

    # --- Phase 4: Final local refinement ---
    if best_params is not None and time_left() > 0.3:
        # Coordinate descent with shrinking step
        step = ranges * 0.001
        x = best_params.copy()
        fx = best
        
        while time_left() > 0.2:
            improved = False
            for i in range(dim):
                if time_left() < 0.1:
                    return best
                
                for direction in [1, -1]:
                    xn = x.copy()
                    xn[i] += direction * step[i]
                    xn = clip(xn)
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn
                        fx = fn
                        improved = True
                        break
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-10:
                    break

    return best
