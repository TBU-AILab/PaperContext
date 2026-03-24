#Looking at the previous algorithm, it scores 211.81. Let me analyze potential improvements:
#
#1. The sep-CMA-ES has some bugs (sigma update interacts poorly with diagD)
#2. DE phase could use better population sizing and more generations
#3. Need more aggressive local search
#4. Better time allocation between phases
#
#Let me implement a cleaner, more robust approach:
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
    evals = 0

    def time_left():
        return max_time - (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Sobol-like stratified sampling ---
    n_init = min(500, max(100, dim * 15))
    for i in range(n_init):
        if time_left() < 0.5:
            return best
        x = lower + np.random.uniform(0, 1, dim) * ranges
        evaluate(x)

    # --- Phase 2: SHADE (Success-History based Adaptive DE) ---
    pop_size = min(100, max(30, dim * 4))
    
    # Initialize population from random + best found
    pop = np.array([lower + np.random.uniform(0, 1, dim) * ranges for _ in range(pop_size)])
    if best_params is not None:
        pop[0] = best_params.copy()
    pop_f = np.array([evaluate(p) for p in pop])
    
    memory_size = 6
    MF = np.full(memory_size, 0.5)
    MCR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    
    # Linear population size reduction parameters
    pop_size_init = pop_size
    pop_size_min = max(4, dim // 2)
    
    gen = 0
    de_time_frac = 0.50
    de_deadline = time_left() * de_time_frac
    de_start_time = datetime.now()
    
    while True:
        elapsed_de = (datetime.now() - de_start_time).total_seconds()
        if elapsed_de > de_deadline or time_left() < 1.5:
            break
        
        # Linear pop reduction
        ratio = min(1.0, elapsed_de / max(de_deadline, 1e-9))
        current_pop_size = max(pop_size_min, int(pop_size_init - (pop_size_init - pop_size_min) * ratio))
        if current_pop_size < len(pop_f):
            idx_keep = np.argsort(pop_f)[:current_pop_size]
            pop = pop[idx_keep]
            pop_f = pop_f[idx_keep]
        
        NP = len(pop_f)
        S_F, S_CR, S_df = [], [], []
        new_pop = pop.copy()
        new_pop_f = pop_f.copy()
        
        for i in range(NP):
            if time_left() < 1.0:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.randn() * 0.1 + MCR[ri], 0.0, 1.0)
            
            # p-best index
            p = max(2, int(0.05 * NP + 1))
            sorted_idx = np.argsort(pop_f)
            pbest = sorted_idx[np.random.randint(p)]
            
            # Select r1 != i
            candidates = [j for j in range(NP) if j != i]
            r1 = np.random.choice(candidates)
            
            # Select r2 from pop + archive, != i, r1
            pool_size = NP + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < NP else archive[r2 - NP]
            
            # current-to-pbest/1
            mutant = pop[i] + Fi * (pop[pbest] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Binomial crossover
            jrand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + pop[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + pop[i][d]) / 2
            
            f_trial = evaluate(trial)
            
            if f_trial < pop_f[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(pop_f[i] - f_trial)
                archive.append(pop[i].copy())
                new_pop[i] = trial
                new_pop_f[i] = f_trial
            elif f_trial == pop_f[i]:
                new_pop[i] = trial
                new_pop_f[i] = f_trial
        
        pop = new_pop
        pop_f = new_pop_f
        
        # Trim archive
        while len(archive) > NP:
            archive.pop(np.random.randint(len(archive)))
        
        # Update memory
        if S_F:
            w = np.array(S_df)
            ws = w.sum()
            if ws > 0:
                w = w / ws
            else:
                w = np.ones(len(w)) / len(w)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            MF[mem_idx] = np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30)
            MCR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        gen += 1

    # --- Phase 3: CMA-ES restarts from best solutions ---
    def run_cma(x0, sigma0, time_budget):
        nonlocal best, best_params
        t0 = datetime.now()
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy().astype(float)
        sigma = sigma0
        C_diag = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        stag = 0
        prev_best_gen = float('inf')
        
        while True:
            if (datetime.now() - t0).total_seconds() > time_budget or time_left() < 0.3:
                break
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = mean[None, :] + sigma * (arz * np.sqrt(C_diag)[None, :])
            for j in range(lam):
                arx[j] = clip(arx[j])
            
            fits = np.array([evaluate(arx[j]) for j in range(lam)])
            if time_left() < 0.3:
                break
            
            order = np.argsort(fits)
            
            gbf = fits[order[0]]
            if gbf >= prev_best_gen - 1e-14:
                stag += 1
            else:
                stag = 0
            prev_best_gen = min(prev_best_gen, gbf)
            if stag > 15 + 10 * n / lam:
                break
            
            # Recombination
            old_mean = mean.copy()
            sel_x = arx[order[:mu]]
            sel_z = arz[order[:mu]]
            mean = weights @ sel_x
            
            zmean = weights @ sel_z
            
            inv_sqrt_C = 1.0 / np.sqrt(np.maximum(C_diag, 1e-30))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C * (mean - old_mean) / sigma)
            
            hsig = np.linalg.norm(ps)**2 / (1 - (1-cs)**(2*(stag + 1 + 10))) / n < 2 + 4.0/(n+1)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Diagonal covariance update
            artmp = (sel_x - old_mean[None, :]) / sigma
            C_diag = ((1 - c1 - cmu_v) * C_diag 
                      + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag)
                      + cmu_v * (weights[:, None] * artmp**2).sum(axis=0))
            C_diag = np.maximum(C_diag, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            if sigma > 2 * np.max(ranges):
                break
            if sigma * np.sqrt(np.max(C_diag)) < 1e-14 * np.max(ranges):
                break

    # Collect top solutions for CMA restarts
    all_sorted = np.argsort(pop_f)
    top_starts = [pop[all_sorted[i]] for i in range(min(5, len(pop_f)))]
    
    restart = 0
    while time_left() > 0.8:
        if restart < len(top_starts):
            x0 = top_starts[restart]
            sigma0 = 0.05 * np.max(ranges)
        elif restart < len(top_starts) + 3:
            # Near-best restarts with increasing sigma
            scale = 0.005 * (5 ** (restart - len(top_starts)))
            x0 = clip(best_params + np.random.randn(dim) * ranges * scale)
            sigma0 = scale * np.max(ranges)
        else:
            # Random restarts
            x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sigma0 = 0.2 * np.max(ranges)
        
        tb = min(time_left() - 0.4, max(1.0, time_left() * 0.25))
        if tb < 0.4:
            break
        run_cma(x0, sigma0, tb)
        restart += 1

    # --- Phase 4: Final Nelder-Mead style simplex on best ---
    if best_params is not None and time_left() > 0.3 and dim <= 30:
        # Quick coordinate descent refinement
        x = best_params.copy()
        fx = best
        step = ranges * 0.0005
        
        for _ in range(5):
            if time_left() < 0.2:
                break
            improved = False
            for d in range(dim):
                if time_left() < 0.1:
                    return best
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] += sign * step[d]
                    xn = clip(xn)
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn
                        fx = fn
                        improved = True
                        break
            if not improved:
                step *= 0.3
                if np.max(step / ranges) < 1e-12:
                    break

    return best
