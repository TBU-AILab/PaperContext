#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
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
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(20 * dim, 200)
    for i in range(n_init):
        if remaining() <= 0:
            return best
        x = lower + np.random.rand(dim) * ranges
        evaluate(x)
    
    if best_x is None:
        return best
    
    # Phase 2: CMA-ES inspired search
    # Simple (1+1)-ES with adaptive step size, then multi-restart
    
    def local_search(x0, sigma0, budget_fraction=0.3):
        nonlocal best, best_x
        x = x0.copy()
        fx = func(clip(x))
        if fx < best:
            best = fx
            best_x = clip(x).copy()
        
        sigma = sigma0
        success_count = 0
        total_count = 0
        
        time_limit = remaining() * budget_fraction
        ls_start = datetime.now()
        
        while True:
            elapsed = (datetime.now() - ls_start).total_seconds()
            if elapsed >= time_limit or remaining() <= 0.1:
                break
            
            # Mutate
            z = np.random.randn(dim)
            x_new = clip(x + sigma * z)
            fx_new = func(x_new)
            total_count += 1
            
            if fx_new < best:
                best = fx_new
                best_x = x_new.copy()
            
            if fx_new < fx:
                x = x_new
                fx = fx_new
                success_count += 1
            
            # 1/5 success rule
            if total_count % (10 * dim) == 0 and total_count > 0:
                rate = success_count / total_count
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma *= 0.7
                success_count = 0
                total_count = 0
                sigma = max(sigma, 1e-15)
        
        return x, fx
    
    # Phase 2a: Local search from best point
    sigma_init = 0.1 * np.mean(ranges)
    local_search(best_x.copy(), sigma_init, budget_fraction=0.3)
    
    # Phase 3: CMA-ES simplified
    def cmaes_simple():
        nonlocal best, best_x
        
        pop_size = 4 + int(3 * np.log(dim))
        mu = pop_size // 2
        
        mean = best_x.copy()
        sigma = 0.3 * np.mean(ranges)
        C = np.eye(dim)
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = 4.0 / (dim + 4)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        gen = 0
        while remaining() > 0.2:
            # Sample population
            try:
                sqrtC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sqrtC = np.eye(dim)
            
            solutions = []
            for i in range(pop_size):
                if remaining() <= 0.1:
                    return
                z = np.random.randn(dim)
                x = clip(mean + sigma * sqrtC.dot(z))
                fx = func(x)
                if fx < best:
                    best = fx
                    best_x = x.copy()
                solutions.append((fx, x))
            
            solutions.sort(key=lambda t: t[0])
            
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * solutions[i][1]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(sqrtC, (mean - old_mean) / sigma)
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2.0 / (dim + 1) else 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = np.zeros((mu, dim))
            for i in range(mu):
                artmp[i] = (solutions[i][1] - old_mean) / sigma
            
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                C += cmu_val * weights[i] * np.outer(artmp[i], artmp[i])
            
            # Symmetrize
            C = (C + C.T) / 2
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.mean(ranges))
            
            gen += 1
            
            if sigma < 1e-12:
                # Restart
                mean = lower + np.random.rand(dim) * ranges
                sigma = 0.3 * np.mean(ranges)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
    
    if dim <= 50:
        cmaes_simple()
    else:
        # For high dimensions, use differential evolution style
        pop_size = min(10 * dim, 100)
        pop = []
        fit = []
        
        # Initialize population
        for i in range(pop_size):
            if remaining() <= 0.1:
                return best
            x = lower + np.random.rand(dim) * ranges
            fx = evaluate(x)
            pop.append(x)
            fit.append(fx)
        
        pop = np.array(pop)
        fit = np.array(fit)
        
        F = 0.8
        CR = 0.9
        
        while remaining() > 0.2:
            for i in range(pop_size):
                if remaining() <= 0.1:
                    return best
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Use best as one vector sometimes
                if np.random.rand() < 0.3:
                    a = best_x.copy()
                
                mutant = clip(a + F * (b - c))
                
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial = clip(trial)
                
                f_trial = func(trial)
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # Final local refinement
    if remaining() > 0.1 and best_x is not None:
        local_search(best_x.copy(), 0.01 * np.mean(ranges), budget_fraction=0.9)
    
    return best
