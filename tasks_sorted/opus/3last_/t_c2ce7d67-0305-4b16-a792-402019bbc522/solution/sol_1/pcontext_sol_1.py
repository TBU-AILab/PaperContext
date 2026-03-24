#Looking at the code, I see a solid CMA-ES implementation with restarts. Let me improve it by:
#
#1. Better initial sampling (Sobol-like stratified)
#2. Fixing the CMA-ES restart budget allocation
#3. Adding differential evolution as a complementary strategy
#4. Better local search (coordinate descent with golden section)
#5. Tuning parameters more carefully
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Initial sampling with stratified random
    n_init = min(300, max(100, dim * 15))
    for i in range(n_init):
        if elapsed() >= max_time * 0.1:
            break
        x = lower + np.random.rand(dim) * ranges
        eval_f(x)

    # Phase 2: Differential Evolution + CMA-ES hybrid
    # DE phase
    pop_size = min(max(10 * dim, 40), 200)
    
    # Initialize population
    pop = lower + np.random.rand(pop_size, dim) * ranges
    fit = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.3:
            break
        fit[i] = eval_f(pop[i])
    
    # Insert best known
    if best_x is not None:
        worst_idx = np.argmax(fit)
        pop[worst_idx] = best_x.copy()
        fit[worst_idx] = best
    
    # DE iterations
    F = 0.8
    CR = 0.9
    de_gen = 0
    while remaining() > max_time * 0.4:
        de_gen += 1
        # Adaptive F and CR
        F_i = 0.5 + 0.3 * np.random.rand()
        CR_i = 0.8 + 0.2 * np.random.rand()
        
        for i in range(pop_size):
            if remaining() <= max_time * 0.4:
                break
            
            # current-to-best/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            best_idx = np.argmin(fit)
            
            mutant = pop[i] + F_i * (pop[best_idx] - pop[i]) + F_i * (pop[r1] - pop[r2])
            mutant = np.clip(mutant, lower, upper)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            
            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial

    # Phase 3: CMA-ES from best solution
    def run_cmaes(x0, sigma0, pop_sz, budget_time):
        nonlocal best, best_x
        cma_start = elapsed()
        n = dim
        lam = pop_sz
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu_p = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        
        mean = x0.copy()
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)
        use_full = n <= 80
        
        if use_full:
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
            D_vals = np.ones(n)
            B = np.eye(n)
            D = np.ones(n)
        else:
            diagC = np.ones(n)
        
        gen = 0
        stag = 0
        local_best = best
        
        while (elapsed() - cma_start) < budget_time and remaining() > 0.05:
            gen += 1
            arz = np.random.randn(lam, n)
            
            if use_full:
                if gen == 1 or (gen - eigeneval) > lam / (c1 + cmu_p) / n / 10:
                    eigeneval = gen
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_vals, B = np.linalg.eigh(C)
                        D_vals = np.maximum(D_vals, 1e-20)
                        D = np.sqrt(D_vals)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n); D_vals = np.ones(n)
                arx = mean + sigma * (arz @ (B * D).T)
            else:
                sqrtD = np.sqrt(diagC)
                arx = mean + sigma * arz * sqrtD
            
            arx = np.clip(arx, lower, upper)
            fvals = np.array([eval_f(arx[i]) for i in range(lam) if remaining() > 0.02])
            if len(fvals) < lam:
                break
            
            idx = np.argsort(fvals)
            if fvals[idx[0]] < local_best - 1e-13:
                local_best = fvals[idx[0]]; stag = 0
            else:
                stag += 1
            
            old_mean = mean.copy()
            mean = np.clip(weights @ arx[idx[:mu]], lower, upper)
            diff = mean - old_mean
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / (sigma * sqrtD)
            
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/E_norm < 1.4+2.0/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mu_eff)*diff/sigma
            
            if use_full:
                artmp = (arx[idx[:mu]] - old_mean)/sigma
                C = (1-c1-cmu_p)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu_p*(artmp.T@np.diag(weights)@artmp)
            else:
                artmp = (arx[idx[:mu]] - old_mean)/sigma
                diagC = (1-c1-cmu_p)*diagC + c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC) + cmu_p*np.sum(weights[:,None]*artmp**2,0)
                diagC = np.maximum(diagC, 1e-20)
            
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/E_norm - 1))
            sigma = min(sigma, np.max(ranges))
            if sigma < 1e-13 or stag > 15+30*n//lam:
                break
    
    for restart in range(20):
        if remaining() < 0.2:
            break
        ps = max(int(4+3*np.log(dim)) * min(2**restart, 10), 10)
        ps = min(ps, 300)
        bt = remaining() * 0.6
        if best_x is not None and (restart == 0 or np.random.rand() < 0.5):
            x0 = best_x + 0.1*(2**restart)*ranges*np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            s0 = 0.2*min(2**restart,5)*np.mean(ranges)
        else:
            x0 = lower + np.random.rand(dim)*ranges
            s0 = 0.3*np.mean(ranges)
        run_cmaes(x0, min(s0, np.mean(ranges)), ps, bt)
    
    # Local refinement
    if best_x is not None:
        step = 0.01 * ranges
        for _ in range(50):
            if remaining() < 0.02:
                break
            improved = False
            for i in range(dim):
                if remaining() < 0.01:
                    break
                for d in [step[i], -step[i]]:
                    xt = best_x.copy()
                    xt[i] = np.clip(best_x[i]+d, lower[i], upper[i])
                    if eval_f(xt) < best:
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-15:
                    break
    
    return best
