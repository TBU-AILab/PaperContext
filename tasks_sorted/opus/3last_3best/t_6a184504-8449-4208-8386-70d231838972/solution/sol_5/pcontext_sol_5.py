#Looking at the results, the best algorithm (27.6) uses a simple CMA-ES with local search and DE for high dimensions. The key issues I see:
#
#1. **Time budget management** - the budget allocation between phases could be better
#2. **Restart strategy** - need better BIPOP-style restarts mixing small and large populations
#3. **Local search refinement** - the (1+1)-ES local search is effective and should be kept
#4. **For high dimensions** - need better DE with adaptive parameters (SHADE-style)
#
#I'll create a more refined hybrid combining the best elements:
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

    # Phase 1: LHS initialization
    n_init = min(max(20 * dim, 100), 500)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, d] = lower[d] + (perm + np.random.rand(n_init)) / n_init * ranges[d]
    
    init_fits = []
    for i in range(n_init):
        if remaining() <= 0.5:
            return best
        f = evaluate(init_points[i])
        init_fits.append(f)
    
    sorted_idx = np.argsort(init_fits)
    top_k = min(15, n_init)
    top_points = [init_points[sorted_idx[i]].copy() for i in range(top_k)]
    top_fits = [init_fits[sorted_idx[i]] for i in range(top_k)]

    # Local search: (1+1)-ES with 1/5 rule and coordinate-wise perturbation
    def local_search_es(x0, sigma0, time_budget):
        nonlocal best, best_x
        x = x0.copy()
        fx = evaluate(x)
        sigma = sigma0
        succ = 0
        total = 0
        ls_start = datetime.now()
        
        while True:
            el = (datetime.now() - ls_start).total_seconds()
            if el >= time_budget or remaining() <= 0.1:
                break
            z = np.random.randn(dim)
            x_new = clip(x + sigma * z)
            fx_new = evaluate(x_new)
            total += 1
            if fx_new < fx:
                x = x_new
                fx = fx_new
                succ += 1
            if total % max(10, 5 * dim) == 0:
                rate = succ / total
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                succ = 0
                total = 0
                sigma = max(sigma, 1e-16)
                if sigma < 1e-14:
                    break
        return x, fx

    # CMA-ES implementation
    def run_cmaes(x0, init_sigma, lam_factor=1, time_budget=None):
        nonlocal best, best_x
        if remaining() <= 0.2:
            return
        t_start = (datetime.now() - start).total_seconds()
        if time_budget is None:
            time_budget = remaining()
        
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        lam = max(int(base_lam * lam_factor), 6)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_c = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = n <= 80
        
        if use_full:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0
        else:
            diagC = np.ones(n)
        
        gen = 0
        no_improve = 0
        best_local = float('inf')
        flat_count = 0
        prev_best_gen = float('inf')
        hist_fits = []
        
        while True:
            t_elapsed = (datetime.now() - start).total_seconds() - t_start
            if t_elapsed >= time_budget or remaining() <= 0.15:
                return
            
            if use_full and eigen_countdown <= 0:
                C = (C + C.T) / 2
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                    if D.max() / D.min() > 1e14:
                        return
                except:
                    return
                eigen_countdown = max(1, int(lam / (c1 + cmu_c + 1e-20) / n / 10))
            
            arxs = []
            fits = []
            for k in range(lam):
                if remaining() <= 0.1:
                    return
                z = np.random.randn(n)
                if use_full:
                    x = mean + sigma * (B @ (D * z))
                else:
                    x = mean + sigma * (np.sqrt(diagC) * z)
                f = evaluate(x)
                arxs.append(clip(x))
                fits.append(f)
            
            idx = np.argsort(fits)
            best_gen = fits[idx[0]]
            
            if best_gen < best_local - 1e-12 * (abs(best_local) + 1):
                best_local = best_gen
                no_improve = 0
            else:
                no_improve += 1
            
            if abs(best_gen - prev_best_gen) < 1e-12 * (abs(best_gen) + 1e-12):
                flat_count += 1
            else:
                flat_count = 0
            prev_best_gen = best_gen
            
            old_mean = mean.copy()
            mean = sum(weights[i] * arxs[idx[i]] for i in range(mu))
            diff = mean - old_mean
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ diff / sigma
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * diff / (np.sqrt(diagC)*sigma + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1-(1-cs)**(2*(gen+1))) / chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*diff/sigma
            
            artmp = np.array([(arxs[idx[i]]-old_mean)/sigma for i in range(mu)])
            
            if use_full:
                C = (1-c1-cmu_c)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)
                for i in range(mu):
                    C += cmu_c*weights[i]*np.outer(artmp[i],artmp[i])
                eigen_countdown -= 1
            else:
                diagC = (1-c1-cmu_c)*diagC + c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)
                for i in range(mu):
                    diagC += cmu_c*weights[i]*artmp[i]**2
                diagC = np.maximum(diagC, 1e-20)
            
            sigma *= np.exp((cs/damps)*(ps_norm/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2*np.max(ranges))
            gen += 1
            if sigma < 1e-14 or flat_count >= 12 or no_improve >= 20+10*n:
                return

    # Main loop: alternate CMA-ES restarts with local search
    pop_factor = 1
    restart = 0
    use_small = True
    
    while remaining() > 0.5:
        if restart < len(top_points):
            x0 = top_points[restart]
            sig0 = np.mean(ranges) / 3.0
            pf = 1
        else:
            if use_small and best_x is not None:
                x0 = best_x + np.random.randn(dim) * ranges * 0.05
                x0 = clip(x0)
                sig0 = np.mean(ranges) / 6.0
                pf = 1
            else:
                x0 = lower + np.random.rand(dim) * ranges
                sig0 = np.mean(ranges) / 2.5
                pf = pop_factor
            use_small = not use_small
        
        budget = min(remaining() * 0.4, remaining())
        run_cmaes(x0, sig0, lam_factor=pf, time_budget=budget)
        
        if remaining() > 0.5 and best_x is not None:
            local_search_es(best_x.copy(), 0.01 * np.mean(ranges), min(remaining()*0.15, 2.0))
        
        restart += 1
        if restart > len(top_points) + 2:
            pop_factor = min(pop_factor * 2, 10)
    
    return best
