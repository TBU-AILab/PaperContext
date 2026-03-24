#Looking at the results, the best algorithm (27.6) uses CMA-ES for low dims and DE for high dims. Key issues I see:
#
#1. **Time budget allocation** - too much time on initial sampling, not enough on optimization
#2. **DE for high dims** needs adaptive parameters (SHADE-style)
#3. **Local search** at the end is very effective
#4. **CMA-ES restarts** need better BIPOP strategy with both small and large populations
#5. **Missing mirrored sampling** and active CMA-ES improvements
#
#Let me create an improved hybrid:
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
    n_init = min(max(10 * dim, 50), 200)
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
    top_k = min(8, n_init)
    top_points = [init_points[sorted_idx[i]].copy() for i in range(top_k)]
    top_fits = [init_fits[sorted_idx[i]] for i in range(top_k)]

    # Local search: (1+1)-ES with 1/5 rule
    def local_search_es(x0, sigma0, time_budget):
        nonlocal best, best_x
        x = x0.copy()
        fx = evaluate(x)
        sigma = sigma0
        succ = 0
        total = 0
        ls_start = datetime.now()
        adapt_interval = max(10, 2 * dim)
        
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
            if total % adapt_interval == 0:
                rate = succ / total
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                succ = 0
                total = 0
                if sigma < 1e-15:
                    sigma = sigma0 * 0.01
        return x, fx

    # CMA-ES implementation
    def run_cmaes(x0, init_sigma, lam_factor=1, time_budget=None):
        nonlocal best, best_x
        if remaining() <= 0.3:
            return
        t_start = (datetime.now() - start).total_seconds()
        if time_budget is None:
            time_budget = remaining()
        
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        lam = max(int(base_lam * lam_factor), 6)
        mu = lam // 2
        
        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights_raw / np.sum(weights_raw)
        mueff = 1.0 / np.sum(weights**2)
        
        # Active CMA weights (negative weights for worst solutions)
        n_neg = lam - mu
        if n_neg > 0:
            neg_weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, n_neg + 1))
            neg_weights_raw = neg_weights_raw / np.sum(neg_weights_raw)
            mueff_neg = 1.0 / np.sum(neg_weights_raw**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_c = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Active CMA: scale for negative weights
        alpha_mu_neg = 1 + c1 / (cmu_c + 1e-30)
        alpha_mueff = 1 + 2 * mueff_neg / (mueff + 2) if n_neg > 0 else 1
        alpha_pos = (1 - c1 - cmu_c) / (n * cmu_c + 1e-30)
        alpha_min = min(alpha_mu_neg, alpha_mueff, alpha_pos) if n_neg > 0 else 1
        
        mean = x0.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = n <= 100
        
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
            
            artmp_pos = np.array([(arxs[idx[i]]-old_mean)/sigma for i in range(mu)])
            
            if use_full:
                C_new = (1-c1-cmu_c)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)
                for i in range(mu):
                    C_new += cmu_c*weights[i]*np.outer(artmp_pos[i],artmp_pos[i])
                # Active CMA: subtract contribution from worst
                if n_neg > 0 and alpha_min > 0:
                    for i in range(min(n_neg, mu)):
                        worst_idx = idx[lam - 1 - i]
                        y_neg = (arxs[worst_idx] - old_mean) / sigma
                        if use_full:
                            Cy = invsqrtC @ y_neg
                            norm_Cy_sq = np.dot(Cy, Cy)
                            if norm_Cy_sq > 1e-30:
                                scaled = y_neg / norm_Cy_sq
                            else:
                                scaled = y_neg
                        else:
                            scaled = y_neg
                        C_new -= cmu_c * alpha_min * neg_weights_raw[i] * np.outer(scaled, scaled) * n
                C = C_new
                eigen_countdown -= 1
            else:
                diagC = (1-c1-cmu_c)*diagC + c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)
                for i in range(mu):
                    diagC += cmu_c*weights[i]*artmp_pos[i]**2
                diagC = np.maximum(diagC, 1e-20)
            
            sigma *= np.exp((cs/damps)*(ps_norm/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2*np.max(ranges))
            gen += 1
            if sigma < 1e-14 or flat_count >= 12 or no_improve >= 20+10*n:
                return

    # SHADE-style DE for high dimensions
    def run_shade(time_budget):
        nonlocal best, best_x
        if remaining() <= 0.3:
            return
        
        de_start = datetime.now()
        n = dim
        NP = min(max(5 * n, 50), 200)
        H = 100  # memory size
        
        pop = np.zeros((NP, n))
        fit = np.zeros(NP)
        for i in range(NP):
            if remaining() <= 0.1:
                return
            pop[i] = lower + np.random.rand(n) * ranges
            fit[i] = evaluate(pop[i])
        
        # Insert best known points
        if best_x is not None:
            worst_idx = np.argmax(fit)
            pop[worst_idx] = best_x.copy()
            fit[worst_idx] = best
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        archive = []
        max_archive = NP
        
        while True:
            el = (datetime.now() - de_start).total_seconds()
            if el >= time_budget or remaining() <= 0.15:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            # Sort for current-to-pbest
            fit_sorted_idx = np.argsort(fit)
            
            for i in range(NP):
                if remaining() <= 0.1:
                    return
                
                r_idx = np.random.randint(0, H)
                mu_F = M_F[r_idx]
                mu_CR = M_CR[r_idx]
                
                # Generate F from Cauchy
                while True:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                    if Fi > 0:
                        break
                Fi = min(Fi, 1.0)
                
                # Generate CR from Normal
                CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
                
                # current-to-pbest/1
                p = max(2, int(0.1 * NP))
                p_best_idx = fit_sorted_idx[np.random.randint(0, p)]
                
                candidates = list(range(NP))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                # r2 from pop + archive
                all_pool_size = NP + len(archive)
                r2 = np.random.randint(0, all_pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, all_pool_size)
                
                if r2 < NP:
                    xr2 = pop[r2]
                else:
                    xr2 = archive[r2 - NP]
                
                mutant = pop[i] + Fi * (pop[p_best_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Binomial crossover
                cross_points = np.random.rand(n) < CRi
                j_rand = np.random.randint(0, n)
                cross_points[j_rand] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Bounce-back boundary handling
                for dd in range(n):
                    if trial[dd] < lower[dd]:
                        trial[dd] = (lower[dd] + pop[i][dd]) / 2
                    elif trial[dd] > upper[dd]:
                        trial[dd] = (upper[dd] + pop[i][dd]) / 2
                
                trial = clip(trial)
                f_trial = evaluate(trial
