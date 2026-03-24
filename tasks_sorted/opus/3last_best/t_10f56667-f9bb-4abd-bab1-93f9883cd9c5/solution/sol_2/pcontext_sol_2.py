#Looking at the best algorithm (score 14.88), I can identify several weaknesses:
#
#1. The CMA-ES local search doesn't properly update the covariance matrix - it's just doing random sampling with shrinking sigma
#2. The SHADE loop has slow Python-level per-individual operations
#3. Time allocation could be better - need more exploitation time
#4. No proper restart mechanism when stuck
#
#Let me implement a proper hybrid with L-SHADE + a real CMA-ES with covariance adaptation + multi-restart strategy:
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left(frac=1.0):
        return elapsed() < max_time * frac
    
    def ev(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # ---- Phase 1: L-SHADE ----
    N_init = min(max(40, 10 * dim), 400)
    N_min = max(4, dim)
    pop_size = N_init
    H = 100
    
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([ev(pop[i]) for i in range(pop_size) if time_left(0.95)])
    if len(fit) < pop_size:
        return best
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    gen = 0
    max_gen_est = max(1, int(max_time * 0.6 / (pop_size * 0.001 + 0.01)))
    
    while time_left(0.60):
        gen += 1
        S_F, S_CR, S_w = [], [], []
        sorted_idx = np.argsort(fit)
        p_max = max(2, int(0.2 * pop_size))
        
        trials = np.empty_like(pop)
        trial_from = np.zeros(pop_size, dtype=bool)
        
        for i in range(pop_size):
            if not time_left(0.60):
                break
            ri = np.random.randint(H)
            
            # Cauchy for F
            F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            pi = np.random.randint(1, p_max + 1)
            pbest = sorted_idx[np.random.randint(0, pi)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            union_size = pop_size + len(archive)
            r2_idx = np.random.randint(union_size - 2)
            pool = [j for j in range(pop_size) if j != i and j != r1]
            if r2_idx < len(pool):
                r2_vec = pop[pool[r2_idx]]
            else:
                r2_vec = archive[(r2_idx - len(pool)) % max(1, len(archive))] if archive else pop[pool[np.random.randint(len(pool))]]
            
            mutant = pop[i] + F_i * (pop[pbest] - pop[i]) + F_i * (pop[r1] - r2_vec)
            
            mask = np.random.random(dim) < CR_i
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            out_lo = trial < lower; out_hi = trial > upper
            trial[out_lo] = (lower[out_lo] + pop[i][out_lo]) / 2
            trial[out_hi] = (upper[out_hi] + pop[i][out_hi]) / 2
            
            f_trial = ev(trial)
            if f_trial <= fit[i]:
                d = fit[i] - f_trial
                if d > 0:
                    S_F.append(F_i); S_CR.append(CR_i); S_w.append(d)
                archive.append(pop[i].copy())
                if len(archive) > N_init:
                    archive.pop(np.random.randint(len(archive)))
                pop[i] = trial; fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        # LPSR
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * gen / max(1, max_gen_est))))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # ---- Phase 2: CMA-ES local search with restarts ----
    while time_left(0.98) and best_x is not None:
        sigma = 0.05 * np.mean(ranges)
        xmean = best_x.copy()
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n+1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        ds = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        pc = np.zeros(n); ps = np.zeros(n)
        B = np.eye(n); D = np.ones(n); C = np.eye(n)
        eigeneval = 0
        
        while time_left(0.98):
            arz = np.random.randn(lam, n)
            arx = xmean + sigma * (arz @ (B * D).T)
            arfit = np.array([ev(arx[j]) for j in range(lam) if time_left(0.98)])
            if len(arfit) < lam: break
            idx = np.argsort(arfit)
            xold = xmean.copy()
            xmean = np.dot(w, arx[idx[:mu_c]])
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * np.linalg.solve(B * D[None,:], (xmean - xold)/sigma)
            hsig = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lam))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma
            artmp = (arx[idx[:mu_c]] - xold) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(w) @ artmp)
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5*np.mean(ranges))
            eigeneval += lam
            if eigeneval >= lam*(n+2):
                C = np.triu(C) + np.triu(C,1).T
                D_sq, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_sq, 1e-20))
                eigeneval = 0
            if sigma < 1e-12: break
    
    return best
