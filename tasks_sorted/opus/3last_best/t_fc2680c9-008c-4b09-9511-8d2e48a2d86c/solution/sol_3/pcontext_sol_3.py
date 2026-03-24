#Looking at the results, algorithm 1 (SHADE + CMA-ES) performed best at 8.08. I'll improve it by:
#
#1. **Fixing CMA-ES implementation** - the covariance update had issues with using sqrtC instead of proper inverse
#2. **Better time allocation** - more time for CMA-ES local search which seems effective
#3. **Population size reduction (L-SHADE style)** to focus resources
#4. **Multiple CMA-ES restarts with decreasing sigma**
#5. **Better boundary handling in mutations**
#
import numpy as np
from datetime import datetime

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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = [0]
    def ev(x):
        nonlocal best, best_x
        evals[0] += 1
        v = func(x)
        if v < best:
            best = v
            best_x = x.copy()
        return v

    # --- SHADE Differential Evolution ---
    pop_size_init = min(max(8 * dim, 40), 120)
    pop_size = pop_size_init
    H = 50
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    k = 0
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    pop = lower + pop * ranges
    pop = clip(pop)
    
    fit = np.array([ev(pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fit) < pop_size:
        fit = np.append(fit, [float('inf')] * (pop_size - len(fit)))
    
    archive = []
    gen = 0
    max_evals_de = float('inf')
    
    while elapsed() < max_time * 0.45:
        S_F, S_CR, S_delta = [], [], []
        gen += 1
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.45:
                break
            
            ri = np.random.randint(H)
            mu_F, mu_CR = memory_F[ri], memory_CR[ri]
            Fi = np.clip(mu_F + 0.1 * np.random.standard_cauchy(), 0.01, 1.5)
            CRi = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fit)[:p])
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            r2_pool = [j for j in combined if j != i and j != r1]
            r2 = np.random.choice(r2_pool)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Bounce-back boundary
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.random() * (pop[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.random() * (upper[d] - pop[i][d])
            mutant = clip(mutant)
            
            mask = np.random.random(dim) < CRi
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            trial_fit = ev(trial)
            
            if trial_fit <= fit[i]:
                delta = fit[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                if len(archive) < pop_size_init:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fit[i] = trial_fit
        
        if S_F:
            w = np.array(S_delta); w = w / (w.sum() + 1e-30)
            memory_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        # L-SHADE: reduce population
        new_size = max(4, int(pop_size_init - (pop_size_init - 4) * elapsed() / (max_time * 0.45)))
        if new_size < pop_size:
            idx_keep = np.argsort(fit)[:new_size]
            pop = pop[idx_keep]; fit = fit[idx_keep]; pop_size = new_size

    # --- CMA-ES local search from top solutions ---
    top_k = min(8, len(pop))
    top_indices = np.argsort(fit)[:top_k]
    candidates = [pop[ti].copy() for ti in top_indices]
    if best_x is not None and not any(np.allclose(best_x, c) for c in candidates):
        candidates.insert(0, best_x.copy())
    
    for ci, cand in enumerate(candidates):
        if elapsed() >= max_time * 0.97:
            break
        time_per = (max_time * 0.97 - elapsed()) / max(1, len(candidates) - ci)
        deadline = elapsed() + time_per
        
        sigma = 0.02 * np.max(ranges)
        mean = cand.copy()
        lam = max(4 + int(3 * np.log(dim)), 8)
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1)); w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2*max(0, np.sqrt((mu_eff-1)/(dim+1))-1) + cs
        cc = (4+mu_eff/dim)/(dim+4+2*mu_eff/dim)
        c1 = 2/((dim+1.3)**2+mu_eff)
        cmu_v = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((dim+2)**2+mu_eff))
        ps = np.zeros(dim); pc = np.zeros(dim)
        C = np.eye(dim)
        chiN = np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))
        
        for g in range(50000):
            if elapsed() >= deadline: break
            try: eig_vals, eig_vecs = np.linalg.eigh(C); eig_vals = np.maximum(eig_vals, 1e-20); sqrtC = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T; invsqrtC = eig_vecs @ np.diag(1.0/np.sqrt(eig_vals)) @ eig_vecs.T
            except: C = np.eye(dim); sqrtC = np.eye(dim); invsqrtC = np.eye(dim)
            zs = np.random.randn(lam, dim)
            ys = zs @ sqrtC.T
            xs = np.array([clip(mean + sigma * y) for y in ys])
            fs = np.array([ev(x) for x in xs])
            idx = np.argsort(fs)
            ymean = np.sum(w[:, None] * ys[idx[:mu_c]], axis=0)
            mean = clip(mean + sigma * ymean)
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ ymean)
            hn = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1))) < (1.4+2/(dim+1))*chiN)
            pc = (1-cc)*pc + hn*np.sqrt(cc*(2-cc)*mu_eff)*ymean
            C = (1-c1-cmu_v)*C + c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C)
            for j in range(mu_c): C += cmu_v*w[j]*np.outer(ys[idx[j]], ys[idx[j]])
            C = (C+C.T)/2 + 1e-12*np.eye(dim)
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-15, np.max(ranges))
            if sigma < 1e-12: break

    while elapsed() < max_time * 0.999:
        ev(clip(best_x + np.random.randn(dim) * ranges * 0.001))
    
    return best
