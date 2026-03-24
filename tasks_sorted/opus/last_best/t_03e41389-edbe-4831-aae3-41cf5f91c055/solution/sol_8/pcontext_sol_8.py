#Looking at the previous implementations, I can see several areas for improvement:
#
#1. The CMA-ES has issues with the Cholesky-based approach and covariance update
#2. DE phase could be more effective with better strategies
#3. Need better restart strategies and time management
#
#Let me build a more robust optimizer combining SHADE-style DE with proper CMA-ES and Nelder-Mead refinement.
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: LHS sampling
    n_init = min(max(50, 20 * dim), 500)
    pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, i] = (perm + np.random.rand(n_init)) / n_init
    pop = lower + pop * ranges
    
    init_data = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.07:
            break
        f = evaluate(pop[i])
        init_data.append((f, pop[i].copy()))
    init_data.sort(key=lambda x: x[0])

    # Phase 2: SHADE-style DE
    NP = min(max(30, 7 * dim), 120)
    de_pop = np.array([x[1] for x in init_data[:min(NP, len(init_data))]])
    while len(de_pop) < NP:
        de_pop = np.vstack([de_pop, lower + np.random.rand(dim) * ranges])
    de_fit = np.array([init_data[i][0] if i < len(init_data) else evaluate(de_pop[i]) for i in range(NP)])
    
    archive = []
    H = 6
    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.5)
    k = 0
    
    de_deadline = max_time * 0.30
    while elapsed() < de_deadline:
        S_F, S_CR, S_df = [], [], []
        p_min = max(2, int(0.05 * NP))
        p_max = max(p_min, int(0.2 * NP))
        
        new_pop = de_pop.copy()
        new_fit = de_fit.copy()
        
        for i in range(NP):
            if elapsed() >= de_deadline:
                break
            ri = np.random.randint(H)
            Fi = np.clip(mem_F[ri] + 0.1 * np.random.standard_cauchy(), 0.05, 1.0)
            CRi = np.clip(mem_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = np.random.randint(p_min, p_max + 1)
            p_idx = np.argsort(de_fit)[:p]
            xpbest = de_pop[np.random.choice(p_idx)]
            
            candidates = [j for j in range(NP) if j != i]
            r1 = np.random.choice(candidates)
            all_pool = list(range(NP)) + list(range(NP, NP + len(archive)))
            all_pool = [j for j in all_pool if j != i and j != r1]
            r2 = np.random.choice(all_pool) if all_pool else r1
            xr2 = de_pop[r2] if r2 < NP else archive[r2 - NP]
            
            v = de_pop[i] + Fi * (xpbest - de_pop[i]) + Fi * (de_pop[r1] - xr2)
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, v, de_pop[i])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial < de_fit[i]:
                S_F.append(Fi); S_CR.append(CRi); S_df.append(de_fit[i] - f_trial)
                archive.append(de_pop[i].copy())
                if len(archive) > NP: archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial; new_fit[i] = f_trial
        
        de_pop, de_fit = new_pop, new_fit
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            mem_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            mem_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H

    # Phase 3: CMA-ES with restarts
    def run_cmaes(x0, sigma0, deadline, lam=None):
        n = dim
        lam = lam or (4 + int(3 * np.log(n)))
        lam = max(lam, 8)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cs = (mueff + 2)/(n + mueff + 5); cc = (4 + mueff/n)/(n + 4 + 2*mueff/n)
        c1 = 2/((n+1.3)**2 + mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs; chiN = n**0.5*(1-1/(4*n)+1/(21*n*n))
        m = x0.copy(); sig = sigma0; C = np.eye(n); ps = np.zeros(n); pc = np.zeros(n)
        B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n); eigcnt = 0; g = 0
        while elapsed() < min(deadline, max_time*0.95):
            g += 1
            z = np.random.randn(lam, n)
            y = z * D[None, :] @ B.T
            x = m[None, :] + sig * y
            x = np.clip(x, lower, upper)
            fits = np.array([evaluate(x[k]) for k in range(lam)])
            idx = np.argsort(fits); xw = w @ x[idx[:mu]]; yw = w @ y[idx[:mu]]
            old_m = m.copy(); m = xw
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ (m-old_m)/sig)
            hn = int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*g))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hn*np.sqrt(cc*(2-cc)*mueff)*(m-old_m)/sig
            artmp = (x[idx[:mu]]-old_m)/sig
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C) + cmu*(artmp.T@np.diag(w)@artmp)
            sig *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1))
            eigcnt += lam
            if eigcnt > lam/(c1+cmu+1e-30)/n/10:
                eigcnt = 0; C = np.triu(C)+np.triu(C,1).T
                try:
                    D2, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(D2,1e-20)); invsqrtC = B@np.diag(1/D)@B.T
                except: C=np.eye(n); B=np.eye(n); D=np.ones(n); invsqrtC=np.eye(n)
            if sig < 1e-16 or sig > 1e8: break

    cma_budget = max_time * 0.93 - elapsed()
    for r in range(8):
        rem = max_time * 0.93 - elapsed()
        if rem < 0.3: break
        bud = rem / max(1, 4 - min(r, 3))
        if r == 0: x0 = best_x.copy(); sig = 0.2*np.mean(ranges)
        elif r % 2 == 1: x0 = best_x + 0.03*ranges*(np.random.rand(dim)-0.5); x0 = np.clip(x0,lower,upper); sig = 0.05*np.mean(ranges)
        else: x0 = lower+np.random.rand(dim)*ranges; sig = 0.3*np.mean(ranges)
        run_cmaes(x0, sig, elapsed()+bud)

    # Phase 4: Fine local search
    if best_x is not None:
        sc = 0.002*ranges
        for _ in range(50):
            if elapsed() >= max_time*0.998: break
            imp = False
            for d in np.random.permutation(dim):
                if elapsed() >= max_time*0.998: break
                for dr in [1,-1]:
                    t = best_x.copy(); t[d] += dr*sc[d]
                    if evaluate(t) < best:
                        imp = True
                        for _ in range(8):
                            t2 = best_x.copy(); t2[d] += dr*sc[d]*2
                            if evaluate(t2) < best: sc[d]*=1.5
                            else: break
            if not imp: sc *= 0.4
            if np.max(sc) < 1e-16: break
    return best
