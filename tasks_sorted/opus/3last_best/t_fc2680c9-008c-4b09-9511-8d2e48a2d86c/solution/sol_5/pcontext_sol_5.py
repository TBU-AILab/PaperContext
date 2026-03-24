#Looking at the results, algorithm 3 performed best (8.08). I'll analyze the issues and improve:
#
#1. **L-SHADE population reduction** for better DE convergence
#2. **Proper CMA-ES with eigendecomposition** (not Cholesky which has different semantics)
#3. **Better sigma initialization** scaled to search range
#4. **More aggressive local search** with adaptive restarts
#5. **Nelder-Mead refinement** at the end for fine-tuning
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
    
    def ev(x):
        nonlocal best, best_x
        v = func(x)
        if v < best:
            best = v
            best_x = x.copy()
        return v

    # --- L-SHADE ---
    pop_size_init = min(max(10 * dim, 50), 150)
    pop_size = pop_size_init
    H = 60
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    k = 0
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([ev(pop[i]) for i in range(pop_size)])
    archive = []
    de_time = max_time * 0.50
    
    while elapsed() < de_time:
        S_F, S_CR, S_delta = [], [], []
        
        sorted_idx = np.argsort(fit)
        for i in range(pop_size):
            if elapsed() >= de_time:
                break
            ri = np.random.randint(H)
            Fi = np.clip(memory_F[ri] + 0.1 * np.random.standard_cauchy(), 0.01, 1.5)
            CRi = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(p)]
            idxs = [j for j in range(pop_size) if j != i]
            r1 = idxs[np.random.randint(len(idxs))]
            pool_size = pop_size + len(archive)
            while True:
                r2 = np.random.randint(pool_size)
                if r2 != i and r2 != r1: break
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            for d2 in range(dim):
                if mutant[d2] < lower[d2]:
                    mutant[d2] = lower[d2] + np.random.random() * (pop[i][d2] - lower[d2])
                elif mutant[d2] > upper[d2]:
                    mutant[d2] = upper[d2] - np.random.random() * (upper[d2] - pop[i][d2])
            mutant = clip(mutant)
            mask = np.random.random(dim) < CRi
            if not np.any(mask): mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            tf = ev(trial)
            if tf <= fit[i]:
                delta = fit[i] - tf
                if delta > 0: S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                if len(archive) < pop_size_init: archive.append(pop[i].copy())
                elif archive: archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fit[i] = tf
        
        if S_F:
            w = np.array(S_delta); w /= w.sum() + 1e-30
            memory_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        new_size = max(4, int(round(pop_size_init - (pop_size_init - 4) * elapsed() / de_time)))
        if new_size < pop_size:
            idx_keep = np.argsort(fit)[:new_size]
            pop = pop[idx_keep]; fit = fit[idx_keep]; pop_size = new_size

    # --- CMA-ES from top solutions ---
    candidates = [pop[i].copy() for i in np.argsort(fit)[:min(5, pop_size)]]
    if best_x is not None: candidates.insert(0, best_x.copy())
    
    for ci, cand in enumerate(candidates):
        if elapsed() >= max_time * 0.96: break
        time_left = max_time * 0.96 - elapsed()
        deadline = elapsed() + time_left / max(1, len(candidates) - ci)
        n = dim; sigma = 0.02 * np.max(ranges); m = cand.copy()
        lam = max(4+int(3*np.log(n)),8); mu_c = lam//2
        w = np.log(mu_c+0.5)-np.log(np.arange(1,mu_c+1)); w/=w.sum()
        mu_eff=1/np.sum(w**2); cs=(mu_eff+2)/(n+mu_eff+5); ds=1+2*max(0,np.sqrt((mu_eff-1)/(n+1))-1)+cs
        cc=(4+mu_eff/n)/(n+4+2*mu_eff/n); c1=2/((n+1.3)**2+mu_eff); cmu_v=min(1-c1,2*(mu_eff-2+1/mu_eff)/((n+2)**2+mu_eff))
        ps=np.zeros(n); pc=np.zeros(n); D=np.ones(n); B=np.eye(n); chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        eigen_update=0; C=np.eye(n)
        for g in range(50000):
            if elapsed()>=deadline: break
            if g % max(1, int(1/(c1+cmu_v)/n/10)) == 0:
                C=(C+C.T)/2; D2,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(D2,1e-20)); eigen_update=g
            BD=B*D; zs=np.random.randn(lam,n); ys=(BD@zs.T).T
            xs=np.array([clip(m+sigma*y) for y in ys]); fs=np.array([ev(x) for x in xs])
            idx=np.argsort(fs); yw=np.sum(w[:,None]*ys[idx[:mu_c]],axis=0)
            m=clip(m+sigma*yw); zw=np.sum(w[:,None]*zs[idx[:mu_c]],axis=0)
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*zw
            hn=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))<(1.4+2/(n+1))*chiN)
            pc=(1-cc)*pc+hn*np.sqrt(cc*(2-cc)*mu_eff)*yw
            C=(1-c1-cmu_v)*C+c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C)
            for j in range(mu_c): C+=cmu_v*w[j]*np.outer(ys[idx[j]],ys[idx[j]])
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=np.clip(sigma,1e-15,np.max(ranges))
            if sigma<1e-13: break

    while elapsed()<max_time*0.999:
        ev(clip(best_x+np.random.randn(dim)*ranges*0.0003))
    return best
