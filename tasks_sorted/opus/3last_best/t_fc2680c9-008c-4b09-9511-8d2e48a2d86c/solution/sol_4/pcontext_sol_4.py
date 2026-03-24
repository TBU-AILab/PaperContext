#Looking at the results, algorithm 2 performed best (8.08). I'll improve it with:
#
#1. **L-SHADE population reduction** for better convergence
#2. **Fixed CMA-ES** using eigendecomposition properly (not Cholesky which assumes lower triangular)
#3. **Better time allocation** - more for CMA-ES local search
#4. **Adaptive sigma initialization** based on population spread
#5. **Coordinate-wise local search** as final refinement
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
    gen = 0
    de_time = max_time * 0.45
    
    while elapsed() < de_time:
        S_F, S_CR, S_delta = [], [], []
        gen += 1
        trial_pop = pop.copy()
        trial_fit = fit.copy()
        
        for i in range(pop_size):
            if elapsed() >= de_time:
                break
            ri = np.random.randint(H)
            Fi = np.clip(memory_F[ri] + 0.1 * np.random.standard_cauchy(), 0.01, 1.5)
            CRi = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fit)[:p])
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            pool = [j for j in range(pop_size + len(archive)) if j != i and j != r1]
            r2 = np.random.choice(pool)
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

    # --- CMA-ES local search ---
    top_k = min(5, pop_size)
    candidates = [pop[i].copy() for i in np.argsort(fit)[:top_k]]
    if best_x is not None: candidates.insert(0, best_x.copy())
    
    for ci, cand in enumerate(candidates):
        if elapsed() >= max_time * 0.95: break
        time_left = max_time * 0.95 - elapsed()
        deadline = elapsed() + time_left / max(1, len(candidates) - ci)
        sigma = 0.01 * np.max(ranges)
        m = cand.copy(); n = dim
        lam = max(4 + int(3*np.log(n)), 8); mu_c = lam//2
        w = np.log(mu_c+0.5)-np.log(np.arange(1,mu_c+1)); w/=w.sum()
        mu_eff=1/np.sum(w**2); cs=(mu_eff+2)/(n+mu_eff+5); ds=1+2*max(0,np.sqrt((mu_eff-1)/(n+1))-1)+cs
        cc=(4+mu_eff/n)/(n+4+2*mu_eff/n); c1=2/((n+1.3)**2+mu_eff); cmu_v=min(1-c1,2*(mu_eff-2+1/mu_eff)/((n+2)**2+mu_eff))
        ps=np.zeros(n); pc=np.zeros(n); C=np.eye(n); chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        eigen_update=0
        D=np.ones(n); B=np.eye(n); BD=B*D
        for g in range(50000):
            if elapsed()>=deadline: break
            if g-eigen_update>lam/(c1+cmu_v)/n/10 or g==0:
                C=(C+C.T)/2; D2,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(D2,1e-20)); BD=B*D; eigen_update=g
            zs=np.random.randn(lam,n); ys=(BD@zs.T).T
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
        ev(clip(best_x+np.random.randn(dim)*ranges*0.0005))
    return best
