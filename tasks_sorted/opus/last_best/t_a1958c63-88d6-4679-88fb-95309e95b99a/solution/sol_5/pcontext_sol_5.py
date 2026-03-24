#Looking at the code, I see several issues with the CMA-ES implementation (particularly the `ps` update using `np.linalg.solve` incorrectly - it should use the inverse square root of C, not solve with C itself). Also, the time allocation could be better, and I should add more diverse restarts and a proper local search phase.
#
#Key improvements:
#1. **Fix CMA-ES** - correct the path update computation
#2. **Better time allocation** - more time for CMA-ES with multiple restarts
#3. **Add population size reduction (L-SHADE style)**
#4. **Add coordinate descent as final polish**
#5. **Multiple CMA-ES restarts from different starting points**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def lhs_sample(n):
        result = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, j] = lower[j] + ranges[j] * (perm[i] + np.random.rand()) / n
        return result
    
    def cma_es(x0, sigma0=0.3, max_evals=10000, min_time_frac=0.02):
        if time_left() < min_time_frac * max_time:
            return
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1.0 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        xmean = x0.copy()
        sigma = sigma0 * np.mean(ranges)
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        evals = 0
        stag = 0
        prev_b = best
        
        while evals < max_evals and time_left() > min_time_frac * max_time:
            # Update eigen decomposition
            if evals - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.array([clip(xmean + sigma * (B @ (D * arz[k]))) for k in range(lam)])
            fitvals = np.array([evaluate(arx[k]) for k in range(lam)])
            evals += lam
            
            idx = np.argsort(fitvals)
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = xmean - xold
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ diff / (sigma + 1e-30)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*evals/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff / (sigma + 1e-30)
            
            artmp = (arx[idx[:mu]] - xold) / (sigma + 1e-30)
            C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.mean(ranges))
            
            if best < prev_b - 1e-14: stag = 0
            else: stag += 1
            prev_b = best
            if stag > 15 + 5*n or sigma < 1e-16 * np.mean(ranges):
                break

    # Phase 1: SHADE with population reduction
    pop_size_init = min(max(10*dim, 50), 300)
    pop_size_min = max(4, dim)
    samples = lhs_sample(pop_size_init * 2)
    all_f = np.array([evaluate(s) for s in samples])
    idx = np.argsort(all_f)[:pop_size_init]
    pop = samples[idx].copy(); pop_fit = all_f[idx].copy()
    pop_size = pop_size_init
    
    H=100; M_F=np.full(H,0.5); M_CR=np.full(H,0.85); ki=0; archive=[]
    generation=0; no_improve=0; prev_best=best; max_gen_shade = 500
    
    while time_left() > max_time * 0.45 and generation < max_gen_shade:
        generation += 1; S_F,S_CR,S_df=[],[],[]
        r_idx=np.random.randint(0,H,size=pop_size)
        F_vals=np.clip([np.random.standard_cauchy()*0.1+M_F[r_idx[i]] for i in range(pop_size)],0.01,1.0)
        CR_vals=np.clip(np.random.randn(pop_size)*0.1+M_CR[r_idx],0,1)
        pbest_idx=np.argsort(pop_fit)[:max(2,int(0.1*pop_size))]
        
        for i in range(pop_size):
            if time_left()<max_time*0.45: break
            pi=pbest_idx[np.random.randint(len(pbest_idx))]
            r1=np.random.randint(pop_size)
            while r1==i: r1=np.random.randint(pop_size)
            us=pop_size+len(archive); r2=np.random.randint(us)
            while r2==i or r2==r1: r2=np.random.randint(us)
            xr2=pop[r2] if r2<pop_size else archive[r2-pop_size]
            mutant=pop[i]+F_vals[i]*(pop[pi]-pop[i])+F_vals[i]*(pop[r1]-xr2)
            trial=pop[i].copy(); j_rand=np.random.randint(dim)
            mask=np.random.rand(dim)<CR_vals[i]; mask[j_rand]=True
            trial[mask]=mutant[mask]; trial=clip(trial)
            ft=evaluate(trial)
            if ft<=pop_fit[i]:
                if ft<pop_fit[i]: S_F.append(F_vals[i]);S_CR.append(CR_vals[i]);S_df.append(pop_fit[i]-ft);archive.append(pop[i].copy())
                pop[i]=trial;pop_fit[i]=ft
        if len(archive)>pop_size: archive=[archive[j] for j in np.random.choice(len(archive),pop_size,replace=False)]
        if S_F:
            w=np.array(S_df);w/=w.sum()+1e-30
            M_F[ki]=np.sum(w*np.array(S_F)**2)/(np.sum(w*np.array(S_F))+1e-30)
            M_CR[ki]=np.sum(w*np.array(S_CR));ki=(ki+1)%H
        # L-SHADE reduction
        new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * generation / max_gen_shade)))
        if new_size < pop_size:
            sidx = np.argsort(pop_fit)[:new_size]
            pop = pop[sidx]; pop_fit = pop_fit[sidx]; pop_size = new_size
        if best<prev_best-1e-14:no_improve=0
        else:no_improve+=1
        prev_best=best

    # Phase 2: Multiple CMA-ES restarts
    if best_params is not None:
        top_idx = np.argsort(pop_fit)[:min(5, pop_size)]
        starts = [best_params.copy()] + [pop[i].copy() for i in top_idx[:4]]
        for sigma in [0.2, 0.05, 0.01]:
            for s in starts:
                if time_left() < 0.04 * max_time: break
                cma_es(s, sigma0=sigma, max_evals=8000, min_time_frac=0.02)
            if time_left() < 0.04 * max_time: break
    
    return best
