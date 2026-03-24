#Looking at the current best algorithm, I can identify several areas for improvement:
#
#1. **Add CMA-ES as local search** - it's far superior to Nelder-Mead for continuous optimization
#2. **Better initialization** - use Latin Hypercube Sampling instead of random
#3. **More aggressive exploitation** - coordinate-wise golden section search for fine-tuning
#4. **Better restart strategy** in SHADE
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
    
    # LHS initialization
    def lhs_sample(n):
        result = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, j] = lower[j] + ranges[j] * (perm[i] + np.random.rand()) / n
        return result
    
    # CMA-ES local search
    def cma_es(x0, sigma0=0.2, max_evals=5000):
        if time_left() < 0.03 * max_time:
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
        C = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        evals = 0
        stag = 0
        prev_b = best
        
        while evals < max_evals and time_left() > 0.02 * max_time:
            if evals - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n)
            else:
                if 'D' not in dir():
                    D = np.ones(n); B = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.array([clip(xmean + sigma * (B @ (D * arz[k]))) for k in range(lam)])
            fitvals = np.array([evaluate(arx[k]) for k in range(lam)])
            evals += lam
            
            idx = np.argsort(fitvals)
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = (xmean - xold) / (sigma + 1e-30)
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * np.linalg.solve(B @ np.diag(D) @ B.T + 1e-20*np.eye(n), diff) if sigma > 1e-20 else ps * (1-cs)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*evals/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff
            
            artmp = (arx[idx[:mu]] - xold) / (sigma + 1e-30)
            C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.mean(ranges))
            
            if best < prev_b - 1e-14: stag = 0
            else: stag += 1
            prev_b = best
            if stag > 10 + 3*n or sigma < 1e-16 * np.mean(ranges):
                break
    
    # Phase 1: SHADE
    pop_size = min(max(8*dim, 40), 200)
    samples = lhs_sample(pop_size * 2)
    all_f = np.array([evaluate(s) for s in samples])
    idx = np.argsort(all_f)[:pop_size]
    pop = samples[idx].copy(); pop_fit = all_f[idx].copy()
    
    H=100; M_F=np.full(H,0.5); M_CR=np.full(H,0.8); ki=0; archive=[]
    generation=0; no_improve=0; prev_best=best
    
    while time_left() > max_time * 0.35:
        generation += 1; S_F,S_CR,S_df=[],[],[]
        r_idx=np.random.randint(0,H,size=pop_size)
        F_vals=np.array([min(max(0.1*np.random.standard_cauchy()+M_F[r_idx[i]],0.01),1.0) for i in range(pop_size)])
        CR_vals=np.clip(np.random.randn(pop_size)*0.1+M_CR[r_idx],0,1)
        pbest_idx=np.argsort(pop_fit)[:max(2,int(0.11*pop_size))]
        
        for i in range(pop_size):
            if time_left()<max_time*0.35: break
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
        if best<prev_best-1e-14:no_improve=0
        else:no_improve+=1
        prev_best=best
        if no_improve>15+dim:
            for i in range(pop_size):
                if pop_fit[i]>best+1e-10:
                    pop[i]=best_params+0.2*ranges*(np.random.rand(dim)-0.5);pop[i]=clip(pop[i]);pop_fit[i]=evaluate(pop[i])
            no_improve=0
    
    # Phase 2: CMA-ES refinement
    if best_params is not None:
        for sigma in [0.15, 0.05, 0.01, 0.002]:
            if time_left() < 0.03 * max_time: break
            cma_es(best_params.copy(), sigma0=sigma, max_evals=5000)
    
    return best
