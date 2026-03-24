#Looking at the results, algorithm 1 (score 36.64) significantly outperforms the others. I'll build on its structure but make key improvements:
#
#1. **Better DE with SHADE-like adaptation** (self-adaptive F and CR)
#2. **Improved CMA-ES restart diversity** - truly alternate between small/large populations (BIPOP)
#3. **Better time allocation** - more time to DE which seems effective
#4. **Nelder-Mead local search** for final refinement
#5. **Opposition-based learning** in initialization
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: LHS + opposition-based
    n_init = min(max(20 * dim, 100), 500)
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fit = []
    for i in range(n_init):
        if remaining() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fit.append((f, i))
        # Opposition-based point
        if remaining() > 0 and i < n_init // 3:
            opp = lower + upper - init_points[i]
            eval_func(opp)
    
    init_fit.sort()
    top_k = min(20, n_init)
    top_points = [init_points[init_fit[i][1]].copy() for i in range(top_k)]

    # Phase 2: SHADE-style DE
    def run_de(time_fraction=0.4):
        nonlocal best, best_x
        deadline = elapsed() + max_time * time_fraction
        
        pop_size = max(min(8 * dim, 80), 25)
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(pop_size):
            if i < len(top_points):
                pop[i] = top_points[i].copy()
            else:
                pop[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
            fit[i] = eval_func(pop[i])
            if remaining() <= 0 or elapsed() >= deadline:
                return
        
        # SHADE memory
        H = 20
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        archive = []
        max_archive = pop_size
        
        while elapsed() < deadline and remaining() > 0:
            S_F, S_CR, S_df = [], [], []
            
            for i in range(pop_size):
                if elapsed() >= deadline or remaining() <= 0:
                    return
                
                ri = np.random.randint(H)
                Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.1, 1.0)
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
                
                # DE/current-to-pbest/1
                p = max(2, int(0.1 * pop_size))
                pbest_idx = np.argsort(fit)[:p]
                pb = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                a = np.random.choice(idxs)
                
                union = list(range(pop_size)) + list(range(len(archive)))
                b_idx = np.random.randint(pop_size + len(archive))
                xb = pop[b_idx] if b_idx < pop_size else archive[b_idx - pop_size]
                
                mutant = clip(pop[i] + Fi * (pb - pop[i]) + Fi * (pop[a] - xb))
                
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                f_trial = eval_func(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi); S_CR.append(CRi)
                        S_df.append(abs(fit[i] - f_trial))
                        if len(archive) < max_archive:
                            archive.append(pop[i].copy())
                        elif archive:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    pop[i] = trial; fit[i] = f_trial
            
            if S_F:
                w = np.array(S_df); w = w / (w.sum() + 1e-30)
                M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[k] = np.sum(w * np.array(S_CR))
                k = (k + 1) % H
    
    run_de(time_fraction=0.40)

    # Phase 3: CMA-ES with BIPOP restarts
    def run_cmaes(x0, initial_sigma, lam_mult=1):
        nonlocal best, best_x
        sigma = initial_sigma; mean = x0.copy(); n = dim
        lam = max(int((4 + int(3 * np.log(n))) * lam_mult), 6); mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n); cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN = n**0.5*(1-1/(4*n)+1/(21*n**2))
        pc=np.zeros(n); ps=np.zeros(n); B=np.eye(n); D=np.ones(n); C=np.eye(n)
        invsqrtC=np.eye(n); eigeneval=0; counteval=0; stag=0; best_gen=float('inf')
        for gen in range(1,10000):
            if remaining()<=0: return
            if counteval-eigeneval > lam/(c1+cmu+1e-30)/n/10:
                eigeneval=counteval; C=(C+C.T)/2
                try: Dv,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(Dv,1e-20)); invsqrtC=B@np.diag(1/D)@B.T
                except: return
            arx=np.zeros((lam,n)); af=np.zeros(lam)
            for k in range(lam):
                if remaining()<=0: return
                arx[k]=clip(mean+sigma*(B@(D*np.random.randn(n)))); af[k]=eval_func(arx[k]); counteval+=1
            idx=np.argsort(af); om=mean.copy(); mean=w@arx[idx[:mu]]; d=(mean-om)/sigma
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*invsqrtC@d
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*d
            at=(arx[idx[:mu]]-om)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(at.T@np.diag(w)@at)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=np.clip(sigma,1e-20,np.max(ranges)*2)
            if gen%10==0:
                if af[idx[0]]<best_gen-1e-10*(abs(best_gen)+1): best_gen=af[idx[0]]; stag=0
                else: stag+=1
                if stag>=4 or sigma<1e-16: return

    restart=0; lam_mult=1; small_budget_used=0; large_budget_used=0
    while remaining()>0.5:
        use_large = restart >= len(top_points) and large_budget_used <= small_budget_used
        if restart < len(top_points):
            x0=top_points[restart]; sig=np.mean(ranges)/4; lm=1
        elif use_large:
            x0=np.array([np.random.uniform(l,u) for l,u in bounds]); sig=np.mean(ranges)/2; lm=min(lam_mult*2,8); lam_mult=lm
        elif best_x is not None and np.random.rand()<0.5:
            x0=clip(best_x+np.random.randn(dim)*ranges*0.03); sig=np.mean(ranges)/10; lm=1
        else:
            x0=np.array([np.random.uniform(l,u) for l,u in bounds]); sig=np.mean(ranges)/4; lm=1
        t0=elapsed(); run_cmaes(x0,sig,lm); dt=elapsed()-t0
        if use_large: large_budget_used+=dt
        else: small_budget_used+=dt
        restart+=1

    return best
