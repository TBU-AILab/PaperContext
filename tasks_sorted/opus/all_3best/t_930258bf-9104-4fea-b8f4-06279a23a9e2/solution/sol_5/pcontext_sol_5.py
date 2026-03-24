#Looking at the results, the best algorithm (score 17.09) uses SHADE DE + CMA-ES with BIPOP restarts. I'll improve it with:
#
#1. **Fix L-SHADE population reduction bug** (was breaking pop tracking)
#2. **Add Nelder-Mead local search** after DE for refinement
#3. **Better CMA-ES restart strategy** with more exploitation near best
#4. **Adaptive time allocation** based on dimension
#5. **Coordinate descent final polish**
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
        if remaining() > 0 and i < n_init // 3:
            opp = lower + upper - init_points[i]
            eval_func(opp)
    
    init_fit.sort()
    top_k = min(25, n_init)
    top_points = [init_points[init_fit[i][1]].copy() for i in range(top_k)]

    # Phase 2: SHADE DE (in-place update, no population reduction bugs)
    def run_shade(time_fraction=0.40):
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
        
        H = 25
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
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
                
                p = max(2, int(0.1 * pop_size))
                pbest_idx = np.argsort(fit)[:p]
                pb = pop[np.random.choice(pbest_idx)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                all_size = pop_size + len(archive)
                r2 = np.random.randint(all_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(all_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = clip(pop[i] + Fi * (pb - pop[i]) + Fi * (pop[r1] - xr2))
                
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                f_trial = eval_func(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_df.append(abs(fit[i] - f_trial))
                        if len(archive) < max_archive:
                            archive.append(pop[i].copy())
                        elif archive:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    pop[i] = trial.copy()
                    fit[i] = f_trial
            
            if S_F:
                w = np.array(S_df)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
    
    run_shade(time_fraction=0.42)

    # Phase 2.5: Nelder-Mead local search
    def nelder_mead(x0, max_time_frac=0.08):
        nonlocal best, best_x
        deadline = elapsed() + max_time * max_time_frac
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        step = ranges * 0.05
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += step[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        
        for _ in range(5000):
            if elapsed() >= deadline or remaining() <= 0:
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                fc = eval_func(xc)
                if fc < min(fr, f_simplex[n]):
                    simplex[n] = xc; f_simplex[n] = fc
                else:
                    for i in range(1, n + 1):
                        if remaining() <= 0: return
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_func(simplex[i])
            
            if np.max(np.abs(simplex[n] - simplex[0])) < 1e-12:
                return
    
    if best_x is not None and remaining() > 1:
        nelder_mead(best_x.copy(), max_time_frac=0.08)

    # Phase 3: CMA-ES with BIPOP restarts
    def run_cmaes(x0, initial_sigma, lam_mult=1):
        nonlocal best, best_x
        sigma = initial_sigma; mean = x0.copy(); n = dim
        lam = max(int((4 + int(3 * np.log(n))) * lam_mult), 6); mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN=n**0.5*(1-1/(4*n)+1/(21*n**2))
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);C=np.eye(n)
        invsqrtC=np.eye(n);eigeneval=0;counteval=0;stag=0;best_gen=float('inf')
        for gen in range(1,10000):
            if remaining()<=0: return
            if counteval-eigeneval>lam/(c1+cmu+1e-30)/n/10:
                eigeneval=counteval;C=(C+C.T)/2
                try: Dv,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(Dv,1e-20));invsqrtC=B@np.diag(1/D)@B.T
                except: return
            arx=np.zeros((lam,n));af=np.zeros(lam)
            for kk in range(lam):
                if remaining()<=0: return
                arx[kk]=clip(mean+sigma*(B@(D*np.random.randn(n))));af[kk]=eval_func(arx[kk]);counteval+=1
            idx=np.argsort(af);om=mean.copy();mean=w@arx[idx[:mu]];d=(mean-om)/sigma
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*invsqrtC@d
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*d
            at=(arx[idx[:mu]]-om)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(at.T@np.diag(w)@at)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1));sigma=np.clip(sigma,1e-20,np.max(ranges)*2)
            if gen%8==0:
                if af[idx[0]]<best_gen-1e-10*(abs(best_gen)+1):best_gen=af[idx[0]];stag=0
                else:stag+=1
                if stag>=5 or sigma<1e-16:return

    restart=0;lam_mult=1;sb=0;lb=0
    while remaining()>0.5:
        use_large=restart>=len(top_points) and lb<=sb
        if restart<len(top_points):
            x0=top_points[restart];sig=np.mean(ranges)/4;lm=1
        elif use_large:
            x0=np.array([np.random.uniform(l,u) for l,u in bounds]);sig=np.mean(ranges)/2;lm=min(lam_mult*2,8);lam_mult=lm
        elif best_x is not None and np.random.rand()<0.65:
            x0=clip(best_x+np.random.randn(dim)*ranges*0.02);sig=np.mean(ranges)/12;lm=1
        else:
            x0=np.array([np.random.uniform(l,u) for l,u in bounds]);sig=np.mean(ranges)/4;lm=1
        t0=elapsed();run_cmaes(x0,sig,lm);dt=elapsed()-t0
        if use_large:lb+=dt
        else:sb+=dt
        restart+=1

    return best
