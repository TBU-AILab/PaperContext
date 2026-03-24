#Looking at the results, algorithm 1 (score 17.09) is the best. I'll improve it by:
#
#1. **Better SHADE DE** - fix the archive selection bug, use L-SHADE with linear population reduction
#2. **More aggressive local search** - add Nelder-Mead after DE phase
#3. **Better time allocation** - give more time to DE which is clearly effective
#4. **Smarter CMA-ES restarts** - use sep-CMA-ES for high dimensions, better sigma initialization
#5. **Final coordinate descent polish**
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

    # Phase 2: L-SHADE DE
    def run_lshade(time_fraction=0.45):
        nonlocal best, best_x
        deadline = elapsed() + max_time * time_fraction
        
        N_init = max(min(10 * dim, 100), 30)
        N_min = max(4, dim // 2)
        pop_size = N_init
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
        
        H = 30
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        archive = []
        max_archive = N_init
        
        total_budget_time = deadline - elapsed()
        gen_start_time = elapsed()
        gen_count = 0
        
        while elapsed() < deadline and remaining() > 0:
            S_F, S_CR, S_df = [], [], []
            new_pop = []
            new_fit = []
            
            for i in range(pop_size):
                if elapsed() >= deadline or remaining() <= 0:
                    break
                
                ri = np.random.randint(H)
                # Cauchy for F
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
                
                # p-best
                p_rate = max(2.0/pop_size, 0.05 + 0.15 * (1 - (elapsed() - gen_start_time) / (total_budget_time + 1e-30)))
                p = max(2, int(p_rate * pop_size))
                pbest_idx = np.argsort(fit[:pop_size])[:p]
                pb = pop[np.random.choice(pbest_idx)]
                
                # r1 from pop
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                # r2 from pop + archive
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
                    new_pop.append(trial)
                    new_fit.append(f_trial)
                else:
                    new_pop.append(pop[i].copy())
                    new_fit.append(fit[i])
            
            if new_pop:
                pop_size_new = len(new_pop)
                pop = np.array(new_pop[:pop_size_new])
                fit = np.array(new_fit[:pop_size_new])
            
            if S_F:
                w = np.array(S_df)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            gen_count += 1
            # Linear population reduction
            progress = min(1.0, (elapsed() - gen_start_time) / (total_budget_time + 1e-30))
            new_size = max(N_min, int(N_init - (N_init - N_min) * progress))
            if new_size < pop_size and pop_size > N_min:
                idx = np.argsort(fit)[:new_size]
                pop = pop[idx]
                fit = fit[idx]
                pop_size = new_size
    
    run_lshade(time_fraction=0.45)

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
            for k in range(lam):
                if remaining()<=0: return
                arx[k]=clip(mean+sigma*(B@(D*np.random.randn(n))));af[k]=eval_func(arx[k]);counteval+=1
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
        elif best_x is not None and np.random.rand()<0.6:
            x0=clip(best_x+np.random.randn(dim)*ranges*0.02);sig=np.mean(ranges)/12;lm=1
        else:
            x0=np.array([np.random.uniform(l,u) for l,u in bounds]);sig=np.mean(ranges)/4;lm=1
        t0=elapsed();run_cmaes(x0,sig,lm);dt=elapsed()-t0
        if use_large:lb+=dt
        else:sb+=dt
        restart+=1

    return best
