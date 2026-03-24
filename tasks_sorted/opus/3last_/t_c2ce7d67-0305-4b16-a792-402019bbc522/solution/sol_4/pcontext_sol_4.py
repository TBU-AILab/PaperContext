#Looking at the results, algorithm 2 performed best (8.47). Let me analyze what worked and improve:
#
#1. L-SHADE DE was effective but algorithm 1 had bugs in initialization
#2. CMA-ES restarts need better budget allocation
#3. Need better balance between exploration and exploitation
#4. Local search at the end helps
#
#Key improvements:
#- Fix L-SHADE population initialization properly
#- Better time budget: more time for DE, smarter CMA-ES restarts
#- Add Nelder-Mead as additional local search
#- Better restart strategy for CMA-ES (start from diverse good points)
#
import numpy as np
from datetime import datetime, timedelta

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
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Latin Hypercube Sampling
    n_init = min(500, max(80, dim * 12))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    init_samples = (perms + np.random.rand(n_init, dim)) / n_init
    
    top_k_x = []
    top_k_f = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.07:
            break
        x = lower + init_samples[i] * ranges
        f = eval_f(x)
        top_k_x.append(x.copy())
        top_k_f.append(f)
    
    # Keep top solutions for diversity
    if top_k_f:
        sorted_idx = np.argsort(top_k_f)
        top_k_x = [top_k_x[j] for j in sorted_idx[:max(20, dim)]]
        top_k_f = [top_k_f[j] for j in sorted_idx[:max(20, dim)]]

    # Phase 2: L-SHADE
    pop_size_init = min(max(8 * dim, 50), 250)
    pop_size = pop_size_init
    pop = lower + np.random.rand(pop_size, dim) * ranges
    fit = np.array([eval_f(pop[i]) for i in range(pop_size)])
    
    # Inject top solutions from sampling
    n_inject = min(len(top_k_x), pop_size // 3)
    sort_fit = np.argsort(fit)[::-1]  # worst first
    for j in range(n_inject):
        idx_w = sort_fit[j]
        if top_k_f[j] < fit[idx_w]:
            pop[idx_w] = np.array(top_k_x[j])
            fit[idx_w] = top_k_f[j]
    
    if best_x is not None:
        worst_idx = np.argmax(fit)
        pop[worst_idx] = best_x.copy()
        fit[worst_idx] = best

    H = 6
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    mem_idx = 0
    archive = []
    nfe = 0
    max_nfe_de = pop_size_init * 600
    min_pop_size = max(4, dim // 2)
    
    de_end_fraction = 0.48
    
    while remaining() > max_time * (1.0 - de_end_fraction):
        S_F, S_CR, S_df = [], [], []
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(pop_size):
            if remaining() <= max_time * (1.0 - de_end_fraction):
                break
            
            ri = np.random.randint(H)
            while True:
                F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                if F_i > 0:
                    break
            F_i = min(F_i, 1.0)
            CR_i = np.clip(np.random.randn() * 0.1 + memory_CR[ri], 0.0, 1.0)
            
            p = max(2, int(max(0.05, 0.25 - 0.2 * nfe / max_nfe_de) * pop_size))
            p_best_idx = np.argsort(fit)[:p]
            pb = np.random.choice(p_best_idx)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            attempts = 0
            while (r2 == i or r2 == r1) and attempts < 25:
                r2 = np.random.randint(combined_size)
                attempts += 1
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + F_i * (pop[pb] - pop[i]) + F_i * (pop[r1] - xr2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + pop[i][d]) / 2.0
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + pop[i][d]) / 2.0
            
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            f_trial = eval_f(trial)
            nfe += 1
            if f_trial <= fit[i]:
                df = fit[i] - f_trial
                if df > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_df.append(df)
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop = new_pop
        fit = new_fit
        
        if S_F:
            w = np.array(S_df)
            w /= w.sum() + 1e-30
            sf = np.array(S_F)
            scr = np.array(S_CR)
            memory_F[mem_idx] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
            memory_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % H
        
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * nfe / max_nfe_de)))
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fit)
            pop = pop[sort_idx[:new_pop_size]]
            fit = fit[sort_idx[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))

    # Collect diverse elite set from DE
    elite_x = [pop[j].copy() for j in np.argsort(fit)[:min(5, pop_size)]]

    # Phase 3: CMA-ES with IPOP restarts
    def run_cmaes(x0, sigma0, lam, budget_time):
        nonlocal best, best_x
        cma_start = elapsed()
        n = dim
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n))
        cc = (4.0 + mu_eff/n) / (n + 4.0 + 2.0*mu_eff/n)
        c1 = 2.0 / ((n+1.3)**2 + mu_eff)
        cmu_v = min(1.0-c1, 2.0*(mu_eff-2.0+1.0/mu_eff)/((n+2.0)**2+mu_eff))
        mean = x0.copy(); sigma = sigma0
        ps = np.zeros(n); pc = np.zeros(n)
        uf = n <= 80
        if uf: C=np.eye(n);iC=np.eye(n);ee=0;Dv=np.ones(n);B=np.eye(n);DD=np.ones(n)
        else: dC=np.ones(n)
        g=0;st=0;lb=best
        while (elapsed()-cma_start)<budget_time and remaining()>0.05:
            g+=1; arz=np.random.randn(lam,n)
            if uf:
                if g==1 or (g-ee)>lam/(c1+cmu_v)/n/10:
                    ee=g;C=np.triu(C)+np.triu(C,1).T
                    try: Dv,B=np.linalg.eigh(C);Dv=np.maximum(Dv,1e-20);DD=np.sqrt(Dv);iC=B@np.diag(1.0/DD)@B.T
                    except: C=np.eye(n);DD=np.ones(n);B=np.eye(n);iC=np.eye(n);Dv=np.ones(n)
                arx=mean+sigma*(arz@(B*DD).T)
            else: sD=np.sqrt(dC);arx=mean+sigma*arz*sD
            arx=np.clip(arx,lower,upper)
            fv=np.array([eval_f(arx[k]) for k in range(lam) if remaining()>0.03])
            if len(fv)<lam: break
            ix=np.argsort(fv)
            if fv[ix[0]]<lb-1e-13: lb=fv[ix[0]];st=0
            else: st+=1
            om=mean.copy();mean=np.clip(weights@arx[ix[:mu]],lower,upper);df=mean-om
            if uf: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*(iC@df)/sigma
            else: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*df/(sigma*sD)
            hs=np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*g))/E_norm<1.4+2.0/(n+1)
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mu_eff)*df/sigma
            if uf: at=(arx[ix[:mu]]-om)/sigma;C=(1-c1-cmu_v)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu_v*(at.T@np.diag(weights)@at)
            else: at=(arx[ix[:mu]]-om)/sigma;dC=(1-c1-cmu_v)*dC+c1*(pc**2+(1-hs)*cc*(2-cc)*dC)+cmu_v*np.sum(weights[:,None]*at**2,0);dC=np.maximum(dC,1e-20)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/E_norm-1));sigma=min(sigma,np.max(ranges))
            if sigma<1e-13 or st>20+30*n//lam: break

    for restart in range(30):
        if remaining()<0.15: break
        pm=min(2**restart,12)
        ps_cma=max(int(4+3*np.log(dim))*pm,8);ps_cma=min(ps_cma,400)
        bt=remaining()*0.45
        if restart<len(elite_x):
            x0=elite_x[restart]+0.05*(1.5**restart)*ranges*np.random.randn(dim);x0=np.clip(x0,lower,upper);s0=0.1*min(1.5**restart,4)*np.mean(ranges)
        elif best_x is not None and np.random.rand()<0.5:
            x0=best_x+0.05*(1.5**restart)*ranges*np.random.randn(dim);x0=np.clip(x0,lower,upper);s0=0.15*min(1.5**restart,4)*np.mean(ranges)
        else:
            x0=lower+np.random.rand(dim)*ranges;s0=0.3*np.mean(ranges)
        run_cmaes(x0,min(s0,np.mean(ranges)),ps_cma,bt)

    # Phase 4: Nelder-Mead simplex from best
    if best_x is not None and remaining()>0.1:
        n=dim; alpha=1.0; gamma=2.0; rho=0.5; sig=0.5
        simplex=np.zeros((n+1,n)); sf=np.zeros(n+1)
        simplex[0]=best_x.copy(); sf[0]=best
        step_s=0.02*ranges
        for j in range(n):
            simplex[j+1]=best_x.copy(); simplex[j+1][j]+=step_s[j]
            simplex[j+1]=np.clip(simplex[j+1],lower,upper); sf[j+1]=eval_f(simplex[j+1])
        for _ in range(500*dim):
            if remaining()<0.03: break
            oi=np.argsort(sf); simplex=simplex[oi]; sf=sf[oi]
            cent=np.mean(simplex[:-1],axis=0)
            xr=np.clip(cent+alpha*(cent-simplex[-1]),lower,upper); fr=eval_f(xr)
            if fr
