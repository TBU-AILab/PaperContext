#Looking at the results, algorithm 3 performed best (8.47). Let me analyze what worked and improve:
#
#1. SHADE DE + CMA-ES with IPOP restarts was the winning combination
#2. Need better time allocation - more time for CMA-ES which seems most effective
#3. Better restart strategy for CMA-ES - more restarts from best, adaptive sigma
#4. Improve DE phase with better population management
#5. Add Nelder-Mead for fine-tuning
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
    n_init = min(800, max(100, dim * 15))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    init_samples = (perms + np.random.rand(n_init, dim)) / n_init
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.06:
            break
        x = lower + init_samples[i] * ranges
        eval_f(x)

    # Phase 2: SHADE-based DE
    pop_size_init = min(max(8 * dim, 50), 250)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 3)
    pop = lower + np.random.rand(pop_size, dim) * ranges
    fit = np.zeros(pop_size)
    for i in range(pop_size):
        if elapsed() >= max_time * 0.12:
            pop = pop[:i]; fit = fit[:i]; pop_size = i
            break
        fit[i] = eval_f(pop[i])
    
    if best_x is not None and pop_size > 0:
        worst_idx = np.argmax(fit[:pop_size])
        pop[worst_idx] = best_x.copy()
        fit[worst_idx] = best

    H = 6
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    mem_idx = 0
    archive = []
    nfe_de = 0
    max_nfe_de = pop_size_init * 500
    
    de_end_fraction = 0.38
    
    while remaining() > max_time * (1.0 - de_end_fraction) and pop_size >= min_pop_size:
        S_F, S_CR, S_df = [], [], []
        
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
            
            p = max(2, int(max(0.05, 0.2 - 0.15 * nfe_de / max_nfe_de) * pop_size))
            p_best_idx = np.argsort(fit[:pop_size])[:p]
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
            nfe_de += 1
            if f_trial <= fit[i]:
                df = fit[i] - f_trial
                if df > 0:
                    S_F.append(F_i); S_CR.append(CR_i); S_df.append(df)
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                pop[i] = trial.copy()
                fit[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            sf = np.array(S_F); scr = np.array(S_CR)
            memory_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            memory_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % H
        
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * nfe_de / max_nfe_de)))
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fit[:pop_size])
            pop = pop[sort_idx[:new_pop_size]].copy()
            fit = fit[sort_idx[:new_pop_size]].copy()
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))

    # Phase 3: CMA-ES with IPOP restarts
    def run_cmaes(x0, sigma0, lam, budget_time):
        nonlocal best, best_x
        cma_start = elapsed()
        n = dim
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n))
        cc = (4.0 + mu_eff/n) / (n + 4.0 + 2.0*mu_eff/n)
        c1 = 2.0 / ((n+1.3)**2 + mu_eff)
        cmu_v = min(1.0-c1, 2.0*(mu_eff-2.0+1.0/mu_eff)/((n+2.0)**2+mu_eff))
        
        mean = x0.copy(); sigma = sigma0
        ps = np.zeros(n); pc = np.zeros(n)
        use_full = n <= 100
        if use_full:
            C = np.eye(n); invsqrtC = np.eye(n); eigeneval = 0; B = np.eye(n); D = np.ones(n)
        else:
            diagC = np.ones(n)
        gen = 0; stag = 0; local_best = best
        
        while (elapsed()-cma_start) < budget_time and remaining() > 0.05:
            gen += 1; arz = np.random.randn(lam, n)
            if use_full:
                if gen == 1 or (gen-eigeneval) > lam/(c1+cmu_v)/n/10:
                    eigeneval = gen; C = np.triu(C)+np.triu(C,1).T
                    try:
                        Dv, B = np.linalg.eigh(C); Dv=np.maximum(Dv,1e-20); D=np.sqrt(Dv); invsqrtC=B@np.diag(1.0/D)@B.T
                    except: C=np.eye(n);D=np.ones(n);B=np.eye(n);invsqrtC=np.eye(n)
                arx = mean + sigma*(arz@(B*D).T)
            else:
                sqrtD=np.sqrt(diagC); arx=mean+sigma*arz*sqrtD
            arx=np.clip(arx,lower,upper)
            fvals=np.array([eval_f(arx[k]) for k in range(lam) if remaining()>0.03])
            if len(fvals)<lam: break
            idx=np.argsort(fvals)
            if fvals[idx[0]]<local_best-1e-14: local_best=fvals[idx[0]]; stag=0
            else: stag+=1
            old_mean=mean.copy(); mean=np.clip(weights@arx[idx[:mu]],lower,upper); diff=mean-old_mean
            if use_full: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*(invsqrtC@diff)/sigma
            else: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*diff/(sigma*sqrtD)
            hsig=np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/E_norm<1.4+2.0/(n+1)
            pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mu_eff)*diff/sigma
            if use_full:
                artmp=(arx[idx[:mu]]-old_mean)/sigma; C=(1-c1-cmu_v)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)+cmu_v*(artmp.T@np.diag(weights)@artmp)
            else:
                artmp=(arx[idx[:mu]]-old_mean)/sigma; diagC=(1-c1-cmu_v)*diagC+c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)+cmu_v*np.sum(weights[:,None]*artmp**2,0); diagC=np.maximum(diagC,1e-20)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/E_norm-1)); sigma=min(sigma,np.max(ranges)*2)
            if sigma<1e-14 or stag>20+40*n//lam: break

    for restart in range(50):
        if remaining()<0.12: break
        pm=min(2**restart, 20)
        ps_cma=max(int(4+3*np.log(dim))*pm,8); ps_cma=min(ps_cma,500)
        bt=remaining()*0.4
        if best_x is not None and (restart<2 or np.random.rand()<0.5):
            scale=0.05*(1.5**restart); x0=np.clip(best_x+scale*ranges*np.random.randn(dim),lower,upper)
            s0=0.15*min(1.5**restart,6)*np.mean(ranges)
        else:
            x0=lower+np.random.rand(dim)*ranges; s0=0.3*np.mean(ranges)
        run_cmaes(x0,min(s0,np.mean(ranges)),ps_cma,bt)

    # Phase 4: Pattern search
    if best_x is not None:
        step=0.005*ranges
        for _ in range(300):
            if remaining()<0.02: break
            improved=False
            perm=np.random.permutation(dim)
            for i in perm:
                if remaining()<0.01: break
                for d in [step[i],-step[i]]:
                    xt=best_x.copy(); xt[i]=np.clip(best_x[i]+d,lower[i],upper[i])
                    if eval_f(xt)<best: improved=True; break
            if not improved:
                step*=0.5
                if np.max(step/ranges)<1e-16: break
    
    return best
