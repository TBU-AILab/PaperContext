#Looking at the two algorithms, #2 performed much better (17.6 vs 55.0). Let me analyze the key differences and improve upon #2:
#
#1. Better initial sampling with Latin Hypercube
#2. More aggressive DE phase before CMA-ES
#3. Improved CMA-ES restart strategy (IPOP-CMA-ES style)
#4. Better time budget allocation
#5. Enhanced local search with pattern search + coordinate descent
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
    
    evals = [0]
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        evals[0] += 1
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
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.08:
            break
        x = lower + init_samples[i] * ranges
        eval_f(x)

    # Phase 2: Differential Evolution
    pop_size = min(max(8 * dim, 40), 200)
    pop = lower + np.random.rand(pop_size, dim) * ranges
    fit = np.array([eval_f(pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.15])
    if len(fit) < pop_size:
        pop = pop[:len(fit)]
        pop_size = len(fit)
    
    if best_x is not None:
        worst_idx = np.argmax(fit)
        pop[worst_idx] = best_x.copy()
        fit[worst_idx] = best

    # DE/current-to-pbest/1 with archive (SHADE-like)
    memory_F = np.full(5, 0.5)
    memory_CR = np.full(5, 0.8)
    mem_idx = 0
    archive = []
    
    while remaining() > max_time * 0.55:
        S_F, S_CR, S_df = [], [], []
        
        for i in range(pop_size):
            if remaining() <= max_time * 0.55:
                break
            
            ri = np.random.randint(5)
            F_i = np.clip(np.random.standard_cauchy() * 0.1 + memory_F[ri], 0.1, 1.0)
            CR_i = np.clip(np.random.randn() * 0.1 + memory_CR[ri], 0.0, 1.0)
            
            # p-best
            p = max(2, int(0.1 * pop_size))
            p_best_idx = np.argsort(fit)[:p]
            pb = np.random.choice(p_best_idx)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [c for c in combined if c != i and c != r1]
            if len(combined) == 0:
                combined = [r for r in range(pop_size) if r != i]
            r2_idx = np.random.choice(combined)
            if r2_idx < pop_size:
                xr2 = pop[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = pop[i] + F_i * (pop[pb] - pop[i]) + F_i * (pop[r1] - xr2)
            mutant = np.clip(mutant, lower, upper)
            
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                df = fit[i] - f_trial
                if df > 0:
                    S_F.append(F_i); S_CR.append(CR_i); S_df.append(df)
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                pop[i] = trial
                fit[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            memory_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % 5

    # Phase 3: CMA-ES with IPOP restarts
    def run_cmaes(x0, sigma0, pop_size, budget_time):
        nonlocal best, best_x
        cma_start = elapsed()
        n = dim
        lam = pop_size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n))
        cc = (4.0 + mu_eff/n) / (n + 4.0 + 2.0*mu_eff/n)
        c1 = 2.0 / ((n+1.3)**2 + mu_eff)
        cmu_val = min(1.0-c1, 2.0*(mu_eff-2.0+1.0/mu_eff)/((n+2.0)**2+mu_eff))
        
        mean = x0.copy()
        sigma = sigma0
        ps = np.zeros(n); pc = np.zeros(n)
        use_full = n <= 80
        
        if use_full:
            C = np.eye(n); invsqrtC = np.eye(n)
            eigeneval = 0; D_vals = np.ones(n); B = np.eye(n); D = np.ones(n)
        else:
            diagC = np.ones(n)
        
        gen = 0; stag = 0; local_best = best
        
        while (elapsed()-cma_start) < budget_time and remaining() > 0.05:
            gen += 1
            arz = np.random.randn(lam, n)
            if use_full:
                if gen == 1 or (gen-eigeneval) > lam/(c1+cmu_val)/n/10:
                    eigeneval = gen; C = np.triu(C)+np.triu(C,1).T
                    try:
                        D_vals, B = np.linalg.eigh(C); D_vals = np.maximum(D_vals,1e-20); D = np.sqrt(D_vals); invsqrtC = B@np.diag(1.0/D)@B.T
                    except:
                        C=np.eye(n);D=np.ones(n);B=np.eye(n);invsqrtC=np.eye(n);D_vals=np.ones(n)
                arx = mean + sigma*(arz@(B*D).T)
            else:
                sqrtD = np.sqrt(diagC); arx = mean + sigma*arz*sqrtD
            arx = np.clip(arx, lower, upper)
            fvals = np.array([eval_f(arx[i]) for i in range(lam) if remaining()>0.02])
            if len(fvals)<lam: break
            idx = np.argsort(fvals)
            if fvals[idx[0]]<local_best-1e-13: local_best=fvals[idx[0]]; stag=0
            else: stag+=1
            old_mean = mean.copy(); mean = np.clip(weights@arx[idx[:mu]], lower, upper); diff = mean-old_mean
            if use_full: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*(invsqrtC@diff)/sigma
            else: ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*diff/(sigma*sqrtD)
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/E_norm < 1.4+2.0/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mu_eff)*diff/sigma
            if use_full:
                artmp=(arx[idx[:mu]]-old_mean)/sigma; C=(1-c1-cmu_val)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)+cmu_val*(artmp.T@np.diag(weights)@artmp)
            else:
                artmp=(arx[idx[:mu]]-old_mean)/sigma; diagC=(1-c1-cmu_val)*diagC+c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)+cmu_val*np.sum(weights[:,None]*artmp**2,0); diagC=np.maximum(diagC,1e-20)
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/E_norm-1)); sigma=min(sigma, np.max(ranges))
            if sigma<1e-13 or stag>15+30*n//lam: break

    for restart in range(30):
        if remaining()<0.15: break
        pm = min(2**restart, 16)
        ps_cma = max(int(4+3*np.log(dim))*pm, 10); ps_cma = min(ps_cma, 400)
        bt = remaining()*0.5
        if best_x is not None and (restart==0 or np.random.rand()<0.4):
            x0 = best_x + 0.1*(2**(restart*0.5))*ranges*np.random.randn(dim); x0 = np.clip(x0, lower, upper); s0 = 0.2*min(2**restart,5)*np.mean(ranges)
        else:
            x0 = lower+np.random.rand(dim)*ranges; s0 = 0.3*np.mean(ranges)
        run_cmaes(x0, min(s0, np.mean(ranges)), ps_cma, bt)

    # Phase 4: Pattern search local refinement
    if best_x is not None:
        step = 0.01*ranges
        for _ in range(100):
            if remaining()<0.02: break
            improved = False
            for i in range(dim):
                if remaining()<0.01: break
                for d in [step[i], -step[i]]:
                    xt = best_x.copy(); xt[i] = np.clip(best_x[i]+d, lower[i], upper[i])
                    if eval_f(xt) < best: improved=True; break
            if not improved:
                step *= 0.5
                if np.max(step/ranges)<1e-15: break
    
    return best
