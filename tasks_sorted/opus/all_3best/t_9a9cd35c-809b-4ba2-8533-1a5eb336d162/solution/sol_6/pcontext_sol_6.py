#Looking at the results, the best algorithm (score 12.19) uses L-SHADE + CMA-ES + local search. I'll analyze what can be improved:
#
#1. **Better L-SHADE tuning** - higher initial CR memory, better NFE estimation
#2. **More efficient DE** - avoid expensive list operations, use vectorized boundary handling
#3. **Better CMA-ES restart balance** - more focused local restarts since DE already provides global exploration
#4. **Multi-scale local search** with adaptive step sizes
#5. **Better time allocation** - give more time to DE which is the strongest phase
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
    
    def time_left():
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
    
    # --- Phase 1: LHS initialization ---
    n_init = min(max(20 * dim, 100), 500)
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fits = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fits.append((f, i))
    
    init_fits.sort()
    top_k = min(15, len(init_fits))
    start_points = [init_points[init_fits[i][1]].copy() for i in range(top_k)]
    
    if best_x is None:
        best_x = (lower + upper) / 2.0
    
    # --- Phase 2: L-SHADE ---
    pop_size_init = min(max(14 * dim, 70), 350)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    pop = np.zeros((pop_size, dim))
    pop_f = np.full(pop_size, float('inf'))
    
    for i in range(min(top_k, pop_size)):
        pop[i] = start_points[i]
        pop_f[i] = init_fits[i][0]
    
    for i in range(top_k, pop_size):
        pop[i] = lower + np.random.rand(dim) * ranges
        if time_left() <= 0:
            return best
        pop_f[i] = eval_func(pop[i])
    
    H = max(6, dim)
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k_idx = 0
    archive = []
    
    nfe = n_init + pop_size - top_k
    max_nfe_estimate = pop_size_init * 300
    
    de_time_frac = 0.55
    de_end_time = elapsed() + time_left() * de_time_frac
    
    gen = 0
    while elapsed() < de_end_time and time_left() > 1.0 and pop_size >= min_pop_size:
        S_F = []
        S_CR = []
        S_df = []
        
        sort_idx = np.argsort(pop_f)
        
        new_pop = pop.copy()
        new_pop_f = pop_f.copy()
        
        for i in range(pop_size):
            if time_left() <= 0.5:
                pop = new_pop
                pop_f = new_pop_f
                break
            
            ri = np.random.randint(H)
            mu_f = M_F[ri]
            mu_cr = M_CR[ri]
            
            F = -1
            attempts = 0
            while F <= 0 and attempts < 20:
                F = mu_f + 0.1 * np.random.standard_cauchy()
                attempts += 1
                if F >= 1.0:
                    F = 1.0
                    break
            if F <= 0:
                F = 0.1
            F = min(F, 1.0)
            
            CR = np.clip(mu_cr + 0.1 * np.random.randn(), 0.0, 1.0)
            if mu_cr < 0:
                CR = 0.0
            
            p_rate = max(0.05, 0.25 - 0.20 * nfe / max(1, max_nfe_estimate))
            p = max(2, int(p_rate * pop_size))
            pbest_idx = sort_idx[np.random.randint(0, p)]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i:
                r1 += 1
            
            pool_size = pop_size + len(archive)
            if pool_size > 2:
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
            else:
                r2 = r1
            
            if r2 < pop_size:
                r2_vec = pop[r2]
            else:
                r2_vec = archive[r2 - pop_size]
            
            mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - r2_vec)
            
            # Bounce-back boundary handling
            for dd in range(dim):
                if mutant[dd] < lower[dd]:
                    mutant[dd] = lower[dd] + np.random.rand() * (pop[i][dd] - lower[dd])
                elif mutant[dd] > upper[dd]:
                    mutant[dd] = upper[dd] - np.random.rand() * (upper[dd] - pop[i][dd])
            
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            f_trial = eval_func(trial)
            nfe += 1
            
            if f_trial <= pop_f[i]:
                if f_trial < pop_f[i]:
                    archive.append(pop[i].copy())
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(abs(pop_f[i] - f_trial))
                new_pop[i] = trial
                new_pop_f[i] = f_trial
        
        pop = new_pop
        pop_f = new_pop_f
        
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        if S_F and sum(S_df) > 0:
            w = np.array(S_df) / sum(S_df)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k_idx] = float(np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30))
            if np.max(scr) <= 0:
                M_CR[k_idx] = -1.0
            else:
                M_CR[k_idx] = float(np.sum(w * scr))
            k_idx = (k_idx + 1) % H
        
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * nfe / max_nfe_estimate)))
        if new_pop_size < pop_size:
            sidx = np.argsort(pop_f)
            pop = pop[sidx[:new_pop_size]]
            pop_f = pop_f[sidx[:new_pop_size]]
            pop_size = new_pop_size
        
        gen += 1
    
    # --- Phase 3: CMA-ES with restarts ---
    base_lam = max(4 + int(3 * np.log(dim)), 12)
    use_sep = dim > 40
    restart_count = 0
    
    def run_cmaes(x0, sigma0, lam):
        nonlocal best, best_x
        mean = x0.copy(); sigma = sigma0; n = dim
        mu_count = lam // 2
        weights = np.log(mu_count + 0.5) - np.log(np.arange(1, mu_count + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4+mueff/n)/(n+4+2*mueff/n)
        cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff)
        cmu_val = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        pc=np.zeros(n); ps=np.zeros(n)
        if use_sep: diagC=np.ones(n)
        else: B=np.eye(n);D=np.ones(n);C=np.eye(n);invsqrtC=np.eye(n);eigeneval=0
        counteval=0;g=0;best_local=float('inf');stag=0
        while time_left()>0.3:
            arx=np.zeros((lam,n));fit=np.zeros(lam)
            for kk in range(lam):
                if time_left()<=0.2: return
                z=np.random.randn(n)
                if use_sep: x=mean+sigma*np.sqrt(diagC)*z
                else: x=mean+sigma*(B@(D*z))
                arx[kk]=clip(x);fit[kk]=eval_func(arx[kk]);counteval+=1
            idx=np.argsort(fit);arx=arx[idx];fit=fit[idx]
            if fit[0]<best_local: best_local=fit[0];stag=0
            else: stag+=1
            old=mean.copy();mean=weights@arx[:mu_count];d_vec=mean-old
            if use_sep:
                inv=1/np.sqrt(diagC);ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(inv*d_vec)/sigma
                hsig=int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN<1.4+2/(n+1))
                pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*d_vec/sigma
                art=(arx[:mu_count]-old)/sigma
                diagC=(1-c1-cmu_val)*diagC+c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)+cmu_val*np.sum(weights[:,None]*art**2,axis=0)
                diagC=np.maximum(diagC,1e-20)
            else:
                ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@d_vec)/sigma
                hsig=int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN<1.4+2/(n+1))
                pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*d_vec/sigma
                art=(arx[:mu_count]-old)/sigma
                C=(1-c1-cmu_val)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)+cmu_val*(art.T@np.diag(weights)@art)
                if counteval-eigeneval>lam/(c1+cmu_val)/n/10:
                    eigeneval=counteval;C=np.triu(C)+np.triu(C,1).T
                    try: D2,B=np.linalg.eigh(C);D2=np.maximum(D2,1e-20);D=np.sqrt(D2);invsqrtC=B@np.diag(1/D)@B.T
                    except: return
            sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            sigma=np.clip(sigma,1e-20,np.max(ranges)*2);g+=1
            if sigma<1e-14 or stag>10+30*n/lam: return
            cond=sigma*np.max(np.sqrt(diagC)) if use_sep else sigma*np.max(D)
            if cond<1e-12*np.max(ranges): return
    
    de_sort=np.argsort(pop_f)
    all_starts=[pop[de_sort[i]].copy() for i in range(min(5,pop_size))]
    sp_idx=0
    while time_left()>0.8:
        if restart_count<3:
            if sp_idx<len(all_starts): x0=all_starts[sp_idx];sp_idx+=1;lam=base_lam;s0=np.mean(ranges)/6
            else: x0=best_x+np.random.randn(dim)*ranges*0.02;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/10
        elif restart_count%5==0:
            x0=lower+np.random.rand(dim)*ranges;lam=min(int(base_lam*2**(restart_count*0.25)),200);s0=np.mean(ranges)/3
        elif restart_count%5<=2:
            x0=best_x+np.random.randn(dim)*ranges*0.01;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/15
        else:
            x0=best_x+np.random.randn(dim)*ranges*0.05;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/8
        run_cmaes(x0,s0,lam);restart_count+=1
    
    # --- Phase 4: Multi-scale local search ---
    if time_left()>0.2 and best_x is not None:
        for scale in [0.01, 0.003, 0.001, 0.0003, 0.0001]:
            if time_left()<=0.1: break
            step=ranges*scale; improved=True
            while improved and time_left()>0.1:
                improved=False
                perm=np.random.permutation(dim)
                for d in perm:
                    if time_left()<=0.05: break
                    for sign in [1,-1]:
                        trial=best_x.copy();trial[d]+=sign*step[d];trial=clip(trial)
                        f=eval_func(trial)
                        if f<best: improved=True; break
    
    return best
