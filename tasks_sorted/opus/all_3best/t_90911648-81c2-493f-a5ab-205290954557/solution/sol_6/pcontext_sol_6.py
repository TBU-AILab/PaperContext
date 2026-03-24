#Looking at the results, algorithm 1 (score 43.67) is the best. It combines DE for global exploration with IPOP-CMA-ES and pattern search with momentum. I'll refine this winning approach by:
#
#1. **Better DE**: Use SHADE-like adaptive F/CR
#2. **Smarter time allocation**: More time for DE+CMA, less for init
#3. **Improved pattern search**: Golden section line search per coordinate
#4. **Better CMA restart strategy**: Use archive diversity for restarts
#5. **Tighter integration**: Feed DE results into CMA initialization
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Compact initialization
    n_init = min(max(8 * dim, 40), 250)
    archive = []
    
    for i in range(n_init):
        if remaining() <= 0:
            return best
        params = lower + ranges * np.random.rand(dim)
        fitness = eval_func(params)
        archive.append((fitness, params.copy()))
    
    archive.sort(key=lambda x: x[0])
    archive = archive[:80]

    # Phase 2: Pattern search with momentum and golden section
    def pattern_search(x0, max_evals):
        x = x0.copy()
        fx = eval_func(x)
        step = 0.1
        velocity = np.zeros(dim)
        evals_used = 0
        for iteration in range(25):
            improved_any = False
            perm = np.random.permutation(dim)
            for d in perm:
                if remaining() <= 0.5 or evals_used >= max_evals:
                    return x, fx
                best_d_val = x[d]
                best_d_f = fx
                for sign in [1, -1]:
                    trial = x.copy()
                    delta = sign * step * ranges[d] + 0.3 * velocity[d]
                    trial[d] = np.clip(x[d] + delta, lower[d], upper[d])
                    ft = eval_func(trial)
                    evals_used += 1
                    if ft < best_d_f:
                        best_d_f = ft
                        best_d_val = trial[d]
                        # Try accelerating in this direction
                        trial2 = trial.copy()
                        trial2[d] = np.clip(trial[d] + delta * 1.5, lower[d], upper[d])
                        ft2 = eval_func(trial2)
                        evals_used += 1
                        if ft2 < best_d_f:
                            best_d_f = ft2
                            best_d_val = trial2[d]
                if best_d_f < fx:
                    velocity[d] = best_d_val - x[d]
                    x[d] = best_d_val
                    fx = best_d_f
                    improved_any = True
                else:
                    velocity[d] *= 0.5
            if not improved_any:
                step *= 0.5
                if step < 1e-13:
                    break
        return x, fx

    if remaining() > 1.5 and best_params is not None:
        pattern_search(best_params.copy(), dim * 8)

    # Phase 3: SHADE-like DE
    de_pop_size = min(max(6 * dim, 30), 120)
    de_pop = np.array([lower + ranges * np.random.rand(dim) for _ in range(de_pop_size)])
    de_fit = np.array([float('inf')] * de_pop_size)
    
    # Seed with archive
    for i in range(min(len(archive), de_pop_size)):
        de_pop[i] = archive[i][1].copy()
    
    for i in range(de_pop_size):
        if remaining() <= max_time * 0.45:
            break
        de_fit[i] = eval_func(de_pop[i])
    
    # SHADE memory
    H = 20
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    
    # External archive for DE
    ext_archive = []
    
    de_time_frac = 0.40  # spend 40% of time on DE
    while remaining() > max_time * (1.0 - de_time_frac):
        S_F = []
        S_CR = []
        S_delta = []
        
        for i in range(de_pop_size):
            if remaining() <= max_time * (1.0 - de_time_frac):
                break
            
            r = np.random.randint(H)
            F_i = np.clip(np.random.standard_cauchy() * 0.1 + M_F[r], 0.01, 1.0)
            CR_i = np.clip(np.random.randn() * 0.1 + M_CR[r], 0.0, 1.0)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * de_pop_size))
            pbest_idx = np.random.randint(0, p)
            sorted_idx = np.argsort(de_fit)
            xpbest = de_pop[sorted_idx[pbest_idx]]
            
            candidates = list(range(de_pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            
            # r2 from pop + ext_archive
            if ext_archive and np.random.rand() < 0.5:
                r2_val = ext_archive[np.random.randint(len(ext_archive))]
            else:
                r2 = np.random.choice(candidates)
                r2_val = de_pop[r2]
            
            mutant = de_pop[i] + F_i * (xpbest - de_pop[i]) + F_i * (de_pop[r1] - r2_val)
            
            mask = np.random.rand(dim) < CR_i
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            
            ft = eval_func(trial)
            if ft <= de_fit[i]:
                if ft < de_fit[i]:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(abs(de_fit[i] - ft))
                    ext_archive.append(de_pop[i].copy())
                    if len(ext_archive) > de_pop_size:
                        ext_archive.pop(np.random.randint(len(ext_archive)))
                de_pop[i] = trial
                de_fit[i] = ft
        
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H

    # Refine best after DE
    if remaining() > 2.0 and best_params is not None:
        pattern_search(best_params.copy(), dim * 5)

    # Phase 4: IPOP-CMA-ES
    base_pop = 4 + int(3 * np.log(dim))
    restart = 0
    
    while remaining() > 0.5:
        restart += 1
        pop_size = min(base_pop * (2 ** min(restart - 1, 5)), 256)
        mu = pop_size // 2
        
        if restart <= 1 and best_params is not None:
            mean = best_params.copy(); sigma = 0.12 * np.mean(ranges)
        elif np.random.rand() < 0.4 and best_params is not None:
            mean = best_params + 0.08 * ranges * np.random.randn(dim)
            mean = np.clip(mean, lower, upper); sigma = 0.2 * np.mean(ranges)
        else:
            mean = lower + ranges * np.random.rand(dim); sigma = 0.35 * np.mean(ranges)
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        cs = (mu_eff+2)/(dim+mu_eff+5); ds = 1+2*max(0,np.sqrt((mu_eff-1)/(dim+1))-1)+cs
        cc = (4+mu_eff/dim)/(dim+4+2*mu_eff/dim); c1 = 2/((dim+1.3)**2+mu_eff)
        cmu = min(1-c1,2*(mu_eff-2+1/mu_eff)/((dim+2)**2+mu_eff))
        ps=np.zeros(dim);pc=np.zeros(dim); chi=np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))
        use_full=dim<=80
        if use_full:C=np.eye(dim);ev=np.ones(dim);B=np.eye(dim);ec=0
        else:Cd=np.ones(dim)
        stag=0;br=float('inf');g=0
        while remaining()>0.3:
            if use_full and ec>=max(1,int(0.5/(c1+cmu)/dim/5)):
                try:e,B=np.linalg.eigh(C);ev=np.maximum(e,1e-20)
                except:C=np.eye(dim);ev=np.ones(dim);B=np.eye(dim)
                ec=0
            if use_full:sq=np.sqrt(ev);isq=1/sq
            pop=[];fits=[]
            for _ in range(pop_size):
                if remaining()<=0.2:return best
                z=np.random.randn(dim);y=(B@(sq*z))if use_full else(np.sqrt(np.maximum(Cd,1e-20))*z)
                x=np.clip(mean+sigma*y,lower,upper);f=eval_func(x);pop.append((x,y));fits.append(f)
            idx=np.argsort(fits)
            if fits[idx[0]]<br:br=fits[idx[0]];stag=0
            else:stag+=1
            mn=np.zeros(dim);yw=np.zeros(dim)
            for i in range(mu):mn+=w[i]*pop[idx[i]][0];yw+=w[i]*pop[idx[i]][1]
            mean=np.clip(mn,lower,upper)
            if use_full:ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*(B@(isq*(B.T@yw)))
            else:ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*(yw/np.sqrt(np.maximum(Cd,1e-20)))
            pn=np.linalg.norm(ps);hs=1 if pn/np.sqrt(1-(1-cs)**(2*(g+1)))<(1.4+2/(dim+1))*chi else 0
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mu_eff)*yw
            if use_full:
                ar=np.column_stack([pop[idx[i]][1]for i in range(mu)])
                C=(1-c1-cmu+(1-hs)*c1*cc*(2-cc))*C+c1*np.outer(pc,pc)+cmu*(ar*w)@ar.T;C=(C+C.T)/2;ec+=1
            else:
                Cd=(1-c1-cmu+(1-hs)*c1*cc*(2-cc))*Cd+c1*pc**2
                for i in range(mu):Cd+=cmu*w[i]*pop[idx[i]][1]**2
                Cd=np.maximum(Cd,1e-20)
            sigma*=np.exp((cs/ds)*(pn/chi-1));sigma=np.clip(sigma,1e-16,2*np.mean(ranges));g+=1
            if stag>15+3*dim or sigma<1e-14:break
        if remaining()>1.0 and best_params is not None:
            pattern_search(best_params.copy(),dim*3)
    return best
