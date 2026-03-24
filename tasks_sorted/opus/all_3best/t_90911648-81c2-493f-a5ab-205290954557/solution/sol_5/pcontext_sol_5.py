#Looking at the results, the best algorithm (112.07) uses IPOP-CMA-ES. I'll analyze the key issues and make targeted improvements:
#
#1. **Too much time on initialization** - reduce and use smarter sampling
#2. **Local search is weak** - random directions are wasteful; use proper Powell/pattern search
#3. **Eigendecomposition every generation wastes time for large dim** - cache better
#4. **No DE for diversity** - add a DE phase for exploration
#5. **Restart strategy needs refinement** - better BIPOP budget tracking
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

    # Phase 1: Compact initialization with opposition
    n_init = min(max(10 * dim, 50), 300)
    archive = []
    
    for i in range(n_init):
        if remaining() <= 0:
            return best
        params = lower + ranges * np.random.rand(dim)
        fitness = eval_func(params)
        archive.append((fitness, params.copy()))
    
    archive.sort(key=lambda x: x[0])
    archive = archive[:100]

    # Phase 2: Pattern search (accelerated coordinate descent) on best
    def pattern_search(x0, max_evals):
        x = x0.copy()
        fx = eval_func(x)
        step = 0.1
        velocity = np.zeros(dim)
        evals_used = 0
        for iteration in range(20):
            improved_any = False
            perm = np.random.permutation(dim)
            for d in perm:
                if remaining() <= 0.5 or evals_used >= max_evals:
                    return x, fx
                # Try with momentum
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
        pattern_search(best_params.copy(), dim * 6)

    # Phase 3: DE phase for global diversity
    de_pop_size = min(max(8 * dim, 40), 150)
    de_pop = np.array([lower + ranges * np.random.rand(dim) for _ in range(de_pop_size)])
    de_fit = np.array([float('inf')] * de_pop_size)
    
    # Seed with archive
    for i in range(min(len(archive), de_pop_size)):
        de_pop[i] = archive[i][1].copy()
    
    for i in range(de_pop_size):
        if remaining() <= max_time * 0.4:
            break
        de_fit[i] = eval_func(de_pop[i])
    
    F, CR = 0.8, 0.9
    de_gens = 0
    while remaining() > max_time * 0.35:
        for i in range(de_pop_size):
            if remaining() <= max_time * 0.35:
                break
            idxs = np.random.choice(de_pop_size, 3, replace=False)
            while i in idxs:
                idxs = np.random.choice(de_pop_size, 3, replace=False)
            a, b, c = de_pop[idxs[0]], de_pop[idxs[1]], de_pop[idxs[2]]
            # current-to-best
            best_idx = np.argmin(de_fit)
            mutant = de_pop[i] + F * (de_pop[best_idx] - de_pop[i]) + F * (b - c)
            mask = np.random.rand(dim) < CR
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            ft = eval_func(trial)
            if ft <= de_fit[i]:
                de_pop[i] = trial
                de_fit[i] = ft
        de_gens += 1

    # Refine best after DE
    if remaining() > 2.0 and best_params is not None:
        pattern_search(best_params.copy(), dim * 4)

    # Phase 4: IPOP-CMA-ES
    base_pop = 4 + int(3 * np.log(dim))
    restart = 0
    
    while remaining() > 0.5:
        restart += 1
        pop_size = min(base_pop * (2 ** min(restart - 1, 5)), 256)
        mu = pop_size // 2
        
        if restart <= 1 and best_params is not None:
            mean = best_params.copy()
            sigma = 0.15 * np.mean(ranges)
        elif np.random.rand() < 0.4 and best_params is not None:
            mean = best_params + 0.1 * ranges * np.random.randn(dim)
            mean = np.clip(mean, lower, upper)
            sigma = 0.2 * np.mean(ranges)
        else:
            mean = lower + ranges * np.random.rand(dim)
            sigma = 0.35 * np.mean(ranges)
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        cs = (mu_eff+2)/(dim+mu_eff+5)
        ds = 1+2*max(0,np.sqrt((mu_eff-1)/(dim+1))-1)+cs
        cc = (4+mu_eff/dim)/(dim+4+2*mu_eff/dim)
        c1 = 2/((dim+1.3)**2+mu_eff)
        cmu = min(1-c1,2*(mu_eff-2+1/mu_eff)/((dim+2)**2+mu_eff))
        ps=np.zeros(dim);pc=np.zeros(dim)
        chi=np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))
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
                z=np.random.randn(dim)
                y=(B@(sq*z))if use_full else(np.sqrt(np.maximum(Cd,1e-20))*z)
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
