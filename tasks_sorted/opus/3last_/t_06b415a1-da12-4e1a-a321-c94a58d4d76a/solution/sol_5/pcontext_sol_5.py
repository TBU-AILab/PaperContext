#Looking at the results, algorithm 2 (score 20.53) is clearly the best. I'll analyze what makes it work and improve:
#
#1. **Better time allocation** - algorithm 2 spends 42% on L-SHADE which seems good
#2. **More CMA-ES restarts** including random restarts near best
#3. **Avoid caching overhead** (tobytes() is expensive)
#4. **Better initial sigma for CMA-ES**
#5. **Nelder-Mead as additional local search**
#6. **Larger initial population with better exploration**
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
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
    
    # --- Phase 1: LHS Initialization + Opposition ---
    pop_size_init = min(max(18 * dim, 100), 400)
    pop_size = pop_size_init
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok(0.95):
            return best
        fitness[i] = evaluate(pop[i])
    
    opp_pop = lower + upper - pop
    opp_fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok(0.92):
            break
        opp_fitness[i] = evaluate(opp_pop[i])
    
    all_pop = np.vstack([pop, opp_pop])
    all_fit = np.concatenate([fitness, opp_fitness])
    order = np.argsort(all_fit)[:pop_size]
    pop = all_pop[order].copy()
    fitness = all_fit[order].copy()
    
    # --- Phase 2: L-SHADE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    nfe = 0
    max_nfe_shade = pop_size_init * 200
    
    while time_ok(0.38):
        S_F, S_CR, S_df = [], [], []
        sorted_idx = np.argsort(fitness)
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if not time_ok(0.38):
                break
            
            ri = np.random.randint(memory_size)
            F_i = -1
            while F_i <= 0:
                F_i = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F_i >= 1.0:
                    F_i = 1.0
                    break
            F_i = min(F_i, 1.0)
            
            if M_CR[ri] < 0:
                CR_i = 0.0
            else:
                CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(np.random.uniform(0.05, 0.2) * pop_size))
            pbest_idx = sorted_idx[np.random.randint(p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            pool = [x for x in pool if x != i and x != r1]
            if not pool:
                pool = [xx for xx in range(pop_size) if xx != i]
            r2_idx = np.random.choice(pool)
            xr2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CR_i)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = (lower[j] + pop[i][j]) / 2
                elif trial[j] > upper[j]:
                    trial[j] = (upper[j] + pop[i][j]) / 2
            
            f_trial = evaluate(trial)
            nfe += 1
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_df.append(abs(fitness[i] - f_trial))
                    archive.append(pop[i].copy())
                new_pop[i] = clip(trial)
                new_fitness[i] = f_trial
        
        pop = new_pop
        fitness = new_fitness
        
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % memory_size
        
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        new_pop_size = max(4, int(round(pop_size_init - (pop_size_init - 4) * nfe / max_nfe_shade)))
        if new_pop_size < pop_size:
            order = np.argsort(fitness)[:new_pop_size]
            pop = pop[order]
            fitness = fitness[order]
            pop_size = new_pop_size
    
    # --- Phase 3: CMA-ES restarts ---
    top_k = min(12, len(pop))
    top_indices = np.argsort(fitness)[:top_k]
    restart = 0
    while time_ok(0.88) and restart < top_k + 8:
        if restart < top_k:
            x0 = pop[top_indices[restart]].copy()
            sig0 = np.mean(ranges) * 0.12 / (1 + restart * 0.15)
        else:
            x0 = best_params + np.random.randn(dim) * ranges * 0.03
            x0 = clip(x0)
            sig0 = np.mean(ranges) * 0.05
        restart += 1
        n=dim;lam=4+int(3*np.log(n));mu=lam//2;w=np.log(mu+.5)-np.log(np.arange(1,mu+1));w/=w.sum();mueff=1./np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n);cs=(mueff+2)/(n+mueff+5);c1=2/((n+1.3)**2+mueff);cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff));ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs;chiN=n**.5*(1-1/(4*n)+1/(21*n**2))
        m=x0.copy();s=sig0;pc_=np.zeros(n);ps_=np.zeros(n);C=np.eye(n);ce=0;stag=0;bf=1e30
        for g in range(3000):
            if not time_ok(0.88):break
            try:C=(C+C.T)/2;Dv,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(Dv,1e-20))
            except:break
            ax=np.zeros((lam,n));af=np.zeros(lam)
            for j in range(lam):
                if not time_ok(0.88):break
                ax[j]=clip(m+s*(B@(D*np.random.randn(n))));af[j]=evaluate(ax[j]);ce+=1
            ix=np.argsort(af);om=m.copy();m=np.sum(w[:,None]*ax[ix[:mu]],axis=0);ym=(m-om)/s
            ps_=(1-cs)*ps_+np.sqrt(cs*(2-cs)*mueff)*(B@(1./D*(B.T@ym)));hs=float(np.linalg.norm(ps_)/np.sqrt(1-(1-cs)**(2*ce/lam))/chiN<1.4+2/(n+1))
            pc_=(1-cc)*pc_+hs*np.sqrt(cc*(2-cc)*mueff)*ym;at=(ax[ix[:mu]]-om)/s
            C=(1-c1-cmu)*C+c1*(np.outer(pc_,pc_)+(1-hs)*cc*(2-cc)*C)+cmu*(at.T@np.diag(w)@at)
            s*=np.exp((cs/ds)*(np.linalg.norm(ps_)/chiN-1));s=np.clip(s,1e-16,np.max(ranges))
            if af[ix[0]]<bf-1e-10:bf=af[ix[0]];stag=0
            else:stag+=1
            if stag>15+30*n//lam or s<1e-14:break
    
    # --- Phase 4: Nelder-Mead ---
    if best_params is not None and time_ok(0.92):
        n=dim;simplex=np.zeros((n+1,n));simplex[0]=best_params.copy()
        for i in range(n):
            simplex[i+1]=best_params.copy();simplex[i+1,i]+=ranges[i]*0.005
            simplex[i+1]=clip(simplex[i+1])
        fs=np.array([evaluate(simplex[i]) for i in range(n+1) if time_ok(0.96)])
        if len(fs)==n+1:
            for _ in range(2000):
                if not time_ok(0.96):break
                oi=np.argsort(fs);simplex=simplex[oi];fs=fs[oi]
                centroid=simplex[:-1].mean(axis=0)
                xr=clip(centroid+1.0*(centroid-simplex[-1]));fr=evaluate(xr)
                if fr<fs[0]:
                    xe=clip(centroid+2.0*(xr-centroid));fe=evaluate(xe)
                    if fe<fr:simplex[-1]=xe;fs[-1]=fe
                    else:simplex[-1]=xr;fs[-1]=fr
                elif fr<fs[-2]:simplex[-1]=xr;fs[-1]=fr
                else:
                    xc=clip(centroid+0.5*(simplex[-1]-centroid));fc=evaluate(xc)
                    if fc<fs[-1]:simplex[-1]=xc;fs[-1]=fc
                    else:
                        for i in range(1,n+1):
                            simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0]))
                            fs[i]=evaluate(simplex[i])
                if np.max(np.abs(simplex[-1]-simplex[0]))<1e-15:break
    
    # --- Phase 5: Coordinate descent ---
    if best_params is not None:
        x=best_params.copy();f_x=best;step=ranges*0.005
        while time_ok(0.995):
            improved=False
            for d in range(dim):
                if not time_ok(0.995):break
                for sign in [1,-1]:
                    trial=x.copy();trial[d]+=sign*step[d];trial=clip(trial)
                    ft=evaluate(trial)
                    if ft<f_x:x=trial;f_x=ft;improved=True;step[d]*=1.5;break
                else:step[d]*=0.5
            if not improved and np.max(step/ranges)<1e-15:break
    
    return best
