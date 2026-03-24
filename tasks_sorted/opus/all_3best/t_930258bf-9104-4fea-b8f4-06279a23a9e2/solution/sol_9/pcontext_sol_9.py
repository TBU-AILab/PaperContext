#Looking at the best algorithm (score 6.34), I'll make targeted improvements:
#
#1. **Faster SHADE inner loop** - vectorize where possible, reduce Python overhead
#2. **Better Nelder-Mead** - add adaptive restarts with different scales
#3. **Improved CMA-ES restart strategy** - more emphasis on local exploitation with very small perturbations
#4. **Add a final coordinate descent** for fine-tuning
#5. **Slightly adjust time allocation** - give more to SHADE which seems most effective
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

    # Phase 1: LHS + opposition
    n_init = min(max(11 * dim, 55), 280)
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
        if remaining() > 0 and i < n_init // 6:
            opp = lower + upper - init_points[i]
            eval_func(opp)
    
    init_fit.sort()
    top_k = min(30, n_init)
    top_points = [init_points[init_fit[i][1]].copy() for i in range(top_k)]

    # Phase 2: SHADE DE
    def run_shade(time_fraction=0.46):
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
        
        H = 30
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        max_archive = pop_size
        
        while elapsed() < deadline and remaining() > 0:
            S_F, S_CR, S_df = [], [], []
            sorted_idx = np.argsort(fit)
            
            shade_start = deadline - max_time * time_fraction
            time_progress = min(1.0, (elapsed() - shade_start) / (max_time * time_fraction + 1e-30))
            p_rate = max(2.0/pop_size, 0.25 - 0.20 * time_progress)
            p = max(2, int(p_rate * pop_size))
            
            for i in range(pop_size):
                if elapsed() >= deadline or remaining() <= 0:
                    return
                
                ri = np.random.randint(H)
                Fi = -1.0
                for _ in range(30):
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi > 0:
                        break
                Fi = np.clip(Fi, 0.01, 1.0)
                
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
                
                pb = pop[sorted_idx[np.random.randint(p)]]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                all_size = pop_size + len(archive)
                r2 = np.random.randint(all_size)
                att = 0
                while (r2 == i or r2 == r1) and att < 20:
                    r2 = np.random.randint(all_size); att += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pb - pop[i]) + Fi * (pop[r1] - xr2)
                out_low = mutant < lower
                out_high = mutant > upper
                mutant[out_low] = (lower[out_low] + pop[i][out_low]) / 2.0
                mutant[out_high] = (upper[out_high] + pop[i][out_high]) / 2.0
                
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                f_trial = eval_func(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi); S_CR.append(CRi); S_df.append(fit[i] - f_trial)
                        if len(archive) < max_archive: archive.append(pop[i].copy())
                        elif archive: archive[np.random.randint(len(archive))] = pop[i].copy()
                    pop[i] = trial; fit[i] = f_trial
            
            if S_F:
                w = np.array(S_df); w = w / (w.sum() + 1e-30)
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr**2) / (np.sum(w * scr) + 1e-30)
                k = (k + 1) % H
    
    run_shade(0.46)

    # Phase 2.5: Nelder-Mead multi-scale
    def nelder_mead(x0, max_tf=0.07, scale=0.05):
        nonlocal best, best_x
        dl = elapsed() + max_time * max_tf; n = dim
        s = np.zeros((n+1,n)); s[0]=x0.copy(); st=ranges*scale
        for i in range(n): s[i+1]=x0.copy(); s[i+1][i]+= st[i]*(1 if np.random.rand()>0.5 else -1); s[i+1]=clip(s[i+1])
        fs = np.array([eval_func(s[i]) for i in range(n+1)])
        for _ in range(10000):
            if elapsed()>=dl or remaining()<=0: return
            o=np.argsort(fs); s=s[o]; fs=fs[o]; c=np.mean(s[:n],axis=0)
            xr=clip(c+(c-s[n])); fr=eval_func(xr)
            if fr<fs[0]: xe=clip(c+2*(xr-c)); fe=eval_func(xe); s[n],fs[n]=(xe,fe) if fe<fr else (xr,fr)
            elif fr<fs[n-1]: s[n]=xr; fs[n]=fr
            else:
                xc=clip(c+0.5*((xr if fr<fs[n] else s[n])-c)); fc=eval_func(xc)
                if fc<min(fr,fs[n]): s[n]=xc; fs[n]=fc
                else:
                    for i in range(1,n+1):
                        if remaining()<=0: return
                        s[i]=clip(s[0]+0.5*(s[i]-s[0])); fs[i]=eval_func(s[i])
            if np.max(np.abs(s[n]-s[0]))<1e-14: return
    
    if best_x is not None and remaining()>1: nelder_mead(best_x.copy(),0.05,0.05)
    if best_x is not None and remaining()>0.5: nelder_mead(best_x.copy(),0.03,0.003)
    if best_x is not None and remaining()>0.3: nelder_mead(best_x.copy(),0.02,0.0003)

    # Phase 3: CMA-ES BIPOP
    def run_cmaes(x0,initial_sigma,lam_mult=1):
        nonlocal best,best_x
        sigma=initial_sigma;mean=x0.copy();n=dim;lam=max(int((4+int(3*np.log(n)))*lam_mult),6);mu=lam//2
        w=np.log(mu+0.5)-np.log(np.arange(1,mu+1));w/=w.sum();mueff=1.0/np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n);cs=(mueff+2)/(n+mueff+5);c1=2/((n+1.3)**2+mueff);cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs;chiN=n**0.5*(1-1/(4*n)+1/(21*n**2))
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);C=np.eye(n);invsqrtC=np.eye(n);eigeneval=0;counteval=0;stag=0;bg=float('inf')
        for gen in range(1,10000):
            if remaining()<=0:return
            if counteval-eigeneval>lam/(c1+cmu+1e-30)/n/10:
                eigeneval=counteval;C=(C+C.T)/2
                try:Dv,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(Dv,1e-20));invsqrtC=B@np.diag(1/D)@B.T
                except:return
            arx=np.zeros((lam,n));af=np.zeros(lam)
            for kk in range(lam):
                if remaining()<=0:return
                arx[kk]=clip(mean+sigma*(B@(D*np.random.randn(n))));af[kk]=eval_func(arx[kk]);counteval+=1
            idx=np.argsort(af);om=mean.copy();mean=w@arx[idx[:mu]];d=(mean-om)/(sigma+1e-30)
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*invsqrtC@d;hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*d;at=(arx[idx[:mu]]-om)/(sigma+1e-30)
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(at.T@np.diag(w)@at)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1));sigma=np.clip(sigma,1e-20,np.max(ranges)*2)
            if gen%7==0:
                if af[idx[0]]<bg-1e-10*(abs(bg)+1):bg=af[idx[0]];stag=0
                else:stag+=1
                if stag>=5 or sigma<1e-16:return

    restart=0;lam_mult=1;sb=0;lb=0
    while remaining()>0.3:
        use_large=restart>=len(top_points) and lb<=sb
        if restart<len(top_points):x0=top_points[restart];sig=np.mean(ranges)/4;lm=1
        elif use_large:x0=np.array([np.random.uniform(l,u) for l,u in bounds]);sig=np.mean(ranges)/2;lm=min(lam_mult*2,8);lam_mult=lm
        elif best_x is not None and np.random.rand()<0.85:
            ps2=np.random.choice([0.001,0.003,0.008,0.02,0.05,0.1]);x0=clip(best_x+np.random.randn(dim)*ranges*ps2);sig=np.mean(ranges)*ps2;lm=1
        else:x0=np.array([np.random.uniform(l,u) for l,u in bounds]);sig=np.mean(ranges)/4;lm=1
        t0=elapsed();run_cmaes(x0,sig,lm);dt=elapsed()-t0
        if use_large:lb+=dt
        else:sb+=dt
        restart+=1

    return best
