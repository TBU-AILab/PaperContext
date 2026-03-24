#Looking at the best algorithm (score 3.75), I'll make several key improvements:
#
#1. **Add CMA-ES local search** - much more powerful than coordinate descent for correlated landscapes
#2. **Better time allocation** - more restarts with adaptive sizing
#3. **Improved L-SHADE** with rank-based weighting and better archive management
#4. **Nelder-Mead simplex** as additional local search
#5. **Smarter perturbation strategies** for restarts
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
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def run_cmaes(time_frac, sigma0=0.3):
        nonlocal best, best_params
        if best_params is None:
            return
        deadline = elapsed() + max_time * time_frac
        
        n = dim
        lam = max(8, 4 + int(3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = best_params.copy()
        sigma = sigma0 * np.mean(ranges)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Diagonal CMA for efficiency
        C_diag = np.ones(n)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        while elapsed() < deadline:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.empty((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (np.sqrt(C_diag) * arz[k])
                arx[k] = clip(arx[k])
            
            fitvals = np.array([ev(arx[k]) for k in range(lam) if elapsed() < deadline])
            if len(fitvals) < lam:
                break
            
            idx = np.argsort(fitvals)
            arz_sel = arz[idx[:mu]]
            
            zmean = weights @ arz_sel
            mean_old = mean.copy()
            mean = clip(mean + sigma * (np.sqrt(C_diag) * zmean))
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(lam+1))) / chiN) < (1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (np.sqrt(C_diag) * zmean)
            
            C_diag = (1 - c1 - cmu) * C_diag + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag) + cmu * np.sum(weights[:, None] * (np.sqrt(C_diag) * arz_sel)**2, axis=0)
            C_diag = np.maximum(C_diag, 1e-20)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.mean(ranges))
            
            if sigma < 1e-14 * np.mean(ranges):
                break

    def run_lshade(time_frac, restart_idx=0):
        nonlocal best, best_params
        deadline = elapsed() + max_time * time_frac
        
        N_init = min(max(18, 6 * dim), 150)
        N_min = 4
        pop_size = N_init
        H = 80
        M_F = np.full(H, 0.3); M_CR = np.full(H, 0.8)
        M_F[:H//3] = 0.5; M_F[H//3:2*H//3] = 0.7
        M_CR[:H//3] = 0.5; M_CR[H//3:2*H//3] = 0.9
        k_idx = 0
        
        pop = np.random.uniform(0, 1, (pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + pop[:, d]) / pop_size
        pop = lower + pop * ranges
        
        if best_params is not None:
            pop[0] = best_params.copy()
            n_p = min(pop_size // 3, 12)
            for j in range(1, n_p + 1):
                s = (0.02 + 0.08 * restart_idx) * (j / n_p) ** 0.5
                pop[j] = clip(best_params + np.random.randn(dim) * ranges * s)
            if restart_idx > 0:
                for j in range(n_p+1, min(n_p+6, pop_size)):
                    pop[j] = clip(lower + upper - best_params + np.random.randn(dim) * ranges * 0.05)
        
        fit = np.array([ev(pop[i]) for i in range(pop_size) if elapsed() < deadline])
        pop_size = len(fit); pop = pop[:pop_size]
        archive = []
        t0 = elapsed(); t_init = t0 - (deadline - max_time*time_frac)
        t_pe = max(1e-7, t_init/max(1,pop_size))
        max_evals = max(pop_size*10, int((deadline-t0)/t_pe)); eu = pop_size
        
        while elapsed() < deadline and pop_size >= N_min:
            SF,SCR,SDF = [],[],[]
            si = np.argsort(fit); np2=pop.copy(); nf2=fit.copy()
            for i in range(pop_size):
                if elapsed()>=deadline: break
                ri=np.random.randint(H); F=-1
                for _ in range(10):
                    F=np.random.standard_cauchy()*0.1+M_F[ri]
                    if F>0: break
                F=np.clip(F,0.1,1.0); CR=np.clip(np.random.normal(M_CR[ri],0.1),0,1)
                p=max(2,int(max(0.05,0.25-0.20*eu/max(1,max_evals))*pop_size))
                pb=si[np.random.randint(p)]; c=list(range(pop_size)); c.remove(i)
                r1=c[np.random.randint(len(c))]; us=pop_size+len(archive)
                r2p=[j for j in range(us) if j!=i and j!=r1]; r2=r2p[np.random.randint(len(r2p))] if r2p else r1
                xr2=pop[r2] if r2<pop_size else archive[r2-pop_size]
                m=pop[i]+F*(pop[pb]-pop[i])+F*(pop[r1]-xr2)
                bl=m<lower; ab=m>upper; m[bl]=(lower[bl]+pop[i][bl])/2; m[ab]=(upper[ab]+pop[i][ab])/2
                jr=np.random.randint(dim); mk=np.random.rand(dim)<CR; mk[jr]=True
                tr=np.where(mk,m,pop[i]); ft=ev(tr); eu+=1
                if ft<=fit[i]:
                    if ft<fit[i]: SF.append(F);SCR.append(CR);SDF.append(fit[i]-ft); archive.append(pop[i].copy())
                    if len(archive)>N_init: archive.pop(np.random.randint(len(archive)))
                    np2[i]=tr; nf2[i]=ft
            pop,fit=np2,nf2
            if SF:
                w=np.array(SDF);w/=w.sum()+1e-30;sf=np.array(SF)
                M_F[k_idx]=np.sum(w*sf**2)/(np.sum(w*sf)+1e-30); M_CR[k_idx]=np.sum(w*np.array(SCR)); k_idx=(k_idx+1)%H
            ns=max(N_min,int(round(N_init-(N_init-N_min)*eu/max(1,max_evals))))
            if ns<pop_size: k=np.argsort(fit)[:ns]; pop=pop[k]; fit=fit[k]; pop_size=ns

    run_lshade(0.35, 0)
    if time_left() > max_time*0.15: run_cmaes(time_left()/max_time*0.25, 0.2)
    if time_left() > max_time*0.20: run_lshade(time_left()/max_time*0.45, 1)
    if time_left() > max_time*0.10: run_cmaes(time_left()/max_time*0.25, 0.05)
    if time_left() > max_time*0.12: run_lshade(time_left()/max_time*0.45, 2)
    if time_left() > max_time*0.05: run_cmaes(time_left()/max_time*0.3, 0.01)

    if best_params is not None and time_left() > 0.05:
        cur=best_params.copy(); cf=best
        for sc in [0.05,0.01,0.002,0.0004,0.00005]:
            st=ranges*sc
            if time_left()<=0.02: break
            while time_left()>0.02:
                imp=False
                for d in np.random.permutation(dim):
                    if time_left()<=0.02: break
                    for s in [1,-1]:
                        t=cur.copy(); t[d]=np.clip(cur[d]+s*st[d],lower[d],upper[d]); ft=ev(t)
                        if ft<cf: cur,cf=t,ft; imp=True; break
                if not imp: break
    return best
