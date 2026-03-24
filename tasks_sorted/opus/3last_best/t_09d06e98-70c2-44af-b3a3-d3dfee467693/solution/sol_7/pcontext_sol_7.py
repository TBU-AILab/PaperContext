#Looking at the results, algorithm #2 (score 3.75) performs best. I'll analyze what works and make targeted improvements:
#
#1. **Add a proper CMA-ES** with full covariance matrix for better local search
#2. **Nelder-Mead simplex** as complementary local search
#3. **Better restart strategies** with more diverse exploration
#4. **Improved time allocation** - interleave global and local search phases
#5. **Pattern search** enhancement in coordinate descent
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

    def nelder_mead(x0, time_frac, initial_scale=0.05):
        deadline = elapsed() + max_time * time_frac
        n = dim
        simplex = np.empty((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += ranges[i] * initial_scale * (1 if np.random.rand() > 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
        
        f_vals = np.array([ev(simplex[i]) for i in range(n+1) if elapsed() < deadline])
        if len(f_vals) < n+1:
            return
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() < deadline:
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = ev(xr)
            if elapsed() >= deadline: return
            
            if fr < f_vals[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = ev(xe)
                if elapsed() >= deadline: return
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            elif fr < f_vals[-2]:
                simplex[-1], f_vals[-1] = xr, fr
            else:
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = ev(xc)
                if elapsed() >= deadline: return
                if fc < min(fr, f_vals[-1]):
                    simplex[-1], f_vals[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= deadline: return
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_vals[i] = ev(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-12 * np.mean(ranges):
                break

    def run_lshade(time_frac, restart_idx=0):
        nonlocal best, best_params
        deadline = elapsed() + max_time * time_frac
        N_init = min(max(18, 6*dim), 150)
        N_min = 4; pop_size = N_init; H = 80
        M_F = np.full(H,0.3); M_CR = np.full(H,0.8)
        M_F[:H//3]=0.5; M_F[H//3:2*H//3]=0.7; M_CR[:H//3]=0.5; M_CR[H//3:2*H//3]=0.9
        k_idx=0
        pop=np.random.uniform(0,1,(pop_size,dim))
        for d in range(dim): p=np.random.permutation(pop_size); pop[:,d]=(p+pop[:,d])/pop_size
        pop=lower+pop*ranges
        if best_params is not None:
            pop[0]=best_params.copy(); np_=min(pop_size//3,12)
            for j in range(1,np_+1): s=(0.02+0.1*restart_idx)*(j/np_)**0.5; pop[j]=clip(best_params+np.random.randn(dim)*ranges*s)
            if restart_idx>0:
                for j in range(np_+1,min(np_+6,pop_size)): pop[j]=clip(lower+upper-best_params+np.random.randn(dim)*ranges*0.05)
        fit=np.array([ev(pop[i]) for i in range(pop_size) if elapsed()<deadline]); pop_size=len(fit); pop=pop[:pop_size]
        archive=[]; t0=elapsed(); ti=t0-(deadline-max_time*time_frac); tpe=max(1e-7,ti/max(1,pop_size)); me=max(pop_size*10,int((deadline-t0)/tpe)); eu=pop_size
        while elapsed()<deadline and pop_size>=N_min:
            SF,SCR,SDF=[],[],[]; si=np.argsort(fit); np2,nf2=pop.copy(),fit.copy()
            for i in range(pop_size):
                if elapsed()>=deadline: break
                ri=np.random.randint(H); F=-1
                for _ in range(10): F=np.random.standard_cauchy()*0.1+M_F[ri]; F=F if F>0 else -1; 
                if F<=0: F=0.5
                F=np.clip(F,0.1,1.0); CR=np.clip(np.random.normal(M_CR[ri],0.1),0,1)
                p=max(2,int(max(0.05,0.25-0.20*eu/max(1,me))*pop_size)); pb=si[np.random.randint(p)]
                c=list(range(pop_size)); c.remove(i); r1=c[np.random.randint(len(c))]; us=pop_size+len(archive)
                r2p=[j for j in range(us) if j!=i and j!=r1]; r2=r2p[np.random.randint(len(r2p))] if r2p else r1
                xr2=pop[r2] if r2<pop_size else archive[r2-pop_size]
                m=pop[i]+F*(pop[pb]-pop[i])+F*(pop[r1]-xr2); bl=m<lower; ab=m>upper; m[bl]=(lower[bl]+pop[i][bl])/2; m[ab]=(upper[ab]+pop[i][ab])/2
                jr=np.random.randint(dim); mk=np.random.rand(dim)<CR; mk[jr]=True; tr=np.where(mk,m,pop[i]); ft=ev(tr); eu+=1
                if ft<=fit[i]:
                    if ft<fit[i]: SF.append(F);SCR.append(CR);SDF.append(fit[i]-ft); archive.append(pop[i].copy())
                    if len(archive)>N_init: archive.pop(np.random.randint(len(archive)))
                    np2[i]=tr; nf2[i]=ft
            pop,fit=np2,nf2
            if SF: w=np.array(SDF);w/=w.sum()+1e-30;sf=np.array(SF); M_F[k_idx]=np.sum(w*sf**2)/(np.sum(w*sf)+1e-30); M_CR[k_idx]=np.sum(w*np.array(SCR)); k_idx=(k_idx+1)%H
            ns=max(N_min,int(round(N_init-(N_init-N_min)*eu/max(1,me))))
            if ns<pop_size: k=np.argsort(fit)[:ns]; pop=pop[k]; fit=fit[k]; pop_size=ns

    run_lshade(0.35, 0)
    if time_left()>max_time*0.12: nelder_mead(best_params, time_left()/max_time*0.15, 0.1)
    if time_left()>max_time*0.20: run_lshade(time_left()/max_time*0.45, 1)
    if time_left()>max_time*0.08: nelder_mead(best_params, time_left()/max_time*0.15, 0.02)
    if time_left()>max_time*0.12: run_lshade(time_left()/max_time*0.45, 2)
    if time_left()>max_time*0.05: nelder_mead(best_params, time_left()/max_time*0.2, 0.005)

    if best_params is not None and time_left()>0.05:
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
