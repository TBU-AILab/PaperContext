#Looking at the results, algorithm #5 (score 12.19) performs best with L-SHADE + CMA-ES + Nelder-Mead. Key observations:
#
#1. **M_CR initialized to 0.5** in #5 outperformed M_CR=0.8 in others (12.19 vs 20.66+)
#2. **Nelder-Mead** local search helped vs coordinate descent
#3. The main bottleneck seems to be the DE phase efficiency and CMA-ES restart strategy
#
#I'll improve by: faster DE (avoid list comprehensions for r1/r2), better population reduction, more aggressive CMA-ES local restarts, and a hybrid local search combining Nelder-Mead with coordinate descent.
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
    M_CR = np.full(H, 0.5)
    k_idx = 0
    archive = []
    
    nfe = n_init + pop_size - top_k
    max_nfe_estimate = pop_size_init * 250
    
    de_time_frac = 0.50
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
            att = 0
            while F <= 0 and att < 30:
                F = mu_f + 0.1 * np.random.standard_cauchy()
                att += 1
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
            if r1 >= i: r1 += 1
            
            pool_size = pop_size + len(archive)
            if pool_size > 2:
                r2 = np.random.randint(pool_size)
                ct = 0
                while (r2 == i or r2 == r1) and ct < 20:
                    r2 = np.random.randint(pool_size); ct += 1
            else:
                r2 = r1
            
            r2_vec = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - r2_vec)
            
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
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k_idx] = float(np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30))
            M_CR[k_idx] = -1.0 if np.max(scr) <= 0 else float(np.sum(w * scr))
            k_idx = (k_idx + 1) % H
        
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * nfe / max_nfe_estimate)))
        if new_ps < pop_size:
            sidx = np.argsort(pop_f); pop = pop[sidx[:new_ps]]; pop_f = pop_f[sidx[:new_ps]]; pop_size = new_ps
        gen += 1
    
    base_lam = max(4 + int(3 * np.log(dim)), 12); use_sep = dim > 40; rc = 0
    def run_cmaes(x0, sigma0, lam):
        nonlocal best, best_x; mean=x0.copy();sigma=sigma0;n=dim;mc=lam//2;w=np.log(mc+.5)-np.log(np.arange(1,mc+1));w/=np.sum(w);me=1/np.sum(w**2);cc=(4+me/n)/(n+4+2*me/n);cs=(me+2)/(n+me+5);c1=2/((n+1.3)**2+me);cm=min(1-c1,2*(me-2+1/me)/((n+2)**2+me));da=1+2*max(0,np.sqrt((me-1)/(n+1))-1)+cs;ch=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2));pc=np.zeros(n);ps=np.zeros(n)
        if use_sep:dC=np.ones(n)
        else:B=np.eye(n);D=np.ones(n);C=np.eye(n);iC=np.eye(n);ee=0
        ce=0;g=0;bl=float('inf');st=0
        while time_left()>.3:
            ax=np.zeros((lam,n));fi=np.zeros(lam)
            for k in range(lam):
                if time_left()<=.2:return
                z=np.random.randn(n);x=mean+sigma*(np.sqrt(dC)*z if use_sep else B@(D*z));ax[k]=clip(x);fi[k]=eval_func(ax[k]);ce+=1
            ix=np.argsort(fi);ax=ax[ix];fi=fi[ix]
            if fi[0]<bl:bl=fi[0];st=0
            else:st+=1
            o=mean.copy();mean=w@ax[:mc];d=mean-o
            if use_sep:inv=1/np.sqrt(dC);ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*me)*(inv*d)/sigma;hs=int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/ch<1.4+2/(n+1));pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*me)*d/sigma;art=(ax[:mc]-o)/sigma;dC=(1-c1-cm)*dC+c1*(pc**2+(1-hs)*cc*(2-cc)*dC)+cm*np.sum(w[:,None]*art**2,axis=0);dC=np.maximum(dC,1e-20)
            else:ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*me)*(iC@d)/sigma;hs=int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/ch<1.4+2/(n+1));pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*me)*d/sigma;art=(ax[:mc]-o)/sigma;C=(1-c1-cm)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cm*(art.T@np.diag(w)@art)
            if not use_sep and ce-ee>lam/(c1+cm)/n/10:
                ee=ce;C=np.triu(C)+np.triu(C,1).T
                try:D2,B=np.linalg.eigh(C);D2=np.maximum(D2,1e-20);D=np.sqrt(D2);iC=B@np.diag(1/D)@B.T
                except:return
            sigma*=np.exp((cs/da)*(np.linalg.norm(ps)/ch-1));sigma=np.clip(sigma,1e-20,np.max(ranges)*2);g+=1
            if sigma<1e-14 or st>10+30*n/lam:return
            cd=sigma*np.max(np.sqrt(dC)) if use_sep else sigma*np.max(D)
            if cd<1e-12*np.max(ranges):return
    ds=np.argsort(pop_f);als=[pop[ds[i]].copy() for i in range(min(5,pop_size))];si=0
    while time_left()>1.0:
        if rc<3:
            if si<len(als):x0=als[si];si+=1;lam=base_lam;s0=np.mean(ranges)/6
            else:x0=best_x+np.random.randn(dim)*ranges*.02;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/10
        elif rc%4==0:x0=best_x+np.random.randn(dim)*ranges*.01;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/15
        elif rc%4==1:x0=lower+np.random.rand(dim)*ranges;lam=min(int(base_lam*2**(rc*.3)),250);s0=np.mean(ranges)/3
        else:x0=best_x+np.random.randn(dim)*ranges*.05;x0=clip(x0);lam=base_lam;s0=np.mean(ranges)/8
        run_cmaes(x0,s0,lam);rc+=1
    if time_left()>.3 and best_x is not None:
        n=dim;simplex=np.zeros((n+1,n));simplex_f=np.zeros(n+1);simplex[0]=best_x.copy();simplex_f[0]=best
        for i in range(n):
            p=best_x.copy();p[i]+=ranges[i]*0.005;simplex[i+1]=clip(p)
            if time_left()<=.1:return best
            simplex_f[i+1]=eval_func(simplex[i+1])
        for _ in range(500):
            if time_left()<=.1:break
            order=np.argsort(simplex_f);simplex=simplex[order];simplex_f=simplex_f[order];centroid=np.mean(simplex[:n],axis=0)
            xr=clip(centroid+1.0*(centroid-simplex[n]));fr=eval_func(xr)
            if fr<simplex_f[0]:xe=clip(centroid+2.0*(xr-centroid));fe=eval_func(xe);simplex[n],simplex_f[n]=(xe,fe) if fe<fr else (xr,fr)
            elif fr<simplex_f[n-1]:simplex[n]=xr;simplex_f[n]=fr
            else:
                xc=clip(centroid+0.5*(simplex[n]-centroid));fc=eval_func(xc)
                if fc<simplex_f[n]:simplex[n]=xc;simplex_f[n]=fc
                else:
                    for i in range(1,n+1):
                        if time_left()<=.05:return best
                        simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0]));simplex_f[i]=eval_func(simplex[i])
    return best
