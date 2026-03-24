#Looking at the results, algorithm 6 (score 36.62) and algorithm 8 (score 28.52) performed best. Algorithm 4 (score 197.53) was the best among the ~200 range. Let me analyze what made algorithms 6 and 8 work well and what can be improved:
#
#Key observations:
#1. Algorithm 8 used simpler DE + CMA-ES and performed best (28.52)
#2. Algorithm 6 used SHADE + CMA-ES + Nelder-Mead (36.62)
#3. The more complex algorithms (1-5) with L-SHADE, sep-CMA, etc. performed worse (~200)
#4. Simpler, cleaner implementations seem to work better
#5. The CR adaptation in SHADE using `M_CR[k] = np.sum(w * S_CR)` (weighted arithmetic mean) might be wrong - should use Lehmer mean
#
#Let me build a cleaner, more focused algorithm:
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
    
    def time_ok(frac=0.97):
        return elapsed() < max_time * frac
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    
    # Top solutions archive
    top_k = min(30, max(8, 2*dim))
    top_x = []
    top_f = []
    
    def update_top(x, f):
        if len(top_x) < top_k:
            top_x.append(x.copy())
            top_f.append(f)
        elif f < max(top_f):
            idx = np.argmax(top_f)
            top_x[idx] = x.copy()
            top_f[idx] = f
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        update_top(x, f)
        return f
    
    # Phase 1: LHS initialization
    n_init = min(max(20 * dim, 200), 1500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fit = np.full(n_init, float('inf'))
    actual = 0
    for i in range(n_init):
        if elapsed() > max_time * 0.07:
            break
        init_fit[i] = ev(init_pop[i])
        actual = i + 1
    
    # Center point
    if time_ok(0.08):
        ev((lower + upper) / 2.0)
    
    # Opposition on best
    if actual > 10 and time_ok(0.10):
        si = np.argsort(init_fit[:actual])
        n_opp = min(actual // 4, 30)
        for i in range(n_opp):
            if elapsed() > max_time * 0.10:
                break
            ev(lower + upper - init_pop[si[i]])
    
    si = np.argsort(init_fit[:actual])
    
    # Phase 2: SHADE
    def shade_phase(time_budget):
        nonlocal best, best_params
        if time_budget < 0.1:
            return
        deadline = elapsed() + time_budget
        
        N = min(max(8 * dim, 50), 250)
        H = 100
        n_elite = min(N // 3, actual)
        pop = np.zeros((N, dim))
        fit = np.full(N, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[si[i]].copy()
            fit[i] = init_fit[si[i]]
        for i in range(n_elite, N):
            pop[i] = lower + np.random.random(dim) * ranges
        for i in range(n_elite, N):
            if not time_ok(0.96) or elapsed() > deadline:
                return
            fit[i] = ev(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.8)
        k = 0
        archive = []
        
        while time_ok(0.96) and elapsed() < deadline:
            S_F, S_CR, S_df = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(N):
                if not time_ok(0.96) or elapsed() > deadline:
                    pop[:] = new_pop; fit[:] = new_fit
                    return
                
                ri = np.random.randint(H)
                Fi = -1
                for _ in range(10):
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi > 0: break
                Fi = min(max(Fi, 0.01), 1.0)
                
                CRi = M_CR[ri]
                CRi = 0.0 if CRi < 0 else np.clip(CRi + 0.1 * np.random.randn(), 0, 1)
                
                p = max(2, int(N * 0.11))
                pb = np.argsort(fit)[:p]
                xpb = pop[pb[np.random.randint(p)]]
                
                idxs = [j for j in range(N) if j != i]
                r1 = idxs[np.random.randint(len(idxs))]
                pool = N + len(archive)
                r2 = np.random.randint(pool)
                for _ in range(25):
                    if r2 != i and r2 != r1: break
                    r2 = np.random.randint(pool)
                xr2 = pop[r2] if r2 < N else archive[r2 - N]
                
                v = pop[i] + Fi * (xpb - pop[i]) + Fi * (pop[r1] - xr2)
                for d in range(dim):
                    if v[d] < lower[d]: v[d] = (lower[d] + pop[i,d]) / 2
                    elif v[d] > upper[d]: v[d] = (upper[d] + pop[i,d]) / 2
                
                trial = pop[i].copy()
                jr = np.random.randint(dim)
                mask = np.random.random(dim) < CRi; mask[jr] = True
                trial[mask] = v[mask]
                trial = clip(trial)
                
                ft = ev(trial)
                if ft < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_df.append(fit[i]-ft)
                    archive.append(pop[i].copy())
                    if len(archive) > N: archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial; new_fit[i] = ft
                elif ft == fit[i]:
                    new_pop[i] = trial; new_fit[i] = ft
            
            pop[:] = new_pop; fit[:] = new_fit
            if S_F:
                sd = np.array(S_df); w = sd/(sd.sum()+1e-30)
                sf = np.array(S_F); sc = np.array(S_CR)
                M_F[k] = (w*sf**2).sum()/((w*sf).sum()+1e-30)
                M_CR[k] = -1.0 if sc.max()<=0 else (w*sc**2).sum()/((w*sc).sum()+1e-30)
                k = (k+1) % H
    
    def cma_run(x0, sig0, tb, pm=1):
        nonlocal best, best_params
        if tb < 0.05: return
        dl = elapsed() + tb; n = dim
        lam = max(4+int(3*np.log(n)),6)*pm; mu = lam//2
        w = np.log(mu+.5)-np.log(np.arange(1,mu+1)); w /= w.sum()
        me = 1./np.sum(w**2)
        cc=(4+me/n)/(n+4+2*me/n); cs=(me+2)/(n+me+5)
        c1=2/((n+1.3)**2+me); cm=min(1-c1,2*(me-2+1/me)/((n+2)**2+me))
        ds=1+2*max(0,np.sqrt((me-1)/(n+1))-1)+cs
        chi=np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
        pc=np.zeros(n); ps=np.zeros(n)
        B=np.eye(n); D=np.ones(n); C=np.eye(n); iC=np.eye(n)
        m=x0.copy(); s=sig0; g=0; ni=0; pb=best
        ef=max(1,int(1/(c1+cm+1e-20)/n/10))
        while time_ok(0.96) and elapsed()<dl:
            z=np.random.randn(lam,n); ax=np.zeros((lam,n))
            BD=B*D[None,:]
            for ki in range(lam):
                ax[ki]=m+s*(BD@z[ki])
                for _ in range(3):
                    ob=(ax[ki]<lower)|(ax[ki]>upper)
                    if not np.any(ob): break
                    ax[ki]=np.where(ob,m+s*np.random.randn(n)*D,ax[ki])
                ax[ki]=clip(ax[ki])
            fs=np.zeros(lam)
            for ki in range(lam):
                if not time_ok(0.96) or elapsed()>dl: return
                fs[ki]=ev(ax[ki])
            ix=np.argsort(fs); om=m.copy()
            m=np.sum(w[:,None]*ax[ix[:mu]],axis=0)
            d2=m-om
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*me)*(iC@d2)/max(s,1e-30)
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chi<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*me)*d2/max(s,1e-30)
            at=(ax[ix[:mu]]-om)/max(s,1e-30)
            C=(1-c1-cm)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cm*np.einsum('k,ki,kj->ij',w,at,at)
            s*=np.exp((cs/ds)*(np.linalg.norm(ps)/chi-1))
            s=min(s,2*np.max(ranges)); g+=1
            if g%ef==0:
                C=np.triu(C)+np.triu(C,1).T
                try:
                    Ds,Bn=np.linalg.eigh(C); Ds=np.maximum(Ds,1e-20)
                    D=np.sqrt(Ds); B=Bn; iC=B@np.diag(1./D)@B.T
                except: C=np.eye(n);B=np.eye(n);D=np.ones(n);iC=np.eye(n)
            if best<pb-1e-10: ni=0; pb=best
            else: ni+=1
            if s*np.max(D)<1e-14*np.max(ranges) or ni>25+8*n: break
    
    def nm(x0, sc, tb):
        nonlocal best, best_params
        if tb<0.05: return
        dl=elapsed()+tb; n=dim; s=sc*ranges
        sx=np.zeros((n+1,n)); sx[0]=x0.copy()
        for i in range(n):
            sx[i+1]=x0.copy(); sx[i+1,i]+=s[i]*(1 if np.random.random()<.5 else -1)
            sx[i+1]=clip(sx[i+1])
        fs=np.zeros(n+1)
        for i in range(n+1):
            if not time_ok(0.96) or elapsed()>dl: return
            fs[i]=ev(sx[i])
        while time_ok(0.97) and elapsed()<dl:
            o=np.argsort(fs); sx=sx[o]; fs=fs[o]
            c=np.mean(sx[:-1],axis=0)
            xr=clip(c+(c-sx[-1])); fr=ev(xr)
            if fs[0]<=fr<fs[-2]: sx[-1]=xr;fs[-1]=fr
            elif fr<fs[0]:
                xe=clip(c+2*(xr-c))
                if not time_ok(0.97) or elapsed()>dl: break
                fe=ev(xe)
                if fe<fr: sx[-1]=xe;fs[-1]=fe
                else: sx[-1]=xr;fs[-1]=fr
            else:
                xc=clip(c+.5*(sx[-1]-c)) if fr>=fs[-1] else clip(c+.5*(xr-c))
                if not time_ok(0.97) or elapsed()>dl: break
                fc=ev(xc)
                if fc<min(fr,fs[-1]): sx[-1]=xc;fs[-1]=fc
                else:
                    for i in range(1,n+1):
                        if not time_ok(0.97) or elapsed()>dl: return
                        sx[i]=sx[0]+.5*(sx[i]-sx[0]);sx[i]=clip(sx[i]);fs[i]=ev(sx[i])
            if np.max(np.abs(fs-fs[0]))<1e-16: break
    
    rem = max_time - elapsed()
    shade_phase(rem * 0.30)
    
    r = 0
    while time_ok(0.80):
        rc = max_time*0.80 - elapsed()
        if rc < 0.15: break
        nl = max(1, min(8, int(rc/max(0.2, rc/6))))
        tb = rc/nl*0.9
        if r==0: x0=best_params.copy() if best_params is not None else lower+np.random.random(dim)*ranges; sg=.2*np.max(ranges); pm=1
        elif r%3==1:
            if top_x: st=np.argsort(top_f); x0=top_x[st[min(r//3,len(top_x)-1)]].copy()
            else: x0=best_params.copy() if best_params is not None else lower+np.random.random(dim)*ranges
            x0=clip(x0+.04*ranges*np.random.randn(dim)); sg=.07*np.max(ranges); pm=1
        elif r%3==2: x0=lower+np.random.random(dim)*ranges; sg=.4*np.max(ranges); pm=2
        else: x0=clip(best_params+.1*ranges*np.random.randn(dim)) if best_params is not None else lower+np.random.random(dim)*ranges; sg=.15*np.max(ranges); pm=1
        cma_run(x0, min(sg, np.max(ranges)), tb, pm); r+=1
    
    if time_ok(0.82) and top_x:
        st=np.argsort(top_f)
        for ri in range(min(3,len(top_x))):
            if not time_ok(0.90): break
            rn=max_time*.92-elapsed()
            if rn<0.1: break
            nm(top_x[st[ri]], .02/(1+ri), rn/(min(3,len(top_x))-ri)*.7)
    
    if time_ok(0.92) and best_params is not None:
        nm(best_params, .004, (max_time*.96-elapsed())*.5)
    
    if time_ok(0.96) and best_params is not None:
        dl=max_time*.995; x=best_params.copy(); fx=ev(x); st=.02*ranges.copy()
        while time_ok(0.98) and elapsed()<dl:
            imp=False
            for di in np.random.permutation(dim):
                if not time_ok(0.98) or elapsed()>dl: break
                xn=x.copy();xn[di]=min(x[di]+st[di],upper[di]);fn=ev(xn)
                if fn<fx: x=xn;fx=fn;st[di]*=1.3;imp=True;continue
                xn=x.copy();xn[di]=max(x[di]-st[di],lower[di])
                if not time_ok(0.98) or elapsed()>dl: break
                fn=ev(xn)
                if fn<fx: x=xn;fx=fn;st[di]*=1.3;imp=True
                else: st[di]*=.5
            if not imp or np.max(st/ranges)<1e-14: break
    
    return best
