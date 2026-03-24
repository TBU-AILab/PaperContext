#Looking at the results, the best algorithm (146.4) uses IPOP-CMA-ES. Let me analyze what can be improved:
#
#1. **Opposition-based learning wastes evaluations** (~200 evals for marginal benefit)
#2. **Eigendecomposition frequency** should use the standard formula `1/(c1+cmu)/n/10`
#3. **Need BIPOP** - mix of large population (exploration) and small population (exploitation) restarts
#4. **Local search (Nelder-Mead)** after CMA-ES converges can squeeze out improvements
#5. **Better archive management** for restart initialization
#6. **Reduce initial sampling** to leave more budget for CMA-ES
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

    def time_left():
        return max_time * 0.97 - (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    archive = []
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def add_archive(x, f):
        archive.append((x.copy(), f))
        archive.sort(key=lambda t: t[1])
        if len(archive) > 50: archive.pop()

    # LHS init - reduced to save budget
    n_init = min(max(12*dim, 60), 300)
    pop = np.zeros((n_init, dim))
    for d in range(dim):
        p = np.random.permutation(n_init)
        pop[:, d] = lower[d] + (p + np.random.random(n_init)) / n_init * ranges[d]
    for i in range(n_init):
        if time_left() <= 0: return best
        f = eval_func(pop[i])
        add_archive(pop[i], f)

    def nelder_mead(x0, scale=0.03, max_iter=1500):
        n = dim
        simplex = np.zeros((n+1, n))
        simplex[0] = clip(x0)
        for i in range(n):
            p = x0.copy()
            p[i] += scale * ranges[i] * (1 if x0[i] < (lower[i]+upper[i])/2 else -1)
            simplex[i+1] = clip(p)
        fs = np.array([eval_func(simplex[i]) for i in range(n+1) if time_left() > 0.2])
        if len(fs) < n+1: return
        for _ in range(max_iter):
            if time_left() <= 0.2: return
            o = np.argsort(fs); simplex=simplex[o]; fs=fs[o]
            c = np.mean(simplex[:n], axis=0)
            xr = clip(2*c - simplex[n]); fr = eval_func(xr)
            if fr < fs[0]:
                xe = clip(c + 2*(xr-c)); fe = eval_func(xe)
                if fe < fr: simplex[n],fs[n]=xe,fe
                else: simplex[n],fs[n]=xr,fr
            elif fr < fs[n-1]: simplex[n],fs[n]=xr,fr
            else:
                xc = clip(c+0.5*((fr<fs[n])*(xr-c)+(fr>=fs[n])*(simplex[n]-c)))
                fc = eval_func(xc)
                if fc<=min(fr,fs[n]): simplex[n],fs[n]=xc,fc
                else:
                    for i in range(1,n+1):
                        if time_left()<=0.2: return
                        simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); fs[i]=eval_func(simplex[i])
            if np.max(np.abs(simplex[-1]-simplex[0]))<1e-14: return

    def cma_es(x0, sigma0, lam):
        n=dim; mu=lam//2; w=np.log(mu+.5)-np.log(np.arange(1,mu+1)); w/=w.sum(); mueff=1/np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5); c1=2/((n+1.3)**2+mueff)
        cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff)); ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);C=np.eye(n);invC=np.eye(n)
        chiN=n**.5*(1-1/(4*n)+1/(21*n**2));mean=x0.copy();sigma=sigma0;g=0;stag=0;pbest=1e30
        efreq=max(1,int(1/(c1+cmu+1e-23)/n/10))
        hist_med=[]
        while time_left()>0.4:
            arz=np.random.randn(lam,n);arx=mean+sigma*(arz@(B*D).T);arx=np.array([clip(x) for x in arx])
            fit=np.array([eval_func(arx[k]) for k in range(lam)]);
            if time_left()<0.2: return
            o=np.argsort(fit);arx=arx[o];arz=arz[o];fit=fit[o];add_archive(arx[0],fit[0])
            hist_med.append(np.median(fit))
            if len(hist_med)>10 and hist_med[-1]>=hist_med[-10]-1e-12*abs(hist_med[-10]+1e-30): stag+=1
            elif fit[0]>=pbest-1e-12*max(1,abs(pbest)): stag+=1
            else: stag=0
            pbest=min(pbest,fit[0])
            if stag>10+2*n: return
            om=mean.copy();mean=clip(w@arx[:mu]);zm=w@arz[:mu]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invC@zm)
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*((mean-om)/sigma)
            artmp=(arx[:mu]-om)/sigma;C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(w)@artmp)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1));sigma=min(sigma,2*np.max(ranges));g+=1
            if g%efreq==0:
                try: C=np.triu(C)+np.triu(C,1).T;Ds,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(Ds,1e-20));invC=B@np.diag(1/D)@B.T
                except: return
            if sigma*np.max(D)<1e-12: return

    base=4+int(3*np.log(dim)); r=0; large_pop=base
    while time_left()>0.8:
        if r==0: x0=best_params.copy();sig=0.25*np.mean(ranges);ps=base
        elif r%3==0: large_pop=min(large_pop*2,512);ps=large_pop;x0=clip(archive[min(r//3,len(archive)-1)][0]+np.random.randn(dim)*0.3*ranges);sig=0.4*np.mean(ranges)
        elif r%3==1: ps=max(int(base*0.5*(0.5+np.random.random())),4);x0=clip(best_params+np.random.randn(dim)*0.04*ranges);sig=0.06*np.mean(ranges)
        else: ps=base;x0=clip(np.array([np.random.uniform(l,h) for l,h in bounds]));sig=0.3*np.mean(ranges)
        cma_es(x0,sig,ps)
        if time_left()>1.0 and r%2==0: nelder_mead(best_params,scale=0.02)
        r+=1
    return best
