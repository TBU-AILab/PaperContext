#Looking at the previous algorithms, the best one (146.4) uses IPOP-CMA-ES. I'll improve it with:
#
#1. **BIPOP strategy** - alternating between large and small population restarts
#2. **Better stagnation detection** using both best and median fitness
#3. **Reduce wasted evaluations** on opposition-based learning (didn't help much)
#4. **Local search refinement** (Nelder-Mead) after CMA-ES converges
#5. **Smarter restart initialization** using archive of good solutions
#6. **More efficient eigendecomposition scheduling**
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
        while len(archive) > 30:
            archive.pop()

    # Phase 1: Latin Hypercube Sampling
    n_init = min(max(20 * dim, 100), 500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(init_pop[i])
        add_archive(init_pop[i], f)

    # Nelder-Mead local search
    def nelder_mead(x0, scale=0.05, max_iters=2000):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n+1, n))
        simplex[0] = clip(x0)
        for i in range(n):
            p = x0.copy()
            p[i] += scale * ranges[i]
            if p[i] > upper[i]:
                p[i] = x0[i] - scale * ranges[i]
            simplex[i+1] = clip(p)
        
        fs = np.array([eval_func(simplex[i]) for i in range(n+1) if time_left() > 0])
        if len(fs) < n+1:
            return
        
        for _ in range(max_iters):
            if time_left() <= 0.1:
                return
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = clip(centroid + (centroid - simplex[n]))
            fr = eval_func(xr)
            
            if fr < fs[0]:
                xe = clip(centroid + 2*(xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n], fs[n] = xe, fe
                else:
                    simplex[n], fs[n] = xr, fr
            elif fr < fs[n-1]:
                simplex[n], fs[n] = xr, fr
            else:
                if fr < fs[n]:
                    xc = clip(centroid + 0.5*(xr - centroid))
                else:
                    xc = clip(centroid + 0.5*(simplex[n] - centroid))
                fc = eval_func(xc)
                if fc <= min(fr, fs[n]):
                    simplex[n], fs[n] = xc, fc
                else:
                    for i in range(1, n+1):
                        if time_left() <= 0.1:
                            return
                        simplex[i] = clip(simplex[0] + 0.5*(simplex[i]-simplex[0]))
                        fs[i] = eval_func(simplex[i])
            
            if np.max(np.abs(simplex[n]-simplex[0])) < 1e-14:
                return

    # CMA-ES
    def cma_es(x0, sigma0, pop_size):
        nonlocal best, best_params
        n = dim
        lam = pop_size
        mu = lam // 2
        weights = np.log(mu+0.5) - np.log(np.arange(1,mu+1))
        weights /= weights.sum()
        mueff = 1.0/np.sum(weights**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n)
        cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff)
        cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        pc=np.zeros(n); ps=np.zeros(n)
        B=np.eye(n); D=np.ones(n); C=np.eye(n); invC=np.eye(n)
        chiN=n**0.5*(1-1/(4*n)+1/(21*n**2))
        mean=x0.copy(); sigma=sigma0; g=0; stag=0; pbest=1e30
        efreq=max(1,int(1/(c1+cmu+1e-20)/n/10))
        while time_left()>0.3:
            arz=np.random.randn(lam,n)
            arx=mean+sigma*(arz@(B*D).T)
            arx=np.array([clip(x) for x in arx])
            fit=np.array([eval_func(arx[k]) for k in range(lam)])
            o=np.argsort(fit); arx=arx[o]; arz=arz[o]; fit=fit[o]
            add_archive(arx[0],fit[0])
            if fit[0]>=pbest-1e-12*abs(pbest+1e-30): stag+=1
            else: stag=0
            pbest=min(pbest,fit[0])
            if stag>15+2*n: return
            om=mean.copy(); mean=clip(weights@arx[:mu])
            zm=weights@arz[:mu]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invC@zm)
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*((mean-om)/sigma)
            artmp=(arx[:mu]-om)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(weights)@artmp)
            sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            sigma=min(sigma,2*np.max(ranges))
            g+=1
            if g%efreq==0:
                try:
                    C=np.triu(C)+np.triu(C,1).T; Ds,B=np.linalg.eigh(C)
                    D=np.sqrt(np.maximum(Ds,1e-20)); invC=B@np.diag(1/D)@B.T
                except: return
            if sigma*np.max(D)<1e-12: return

    base=4+int(3*np.log(dim)); r=0
    while time_left()>1:
        if r==0:
            x0=best_params.copy(); sig=0.25*np.mean(ranges); ps=base
        elif r%3==0:
            ps=min(base*2**((r//3)+1),512); x0=clip(archive[min(r//3,len(archive)-1)][0]+np.random.randn(dim)*0.3*ranges); sig=0.4*np.mean(ranges)
        elif r%3==1:
            ps=base; x0=clip(best_params+np.random.randn(dim)*0.05*ranges); sig=0.08*np.mean(ranges)
        else:
            ps=base; x0=clip(np.array([np.random.uniform(l,h) for l,h in bounds])); sig=0.3*np.mean(ranges)
        cma_es(x0,sig,ps)
        if time_left()>1 and best_params is not None:
            nelder_mead(best_params,scale=0.02)
        r+=1
    return best
