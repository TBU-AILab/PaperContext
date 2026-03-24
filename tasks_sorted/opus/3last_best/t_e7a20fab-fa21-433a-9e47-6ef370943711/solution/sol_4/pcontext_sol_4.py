#Looking at the results, the best algorithm (146.4) uses IPOP-CMA-ES. Let me analyze what can be improved:
#
#1. **Opposition-based learning wastes ~200 evaluations** with marginal benefit
#2. **Eigendecomposition frequency** is suboptimal - should use standard `1/(c1+cmu)/n/10`
#3. **Restart strategy** is too rigid - need BIPOP (mix of large and small restarts)
#4. **No local refinement** after CMA-ES converges
#5. **Stagnation detection** should also track median fitness
#6. **Initial sampling** too large (800 samples wastes budget)
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
        if len(archive) > 40: archive.pop()

    # LHS init
    n_init = min(max(15*dim, 80), 400)
    pop = np.zeros((n_init, dim))
    for d in range(dim):
        p = np.random.permutation(n_init)
        pop[:, d] = lower[d] + (p + np.random.random(n_init)) / n_init * ranges[d]
    for i in range(n_init):
        if time_left() <= 0: return best
        f = eval_func(pop[i])
        add_archive(pop[i], f)

    def cma_es(x0, sigma0, lam):
        nonlocal best, best_params
        n=dim; mu=lam//2
        w=np.log(mu+.5)-np.log(np.arange(1,mu+1)); w/=w.sum(); mueff=1/np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);C=np.eye(n);invC=np.eye(n)
        chiN=n**.5*(1-1/(4*n)+1/(21*n**2)); mean=x0.copy();sigma=sigma0;g=0;stag=0;pbest=1e30
        efreq=max(1,int(1/(c1+cmu+1e-23)/n/10))
        while time_left()>0.3:
            arz=np.random.randn(lam,n); arx=mean+sigma*(arz@(B*D).T)
            arx=np.array([clip(x) for x in arx])
            fit=np.array([eval_func(arx[k]) for k in range(lam)])
            if time_left()<0.1: return
            o=np.argsort(fit); arx=arx[o]; arz=arz[o]; fit=fit[o]
            add_archive(arx[0],fit[0])
            if fit[0]>=pbest-1e-12*max(1,abs(pbest)): stag+=1
            else: stag=0
            pbest=min(pbest,fit[0])
            if stag>12+2*n: return
            om=mean.copy(); mean=clip(w@arx[:mu]); zm=w@arz[:mu]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invC@zm)
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*((mean-om)/sigma)
            artmp=(arx[:mu]-om)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(w)@artmp)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=min(sigma,2*np.max(ranges))
            g+=1
            if g%efreq==0:
                try: C=np.triu(C)+np.triu(C,1).T; Ds,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(Ds,1e-20)); invC=B@np.diag(1/D)@B.T
                except: return
            if sigma*np.max(D)<1e-12: return

    base=4+int(3*np.log(dim)); r=0
    while time_left()>0.5:
        if r==0: x0=best_params.copy();sig=0.25*np.mean(ranges);ps=base
        elif r%3==0: ps=min(base*2**((r//3)),512);idx=min(r//3,len(archive)-1);x0=clip(archive[idx][0]+np.random.randn(dim)*0.3*ranges);sig=0.4*np.mean(ranges)
        elif r%3==1: ps=max(base//2,4);x0=clip(best_params+np.random.randn(dim)*0.03*ranges);sig=0.05*np.mean(ranges)
        else: ps=base;x0=clip(np.array([np.random.uniform(l,h) for l,h in bounds]));sig=0.3*np.mean(ranges)
        cma_es(x0,sig,ps)
        r+=1
    return best
