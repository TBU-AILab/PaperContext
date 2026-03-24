#Looking at the best algorithm (81.55), I can identify several improvements:
#
#1. **Too many initial samples** - wastes evaluations on random points
#2. **Separable CMA-ES path update has a bug** - `diff/sqrtC` should use `invsqrtC` properly
#3. **Need differential evolution** as a complementary strategy for better exploration
#4. **Better restart strategy** - use population diversity information
#5. **Nelder-Mead** for local refinement is more efficient than coordinate search
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Compact initialization
    n_init = min(max(10 * dim, 50), 400)
    pop = np.zeros((n_init, dim))
    fits = np.zeros(n_init)
    for i in range(n_init):
        if remaining() <= 0:
            return best
        pop[i] = lower + np.random.rand(dim) * ranges
        fits[i] = evaluate(pop[i])

    # Phase 2: DE/current-to-best with archive
    sorted_idx = np.argsort(fits)
    de_pop_size = min(max(6 * dim, 30), 200)
    de_pop = pop[sorted_idx[:de_pop_size]].copy()
    de_fits = fits[sorted_idx[:de_pop_size]].copy()
    
    archive = []
    F = 0.5
    CR = 0.9
    
    de_gens = 0
    while remaining() > max_time * 0.4 and de_gens < 300:
        if remaining() <= 0:
            return best
        
        best_idx = np.argmin(de_fits)
        new_pop = de_pop.copy()
        new_fits = de_fits.copy()
        
        for i in range(de_pop_size):
            if remaining() <= 0:
                return best
            
            # DE/current-to-pbest/1
            p = max(2, de_pop_size // 5)
            pbest_idx = sorted_idx_de = np.argsort(de_fits)[:p]
            chosen = pbest_idx[np.random.randint(p)]
            
            idxs = [j for j in range(de_pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            if archive and np.random.rand() < 0.5:
                r2_vec = archive[np.random.randint(len(archive))]
            else:
                idxs2 = [j for j in idxs if j != r1]
                r2_vec = de_pop[np.random.choice(idxs2)] if idxs2 else de_pop[r1]
            
            Fi = F + 0.1 * np.random.randn()
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0, 1)
            
            mutant = de_pop[i] + Fi * (de_pop[chosen] - de_pop[i]) + Fi * (de_pop[r1] - r2_vec)
            
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial <= de_fits[i]:
                if f_trial < de_fits[i]:
                    archive.append(de_pop[i].copy())
                    if len(archive) > de_pop_size:
                        archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fits[i] = f_trial
        
        de_pop = new_pop
        de_fits = new_fits
        de_gens += 1

    # Phase 3: Sep-CMA-ES restarts from best
    def sep_cma_es(x0, sigma0):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n); cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff); cmu_val = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN = n**0.5*(1-1/(4*n)+1/(21*n**2))
        mean=x0.copy(); sigma=sigma0; pc=np.zeros(n); ps=np.zeros(n); diagC=np.ones(n)
        stag=0; best_gen=float('inf'); gen=0
        while gen < 200+150*n//lam and remaining() > 0.3:
            sqrtC=np.sqrt(diagC); arx=mean[None,:]+sigma*np.random.randn(lam,n)*sqrtC[None,:]
            arx=np.clip(arx,lower,upper); af=np.array([evaluate(arx[k]) for k in range(lam)])
            if remaining()<=0: return
            idx=np.argsort(af); bf=af[idx[0]]
            if bf<best_gen-1e-10: best_gen=bf; stag=0
            else: stag+=1
            om=mean.copy(); mean=weights@arx[idx[:mu]]; d=(mean-om)/sigma
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*d/sqrtC
            hs=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1)))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*d
            at=(arx[idx[:mu]]-om)/sigma
            diagC=(1-c1-cmu_val)*diagC+c1*(pc**2+(1-hs)*cc*(2-cc)*diagC)+cmu_val*(weights@(at**2))
            diagC=np.clip(diagC,1e-20,1e6)
            sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1)); sigma=np.clip(sigma,1e-16,np.max(ranges))
            gen+=1
            if sigma<1e-12 or stag>20+30*n//lam: return

    r = 0
    while remaining() > 0.5:
        if r == 0 and best_params is not None: x0 = best_params.copy(); s0 = 0.15*np.mean(ranges)
        elif best_params is not None and np.random.rand()<0.6:
            x0=best_params+0.1*ranges*np.random.randn(dim); x0=np.clip(x0,lower,upper); s0=0.1*np.mean(ranges)
        else: x0=lower+np.random.rand(dim)*ranges; s0=0.3*np.mean(ranges)
        sep_cma_es(x0, s0); r += 1

    return best
