#Looking at the results, algorithm #2 performed best (81.55). Let me analyze what worked and what can be improved:
#
#1. **Separable CMA-ES is good** - avoids expensive eigendecomposition
#2. **Too many initial samples** - wastes budget (up to 800 points)
#3. **Opposition-based learning** wastes evaluations
#4. **Stagnation detection too aggressive** - stopping too early
#5. **Need better time allocation** - more time for CMA-ES restarts
#6. **Local search at end is wasteful** for high-dim (coordinate-wise is O(dim) per pass)
#7. **Missing: adaptive population sizing (IPOP), better sigma initialization per dimension**
#
#Key improvements:
#- Reduce initialization to ~5*dim
#- Use IPOP-style increasing population on restarts
#- Per-dimension sigma initialization based on ranges
#- Better restart diversification
#- More generous stagnation tolerance
#- Replace coordinate search with random directional search
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
    n_init = min(max(5 * dim, 40), 300)
    pop = np.zeros((n_init, dim))
    fits = np.zeros(n_init)
    for i in range(n_init):
        if remaining() <= 0:
            return best
        pop[i] = lower + np.random.rand(dim) * ranges
        fits[i] = evaluate(pop[i])

    # Phase 2: Sep-CMA-ES with IPOP restarts
    def sep_cma_es(x0, sigma0, lam_factor=1):
        nonlocal best, best_params
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        lam = int(base_lam * lam_factor)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n)
        cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff)
        cmu_val = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        chiN = n**0.5*(1-1/(4*n)+1/(21*n**2))
        mean = x0.copy(); sigma = sigma0
        pc = np.zeros(n); ps = np.zeros(n); diagC = (ranges/np.max(ranges))**2
        diagC /= np.mean(diagC)
        best_gen = float('inf'); stag = 0; gen = 0
        max_gen = 100 + 200*n//lam
        while gen < max_gen and remaining() > 0.3:
            sqrtC = np.sqrt(diagC)
            arx = mean[None,:] + sigma * np.random.randn(lam, n) * sqrtC[None,:]
            arx = np.clip(arx, lower, upper)
            af = np.array([evaluate(arx[k]) for k in range(lam)])
            if remaining() <= 0: return
            idx = np.argsort(af)
            if af[idx[0]] < best_gen - 1e-10: best_gen = af[idx[0]]; stag = 0
            else: stag += 1
            om = mean.copy(); mean = weights @ arx[idx[:mu]]
            d = (mean - om) / sigma
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * d / sqrtC
            hs = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1)))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mueff)*d
            at = (arx[idx[:mu]] - om) / sigma
            diagC = (1-c1-cmu_val)*diagC + c1*(pc**2+(1-hs)*cc*(2-cc)*diagC) + cmu_val*(weights@(at**2))
            diagC = np.clip(diagC, 1e-20, 1e6)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges))
            gen += 1
            if sigma < 1e-12 or stag > 20 + 40*n//lam: return

    sorted_idx = np.argsort(fits)
    lam_factor = 1.0
    r = 0
    while remaining() > 0.5:
        if r < 3:
            x0 = pop[sorted_idx[r]].copy(); s0 = 0.2*np.mean(ranges)
        elif best_params is not None and np.random.rand() < 0.5:
            sc = 0.05 + 0.1*np.random.rand()
            x0 = best_params + sc*ranges*np.random.randn(dim)
            x0 = np.clip(x0, lower, upper); s0 = sc*np.mean(ranges)
        else:
            x0 = lower + np.random.rand(dim)*ranges; s0 = 0.3*np.mean(ranges)
            lam_factor = min(lam_factor * 2, 10)
        sep_cma_es(x0, s0, lam_factor if r >= 3 else 1.0)
        r += 1
        if r > 5: lam_factor = min(lam_factor * 1.5, 10)

    return best
