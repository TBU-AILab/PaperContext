#Looking at the results, algorithm #4 (score 10.73) is the best. Let me analyze what's working and what needs improvement:
#
#1. **IPOP lambda growth is applied to ALL restarts after r>5**, including local ones - this wastes evaluations
#2. **Local restart probability (50%) is too low** - need more exploitation near best
#3. **diagC initialization with ranges is good** but ps update needs careful handling
#4. **Missing: tracking multiple promising basins** for diversified restarts
#5. **Missing: adaptive sigma for local restarts based on improvement history**
#6. **Stagnation detection could use fitness history window**
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

    # Track top-k diverse basins
    top_k = min(10, n_init)
    sorted_idx = np.argsort(fits)
    basins = [(pop[sorted_idx[i]].copy(), fits[sorted_idx[i]]) for i in range(top_k)]

    def sep_cma_es(x0, sigma0, lam_override=None):
        nonlocal best, best_params
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        lam = lam_override if lam_override else base_lam
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
        pc = np.zeros(n); ps = np.zeros(n)
        diagC = (ranges/np.max(ranges))**2; diagC /= np.mean(diagC)
        best_gen = float('inf'); stag = 0; gen = 0
        hist = []
        max_gen = 100 + 250*n//lam
        while gen < max_gen and remaining() > 0.2:
            sqrtC = np.sqrt(diagC)
            invsqrtC = 1.0 / sqrtC
            arx = mean[None,:] + sigma * np.random.randn(lam, n) * sqrtC[None,:]
            arx = np.clip(arx, lower, upper)
            af = np.array([evaluate(arx[k]) for k in range(lam)])
            if remaining() <= 0: return
            idx = np.argsort(af)
            if af[idx[0]] < best_gen - 1e-12: best_gen = af[idx[0]]; stag = 0
            else: stag += 1
            hist.append(best_gen)
            om = mean.copy(); mean = weights @ arx[idx[:mu]]
            d = (mean - om) / sigma
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC * d)
            hs = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1)))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mueff)*d
            at = (arx[idx[:mu]] - om) / sigma
            diagC = (1-c1-cmu_val)*diagC + c1*(pc**2+(1-hs)*cc*(2-cc)*diagC) + cmu_val*(weights@(at**2))
            diagC = np.clip(diagC, 1e-20, 1e6)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges))
            gen += 1
            if sigma < 1e-13: return
            if stag > 20 + 50*n//lam: return
            if len(hist) > 40 and abs(hist[-1]-hist[-30]) < 1e-12*(1+abs(hist[-1])): return

    ipop_lam = 4 + int(3 * np.log(dim))
    r = 0
    while remaining() > 0.3:
        if r < 3:
            x0 = basins[r % len(basins)][0].copy()
            sep_cma_es(x0, 0.2 * np.mean(ranges))
        elif np.random.rand() < 0.7 and best_params is not None:
            sc = 0.003 + 0.18 * np.random.rand()
            x0 = best_params + sc * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sep_cma_es(x0, sc * np.mean(ranges))
        else:
            x0 = lower + np.random.rand(dim) * ranges
            ipop_lam = min(int(ipop_lam * 1.5), 8 * (4 + int(3 * np.log(dim))))
            sep_cma_es(x0, 0.3 * np.mean(ranges), lam_override=ipop_lam)
        r += 1

    return best
