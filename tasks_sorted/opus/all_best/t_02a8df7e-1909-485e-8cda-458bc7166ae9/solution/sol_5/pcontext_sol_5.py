#Looking at the best algorithm (score 10.73), I can identify several improvements:
#
#1. **diagC initialization using ranges is good** but the ps update dividing by sqrtC is incorrect for sep-CMA-ES (should use invsqrtC properly)
#2. **IPOP lambda factor grows too fast** - wastes evaluations on huge populations
#3. **Need more local restarts near best** - the 50% random restart rate is too high
#4. **Stagnation threshold could be more adaptive**
#5. **Missing: warm restart from best with tiny sigma for fine-tuning**
#6. **Better time allocation** - reserve time for local refinement
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
    n_init = min(max(5 * dim, 40), 250)
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
        lam = max(base_lam, int(base_lam * lam_factor))
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
        diagC = (ranges/np.max(ranges))**2
        diagC /= np.mean(diagC)
        best_gen = float('inf'); stag = 0; gen = 0
        max_gen = 100 + 300*n//lam
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
            if sigma < 1e-13 or stag > 25 + 50*n//lam: return

    sorted_idx = np.argsort(fits)
    lam_factor = 1.0
    r = 0
    while remaining() > 0.5:
        if r < 3:
            x0 = pop[sorted_idx[r % len(sorted_idx)]].copy(); s0 = 0.2*np.mean(ranges)
            lf = 1.0
        elif best_params is not None and np.random.rand() < 0.7:
            sc = 0.02 + 0.15*np.random.rand()
            x0 = best_params + sc*ranges*np.random.randn(dim)
            x0 = np.clip(x0, lower, upper); s0 = sc*np.mean(ranges); lf = 1.0
        else:
            x0 = lower + np.random.rand(dim)*ranges; s0 = 0.3*np.mean(ranges)
            lam_factor = min(lam_factor * 1.5, 6)
            lf = lam_factor
        sep_cma_es(x0, s0, lf)
        r += 1

    return best
