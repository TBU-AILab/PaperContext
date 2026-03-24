#Looking at the results, algorithm #5 (score 10.73) is the best. Let me analyze the key issues:
#
#1. **IPOP lambda grows for ALL restarts after r>5** - should only grow for global restarts
#2. **Local restart probability (50%) is too low** - need more exploitation
#3. **Missing: proper BIPOP** with separate budgets for local/global
#4. **Stagnation history window** could catch flat landscapes better
#5. **Missing: injection of best solution** into CMA-ES mean periodically
#6. **diagC normalization** could be better tuned
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

    # Track diverse elite set
    sorted_idx = np.argsort(fits)
    elite_points = [pop[sorted_idx[i]].copy() for i in range(min(8, n_init))]
    elite_fits = [fits[sorted_idx[i]] for i in range(min(8, n_init))]

    def sep_cma_es(x0, sigma0, lam_override=None, max_gen_override=None):
        nonlocal best, best_params
        n = dim
        base_lam = 4 + int(3 * np.log(n))
        lam = lam_override if lam_override else base_lam
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        diagC = (ranges / np.max(ranges))**2
        diagC /= np.mean(diagC)
        
        best_gen = float('inf')
        stag = 0
        gen = 0
        hist = []
        max_gen = max_gen_override if max_gen_override else (100 + 250*n // lam)
        
        while gen < max_gen and remaining() > 0.15:
            sqrtC = np.sqrt(diagC)
            invsqrtC = 1.0 / sqrtC
            arx = mean[None,:] + sigma * np.random.randn(lam, n) * sqrtC[None,:]
            arx = np.clip(arx, lower, upper)
            af = np.array([evaluate(arx[k]) for k in range(lam)])
            if remaining() <= 0:
                return
            idx = np.argsort(af)
            if af[idx[0]] < best_gen - 1e-12:
                best_gen = af[idx[0]]
                stag = 0
            else:
                stag += 1
            hist.append(best_gen)
            om = mean.copy()
            mean = weights @ arx[idx[:mu]]
            d = (mean - om) / sigma
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC * d)
            hs = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mueff)*d
            at = (arx[idx[:mu]] - om) / sigma
            diagC = (1-c1-cmu_val)*diagC + c1*(pc**2 + (1-hs)*cc*(2-cc)*diagC) + cmu_val*(weights @ (at**2))
            diagC = np.clip(diagC, 1e-20, 1e6)
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges))
            gen += 1
            if sigma < 1e-13:
                return
            if stag > 20 + 50*n // lam:
                return
            if len(hist) > 40 and abs(hist[-1] - hist[-30]) < 1e-12*(1+abs(hist[-1])):
                return

    ipop_lam = 4 + int(3 * np.log(dim))
    r = 0
    while remaining() > 0.3:
        if r < min(3, len(elite_points)):
            x0 = elite_points[r].copy()
            sep_cma_es(x0, 0.2 * np.mean(ranges))
        elif np.random.rand() < 0.75 and best_params is not None:
            sc = 0.002 + 0.2 * np.random.rand()
            x0 = best_params + sc * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sep_cma_es(x0, sc * np.mean(ranges))
        else:
            x0 = lower + np.random.rand(dim) * ranges
            ipop_lam = min(int(ipop_lam * 1.5), 8 * (4 + int(3 * np.log(dim))))
            sep_cma_es(x0, 0.3 * np.mean(ranges), lam_override=ipop_lam)
        r += 1

    return best
#
#Key changes from best algorithm:
#- **75% local restarts** (up from 50%) with wider sigma range (0.002-0.2)
#- **IPOP lambda only grows for global restarts** - local restarts always use base lambda
#- **Removed the `lam_factor` leak** where it grew for all restarts after r>5
#- **Elite tracking** from initialization for diverse starting points
#- **Tighter time margins** (0.15s instead of 0.3s) to squeeze more iterations
#- **Better stagnation**: both counter-based and history-window-based detection
#- **Wider local search range**: sigma scale from 0.002 to 0.2 covers both fine-tuning and basin exploration
