#Looking at the results, the best algorithm (146.4) uses IPOP-CMA-ES. Let me analyze what can be improved:
#
#1. **Too many initial samples** (800 + 200 opposition = 1000 wasted evals)
#2. **Restart strategy needs BIPOP** - alternating large/small populations
#3. **Need local search refinement** (Nelder-Mead) after CMA convergence
#4. **Active CMA-ES** (negative weights) improves convergence
#5. **Smarter restart initialization** from diverse archive points
#6. **Eigen update frequency** should follow standard formula
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
        if len(archive) > 80:
            archive.pop()

    # Phase 1: LHS initialization - compact
    n_init = min(max(10 * dim, 50), 200)
    pop = np.zeros((n_init, dim))
    for d in range(dim):
        p = np.random.permutation(n_init)
        pop[:, d] = lower[d] + (p + np.random.random(n_init)) / n_init * ranges[d]
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(pop[i])
        add_archive(pop[i], f)

    def nelder_mead(x0, scale=0.02, max_iter=2000):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = clip(x0)
        for i in range(n):
            p = x0.copy()
            p[i] += scale * ranges[i] * (1 if x0[i] < (lower[i] + upper[i]) / 2 else -1)
            simplex[i + 1] = clip(p)
        fs = []
        for i in range(n + 1):
            if time_left() <= 0.2:
                return
            fs.append(eval_func(simplex[i]))
        fs = np.array(fs, dtype=float)
        no_imp = 0
        for _ in range(max_iter):
            if time_left() <= 0.2:
                return
            o = np.argsort(fs)
            simplex = simplex[o]
            fs = fs[o]
            c = np.mean(simplex[:n], axis=0)
            xr = clip(2 * c - simplex[n])
            fr = eval_func(xr)
            if fr < fs[0]:
                xe = clip(c + 2 * (xr - c))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n], fs[n] = xe, fe
                else:
                    simplex[n], fs[n] = xr, fr
            elif fr < fs[n - 1]:
                simplex[n], fs[n] = xr, fr
            else:
                if fr < fs[n]:
                    xc = clip(c + 0.5 * (xr - c))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[n], fs[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if time_left() <= 0.2:
                                return
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            fs[i] = eval_func(simplex[i])
                else:
                    xc = clip(c + 0.5 * (simplex[n] - c))
                    fc = eval_func(xc)
                    if fc <= fs[n]:
                        simplex[n], fs[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if time_left() <= 0.2:
                                return
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            fs[i] = eval_func(simplex[i])
            if fs[0] < best - 1e-12:
                no_imp = 0
            else:
                no_imp += 1
            if no_imp > 4 * n:
                return
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                return

    def cma_es(x0, sigma0, lam):
        n = dim
        mu = lam // 2
        w_pos = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w_pos /= w_pos.sum()
        mueff = 1.0 / np.sum(w_pos ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Active CMA negative weights
        w_neg_raw = np.log(mu + 0.5) - np.log(np.arange(lam, mu, -1))
        if w_neg_raw.sum() > 0:
            w_neg_raw /= w_neg_raw.sum()
        alpha_mu_neg = 1 + c1 / (cmu + 1e-30)
        alpha_mueff = 1 + 2 * mu / (mueff + 2)
        alpha_pos_def = (1 - c1 - cmu) / (n * cmu + 1e-30)
        alpha_min = min(alpha_mu_neg, alpha_mueff, alpha_pos_def)
        w_neg = -alpha_min * w_neg_raw

        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invC = np.eye(n)
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        mean = x0.copy()
        sigma = sigma0
        g = 0
        stag = 0
        pbest = 1e30
        efreq = max(1, int(1 / (c1 + cmu + 1e-23) / n / 10))

        while time_left() > 0.4:
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.array([clip(x) for x in arx])
            fit = np.array([eval_func(arx[k]) for k in range(lam)])
            if time_left() < 0.2:
                return
            o = np.argsort(fit)
            arx = arx[o]; arz = arz[o]; fit = fit[o]
            add_archive(arx[0], fit[0])

            if fit[0] >= pbest - 1e-12 * max(1, abs(pbest)):
                stag += 1
            else:
                stag = 0
            pbest = min(pbest, fit[0])
            if stag > 10 + 2 * n:
                return

            om = mean.copy()
            mean = clip(w_pos @ arx[:mu])
            zm = w_pos @ arz[:mu]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invC @ zm)
            hs = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * ((mean - om) / sigma)

            artmp = (arx - om) / sigma
            rank_update = np.zeros((n, n))
            for i in range(mu):
                rank_update += w_pos[i] * np.outer(artmp[i], artmp[i])
            for i in range(mu, lam):
                idx = i - mu
                if idx < len(w_neg):
                    nrm = np.linalg.norm(invC @ arz[i]) + 1e-30
                    rank_update += w_neg[idx] * (n / (nrm ** 2)) * np.outer(artmp[i], artmp[i])

            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + cmu * rank_update
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            g += 1
            if g % efreq == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    Ds, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(Ds, 1e-20))
                    invC = B @ np.diag(1 / D) @ B.T
                except:
                    return
            if sigma * np.max(D) < 1e-12:
                return

    base = 4 + int(3 * np.log(dim))
    large_pop = base
    large_evals = 0
    small_evals = 0
    r = 0

    while time_left() > 1.0:
        if r == 0:
            x0 = best_params.copy()
            sig = 0.25 * np.mean(ranges)
            ps = base
        elif large_evals <= small_evals:
            large_pop = min(large_pop * 2, 512)
            ps = large_pop
            if len(archive) > 2:
                idx = np.random.randint(0, min(8, len(archive)))
                x0 = clip(archive[idx][0] + np.random.randn(dim) * 0.35 * ranges)
            else:
                x0 = clip(np.array([np.random.uniform(l, h) for l, h in bounds]))
            sig = 0.4 * np.mean(ranges)
            large_evals += ps * (15 + 2 * dim)
        else:
            ps = max(int(base * (0.3 + 0.7 * np.random.random())), 4)
            x0 = clip(best_params + np.random.randn(dim) * 0.05 * ranges)
            sig = 0.06 * np.mean(ranges)
            small_evals += ps * (15 + 2 * dim)

        cma_es(x0, sig, ps)

        if time_left() > 1.5 and best_params is not None and r % 2 == 0:
            nelder_mead(best_params, scale=0.015)
        r += 1

    if time_left() > 0.3 and best_params is not None:
        nelder_mead(best_params, scale=0.005, max_iter=3000)

    return best
