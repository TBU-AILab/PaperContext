#Looking at the results, the best algorithm (146.4) uses IPOP-CMA-ES. Let me analyze what can be improved:
#
#1. **Opposition-based learning wastes evaluations** - the 200 opposition evals don't help much
#2. **Initial sampling too large** (800 samples) - reduce to leave more budget for CMA-ES
#3. **BIPOP strategy** - alternate between large population (exploration) and small population (exploitation)
#4. **Local search (Nelder-Mead)** after CMA-ES for fine-tuning
#5. **Weighted recombination with negative weights** (active CMA-ES)
#6. **Better restart initialization** from archive with diversity
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
    eval_count = [0]

    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def add_archive(x, f):
        archive.append((x.copy(), f))
        archive.sort(key=lambda t: t[1])
        if len(archive) > 60:
            archive.pop()

    # Phase 1: Sobol-like LHS initialization - moderate size
    n_init = min(max(10 * dim, 50), 250)
    pop = np.zeros((n_init, dim))
    for d in range(dim):
        p = np.random.permutation(n_init)
        pop[:, d] = lower[d] + (p + np.random.random(n_init)) / n_init * ranges[d]
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(pop[i])
        add_archive(pop[i], f)

    # Nelder-Mead local search
    def nelder_mead(x0, scale=0.03, max_iters=3000):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = clip(x0)
        for i in range(n):
            p = x0.copy()
            delta = scale * ranges[i]
            p[i] += delta if x0[i] < (lower[i] + upper[i]) / 2 else -delta
            simplex[i + 1] = clip(p)
        fs = []
        for i in range(n + 1):
            if time_left() <= 0.15:
                return
            fs.append(eval_func(simplex[i]))
        fs = np.array(fs)

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        no_improve = 0
        for _ in range(max_iters):
            if time_left() <= 0.15:
                return
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            centroid = np.mean(simplex[:n], axis=0)

            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_func(xr)

            if fr < fs[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n], fs[n] = xe, fe
                else:
                    simplex[n], fs[n] = xr, fr
            elif fr < fs[n - 1]:
                simplex[n], fs[n] = xr, fr
            else:
                if fr < fs[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[n], fs[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if time_left() <= 0.15:
                                return
                            simplex[i] = clip(simplex[0] + shrink * (simplex[i] - simplex[0]))
                            fs[i] = eval_func(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                    fc = eval_func(xc)
                    if fc <= fs[n]:
                        simplex[n], fs[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if time_left() <= 0.15:
                                return
                            simplex[i] = clip(simplex[0] + shrink * (simplex[i] - simplex[0]))
                            fs[i] = eval_func(simplex[i])

            spread = np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30))
            if spread < 1e-15:
                return
            if fs[0] < best - 1e-12:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 5 * n:
                return

    # Active CMA-ES with mirrored sampling
    def cma_es(x0, sigma0, lam):
        n = dim
        mu = lam // 2
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_w / raw_w.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Active CMA: negative weights for worst solutions
        raw_neg = np.log(mu + 0.5) - np.log(np.arange(lam, mu, -1))
        neg_w = -raw_neg / raw_neg.sum()
        mueff_neg = 1.0 / np.sum(neg_w ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Scale negative weights
        alpha_neg = min(1 + c1 / (cmu + 1e-30), 1 + (2 * mueff_neg) / (mueff + 2))
        full_w = np.concatenate([weights, alpha_neg * neg_w])

        pc = np.zeros(n); ps = np.zeros(n)
        B = np.eye(n); D = np.ones(n); C = np.eye(n); invC = np.eye(n)
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        mean = x0.copy(); sigma = sigma0; g = 0; stag = 0; pbest = 1e30
        efreq = max(1, int(1 / (c1 + cmu + 1e-23) / n / 10))
        hist_best = []

        while time_left() > 0.3:
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.array([clip(x) for x in arx])
            fit = np.array([eval_func(arx[k]) for k in range(lam)])
            if time_left() < 0.15:
                return
            o = np.argsort(fit); arx = arx[o]; arz = arz[o]; fit = fit[o]
            add_archive(arx[0], fit[0])
            hist_best.append(fit[0])

            if fit[0] >= pbest - 1e-12 * max(1, abs(pbest)):
                stag += 1
            else:
                stag = 0
            pbest = min(pbest, fit[0])
            if stag > 12 + 2 * n:
                return

            om = mean.copy()
            mean = clip(weights @ arx[:mu])
            zm = weights @ arz[:mu]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invC @ zm)
            hs = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * ((mean - om) / sigma)

            artmp = (arx - om) / sigma
            pos_part = c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
            w_all = np.concatenate([weights, alpha_neg * neg_w])
            rank_mu = np.zeros((n, n))
            for i in range(lam):
                wi = w_all[i] if i < mu else w_all[i]
                if wi < 0:
                    nrm = np.linalg.norm(invC @ arz[i]) + 1e-30
                    wi_adj = wi * n / (nrm ** 2)
                else:
                    wi_adj = wi
                rank_mu += wi_adj * np.outer(artmp[i], artmp[i])

            C = (1 - c1 - cmu) * C + pos_part + cmu * rank_mu
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
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
    r = 0
    total_large_evals = 0
    total_small_evals = 0

    while time_left() > 0.8:
        if r == 0:
            x0 = best_params.copy(); sig = 0.25 * np.mean(ranges); ps = base
        elif total_large_evals <= total_small_evals:
            # Large restart (exploration)
            large_pop = min(large_pop * 2, 512)
            ps = large_pop
            if len(archive) > 1:
                idx = np.random.randint(0, min(5, len(archive)))
                x0 = clip(archive[idx][0] + np.random.randn(dim) * 0.3 * ranges)
            else:
                x0 = clip(np.array([np.random.uniform(l, h) for l, h in bounds]))
            sig = 0.4 * np.mean(ranges)
            total_large_evals += ps * (12 + 2 * dim)
        else:
            # Small restart (exploitation)
            ps = max(int(base * (0.5 + 0.5 * np.random.random())), 4)
            x0 = clip(best_params + np.random.randn(dim) * 0.05 * ranges)
            sig = 0.06 * np.mean(ranges)
            total_small_evals += ps * (12 + 2 * dim)

        ec_before = eval_count[0]
        cma_es(x0, sig, ps)

        # Local refinement after each CMA run
        if time_left() > 1.0 and best_params is not None:
            nelder_mead(best_params, scale=0.015)
        r += 1

    # Final polish
    if time_left() > 0.3 and best_params is not None:
        nelder_mead(best_params, scale=0.005, max_iters=5000)

    return best
