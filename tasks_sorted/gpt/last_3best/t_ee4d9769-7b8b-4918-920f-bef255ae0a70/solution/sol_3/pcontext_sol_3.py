import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained).

    Improvements vs provided code:
    - Uses CMA-ES style adaptation (diagonal covariance) for strong local search without gradients.
    - Multi-start restarts with increasing population when stalled (IPOP-like) for robustness.
    - Elitist selection + mirrored sampling (antithetic) to reduce variance per evaluation.
    - Strict time checks; returns best fitness found.

    Notes:
    - Works with plain Python lists; no numpy required.
    - Handles bounds with reflection (good for Gaussian sampling).
    """

    start = time.time()
    deadline = start + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # avoid zero-span issues
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def now():
        return time.time()

    def reflect_scalar(v, i):
        lo, hi = lows[i], highs[i]
        if lo == hi:
            return lo
        # reflect until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            if v > hi:
                v = hi - (v - hi)
        # safety
        if v < lo: v = lo
        if v > hi: v = hi
        return v

    def reflect_vec(x):
        return [reflect_scalar(x[i], i) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # LHS-like init (cheap, decent coverage)
    def lhs_points(n):
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for j in range(dim):
                u = (perms[j][k] + random.random()) / n
                x[j] = lows[j] + u * spans[j]
            pts.append(x)
        return pts

    # ---------- initial best ----------
    best = float("inf")
    best_x = None

    # Spend a small fraction on diversified init, but keep it time-safe
    init_n = max(8, min(40, 6 * dim + 8))
    for x in lhs_points(init_n):
        if now() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_point()
        best = eval_f(best_x)

    # ---------- diagonal CMA-ES core ----------
    # We implement a simplified (diagonal covariance) CMA-ES for speed and self-containment.
    # State: mean m, global step sigma, per-dim scaling D (sqrt of variances).
    def cma_run(m0, f0, sigma0, pop_mult, time_slice):
        nonlocal best, best_x

        t_end = min(deadline, now() + time_slice)

        # strategy parameters
        n = dim
        # population
        lam = max(8, int(pop_mult * (4 + int(3 * math.log(n + 1.0)))))
        if lam % 2 == 1:
            lam += 1  # to support antithetic pairing
        mu = lam // 2

        # recombination weights (log)
        weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        mueff = 1.0 / sum(w * w for w in weights)

        # learning rates (diagonal)
        # conservative but robust
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        # expectation of ||N(0,I)||
        chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        m = m0[:]
        sigma = float(sigma0)
        # diag std multipliers
        D = [1.0] * n

        # evolution paths (diagonal)
        pc = [0.0] * n
        ps = [0.0] * n

        # best in this run
        if f0 < best:
            best, best_x = f0, m0[:]

        # stall tracking to trigger exit early inside slice
        last_improve = now()
        stall_limit = max(0.15 * time_slice, 0.08)

        # generation loop
        while now() < t_end:
            # sample population
            pop = []   # list of (fitness, x, z) where z ~ N(0,I)
            # antithetic sampling: z and -z
            half = lam // 2

            for k in range(half):
                if now() >= t_end:
                    break

                z = [random.gauss(0.0, 1.0) for _ in range(n)]
                # candidate 1
                x1 = [0.0] * n
                for i in range(n):
                    x1[i] = m[i] + sigma * D[i] * z[i]
                x1 = reflect_vec(x1)
                f1 = eval_f(x1)
                pop.append((f1, x1, z))

                if f1 < best:
                    best, best_x = f1, x1[:]
                    last_improve = now()

                if now() >= t_end:
                    break

                # candidate 2 (mirrored)
                x2 = [0.0] * n
                for i in range(n):
                    x2[i] = m[i] - sigma * D[i] * z[i]
                x2 = reflect_vec(x2)
                f2 = eval_f(x2)
                pop.append((f2, x2, [-zi for zi in z]))

                if f2 < best:
                    best, best_x = f2, x2[:]
                    last_improve = now()

            if len(pop) < mu:
                break

            pop.sort(key=lambda t: t[0])

            # recombine to new mean in z-space (and x-space)
            old_m = m[:]
            m = [0.0] * n
            z_w = [0.0] * n
            for j in range(mu):
                w = weights[j]
                fj, xj, zj = pop[j]
                for i in range(n):
                    m[i] += w * xj[i]
                    z_w[i] += w * zj[i]

            # update evolution path for sigma (ps)
            # since diag: (C^-1/2)(m-old_m)/sigma == z_w elementwise / D? actually:
            # m - old_m ≈ sigma * D * z_w => (m-old_m)/(sigma*D) ≈ z_w
            for i in range(n):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * z_w[i]

            # compute norm(ps)
            ps_norm = math.sqrt(sum(v * v for v in ps))

            # hsig
            # using standard criterion
            t = ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * 1.0))  # approx, gen counter omitted
            hsig = 1.0 if (t / chiN) < (1.4 + 2.0 / (n + 1.0)) else 0.0

            # update evolution path for covariance (pc) in x-space:
            # (m-old_m)/sigma elementwise scaled by 1/D? for pc we need in "C^(−1/2)"? In diag,
            # we can keep pc in "normalized by D" space and apply back through D for update.
            # We'll update pc in z-space (like ps) but separate scaling.
            for i in range(n):
                pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (D[i] * z_w[i])

            # diagonal covariance update: D^2
            # Convert D -> var diag (v_i = D_i^2)
            vdiag = [Di * Di for Di in D]

            # rank-one contribution (pc^2)
            for i in range(n):
                vdiag[i] = (1.0 - c1 - cmu) * vdiag[i] + c1 * (pc[i] * pc[i])

            # rank-mu contribution (sum w * (D*z)^2)
            for j in range(mu):
                w = weights[j]
                _, _, zj = pop[j]
                for i in range(n):
                    dz = D[i] * zj[i]
                    vdiag[i] += cmu * w * (dz * dz)

            # safeguard / rebuild D
            for i in range(n):
                if vdiag[i] <= 1e-30:
                    vdiag[i] = 1e-30
                D[i] = math.sqrt(vdiag[i])

            # update sigma
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))

            # limit sigma to reasonable portion of domain to avoid wasting evaluations
            max_sigma = 0.5 * max(spans) if spans else 1.0
            min_sigma = 1e-15
            if sigma > max_sigma:
                sigma = max_sigma
            if sigma < min_sigma:
                sigma = min_sigma

            # stop early within slice if no improvements for a while and sigma is small
            if (now() - last_improve) > stall_limit and sigma < 1e-6 * (sum(spans) / max(1, n)):
                break

        # return best of this run (global is updated already)
        return best, best_x

    # ---------- scheduler: alternating global restarts + CMA slices ----------
    # Use repeated short slices to stay responsive to time limit.
    # Increase pop_mult on restarts to escape local minima.
    restart = 0
    pop_mult = 1.0

    # initial sigma relative to domain
    base_sigma = 0.25 * (sum(spans) / max(1, dim))

    # Keep a current mean (start at best so far)
    m = best_x[:]
    f_m = best

    # small probability of global random injection when time is short/stalled
    last_global_improve = now()

    while True:
        t = now()
        if t >= deadline:
            return best

        remaining = deadline - t

        # if very little time, just do random sampling around best quickly
        if remaining < 0.02:
            # a few quick perturbations
            for _ in range(6):
                if now() >= deadline:
                    break
                x = m[:]
                for i in range(dim):
                    x[i] = reflect_scalar(x[i] + random.gauss(0.0, 0.05 * spans[i]), i)
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x[:]
            return best

        # occasional global random try to avoid pathological traps
        if (t - last_global_improve) > max(0.25, 0.18 * max_time) and random.random() < 0.35:
            xg = rand_point()
            fg = eval_f(xg)
            if fg < best:
                best, best_x = fg, xg[:]
                m, f_m = best_x[:], best
                last_global_improve = now()

        # allocate a slice of time to CMA
        slice_time = min(0.35, max(0.08, 0.22 * remaining))
        # restart logic: if stalled for a while, restart mean somewhere else
        if (t - last_global_improve) > max(0.40, 0.28 * max_time):
            restart += 1
            pop_mult = min(6.0, 1.0 + 0.7 * restart)

            # restart mean: mostly around best, sometimes global
            if random.random() < 0.75:
                m = best_x[:]
                # jitter
                for i in range(dim):
                    m[i] = reflect_scalar(m[i] + random.gauss(0.0, 0.15 * spans[i]), i)
                f_m = eval_f(m)
            else:
                m = rand_point()
                f_m = eval_f(m)

            if f_m < best:
                best, best_x = f_m, m[:]
            last_global_improve = now()  # reset stall timer after restart

        # run CMA slice
        sigma0 = base_sigma * (1.0 + 0.25 * restart)
        if sigma0 < 1e-15:
            sigma0 = 1e-15

        cma_run(m, f_m, sigma0, pop_mult, slice_time)

        # update current mean to global best for continued exploitation
        m = best_x[:]
        f_m = best
        last_global_improve = now() if best == f_m else last_global_improve
