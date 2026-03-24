import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libraries).

    Hybrid strategy:
      1) Multi-start sampling (Latin-hypercube-like per coordinate) + random
      2) CMA-ES-like diagonal evolution strategy (very fast, good exploitation)
      3) Periodic coordinate/pattern local search around the incumbent
      4) Stagnation-triggered soft restarts (keeps global best)

    Returns:
      best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- helpers ----------------
    def clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect(x, lo, hi):
        if lo == hi:
            return lo
        # reflect until in range (robust for big steps)
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        if x < lo: x = lo
        if x > hi: x = hi
        return x

    def ensure_reflect(vec):
        out = vec[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = reflect(out[i], lo, hi)
        return out

    def safe_eval(vec):
        try:
            v = func(vec)
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                v = float(v)
                if v != v or v == float("inf") or v == float("-inf"):
                    return float("inf")
                return v
            return float("inf")
        except Exception:
            return float("inf")

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if span[i] <= 0.0:
            span[i] = 1.0

    def rand_vec():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # approximate N(0,1) via sum of uniforms
    def gauss01():
        return (sum(random.random() for _ in range(12)) - 6.0)

    def randn_vec():
        return [gauss01() for _ in range(dim)]

    # Latin-hypercube-ish initializer (fast, good coverage)
    def lhs_batch(n):
        # for each dimension create a random permutation of strata
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for j in range(dim):
                # sample within stratum perms[j][i]
                u = (perms[j][i] + random.random()) / float(n)
                x[j] = lo[j] + u * (hi[j] - lo[j])
            pts.append(x)
        return pts

    # ---------------- initialization ----------------
    # Diagonal ES parameters
    lam = max(10, min(40, 4 + 3 * dim))          # offspring per iteration
    mu = max(2, lam // 2)                        # parents
    # recombination weights (log), normalized
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    # effective mu
    mueff = 1.0 / sum(w*w for w in weights)

    # learning rates for diagonal covariance (CMA-ES style)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chiN = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))

    # initial mean: best from a small batch
    best = float("inf")
    best_x = None

    # spend a bit of time on initial coverage but keep bounded
    init_n = max(lam, min(4 * lam, 12 + 6 * dim))
    init_pts = lhs_batch(init_n // 2) + [rand_vec() for _ in range(init_n - init_n // 2)]
    for x in init_pts:
        if time.time() >= deadline:
            return best
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        # extremely pathological case
        x = rand_vec()
        return safe_eval(x)

    m = best_x[:]  # mean
    # initial sigma based on span
    sigma = 0.25 * (sum(span) / float(dim))
    if sigma <= 1e-12:
        sigma = 1.0

    # diagonal "C" represented by D (stddev per dim)
    D = [1.0 for _ in range(dim)]
    pc = [0.0 for _ in range(dim)]
    ps = [0.0 for _ in range(dim)]

    # local search controls
    last_local = time.time()
    local_interval = 0.10  # seconds
    local_step_rel = 0.12  # relative to span (will shrink)

    # stagnation/restart
    last_best = best
    no_improve = 0

    # ---------------- main loop ----------------
    while time.time() < deadline:
        # generate offspring
        pop = []
        for _ in range(lam):
            z = randn_vec()
            x = [0.0] * dim
            for j in range(dim):
                x[j] = m[j] + sigma * D[j] * z[j]
            x = ensure_reflect(x)
            pop.append((x, z))

        # evaluate
        scored = []
        for x, z in pop:
            if time.time() >= deadline:
                return best
            fx = safe_eval(x)
            scored.append((fx, x, z))
            if fx < best:
                best, best_x = fx, x[:]

        scored.sort(key=lambda t: t[0])

        # recombine mean
        old_m = m[:]
        m = [0.0] * dim
        zmean = [0.0] * dim
        for i in range(mu):
            _, xi, zi = scored[i]
            wi = weights[i]
            for j in range(dim):
                m[j] += wi * xi[j]
                zmean[j] += wi * zi[j]

        # update evolution path ps (using diagonal approx => invsqrtC = 1/D)
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + math.sqrt(cs * (2.0 - cs) * mueff) * (zmean[j])
        ps_norm = math.sqrt(sum(v*v for v in ps))

        # step-size control
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))

        # update pc
        hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0)) < (1.4 + 2.0 / (dim + 1.0)) * chiN) else 0.0
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (m[j] - old_m[j]) / max(1e-12, sigma)

        # diagonal covariance update via D (keep as stddev per coordinate)
        # C_diag <- (1-c1-cmu)*C + c1*pc^2 + cmu*sum(w*(y_i)^2)
        # where y_i = (x_i-old_m)/sigma
        Cdiag = [D[j] * D[j] for j in range(dim)]
        # rank-mu component
        mu_term = [0.0] * dim
        for i in range(mu):
            _, xi, _ = scored[i]
            wi = weights[i]
            for j in range(dim):
                y = (xi[j] - old_m[j]) / max(1e-12, sigma)
                mu_term[j] += wi * (y * y)

        for j in range(dim):
            Cnew = (1.0 - c1 - cmu) * Cdiag[j] + c1 * (pc[j] * pc[j]) + cmu * mu_term[j]
            if Cnew <= 1e-30:
                Cnew = 1e-30
            D[j] = math.sqrt(Cnew)

        # stagnation / restart handling
        if best < last_best - 1e-12:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # soft restart: jump mean near best, reset paths, widen sigma a bit
        if no_improve >= 12:
            no_improve = 0
            m = best_x[:]
            ps = [0.0] * dim
            pc = [0.0] * dim
            # widen sigma moderately to re-explore
            sigma = max(sigma, 0.18 * (sum(span) / float(dim)))
            # reset diagonal scales partially
            for j in range(dim):
                D[j] = max(D[j], 0.35)

        # periodic local search around best (cheap pattern search)
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now
            x0 = best_x[:]
            f0 = best

            # choose a small set of coordinates
            coords = list(range(dim))
            random.shuffle(coords)
            mcoords = min(dim, 10)
            coords = coords[:mcoords]

            # step size shrinks gently over time
            time_frac = (now - t0) / max(1e-12, float(max_time))
            step_rel = max(0.01, local_step_rel * (0.55 ** time_frac))

            for j in coords:
                if time.time() >= deadline:
                    return best
                step = step_rel * span[j]
                if step <= 0.0:
                    continue

                # try +step then -step
                for s in (1.0, -1.0):
                    xt = x0[:]
                    xt[j] = xt[j] + s * step
                    xt = ensure_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft

                # quick refine if improved
                if f0 < best:
                    step2 = 0.35 * step
                    for s in (1.0, -1.0):
                        xt = x0[:]
                        xt[j] = xt[j] + s * step2
                        xt = ensure_reflect(xt)
                        ft = safe_eval(xt)
                        if ft < f0:
                            x0, f0 = xt, ft

            if f0 < best:
                best, best_x = f0, x0[:]
                # pull mean towards improved best to accelerate exploitation
                m = best_x[:]

            # adapt local frequency a bit
            local_interval = clip(local_interval * (0.92 if f0 < best + 1e-12 else 1.03), 0.05, 0.25)

    return best
