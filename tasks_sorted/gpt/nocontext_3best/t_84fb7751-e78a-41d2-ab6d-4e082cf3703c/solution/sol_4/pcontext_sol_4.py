import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Core changes vs your best (Alg #1 style):
      - Uses a proper CMA-ES–like update (diagonal covariance, evolution paths).
      - Better anytime behavior: LHS+opposition seeding, then steady-state CMA loop.
      - Restarts with increasing population (IPOP-lite) + occasional heavy-tail injections.
      - Time-adaptive cache quantization to reduce duplicate evaluations without
        over-collapsing early exploration.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

    # ---------- helpers ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Time-adaptive cache: coarse early, finer later (limits duplicate work from clipping/tiny steps)
    cache = {}
    def quant_key(x, q):
        k = []
        for i in range(dim):
            u = (x[i] - lows[i]) / spans[i]
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0
        # q from 2^16 .. 2^26
        q = 1 << (16 + int(10 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # approx N(0,1), cheap
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12: u = 1e-12
        if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------- Seeding: Latin-hypercube-ish + opposition ----------
    best = float("inf")
    best_x = rand_vec()
    best = eval_cached(best_x)

    # modest seeding, time-aware
    seed_n = max(24, 8 * dim)
    seed_n = min(seed_n, 500)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 12))

    strata = seed_n
    perms = []
    for i in range(dim):
        p = list(range(strata))
        random.shuffle(p)
        perms.append(p)

    for k in range(strata):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            a = perms[i][k] / float(strata)
            b = (perms[i][k] + 1) / float(strata)
            u = a + (b - a) * random.random()
            x[i] = lows[i] + u * spans[i]
        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition point
        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # ---------- Diagonal CMA-ES (approximate) with restarts ----------
    # CMA parameters depend on population size (lambda).
    def setup_cma(lam):
        mu = lam // 2
        # log-weights
        ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(ws)
        w = [wi / wsum for wi in ws]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # learning rates (diag variant)
        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

        # diagonal covariance learning rate (simplified)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

        return mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu

    # base population
    lam0 = 8 if dim <= 8 else (12 if dim <= 25 else 16)
    lam0 = min(max(lam0, 6), 32)

    restart = 0
    xmean = best_x[:]

    # initial global sigma (scaled by span)
    base_sigma = 0.25 * (sum(spans) / float(dim))
    if base_sigma <= 0.0:
        base_sigma = 1.0

    # diagonal "covariance" as std factors (D)
    D = [0.30 * spans[i] + 1e-12 for i in range(dim)]  # per-dim scale
    # ensure not too huge
    for i in range(dim):
        D[i] = min(D[i], 0.75 * spans[i] + 1e-12)

    # evolution paths
    ps = [0.0] * dim
    pc = [0.0] * dim

    sigma = base_sigma
    best_local = best
    last_improve_time = time.time()

    while True:
        if time.time() >= deadline:
            return best

        lam = min(64, int(lam0 * (2 ** min(restart, 4))))  # IPOPlite cap
        mu, weights, mueff, c_sigma, d_sigma, c_c, c1, c_mu = setup_cma(lam)

        # reset state per restart around global best
        xmean = best_x[:]
        ps = [0.0] * dim
        pc = [0.0] * dim

        # restart sigma: broad early, narrower later
        frac = (time.time() - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0
        sigma = (0.35 * (1.0 - frac) + 0.06) * (sum(spans) / float(dim))

        # also reset D moderately (keeps anisotropy but not stuck)
        for i in range(dim):
            D[i] = max(1e-12, min(0.65 * spans[i] + 1e-12, D[i] * 0.9 + 0.10 * (0.25 * spans[i] + 1e-12)))

        best_local = best
        stall_gens = 0

        while True:
            if time.time() >= deadline:
                return best

            # sample population
            pop = []
            heavy_prob = 0.03 + (0.04 if dim > 30 else 0.0)
            for _ in range(lam):
                if time.time() >= deadline:
                    return best

                z = [randn() for _ in range(dim)]

                # occasional heavy-tailed mutation for escape (in-place on z)
                if random.random() < heavy_prob:
                    k = 1 if dim <= 5 else max(1, dim // 3)
                    idxs = random.sample(range(dim), k) if k < dim else range(dim)
                    for j in idxs:
                        z[j] += 0.35 * cauchy()

                x = [0.0] * dim
                for i in range(dim):
                    x[i] = xmean[i] + sigma * (D[i] * z[i])
                clip_inplace(x)
                fx = eval_cached(x)
                pop.append((fx, x, z))

            pop.sort(key=lambda t: t[0])
            if pop[0][0] < best:
                best = pop[0][0]
                best_x = pop[0][1][:]
                last_improve_time = time.time()

            if pop[0][0] < best_local - 1e-15:
                best_local = pop[0][0]
                stall_gens = 0
            else:
                stall_gens += 1

            # recombination
            old_mean = xmean[:]
            xmean = [0.0] * dim
            zmean = [0.0] * dim
            for j in range(mu):
                w = weights[j]
                xj, zj = pop[j][1], pop[j][2]
                for i in range(dim):
                    xmean[i] += w * xj[i]
                    zmean[i] += w * zj[i]

            # update sigma path ps (diag approximation: treat zmean as whitened direction)
            cs = c_sigma
            for i in range(dim):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * zmean[i]

            # sigma adaptation (use ||ps||)
            norm_ps = math.sqrt(sum(p * p for p in ps))
            # expected norm of N(0,I)
            chi_n = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))
            sigma *= math.exp((cs / d_sigma) * (norm_ps / chi_n - 1.0))

            # update covariance path pc
            cc = c_c
            hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (1.0 + 1.0)))) < (1.4 + 2.0/(dim+1.0)) * chi_n else 0.0
            for i in range(dim):
                pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (xmean[i] - old_mean[i]) / max(1e-18, sigma)

            # diagonal covariance update (update D as stddev factors)
            # We maintain v = D^2 implicitly by updating D multiplicatively.
            # Use a stable positive update on log-scale-ish:
            c1_eff = c1
            c_mu_eff = c_mu

            # rank-mu contribution: weighted z^2
            dz2 = [0.0] * dim
            for j in range(mu):
                w = weights[j]
                zj = pop[j][2]
                for i in range(dim):
                    dz2[i] += w * (zj[i] * zj[i])

            for i in range(dim):
                # target variance factor
                pc2 = pc[i] * pc[i]
                v = (1.0 - c1_eff - c_mu_eff) * (D[i] * D[i]) + c1_eff * (D[i] * D[i]) * pc2 + c_mu_eff * (D[i] * D[i]) * dz2[i]
                if v <= 1e-30:
                    v = 1e-30
                D[i] = math.sqrt(v)

                # keep within bounds-related reasonable range
                Dmin = 1e-14 * spans[i] + 1e-12
                Dmax = 2.0 * spans[i] + 1e-12
                if D[i] < Dmin: D[i] = Dmin
                if D[i] > Dmax: D[i] = Dmax

            # termination / restart conditions (time-bounded)
            # restart if stalled, sigma too small, or no global improvement for a while
            if stall_gens > 25 + dim // 2:
                break
            if sigma < 1e-14 * (sum(spans) / float(dim)) + 1e-18:
                break
            if time.time() - last_improve_time > 0.35 * max_time and stall_gens > 10:
                break

            # occasional "opposition of mean" probe
            if random.random() < 0.02 and time.time() < deadline:
                opp = [lows[i] + highs[i] - xmean[i] for i in range(dim)]
                clip_inplace(opp)
                fo = eval_cached(opp)
                if fo < best:
                    best = fo
                    best_x = opp[:]
                    last_improve_time = time.time()

        restart += 1
        # if very little time remains, stop restarting
        if time.time() + 0.02 >= deadline:
            return best
