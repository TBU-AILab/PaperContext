import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Key upgrades vs previous version:
      - Adds a true *CMA-ES style* evolution path step-size control (much stronger than 1/5 on "any success").
      - Uses *whitened sampling* (diagonal covariance) + rank-mu update on variances.
      - Stronger *intensification*: short, cheap Nelder-Mead-like simplex in normalized space near the best.
      - Better *restart logic*: IPOP-style sigma inflation + partial random reinitialization of mean.
      - Keeps cache to avoid near-duplicate evaluations.

    func: callable(list[float]) -> float
    returns: best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        try:
            v = float(func([]))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span = [s if s > 0.0 else 1.0 for s in span]

    def repair_u(u):
        # clamp to [0,1]
        return [0.0 if u[i] < 0.0 else (1.0 if u[i] > 1.0 else u[i]) for i in range(dim)]

    def to_x(u):
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    # ---- cached evaluation on a quantized grid in normalized space ----
    grid = {}
    q = 1e-4 if dim <= 6 else (3e-4 if dim <= 15 else 9e-4)

    def key_u(u):
        return tuple(int(uu / q + 0.5) for uu in u)

    def eval_u(u):
        uu = repair_u(u)
        k = key_u(uu)
        if k in grid:
            return grid[k], uu
        x = to_x(uu)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        grid[k] = v
        return v, uu

    def rand_u():
        return [random.random() for _ in range(dim)]

    # ---- Halton init (low discrepancy) ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def halton_index(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(dim)

    def halton_u(k):
        return [halton_index(k, primes[j]) for j in range(dim)]

    # ---- initialization ----
    best = float("inf")
    best_u = rand_u()

    n_init = max(40, min(320, 26 * dim + 64))
    k = 17
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(halton_u(k))
        k += 1
        if v < best:
            best, best_u = v, uu

    for _ in range(max(12, dim)):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(rand_u())
        if v < best:
            best, best_u = v, uu

    # ---- CMA-ES-like loop in normalized coordinates (diagonal covariance) ----
    mean = best_u[:]

    lam = max(16, 8 * dim)
    lam += (lam % 2)  # even
    mu = max(5, lam // 2)

    # log-weights
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # diagonal std + global sigma
    diag = [0.22] * dim
    diag_min = 1e-7
    diag_max = 0.55
    sigma = 0.8  # global multiplier in normalized coords

    # evolution paths (diagonal variant)
    ps = [0.0] * dim

    # CMA-ish parameters (diagonal)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    c1 = 0.0  # no full covariance
    cmu = min(0.35, (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    # for diagonal variance update use "ccov" style
    ccov = cmu

    # expected length of N(0,I) (approx)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # ---- local intensification: tiny Nelder-Mead-like simplex in normalized space ----
    def simplex_refine(u0, v0, steps):
        # build simplex: u0 plus coordinate offsets
        # scale from current typical step
        base_step = 0.08
        # use current sigma*diag scale (median-ish)
        sc = sigma * sorted(diag)[len(diag)//2]
        step = max(1e-4, min(0.25, base_step * (0.6 + 1.2 * sc)))

        simplex = [(v0, u0[:])]
        for j in range(dim):
            uj = u0[:]
            uj[j] += step
            vj, uej = eval_u(uj)
            simplex.append((vj, uej))

        simplex.sort(key=lambda t: t[0])

        # coefficients
        alpha, gamma, rho, sigma_shrink = 1.0, 2.0, 0.5, 0.5

        for _ in range(steps):
            if time.time() >= deadline:
                break
            simplex.sort(key=lambda t: t[0])
            bestv, bestu = simplex[0]
            worstv, worstu = simplex[-1]
            secondv = simplex[-2][0]

            # centroid of all but worst
            centroid = [0.0] * dim
            for i in range(len(simplex) - 1):
                ui = simplex[i][1]
                for j in range(dim):
                    centroid[j] += ui[j]
            inv = 1.0 / (len(simplex) - 1)
            for j in range(dim):
                centroid[j] *= inv

            # reflect
            ur = [centroid[j] + alpha * (centroid[j] - worstu[j]) for j in range(dim)]
            vr, uer = eval_u(ur)

            if vr < bestv:
                # expand
                ue = [centroid[j] + gamma * (uer[j] - centroid[j]) for j in range(dim)]
                ve, uee = eval_u(ue)
                simplex[-1] = (ve, uee) if ve < vr else (vr, uer)
            elif vr < secondv:
                simplex[-1] = (vr, uer)
            else:
                # contract
                uc = [centroid[j] + rho * (worstu[j] - centroid[j]) for j in range(dim)]
                vc, uec = eval_u(uc)
                if vc < worstv:
                    simplex[-1] = (vc, uec)
                else:
                    # shrink
                    new_simplex = [simplex[0]]
                    for i in range(1, len(simplex)):
                        ui = simplex[i][1]
                        us = [bestu[j] + sigma_shrink * (ui[j] - bestu[j]) for j in range(dim)]
                        vs, ues = eval_u(us)
                        new_simplex.append((vs, ues))
                    simplex = new_simplex

        simplex.sort(key=lambda t: t[0])
        return simplex[0][0], simplex[0][1]

    # ---- restart control ----
    last_improve_t = time.time()
    stall_seconds = max(0.25, 0.12 * max_time)
    restarts = 0

    gen = 0
    while time.time() < deadline:
        gen += 1
        best_before = best

        # sample offspring (antithetic)
        off = []
        zs = []  # store z for ps update
        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            zs.append(z)

            # + and -
            u1 = [mean[j] + sigma * diag[j] * z[j] for j in range(dim)]
            v1, uu1 = eval_u(u1)
            off.append((v1, uu1, z, +1.0))

            u2 = [mean[j] - sigma * diag[j] * z[j] for j in range(dim)]
            v2, uu2 = eval_u(u2)
            off.append((v2, uu2, z, -1.0))

            if v1 < best:
                best, best_u = v1, uu1
            if v2 < best:
                best, best_u = v2, uu2

        if best < best_before:
            last_improve_t = time.time()

        off.sort(key=lambda t: t[0])

        # recombine mean
        old_mean = mean[:]
        mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                mean[j] += w * ui[j]
        mean = repair_u(mean)

        # update evolution path ps (diagonal approx)
        # y = (mean - old_mean) / (sigma * diag)
        for j in range(dim):
            denom = sigma * diag[j]
            if denom < 1e-18:
                yj = 0.0
            else:
                yj = (mean[j] - old_mean[j]) / denom
            ps[j] = (1.0 - cs) * ps[j] + math.sqrt(cs * (2.0 - cs) * mueff) * yj

        # step-size control using ||ps|| (diagonal)
        ps_norm = math.sqrt(sum(p * p for p in ps))
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        if sigma < 1e-6:
            sigma = 1e-6
        if sigma > 3.0:
            sigma = 3.0

        # diagonal variance update from selected steps around old_mean
        # compute weighted variance of (ui - old_mean) / sigma
        for j in range(dim):
            vj = 0.0
            for i in range(mu):
                w = weights[i]
                ui = off[i][1]
                d = (ui[j] - old_mean[j]) / max(1e-18, sigma)
                vj += w * (d * d)
            # update diag[j]^2 -> blend towards vj
            dj2 = diag[j] * diag[j]
            dj2 = (1.0 - ccov) * dj2 + ccov * vj
            # clamp and sqrt
            if dj2 < diag_min * diag_min:
                dj2 = diag_min * diag_min
            if dj2 > diag_max * diag_max:
                dj2 = diag_max * diag_max
            diag[j] = math.sqrt(dj2)

        # periodic intensification near incumbent
        if (gen % 6 == 0) and time.time() < deadline:
            steps = 6 + 2 * dim
            vloc, uloc = simplex_refine(best_u, best, steps)
            if vloc < best:
                best, best_u = vloc, uloc
                mean = best_u[:]
                last_improve_t = time.time()
                # small contraction to exploit
                sigma = max(1e-6, sigma * 0.65)

        # restart if stalled or too small exploration
        stalled = (time.time() - last_improve_t) > stall_seconds
        tiny = (sigma * max(diag)) < 2e-6

        if stalled or tiny:
            restarts += 1
            last_improve_t = time.time()

            # IPOP-ish: increase population and sigma, keep best but randomize part of mean
            lam = min(4000, int(lam * 1.6) + 2)
            lam += (lam % 2)
            mu = max(5, lam // 2)
            weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            wsum = sum(weights)
            weights = [w / wsum for w in weights]
            mueff = 1.0 / sum(w * w for w in weights)

            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
            cmu = min(0.35, (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            ccov = cmu
            chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

            # partial randomization of mean around best (preserve best but diversify)
            mean = best_u[:]
            for j in range(dim):
                if random.random() < 0.35:
                    mean[j] = 0.75 * mean[j] + 0.25 * random.random()

            # inflate exploration
            sigma = min(2.0, max(0.25, sigma * 1.8))
            for j in range(dim):
                diag[j] = min(diag_max, max(diag[j], 0.18))

            ps = [0.0] * dim

            # inject some purely random candidates quickly
            inj = max(10, dim)
            for _ in range(inj):
                if time.time() >= deadline:
                    return best
                v, uu = eval_u(rand_u())
                if v < best:
                    best, best_u = v, uu
                    mean = best_u[:]
                    break

    return best
