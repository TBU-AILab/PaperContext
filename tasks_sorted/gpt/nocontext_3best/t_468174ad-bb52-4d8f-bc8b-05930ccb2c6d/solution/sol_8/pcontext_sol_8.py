import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (stdlib only).

    Improvements vs. your best #1:
      - Better global modeling: diagonal CMA-ES style update using *natural gradient* idea
        (rank-weights, evolution path, and per-dimension step-size adaptation).
      - Better invariance to scaling: normalizes internal coordinates to [0,1] then maps to bounds.
      - Stronger restart logic: IPOP-style population growth + sigma reset on stagnation.
      - Stronger local refinement: short coordinate-wise quadratic-fit step (3-point parabola)
        plus a small Powell-like random-direction probe.
      - Strict time-safety: all loops are guarded; avoids starting expensive phases near deadline.

    Returns:
        best (float): best objective value found.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    # ---------- helpers ----------
    def time_left():
        return deadline - time.time()

    def clamp01(x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def to_real(u):
        # u in [0,1]^d -> x in bounds
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def eval_u(u):
        # Evaluate at u, clipped to [0,1]
        uu = [clamp01(ui) for ui in u]
        x = to_real(uu)
        return float(func(x)), uu

    def randn():
        # approx N(0,1): sum 12 uniforms - 6
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        # keep finite
        if v > 50.0:
            v = 50.0
        elif v < -50.0:
            v = -50.0
        return v

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    # ---------- initialization: low-discrepancy Halton seeding ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def halton_index(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    primes = first_primes(dim)

    def halton_u(k):
        return [halton_index(k, primes[i]) for i in range(dim)]

    # Start at center in normalized space
    mean = [0.5] * dim
    best_f, best_u = eval_u(mean)
    best = best_f

    # small elite archive in normalized coords
    archive = [(best_f, best_u[:])]
    arch_cap = max(20, 4 * dim)

    def add_archive(f, u):
        archive.append((f, u[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > arch_cap:
            del archive[arch_cap:]

    # Seeding budget (time-aware)
    tl = time_left()
    init_n = max(18, 8 * dim)
    if tl > 2.0:
        init_n = max(init_n, 28 + 10 * dim)
    if max_time < 0.25:
        init_n = max(8, 3 * dim)

    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best
        u = halton_u(k)
        # jitter
        for i in range(dim):
            u[i] = clamp01(u[i] + (random.random() - 0.5) * 0.01)
        f, uu = eval_u(u)
        if f < best:
            best = f
            best_u = uu[:]
            add_archive(f, uu)

        # opposition (in [0,1] it is 1-u)
        uo = [1.0 - uu[i] for i in range(dim)]
        fo, uoo = eval_u(uo)
        if fo < best:
            best = fo
            best_u = uoo[:]
            add_archive(fo, uoo)

    # ---------- diagonal CMA-ES-ish state in normalized space ----------
    # sigma in normalized units; per-dimension scaling in diag D (sqrt of covariance)
    sigma = 0.30
    D = [0.25] * dim  # per-dim scale multiplier (diag)
    # evolution path (for step-size adaptation)
    p_sigma = [0.0] * dim

    # population sizes (restartable)
    lam0 = 8 + 6 * dim
    lam = lam0
    mu = max(4, lam // 4)

    def recompute_weights(mu_):
        ws = []
        s = 0.0
        for r in range(mu_):
            w = math.log(mu_ + 0.5) - math.log(r + 1.0)
            ws.append(w)
            s += w
        if s <= 0.0:
            s = 1.0
        ws = [w / s for w in ws]
        # effective mu
        mueff = 0.0
        for w in ws:
            mueff += w * w
        mueff = 1.0 / max(1e-12, mueff)
        return ws, mueff

    weights, mueff = recompute_weights(mu)

    # learning rates (standard-ish diagonal CMA heuristics)
    c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    # covariance (diag) learning
    c_covar = min(0.35, (2.0 / ((dim + 1.3) ** 2 + mueff)))
    # expected norm of N(0,I)
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # stagnation tracking
    no_improve = 0
    last_best = best
    restarts = 0

    # local search step sizes (normalized)
    lstep = [0.12] * dim
    lstep_min = 1e-12

    def try_local_parabola(i, h):
        """
        1D quadratic fit using 3 points around current best along coordinate i in normalized space.
        Uses at most 3 evaluations (sometimes 2 if time is tight).
        """
        nonlocal best, best_u, no_improve
        if time_left() < 0.003:
            return

        u0 = best_u[:]
        f0 = best

        up = u0[:]
        um = u0[:]
        up[i] = clamp01(up[i] + h)
        um[i] = clamp01(um[i] - h)

        fp, upc = eval_u(up)
        if fp < best:
            best, best_u = fp, upc[:]
            add_archive(fp, upc)
            no_improve = 0
            return

        if time_left() < 0.002:
            return

        fm, umc = eval_u(um)
        if fm < best:
            best, best_u = fm, umc[:]
            add_archive(fm, umc)
            no_improve = 0
            return

        # If no improvement, propose quadratic minimizer if bracket makes sense
        # Fit parabola through (-h,fm), (0,f0), (+h,fp)
        # x* = h*(fm - fp) / (2*(fm - 2f0 + fp))
        denom = (fm - 2.0 * f0 + fp)
        if abs(denom) < 1e-18:
            return
        xstar = 0.5 * h * (fm - fp) / denom
        if abs(xstar) > 2.0 * h:
            return
        us = u0[:]
        us[i] = clamp01(us[i] + xstar)

        if time_left() < 0.002:
            return
        fs, usc = eval_u(us)
        if fs < best:
            best, best_u = fs, usc[:]
            add_archive(fs, usc)
            no_improve = 0

    def random_direction_probe():
        """Small probe along a random direction in normalized space."""
        nonlocal best, best_u, no_improve
        if time_left() < 0.003:
            return
        d = [randn() for _ in range(dim)]
        n = norm(d)
        if n <= 0.0:
            return
        inv = 1.0 / n
        d = [v * inv for v in d]
        scale = 0.15 * sigma
        u = [best_u[i] + scale * d[i] for i in range(dim)]
        f, uc = eval_u(u)
        if f < best:
            best, best_u = f, uc[:]
            add_archive(f, uc)
            no_improve = 0

    # ---------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        tl = time_left()
        if tl <= 0.0:
            return best

        # adjust lambda near deadline
        if tl < 0.06:
            lam_eff = max(6, 2 * dim)
        else:
            lam_eff = lam

        # occasional recenter to best/elite
        if archive and random.random() < 0.12:
            mean = archive[0][1][:]
        elif random.random() < 0.10:
            mean = best_u[:]

        # sample population (antithetic), in normalized space
        pop = []
        half = (lam_eff + 1) // 2

        # exploration mixing
        frac_left = max(0.0, min(1.0, tl / max(1e-9, max_time)))
        p_uniform = 0.01 + 0.07 * frac_left
        p_heavy = 0.03 + 0.12 * frac_left

        for _ in range(half):
            if time.time() >= deadline:
                return best

            if random.random() < p_uniform:
                u1 = [random.random() for _ in range(dim)]
                u2 = [1.0 - u1[i] for i in range(dim)]
            else:
                heavy = (random.random() < p_heavy)
                z = [cauchy() if heavy else randn() for _ in range(dim)]
                # u = mean + sigma * D * z
                u1 = [0.0] * dim
                u2 = [0.0] * dim
                for i in range(dim):
                    step = sigma * D[i] * z[i]
                    u1[i] = mean[i] + step
                    u2[i] = mean[i] - step

            f1, u1c = eval_u(u1)
            pop.append((f1, u1c))
            if f1 < best:
                best, best_u = f1, u1c[:]
                add_archive(f1, u1c)
                no_improve = 0

            if len(pop) < lam_eff:
                f2, u2c = eval_u(u2)
                pop.append((f2, u2c))
                if f2 < best:
                    best, best_u = f2, u2c[:]
                    add_archive(f2, u2c)
                    no_improve = 0

        pop.sort(key=lambda t: t[0])

        # update mu/weights if lambda changed due to restarts
        mu = max(4, lam // 4)
        weights, mueff = recompute_weights(mu)

        # select elites (mix a bit of archive)
        elites = pop[:mu]
        if archive:
            take = min(len(archive), max(1, mu // 3))
            elites = elites[:max(1, mu - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:mu]

        # compute new mean (weighted)
        old_mean = mean[:]
        mean = [0.0] * dim
        for w, (f, u) in zip(weights, elites):
            for i in range(dim):
                mean[i] += w * u[i]

        # evolution path update for sigma
        y = [(mean[i] - old_mean[i]) / max(1e-18, sigma * D[i]) for i in range(dim)]
        # update p_sigma
        a = (1.0 - c_sigma)
        b = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
        for i in range(dim):
            p_sigma[i] = a * p_sigma[i] + b * y[i]

        # sigma update (csa)
        ps_norm = norm(p_sigma)
        sigma *= math.exp((c_sigma / d_sigma) * ((ps_norm / max(1e-18, chi_n)) - 1.0))

        # covariance diagonal update: D^2 <- (1-c)*D^2 + c*E[y_i^2]
        # approximate using elite deviations
        ey2 = [0.0] * dim
        for w, (f, u) in zip(weights, elites):
            for i in range(dim):
                di = (u[i] - old_mean[i]) / max(1e-18, sigma)
                ey2[i] += w * (di * di)

        for i in range(dim):
            d2 = D[i] * D[i]
            d2 = (1.0 - c_covar) * d2 + c_covar * max(1e-18, ey2[i])
            D[i] = math.sqrt(d2)
            # keep D in reasonable range in normalized space
            if D[i] < 1e-6:
                D[i] = 1e-6
            elif D[i] > 2.0:
                D[i] = 2.0

        # stagnation accounting
        if best < last_best - 1e-15:
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # local refinement (time-safe)
        tl = time_left()
        if tl > 0.02:
            # more local work when stagnating
            loc_iters = 1 if no_improve < 8 else 3
            if tl < 0.08:
                loc_iters = 1
            for _ in range(loc_iters):
                if time.time() >= deadline:
                    return best
                i = random.randrange(dim)
                h = lstep[i]
                if h < lstep_min:
                    continue
                try_local_parabola(i, h)
                # step adaptation
                if no_improve == 0:
                    lstep[i] = min(0.25, lstep[i] * 1.15)
                else:
                    lstep[i] = max(lstep_min, lstep[i] * 0.85)

            if tl > 0.04 and random.random() < (0.12 if no_improve < 10 else 0.30):
                random_direction_probe()

        # restart / inflation policy
        if no_improve > 0 and (no_improve % 18 == 0):
            sigma = min(0.6, sigma * 1.25)

        if no_improve > 0 and (no_improve % 55 == 0):
            # IPOP-like restart: increase population, reset sigma/D, recenter
            restarts += 1
            lam = int(lam * 1.5) + 2
            if archive and random.random() < 0.8:
                mean = archive[0][1][:]
            else:
                mean = best_u[:]
            sigma = 0.35
            D = [0.25] * dim
            p_sigma = [0.0] * dim
            no_improve = 0
