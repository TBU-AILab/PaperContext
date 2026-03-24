import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (stdlib only).

    Key upgrades vs your current best (#1):
      - Works in normalized [0,1]^d space for scale invariance (better across heterogeneous bounds).
      - Diagonal CMA-ES core (rank-weights + evolution path step-size control + diagonal covariance).
      - Deterministic low-discrepancy seeding (Halton) + opposition + small jitter.
      - Very cheap, time-safe local refinement: coordinate quadratic step (3-point parabola)
        applied adaptively when stagnating.
      - Restarts: IPOP-ish population growth + sigma reset/inflation when stuck.
      - Strict time guarding: avoids starting long phases near the deadline.

    Returns:
        best (float): best objective value found within max_time seconds.
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

    # ---------------- time helpers ----------------
    def time_left():
        return deadline - time.time()

    # ---------------- mapping helpers ----------------
    def clamp01(x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def to_real(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def eval_u(u):
        uu = [clamp01(ui) for ui in u]
        return float(func(to_real(uu))), uu

    # ---------------- random helpers (no numpy) ----------------
    def randn():
        # approx N(0,1)
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        # clamp to avoid numeric blowups
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

    # ---------------- Halton seeding ----------------
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

    # ---------------- archive ----------------
    archive = []  # list of (f, u)
    arch_cap = max(24, 5 * dim)

    def add_archive(f, u):
        archive.append((f, u[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > arch_cap:
            del archive[arch_cap:]

    # ---------------- init ----------------
    mean = [0.5] * dim
    best, best_u = eval_u(mean)
    add_archive(best, best_u)

    # time-aware seeding size
    init_n = max(20, 10 * dim)
    if max_time > 2.0:
        init_n = max(init_n, 34 + 14 * dim)
    if max_time < 0.25:
        init_n = max(8, 3 * dim)

    # seeding: Halton + jitter + opposition
    # also sprinkle a few random points
    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best

        u = halton_u(k)
        # small jitter to avoid exact patterns
        for i in range(dim):
            u[i] = clamp01(u[i] + (random.random() - 0.5) * 0.012)

        f, uu = eval_u(u)
        if f < best:
            best, best_u = f, uu[:]
        add_archive(f, uu)

        uo = [1.0 - uu[i] for i in range(dim)]
        fo, uoo = eval_u(uo)
        if fo < best:
            best, best_u = fo, uoo[:]
        add_archive(fo, uoo)

        if k <= max(4, dim // 2):
            ur = [random.random() for _ in range(dim)]
            fr, urr = eval_u(ur)
            if fr < best:
                best, best_u = fr, urr[:]
            add_archive(fr, urr)

    # ---------------- diagonal CMA-ES-ish state ----------------
    sigma = 0.32  # global step-size in normalized space
    D = [0.22] * dim  # diagonal scaling (sqrt(diag(C)))
    p_sigma = [0.0] * dim  # evolution path for CSA

    lam0 = 10 + 6 * dim
    lam = lam0
    restarts = 0

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
        mueff_inv = 0.0
        for w in ws:
            mueff_inv += w * w
        mueff = 1.0 / max(1e-12, mueff_inv)
        return ws, mueff

    # expected norm of N(0,I)
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # local refinement: coordinate parabola step in normalized space
    lstep = [0.12] * dim
    lstep_min = 1e-12

    def local_parabola_step(i, h):
        """3-point quadratic fit along coordinate i around best_u."""
        nonlocal best, best_u
        if time_left() < 0.003:
            return False

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
            return True

        if time_left() < 0.002:
            return False

        fm, umc = eval_u(um)
        if fm < best:
            best, best_u = fm, umc[:]
            add_archive(fm, umc)
            return True

        denom = (fm - 2.0 * f0 + fp)
        if abs(denom) < 1e-18:
            return False

        xstar = 0.5 * h * (fm - fp) / denom
        if abs(xstar) > 2.0 * h:
            return False

        us = u0[:]
        us[i] = clamp01(us[i] + xstar)

        if time_left() < 0.002:
            return False

        fs, usc = eval_u(us)
        if fs < best:
            best, best_u = fs, usc[:]
            add_archive(fs, usc)
            return True
        return False

    # ---------------- main loop ----------------
    best_seen = best
    no_improve = 0

    while True:
        if time.time() >= deadline:
            return best

        tl = time_left()
        if tl <= 0.0:
            return best

        # keep lambda small near end to avoid overshooting deadline
        if tl < 0.06:
            lam_eff = max(6, 2 * dim)
        else:
            lam_eff = lam

        mu = max(4, lam_eff // 4)
        weights, mueff = recompute_weights(mu)

        # learning rates / parameters (diagonal CMA-style)
        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_covar = min(0.35, (2.0 / (((dim + 1.3) ** 2) + mueff)))

        # occasional recenter to best archive
        if archive and random.random() < 0.15:
            mean = archive[0][1][:]
        elif random.random() < 0.10:
            mean = best_u[:]

        # exploration schedule (more global early, more local late)
        frac_left = max(0.0, min(1.0, tl / max(1e-9, max_time)))
        p_uniform = 0.01 + 0.08 * frac_left
        p_heavy = 0.03 + 0.12 * frac_left

        # sample population (antithetic)
        pop = []
        half = (lam_eff + 1) // 2
        for _ in range(half):
            if time.time() >= deadline:
                return best

            if random.random() < p_uniform:
                u1 = [random.random() for _ in range(dim)]
                u2 = [1.0 - u1[i] for i in range(dim)]
            else:
                heavy = (random.random() < p_heavy)
                z = [cauchy() if heavy else randn() for _ in range(dim)]
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

            if len(pop) < lam_eff:
                f2, u2c = eval_u(u2)
                pop.append((f2, u2c))
                if f2 < best:
                    best, best_u = f2, u2c[:]
                    add_archive(f2, u2c)

        pop.sort(key=lambda t: t[0])

        # mix in some archive into elites (robustness)
        elites = pop[:mu]
        if archive:
            take = min(len(archive), max(1, mu // 3))
            elites = elites[:max(1, mu - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:mu]

        old_mean = mean[:]

        # recombine mean
        mean = [0.0] * dim
        for w, (f, u) in zip(weights, elites):
            for i in range(dim):
                mean[i] += w * u[i]

        # CSA evolution path for sigma
        y = [(mean[i] - old_mean[i]) / max(1e-18, sigma * D[i]) for i in range(dim)]
        a = (1.0 - c_sigma)
        b = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
        for i in range(dim):
            p_sigma[i] = a * p_sigma[i] + b * y[i]

        psn = norm(p_sigma)
        sigma *= math.exp((c_sigma / d_sigma) * ((psn / max(1e-18, chi_n)) - 1.0))
        if sigma < 1e-12:
            sigma = 1e-12
        elif sigma > 0.9:
            sigma = 0.9

        # diagonal covariance update (update D)
        ey2 = [0.0] * dim
        for w, (f, u) in zip(weights, elites):
            for i in range(dim):
                di = (u[i] - old_mean[i]) / max(1e-18, sigma)
                ey2[i] += w * (di * di)

        for i in range(dim):
            d2 = D[i] * D[i]
            d2 = (1.0 - c_covar) * d2 + c_covar * max(1e-18, ey2[i])
            D[i] = math.sqrt(d2)
            if D[i] < 1e-6:
                D[i] = 1e-6
            elif D[i] > 2.5:
                D[i] = 2.5

        # improvement tracking
        if best < best_seen - 1e-15:
            best_seen = best
            no_improve = 0
        else:
            no_improve += 1

        # local refinement (cheap, time-safe)
        tl = time_left()
        if tl > 0.02:
            iters = 1 if no_improve < 8 else 3
            if tl < 0.08:
                iters = 1
            for _ in range(iters):
                if time.time() >= deadline:
                    return best
                i = random.randrange(dim)
                h = lstep[i]
                if h < lstep_min:
                    continue
                improved = local_parabola_step(i, h)
                if improved:
                    lstep[i] = min(0.25, lstep[i] * 1.15)
                else:
                    lstep[i] = max(lstep_min, lstep[i] * 0.85)

        # stagnation handling: inflate sigma sometimes
        if no_improve > 0 and (no_improve % 20 == 0):
            sigma = min(0.9, sigma * 1.25)

        # restart: increase population, reset shape
        if no_improve > 0 and (no_improve % 60 == 0):
            restarts += 1
            lam = int(lam * 1.5) + 2  # IPOP-ish
            mean = (archive[0][1][:] if archive and random.random() < 0.85 else best_u[:])
            sigma = 0.35
            D = [0.22] * dim
            p_sigma = [0.0] * dim
            no_improve = 0
