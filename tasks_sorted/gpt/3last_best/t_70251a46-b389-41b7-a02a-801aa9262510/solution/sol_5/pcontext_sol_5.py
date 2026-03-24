import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained; stdlib only).

    Key upgrades vs the provided best (95.66):
      - Normalized search in [0,1]^dim + robust bound mapping.
      - Better exploration: scrambled Halton + a few random points.
      - Stronger exploitation: adaptive (1+λ)-ES with diagonal covariance (per-dim sigmas)
        updated via success-weighted steps (lightweight CMA-like).
      - Occasional coordinate/pattern polish for fast final squeezing.
      - Restart logic + heavy-tailed escapes when stagnating.
      - Strict time checks.

    Returns:
      best (float): best fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return deadline - time.time()

    # ---- bounds ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    # normalized <-> real mapping
    def u_to_x(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def x_to_u(x):
        out = [0.0] * dim
        for i in range(dim):
            out[i] = (x[i] - lows[i]) / spans[i]
            if out[i] < 0.0:
                out[i] = 0.0
            elif out[i] > 1.0:
                out[i] = 1.0
        return out

    def clip01_inplace(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def eval_u(u):
        # func expects array-like; list is ok for most callers
        return float(func(u_to_x(u)))

    # scale-aware epsilon for improvement checks (objective scale unknown)
    eps = 1e-15

    # ---- RNG helpers ----
    _spare = [None]
    def gauss():
        z = _spare[0]
        if z is not None:
            _spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare[0] = z1
        return z0

    def cauchy_like(scale):
        # ratio of Gaussians -> heavy tail
        g = gauss()
        h = gauss()
        return (g / max(1e-12, abs(h))) * scale

    # ---- scrambled Halton ----
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            ok = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(k)
            k += 1
        return primes

    primes = first_primes(max(1, dim))
    digit_perm = {}
    for j in range(dim):
        base = primes[j]
        perm = list(range(base))
        random.shuffle(perm)
        digit_perm[(j, base)] = perm

    def halton_scrambled(index, base, perm):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        # keep in [0,1)
        if r < 0.0:
            r = 0.0
        elif r >= 1.0:
            r = 1.0 - 1e-16
        return r

    def halton_u(k):
        u = [0.0] * dim
        for j in range(dim):
            base = primes[j]
            u[j] = halton_scrambled(k, base, digit_perm[(j, base)])
        return u

    def rand_u():
        return [random.random() for _ in range(dim)]

    # ---- coordinate polish in real space (but step computed from spans) ----
    def coord_polish(best_u, best_f, step_frac=0.02, rounds=1):
        # operate on x for intuitive steps, then map back to u
        x = u_to_x(best_u)
        f = best_f
        step = [step_frac * spans[i] for i in range(dim)]
        for _ in range(rounds):
            coords = list(range(dim))
            random.shuffle(coords)
            improved = False
            for j in coords:
                if time_left() <= 0:
                    return x_to_u(x), f
                sj = step[j]
                if sj <= 0.0:
                    continue

                xp = x[:]
                xp[j] = min(highs[j], xp[j] + sj)
                fp = float(func(xp))
                if fp + eps < f:
                    x, f = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[j] = max(lows[j], xm[j] - sj)
                fm = float(func(xm))
                if fm + eps < f:
                    x, f = xm, fm
                    improved = True
                    continue

            if not improved:
                for j in range(dim):
                    step[j] *= 0.6
        return x_to_u(x), f

    # ---- initial seeding ----
    # Start from random
    best_u = rand_u()
    best = eval_u(best_u)
    if time_left() <= 0:
        return best

    # Mix of random + Halton (small but helpful)
    H = max(32, min(520, 40 + 16 * dim))
    R = max(10, min(120, 6 + 4 * dim))

    for _ in range(R):
        if time_left() <= 0:
            return best
        u = rand_u()
        f = eval_u(u)
        if f + eps < best:
            best, best_u = f, u

    for k in range(1, H + 1):
        if time_left() <= 0:
            return best
        u = halton_u(k)
        f = eval_u(u)
        if f + eps < best:
            best, best_u = f, u

    # ---- (1+λ)-ES with diagonal adaptation ----
    # mean at best; per-dimension sigma in normalized space
    mean = best_u[:]
    sig = [0.22] * dim  # start fairly global in normalized coordinates
    sig_min = 1e-12
    sig_max = 0.6

    lam = max(12, min(72, 8 + 3 * dim))
    stagn = 0
    halton_idx = H + 1

    # success stats for global sigma nudging
    succ = 0
    gens = 0

    # evolution path-ish accumulators for diagonal adaptation
    # if steps in a dimension consistently help, increase its sigma a bit
    adapt = [0.0] * dim  # signed running signal
    adapt_decay = 0.90

    while time_left() > 0:
        parent = mean[:]  # mean tracks current best basin

        best_off_u = None
        best_off_f = float("inf")
        best_off_z = None  # mutation in sigma units (normalized)

        # decide mutation style
        heavy = (random.random() < 0.12) or (stagn >= 18 and random.random() < 0.35)

        for _ in range(lam):
            if time_left() <= 0:
                return best

            u = parent[:]
            z = [0.0] * dim
            if heavy:
                for i in range(dim):
                    zi = cauchy_like(1.0)
                    z[i] = zi
                    u[i] += zi * sig[i]
            else:
                for i in range(dim):
                    zi = gauss()
                    z[i] = zi
                    u[i] += zi * sig[i]

            clip01_inplace(u)
            f = eval_u(u)
            if f < best_off_f:
                best_off_f = f
                best_off_u = u
                best_off_z = z

        gens += 1

        # selection: (1+λ)
        if best_off_f + eps < best:
            best = best_off_f
            best_u = best_off_u[:]
            mean = best_u[:]  # greedy mean update (fast)
            succ += 1
            stagn = 0

            # diagonal sigma adaptation from successful step direction/magnitude
            # encourage exploration along consistently helpful coordinates
            for i in range(dim):
                adapt[i] = adapt_decay * adapt[i] + (1.0 - adapt_decay) * best_off_z[i]
        else:
            stagn += 1
            for i in range(dim):
                adapt[i] = adapt_decay * adapt[i]

        # periodic global sigma nudging using 1/5 rule
        if gens >= 10:
            rate = succ / float(gens)
            # if too successful, widen; else shrink
            mult = 1.20 if rate > 0.2 else 0.82
            for i in range(dim):
                sig[i] *= mult
                if sig[i] < sig_min:
                    sig[i] = sig_min
                elif sig[i] > sig_max:
                    sig[i] = sig_max
            succ = 0
            gens = 0

        # per-dimension adjustment: small, bounded
        for i in range(dim):
            # if adapt has large magnitude, that dim seems sensitive/structured
            a = abs(adapt[i])
            if a > 0.8:
                sig[i] = min(sig_max, sig[i] * 1.05)
            elif a < 0.15:
                sig[i] = max(sig_min, sig[i] * 0.98)

        # cheap polish when stalling a bit
        if stagn in (8, 14) and time_left() > 0:
            pu, pf = coord_polish(best_u, best, step_frac=0.02, rounds=1)
            if pf + eps < best:
                best, best_u = pf, pu
                mean = best_u[:]
                stagn = 0

        # restart / injection when quite stuck
        if stagn >= 26 and time_left() > 0:
            stagn = 0

            # candidate: halton injection
            for _ in range(2):
                if time_left() <= 0:
                    return best
                u = halton_u(halton_idx)
                halton_idx += 1
                f = eval_u(u)
                if f + eps < best:
                    best, best_u = f, u
                    mean = best_u[:]

            # candidate: random injection
            if time_left() <= 0:
                return best
            u = rand_u()
            f = eval_u(u)
            if f + eps < best:
                best, best_u = f, u
                mean = best_u[:]

            # widen sigmas a bit to escape local basin
            for i in range(dim):
                sig[i] = max(sig[i], 0.18)

    return best
