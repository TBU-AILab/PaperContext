import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (stdlib only).

    Improvements over the provided best:
      - Normalized space [0,1]^dim with safe mapping to bounds.
      - Stronger global coverage: scrambled Halton + random.
      - Core optimizer: compact CMA-ES-like diagonal adaptation:
          * weighted recombination from top mu offspring
          * per-dimension sigma adaptation (log-space) using successful steps
          * global step-size via 1/5 success rule + mild annealing
      - Local refinement: opportunistic coordinate/pattern polishing with shrinking steps.
      - Restarts: IPOP-style (increase lambda) + heavy-tailed injections when stagnating.
      - Strict time checks throughout.

    Returns:
      best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return deadline - time.time()

    # ---- bounds / normalization ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

    def u_to_x(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def clip01_inplace(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def eval_u(u):
        return float(func(u_to_x(u)))

    # ---- RNG helpers ----
    _spare = [None]
    def gauss():
        z = _spare[0]
        if z is not None:
            _spare[0] = None
            return z
        u1 = max(1e-16, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare[0] = z1
        return z0

    def cauchy_like(scale):
        g = gauss()
        h = gauss()
        return (g / max(1e-16, abs(h))) * scale

    # ---- scrambled Halton for seeding/injection ----
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

    # ---- coordinate / pattern polish ----
    # objective scale unknown, keep eps tiny but nonzero
    eps = 1e-15

    def coord_polish(best_u, best_f, step_frac=0.02, rounds=1):
        x = u_to_x(best_u)
        f = best_f
        steps = [step_frac * spans[i] for i in range(dim)]
        for _ in range(rounds):
            coords = list(range(dim))
            random.shuffle(coords)
            improved = False
            for j in coords:
                if time_left() <= 0:
                    # map x back to u approximately
                    u = [(x[i] - lows[i]) / spans[i] for i in range(dim)]
                    return clip01_inplace(u), f

                sj = steps[j]
                if sj <= 0.0:
                    continue

                # + step
                xp = x[:]
                xp[j] = min(highs[j], xp[j] + sj)
                fp = float(func(xp))
                if fp + eps < f:
                    x, f = xp, fp
                    improved = True
                    continue

                # - step
                xm = x[:]
                xm[j] = max(lows[j], xm[j] - sj)
                fm = float(func(xm))
                if fm + eps < f:
                    x, f = xm, fm
                    improved = True
                    continue

            if not improved:
                for j in range(dim):
                    steps[j] *= 0.55

        u = [(x[i] - lows[i]) / spans[i] for i in range(dim)]
        return clip01_inplace(u), f

    # ---- initial seeding ----
    best_u = rand_u()
    best = eval_u(best_u)
    if time_left() <= 0:
        return best

    H = max(48, min(900, 60 + 20 * dim))
    R = max(14, min(160, 8 + 5 * dim))

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

    # ---- compact diagonal CMA-like ES ----
    mean = best_u[:]

    # sigma per-dimension in normalized space, stored in log-space for stability
    # start moderately global, slightly shrink with dim
    base = 0.28 / (1.0 + 0.015 * dim)
    logsig = [math.log(max(1e-6, min(0.6, base))) for _ in range(dim)]
    sig_min, sig_max = 1e-12, 0.65

    # population sizing and recombination
    lam = max(16, min(96, 10 + 4 * dim))
    mu = max(4, lam // 3)

    # log weights (CMA-ES style)
    weights = [0.0] * mu
    s = 0.0
    for i in range(mu):
        w = math.log((mu + 0.5) / (i + 1.0))
        weights[i] = w
        s += w
    invs = 1.0 / s
    for i in range(mu):
        weights[i] *= invs

    # adaptation parameters
    # learning rate for diagonal sigma update
    lr = 0.20 / (1.0 + 0.02 * dim)   # per-generation; modest
    lr = min(0.25, max(0.05, lr))

    # global step control via 1/5 rule (success = improvement over current best)
    succ = 0
    gens = 0

    # stagnation / restarts (IPOP-like)
    stagn = 0
    hal_idx = H + 1
    restarts = 0

    # mild annealing to help fine convergence late
    # (kept weak to avoid harming global search)
    def anneal_factor():
        # from 1.0 down to ~0.85 as time elapses
        frac = (time.time() - t0) / max(1e-12, (deadline - t0))
        return 1.0 - 0.15 * max(0.0, min(1.0, frac))

    while time_left() > 0:
        # sample offspring around mean with diagonal sigmas
        sig = [min(sig_max, max(sig_min, math.exp(ls))) for ls in logsig]
        a = anneal_factor()

        # decide whether to add some heavy-tailed offspring in this generation
        heavy_gen = (random.random() < 0.10) or (stagn >= 16 and random.random() < 0.35)

        pop = []  # (f, u, z)
        for k in range(lam):
            if time_left() <= 0:
                return best
            u = mean[:]
            z = [0.0] * dim
            if heavy_gen and (k < max(2, lam // 6)):
                # a few heavy-tailed samples
                for i in range(dim):
                    zi = cauchy_like(1.0)
                    z[i] = zi
                    u[i] += zi * sig[i] * a
            else:
                for i in range(dim):
                    zi = gauss()
                    z[i] = zi
                    u[i] += zi * sig[i] * a

            clip01_inplace(u)
            f = eval_u(u)
            pop.append((f, u, z))

        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_u, best_off_z = pop[0]

        # recombination (new mean = weighted sum of best mu)
        new_mean = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            uj = pop[j][1]
            for i in range(dim):
                new_mean[i] += w * uj[i]
        clip01_inplace(new_mean)

        # diagonal sigma update: compare selected step magnitudes to expected ~1
        # use z from selected individuals (in sigma units, before clipping effects)
        # we keep it simple: target E[z^2]=1, update logsig toward observed average.
        obs2 = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            zj = pop[j][2]
            for i in range(dim):
                obs2[i] += w * (zj[i] * zj[i])

        for i in range(dim):
            # if obs2 > 1 -> we are taking large successful steps => increase sigma
            # if obs2 < 1 -> decrease sigma
            # update in log-space
            logsig[i] += 0.5 * lr * (obs2[i] - 1.0)
            # hard bounds
            s_i = math.exp(logsig[i])
            if s_i < sig_min:
                logsig[i] = math.log(sig_min)
            elif s_i > sig_max:
                logsig[i] = math.log(sig_max)

        gens += 1

        # (mu,lambda)-like mean update, but keep elitism for "best"
        mean = new_mean

        if best_off_f + eps < best:
            best = best_off_f
            best_u = best_off_u[:]
            succ += 1
            stagn = 0
        else:
            stagn += 1

        # global sigma nudging every ~10 gens (1/5 success rule)
        if gens >= 10:
            rate = succ / float(gens)
            mult = 1.18 if rate > 0.20 else 0.84
            for i in range(dim):
                s_i = min(sig_max, max(sig_min, math.exp(logsig[i]) * mult))
                logsig[i] = math.log(s_i)
            succ = 0
            gens = 0

        # polish on mild stalls
        if stagn in (8, 14) and time_left() > 0:
            pu, pf = coord_polish(best_u, best, step_frac=0.02, rounds=1)
            if pf + eps < best:
                best, best_u = pf, pu
                mean = best_u[:]
                stagn = 0

        # restarts / injections if stuck
        if stagn >= 26 and time_left() > 0:
            stagn = 0
            restarts += 1

            # IPOP-ish: increase lambda a bit (bounded)
            if restarts <= 6:
                lam = min(160, int(lam * 1.35) + 2)
                mu = max(4, lam // 3)
                weights = [0.0] * mu
                s = 0.0
                for i in range(mu):
                    w = math.log((mu + 0.5) / (i + 1.0))
                    weights[i] = w
                    s += w
                invs = 1.0 / s
                for i in range(mu):
                    weights[i] *= invs
                lr = min(0.25, max(0.04, 0.18 / (1.0 + 0.02 * dim)))

            # inject a couple of global candidates
            for _ in range(2):
                if time_left() <= 0:
                    return best
                u = halton_u(hal_idx)
                hal_idx += 1
                f = eval_u(u)
                if f + eps < best:
                    best, best_u = f, u
                    mean = best_u[:]

            if time_left() <= 0:
                return best
            u = rand_u()
            f = eval_u(u)
            if f + eps < best:
                best, best_u = f, u
                mean = best_u[:]

            # widen sigmas to escape basin, but not too much
            for i in range(dim):
                s_i = max(math.exp(logsig[i]), 0.20)
                logsig[i] = math.log(min(sig_max, s_i))

    return best
