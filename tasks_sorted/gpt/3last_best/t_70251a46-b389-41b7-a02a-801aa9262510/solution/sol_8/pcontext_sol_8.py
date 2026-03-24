import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (stdlib only).

    Upgrade over your current best (diagonal CMA-like):
      - Uses reflection (not clipping) in normalized space to reduce boundary bias.
      - Keeps an elite pool for restarts/recentering (multi-basin robustness).
      - Uses mirrored/antithetic sampling around the mean (variance reduction).
      - Adds a very cheap local solver: multi-scale pattern search around best,
        called when progress stalls (good for squeezing).
      - More stable diagonal sigma adaptation using *selected steps* in normalized
        coordinates (uses actual (u-mean)/sigma, less distorted than raw z when
        boundaries trigger).
      - Time-guarded throughout.

    Returns:
      best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return deadline - time.time()

    if dim <= 0:
        return float(func([]))

    # ---------- bounds / normalization ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

    def u_to_x(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def eval_u(u):
        return float(func(u_to_x(u)))

    # reflect to [0,1] with period 2 (removes clamp-sticking)
    def reflect01_inplace(u):
        for i in range(dim):
            x = u[i]
            if 0.0 <= x <= 1.0:
                continue
            y = x % 2.0
            if y > 1.0:
                y = 2.0 - y
            u[i] = y
        return u

    # ---------- RNG helpers ----------
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

    def cauchy_like(scale=1.0):
        g = gauss()
        h = gauss()
        return (g / max(1e-16, abs(h))) * scale

    def rand_u():
        return [random.random() for _ in range(dim)]

    # ---------- scrambled Halton for seeding/injection ----------
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

    # ---------- elite pool ----------
    eps = 1e-15
    elite_u = []
    elite_f = []
    elite_keys = set()

    def elite_key(u):
        # coarse binning for diversity
        bins = 900
        key = []
        for i in range(dim):
            v = u[i]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            key.append(int(v * bins))
        return tuple(key)

    def push_elite(u, f, cap=10):
        k = elite_key(u)
        if k not in elite_keys:
            elite_keys.add(k)
            elite_u.append(u[:])
            elite_f.append(f)
        else:
            # replace within same key if improved
            for idx in range(len(elite_u)):
                if elite_key(elite_u[idx]) == k and f < elite_f[idx]:
                    elite_u[idx] = u[:]
                    elite_f[idx] = f
                    break

        if len(elite_f) > cap:
            idxs = sorted(range(len(elite_f)), key=lambda i: elite_f[i])[:cap]
            elite_u[:] = [elite_u[i] for i in idxs]
            elite_f[:] = [elite_f[i] for i in idxs]
            elite_keys.clear()
            for uu in elite_u:
                elite_keys.add(elite_key(uu))

    # ---------- local pattern search (multi-scale, cheap) ----------
    def local_pattern(best_u, best_f, budget_evals=64):
        if budget_evals <= 0 or time_left() <= 0:
            return best_u, best_f

        u = best_u[:]
        f = best_f

        # step sizes in normalized space (start larger, shrink)
        # scale down with dimension a bit
        base = 0.12 / (1.0 + 0.05 * dim)
        steps = [base, base * 0.5, base * 0.25, base * 0.125]
        evals = 0

        # coordinate order shuffled each call
        coords = list(range(dim))
        random.shuffle(coords)

        for step in steps:
            if time_left() <= 0 or evals >= budget_evals:
                break
            improved = True
            # loop until no improvement at this scale or budget exhausted
            while improved and time_left() > 0 and evals < budget_evals:
                improved = False
                for j in coords:
                    if time_left() <= 0 or evals >= budget_evals:
                        break

                    # try +step, -step
                    cand = u[:]
                    cand[j] += step
                    reflect01_inplace(cand)
                    fc = eval_u(cand)
                    evals += 1
                    if fc + eps < f:
                        u, f = cand, fc
                        improved = True
                        continue

                    if time_left() <= 0 or evals >= budget_evals:
                        break
                    cand = u[:]
                    cand[j] -= step
                    reflect01_inplace(cand)
                    fc = eval_u(cand)
                    evals += 1
                    if fc + eps < f:
                        u, f = cand, fc
                        improved = True

        return u, f

    # ---------- seeding ----------
    best_u = rand_u()
    best = eval_u(best_u)
    push_elite(best_u, best)

    if time_left() <= 0:
        return best

    H = max(64, min(1100, 80 + 24 * dim))
    R = max(16, min(220, 10 + 6 * dim))

    for _ in range(R):
        if time_left() <= 0:
            return best
        u = rand_u()
        f = eval_u(u)
        if f + eps < best:
            best, best_u = f, u
        push_elite(u, f)

    for k in range(1, H + 1):
        if time_left() <= 0:
            return best
        u = halton_u(k)
        f = eval_u(u)
        if f + eps < best:
            best, best_u = f, u
        push_elite(u, f)

    # ---------- diagonal ES core (CMA-like) ----------
    mean = best_u[:]

    base_sig = 0.30 / (1.0 + 0.02 * dim)
    logsig = [math.log(max(1e-8, min(0.7, base_sig))) for _ in range(dim)]
    sig_min, sig_max = 1e-12, 0.80

    lam0 = max(20, min(110, 12 + 5 * dim))
    lam = lam0
    mu = max(4, lam // 3)

    def make_weights(mu_):
        ws = [0.0] * mu_
        s = 0.0
        for i in range(mu_):
            w = math.log((mu_ + 0.5) / (i + 1.0))
            ws[i] = w
            s += w
        inv = 1.0 / max(1e-16, s)
        for i in range(mu_):
            ws[i] *= inv
        return ws

    weights = make_weights(mu)

    lr = 0.18 / (1.0 + 0.03 * dim)
    lr = min(0.22, max(0.04, lr))

    succ = 0
    gens = 0
    stagn = 0
    restarts = 0
    hal_idx = H + 1

    def anneal():
        frac = (time.time() - t0) / max(1e-12, (deadline - t0))
        return 1.0 - 0.18 * max(0.0, min(1.0, frac))

    while time_left() > 0:
        a = anneal()
        sig = [min(sig_max, max(sig_min, math.exp(ls))) for ls in logsig]

        # sometimes re-center at a different elite if stagnating
        if elite_u and random.random() < (0.15 if stagn >= 10 else 0.06):
            mean = elite_u[random.randrange(len(elite_u))][:]

        heavy_gen = (random.random() < 0.08) or (stagn >= 14 and random.random() < 0.40)

        pop = []  # list of (f, u, stepvec) where stepvec approximates normalized step / sigma
        # antithetic sampling: generate in pairs +/-z
        pairs = lam // 2
        for _ in range(pairs):
            if time_left() <= 0:
                return best
            z = [gauss() for _ in range(dim)]

            # occasionally heavy tail (replace some coords)
            if heavy_gen and random.random() < 0.6:
                # change a few coordinates to heavy tail
                for _k in range(max(1, dim // 6)):
                    j = random.randrange(dim)
                    z[j] = cauchy_like(1.0)

            for sgn in (1.0, -1.0):
                u = mean[:]
                for i in range(dim):
                    u[i] += sgn * z[i] * sig[i] * a
                reflect01_inplace(u)
                f = eval_u(u)

                # compute realized step in sigma units (post-reflection distortion exists,
                # but this is still better than raw z when boundaries are active)
                stepvec = [0.0] * dim
                for i in range(dim):
                    stepvec[i] = (u[i] - mean[i]) / max(1e-16, (sig[i] * a))

                pop.append((f, u, stepvec))
                push_elite(u, f)

                if time_left() <= 0:
                    return best

        # if lam is odd, add one extra
        if lam % 2 == 1 and time_left() > 0:
            u = mean[:]
            for i in range(dim):
                u[i] += gauss() * sig[i] * a
            reflect01_inplace(u)
            f = eval_u(u)
            stepvec = [(u[i] - mean[i]) / max(1e-16, (sig[i] * a)) for i in range(dim)]
            pop.append((f, u, stepvec))
            push_elite(u, f)

        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_u, _ = pop[0]

        # recombination -> new mean
        new_mean = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            uj = pop[j][1]
            for i in range(dim):
                new_mean[i] += w * uj[i]
        reflect01_inplace(new_mean)

        # diagonal sigma update from selected step statistics (E[step^2] target ~1)
        obs2 = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            sj = pop[j][2]
            for i in range(dim):
                obs2[i] += w * (sj[i] * sj[i])

        for i in range(dim):
            logsig[i] += 0.5 * lr * (obs2[i] - 1.0)
            si = math.exp(logsig[i])
            if si < sig_min:
                logsig[i] = math.log(sig_min)
            elif si > sig_max:
                logsig[i] = math.log(sig_max)

        mean = new_mean
        gens += 1

        if best_off_f + eps < best:
            best = best_off_f
            best_u = best_off_u[:]
            succ += 1
            stagn = 0
        else:
            stagn += 1

        # 1/5 success rule (global nudge)
        if gens >= 9:
            rate = succ / float(gens)
            mult = 1.16 if rate > 0.20 else 0.86
            for i in range(dim):
                si = math.exp(logsig[i]) * mult
                si = min(sig_max, max(sig_min, si))
                logsig[i] = math.log(si)
            succ = 0
            gens = 0

        # local refinement on stalls
        if stagn in (7, 11, 15) and time_left() > 0:
            # small budget; scaled with dim but capped
            bud = max(24, min(120, 18 + 3 * dim))
            ru, rf = local_pattern(best_u, best, budget_evals=bud)
            if rf + eps < best:
                best, best_u = rf, ru
                mean = best_u[:]
                stagn = 0
                push_elite(best_u, best)

        # restart / injection if stuck
        if stagn >= 22 and time_left() > 0:
            stagn = 0
            restarts += 1

            # modest IPOP-ish lambda growth early
            if restarts <= 5:
                lam = min(200, int(lam * 1.35) + 2)
            else:
                lam = max(lam0, int(lam * 0.95))

            mu = max(4, lam // 3)
            weights = make_weights(mu)

            # re-center
            if elite_u and random.random() < 0.60:
                mean = elite_u[random.randrange(len(elite_u))][:]
            else:
                mean = best_u[:]

            # widen sigmas a bit
            for i in range(dim):
                si = max(math.exp(logsig[i]), 0.20)
                logsig[i] = math.log(min(sig_max, si))

            # a few global injections
            for _ in range(2):
                if time_left() <= 0:
                    return best
                u = halton_u(hal_idx)
                hal_idx += 1
                f = eval_u(u)
                if f + eps < best:
                    best, best_u = f, u
                    mean = best_u[:]
                push_elite(u, f)

            if time_left() <= 0:
                return best
            u = rand_u()
            f = eval_u(u)
            if f + eps < best:
                best, best_u = f, u
                mean = best_u[:]
            push_elite(u, f)

    return best
