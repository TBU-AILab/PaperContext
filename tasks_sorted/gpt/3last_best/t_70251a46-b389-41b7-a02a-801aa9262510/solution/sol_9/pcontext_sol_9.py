import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded black-box minimizer.

    Improvements over your current best (diag ES + Halton + coord polish):
      - Uses reflection (not clipping) in normalized space to reduce boundary stickiness.
      - Keeps an elite pool for multi-basin robustness and restart centering.
      - Uses antithetic (mirrored) sampling to reduce variance.
      - Uses a stronger local optimizer: multi-scale pattern search in normalized space.
      - Adapts diagonal sigmas using *realized steps* (post-reflection) rather than raw z,
        which is more stable near boundaries.
      - Adds occasional "opposition / centroid" candidates and heavy-tail mutations when stuck.
      - Strict time guards.

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

    # reflect to [0,1] with period 2 (reduces clamp-sticking)
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

    def push_elite(u, f, cap=12):
        k = elite_key(u)
        if k not in elite_keys:
            elite_keys.add(k)
            elite_u.append(u[:])
            elite_f.append(f)
        else:
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

    # ---------- local multi-scale pattern search ----------
    def local_pattern(best_u, best_f, budget_evals=80):
        if budget_evals <= 0 or time_left() <= 0:
            return best_u, best_f

        u = best_u[:]
        f = best_f
        evals = 0

        # step sizes in normalized space (shrink with dim)
        base = 0.14 / (1.0 + 0.06 * dim)
        steps = [base, base * 0.5, base * 0.25, base * 0.125]

        coords = list(range(dim))
        random.shuffle(coords)

        for step in steps:
            if time_left() <= 0 or evals >= budget_evals:
                break
            improved = True
            while improved and time_left() > 0 and evals < budget_evals:
                improved = False
                for j in coords:
                    if time_left() <= 0 or evals >= budget_evals:
                        break

                    cand = u[:]
                    cand[j] += step
                    reflect01_inplace(cand)
                    fc = eval_u(cand); evals += 1
                    if fc + eps < f:
                        u, f = cand, fc
                        improved = True
                        continue

                    if time_left() <= 0 or evals >= budget_evals:
                        break
                    cand = u[:]
                    cand[j] -= step
                    reflect01_inplace(cand)
                    fc = eval_u(cand); evals += 1
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

    H = max(64, min(1200, 90 + 26 * dim))
    R = max(18, min(260, 12 + 7 * dim))

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

    base_sig = 0.32 / (1.0 + 0.02 * dim)
    logsig = [math.log(max(1e-8, min(0.75, base_sig))) for _ in range(dim)]
    sig_min, sig_max = 1e-12, 0.90

    lam0 = max(22, min(140, 14 + 6 * dim))
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
    lr = min(0.23, max(0.04, lr))

    succ = 0
    gens = 0
    stagn = 0
    restarts = 0
    hal_idx = H + 1

    def anneal():
        frac = (time.time() - t0) / max(1e-12, (deadline - t0))
        return 1.0 - 0.20 * max(0.0, min(1.0, frac))

    while time_left() > 0:
        a = anneal()
        sig = [min(sig_max, max(sig_min, math.exp(ls))) for ls in logsig]

        # occasional multi-basin recenter
        if elite_u and random.random() < (0.18 if stagn >= 10 else 0.07):
            mean = elite_u[random.randrange(len(elite_u))][:]

        heavy_gen = (random.random() < 0.08) or (stagn >= 14 and random.random() < 0.45)

        pop = []  # (f, u, stepvec)
        pairs = lam // 2

        for _ in range(pairs):
            if time_left() <= 0:
                return best
            z = [gauss() for _ in range(dim)]
            if heavy_gen and random.random() < 0.65:
                for _k in range(max(1, dim // 6)):
                    j = random.randrange(dim)
                    z[j] = cauchy_like(1.0)

            for sgn in (1.0, -1.0):
                u = mean[:]
                for i in range(dim):
                    u[i] += sgn * z[i] * sig[i] * a
                reflect01_inplace(u)
                f = eval_u(u)

                # realized step in sigma-units (robust near boundaries)
                stepvec = [(u[i] - mean[i]) / max(1e-16, sig[i] * a) for i in range(dim)]
                pop.append((f, u, stepvec))

                push_elite(u, f)
                if f + eps < best:
                    best, best_u = f, u[:]

                if time_left() <= 0:
                    return best

        if lam % 2 == 1 and time_left() > 0:
            u = mean[:]
            for i in range(dim):
                u[i] += gauss() * sig[i] * a
            reflect01_inplace(u)
            f = eval_u(u)
            stepvec = [(u[i] - mean[i]) / max(1e-16, sig[i] * a) for i in range(dim)]
            pop.append((f, u, stepvec))
            push_elite(u, f)
            if f + eps < best:
                best, best_u = f, u[:]

        pop.sort(key=lambda t: t[0])

        # recombination
        new_mean = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            uj = pop[j][1]
            for i in range(dim):
                new_mean[i] += w * uj[i]
        reflect01_inplace(new_mean)

        # diagonal sigma update from selected realized steps
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

        # stagnation tracking (use offspring best vs global best already updated)
        if pop[0][0] + eps < best:
            stagn = 0
            succ += 1
        else:
            stagn += 1
        gens += 1

        # 1/5 success rule (global nudge)
        if gens >= 9:
            rate = succ / float(gens)
            mult = 1.17 if rate > 0.20 else 0.86
            for i in range(dim):
                si = math.exp(logsig[i]) * mult
                si = min(sig_max, max(sig_min, si))
                logsig[i] = math.log(si)
            succ = 0
            gens = 0

        # local refinement on stalls
        if stagn in (7, 11, 15) and time_left() > 0:
            bud = max(30, min(140, 22 + 3 * dim))
            ru, rf = local_pattern(best_u, best, budget_evals=bud)
            if rf + eps < best:
                best, best_u = rf, ru
                mean = best_u[:]
                stagn = 0
                push_elite(best_u, best)

        # cheap "opposition" / centroid attempts sometimes (helps on symmetric landscapes)
        if elite_u and time_left() > 0 and random.random() < (0.06 if stagn < 10 else 0.12):
            # centroid of a few best elites
            idxs = sorted(range(len(elite_f)), key=lambda i: elite_f[i])[:min(4, len(elite_f))]
            c = [0.0] * dim
            for ii in idxs:
                uu = elite_u[ii]
                for d in range(dim):
                    c[d] += uu[d]
            inv = 1.0 / float(len(idxs))
            for d in range(dim):
                c[d] *= inv
            reflect01_inplace(c)
            fc = eval_u(c)
            push_elite(c, fc)
            if fc + eps < best:
                best, best_u = fc, c[:]
                mean = best_u[:]
                stagn = 0

            # opposition of current best
            if time_left() > 0:
                opp = [1.0 - best_u[d] for d in range(dim)]
                reflect01_inplace(opp)
                fo = eval_u(opp)
                push_elite(opp, fo)
                if fo + eps < best:
                    best, best_u = fo, opp[:]
                    mean = best_u[:]
                    stagn = 0

        # restart / injection if stuck
        if stagn >= 22 and time_left() > 0:
            stagn = 0
            restarts += 1

            # modest IPOP-ish lambda growth early
            if restarts <= 5:
                lam = min(240, int(lam * 1.35) + 2)
            else:
                lam = max(lam0, int(lam * 0.93))

            mu = max(4, lam // 3)
            weights = make_weights(mu)

            # re-center
            if elite_u and random.random() < 0.65:
                mean = elite_u[random.randrange(len(elite_u))][:]
            else:
                mean = best_u[:]

            # widen sigmas
            for i in range(dim):
                si = max(math.exp(logsig[i]), 0.22)
                logsig[i] = math.log(min(sig_max, si))

            # global injections
            for _ in range(3):
                if time_left() <= 0:
                    return best
                u = halton_u(hal_idx); hal_idx += 1
                f = eval_u(u)
                push_elite(u, f)
                if f + eps < best:
                    best, best_u = f, u[:]
                    mean = best_u[:]

            if time_left() <= 0:
                return best
            u = rand_u()
            f = eval_u(u)
            push_elite(u, f)
            if f + eps < best:
                best, best_u = f, u[:]
                mean = best_u[:]

    return best
