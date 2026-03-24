import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (stdlib only).

    Improvements over your current best:
      - Still uses normalized [0,1]^dim, but handles boundary/clipping bias by
        "reflecting" mutations at edges (instead of hard-clamp).
      - Hybrid core:
          (A) Diagonal CMA-like (mu,lambda) evolution strategy for robust progress
          (B) Periodic "trust-region" local refinement via simultaneous perturbation
              (SPSA-like) to estimate a descent direction with just 2 evaluations,
              then backtracking line-search in normalized space.
      - Stronger restart logic:
          * multi-start with preserved elite pool
          * adaptive population sizing
          * occasional heavy-tailed exploration
      - More consistent time guarding and less overhead in inner loops.

    Returns:
      best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return deadline - time.time()

    if dim <= 0:
        # degenerate case
        return float(func([]))

    # ---------------- bounds / normalization ----------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

    def u_to_x(u):
        # expects u in [0,1]
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def eval_u(u):
        return float(func(u_to_x(u)))

    # reflection to reduce boundary bias vs clamping
    def reflect01(u):
        for i in range(dim):
            x = u[i]
            if 0.0 <= x <= 1.0:
                continue
            # reflect into [0,1] using period 2
            # x' = x mod 2; if >1 -> 2-x'
            y = x % 2.0
            if y > 1.0:
                y = 2.0 - y
            u[i] = y
        return u

    # ---------------- RNG helpers ----------------
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
        # ratio of Gaussians => heavy tail
        g = gauss()
        h = gauss()
        return (g / max(1e-16, abs(h))) * scale

    def rand_u():
        return [random.random() for _ in range(dim)]

    # ---------------- scrambled Halton ----------------
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

    # ---------------- small elite pool ----------------
    # keep a few diverse good points for restarts / recombination
    elite_u = []
    elite_f = []

    def elite_key(u):
        # coarse binning in normalized space
        # higher bins -> more diversity; keep cheap and deterministic
        bins = 1200
        key = []
        for i in range(dim):
            v = u[i]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            key.append(int(v * bins))
        return tuple(key)

    elite_keys = set()

    def push_elite(u, f, cap=8):
        k = elite_key(u)
        if k not in elite_keys:
            elite_keys.add(k)
            elite_u.append(u[:])
            elite_f.append(f)
        else:
            # even if same bin, still allow replacement if clearly better
            # (find one with same key)
            for idx, uu in enumerate(elite_u):
                if elite_key(uu) == k and f < elite_f[idx]:
                    elite_u[idx] = u[:]
                    elite_f[idx] = f
                    break

        if len(elite_f) > cap:
            idxs = sorted(range(len(elite_f)), key=lambda i: elite_f[i])
            idxs = idxs[:cap]
            elite_u[:] = [elite_u[i] for i in idxs]
            elite_f[:] = [elite_f[i] for i in idxs]
            elite_keys.clear()
            for uu in elite_u:
                elite_keys.add(elite_key(uu))

    # ---------------- SPSA-like local refinement ----------------
    # 2 evaluations to estimate a descent direction; then backtracking.
    def local_refine_spsa(u0, f0, base_radius=0.10):
        if time_left() <= 0:
            return u0, f0

        u = u0[:]
        f = f0

        # radius shrinks with time; also keep small in higher dim
        frac = (time.time() - t0) / max(1e-12, (deadline - t0))
        # start moderate, finish small
        rad = base_radius * (0.35 + 0.65 * (1.0 - frac))
        rad *= 1.0 / (1.0 + 0.03 * dim)
        rad = max(1e-6, min(0.25, rad))

        # random +/-1 direction
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

        up = [u[i] + rad * delta[i] for i in range(dim)]
        um = [u[i] - rad * delta[i] for i in range(dim)]
        reflect01(up)
        reflect01(um)

        if time_left() <= 0:
            return u, f
        fp = eval_u(up)
        if time_left() <= 0:
            return (up, fp) if fp < f else (u, f)
        fm = eval_u(um)

        if fp < f:
            u, f = up, fp
        if fm < f:
            u, f = um, fm

        # gradient estimate in normalized coordinates
        # g_i ~ (fp - fm) / (2*rad*delta_i)
        denom = max(1e-16, 2.0 * rad)
        gscale = (fp - fm) / denom

        # try a few step sizes along -sign(gscale)*delta
        # because gscale is scalar times delta, direction is +/- delta.
        # if fp>fm -> gscale>0 -> descent is -delta; else +delta
        sign = -1.0 if gscale > 0.0 else 1.0
        direction = [sign * delta[i] for i in range(dim)]

        # backtracking
        alpha = rad * 1.6
        for _ in range(5):
            if time_left() <= 0:
                break
            cand = [u[i] + alpha * direction[i] for i in range(dim)]
            reflect01(cand)
            fc = eval_u(cand)
            if fc < f:
                u, f = cand, fc
                alpha *= 1.1
            else:
                alpha *= 0.5

        return u, f

    # ---------------- initialize ----------------
    eps = 1e-15

    best_u = rand_u()
    best = eval_u(best_u)
    push_elite(best_u, best)

    if time_left() <= 0:
        return best

    # seeding: mix random + Halton
    H = max(48, min(900, 60 + 20 * dim))
    R = max(14, min(180, 10 + 5 * dim))

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

    # ---------------- diagonal CMA-like ES core ----------------
    # mean starts at best, but we sometimes restart around other elite members
    mean = best_u[:]

    base = 0.30 / (1.0 + 0.015 * dim)
    logsig = [math.log(max(1e-8, min(0.6, base))) for _ in range(dim)]
    sig_min, sig_max = 1e-12, 0.75

    lam0 = max(18, min(100, 12 + 4 * dim))
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

    lr = 0.22 / (1.0 + 0.02 * dim)
    lr = min(0.28, max(0.05, lr))

    succ = 0
    gens = 0
    stagn = 0
    restarts = 0
    hal_idx = H + 1

    def anneal():
        frac = (time.time() - t0) / max(1e-12, (deadline - t0))
        # weak anneal: 1.0 -> ~0.80
        return 1.0 - 0.20 * max(0.0, min(1.0, frac))

    # main loop
    while time_left() > 0:
        a = anneal()

        # choose parent mean: mostly current mean, sometimes a good elite for multi-basin robustness
        if elite_u and random.random() < (0.18 if stagn >= 10 else 0.08):
            idx = min(len(elite_u) - 1, int(random.random() * len(elite_u)))
            mean = elite_u[idx][:]

        sig = [min(sig_max, max(sig_min, math.exp(ls))) for ls in logsig]

        # heavy-tailed injections sometimes (more when stuck)
        heavy_gen = (random.random() < 0.10) or (stagn >= 14 and random.random() < 0.40)

        pop = []  # list of (f, u, z)
        # generate offspring
        for k in range(lam):
            if time_left() <= 0:
                return best
            u = mean[:]  # mutate around mean
            z = [0.0] * dim

            if heavy_gen and (k < max(2, lam // 6)):
                for i in range(dim):
                    zi = cauchy_like(1.0)
                    z[i] = zi
                    u[i] += zi * sig[i] * a
            else:
                for i in range(dim):
                    zi = gauss()
                    z[i] = zi
                    u[i] += zi * sig[i] * a

            reflect01(u)
            f = eval_u(u)
            pop.append((f, u, z))
            push_elite(u, f)

        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_u, _ = pop[0]

        # recombination
        new_mean = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            uj = pop[j][1]
            for i in range(dim):
                new_mean[i] += w * uj[i]
        reflect01(new_mean)

        # sigma update using selected z^2 statistics
        obs2 = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            zj = pop[j][2]
            for i in range(dim):
                obs2[i] += w * (zj[i] * zj[i])

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

        # global 1/5 success rule
        if gens >= 9:
            rate = succ / float(gens)
            mult = 1.20 if rate > 0.20 else 0.85
            for i in range(dim):
                si = math.exp(logsig[i]) * mult
                si = min(sig_max, max(sig_min, si))
                logsig[i] = math.log(si)
            succ = 0
            gens = 0

        # periodic SPSA local refinement (cheap but often very effective)
        if (stagn in (6, 10, 14)) and time_left() > 0:
            ru, rf = local_refine_spsa(best_u, best, base_radius=max(0.03, min(0.14, math.exp(sum(logsig)/dim))))
            if rf + eps < best:
                best, best_u = rf, ru
                mean = best_u[:]
                stagn = 0
                push_elite(best_u, best)

        # restart / injection if stuck
        if stagn >= 22 and time_left() > 0:
            stagn = 0
            restarts += 1

            # increase population a bit early on (IPOP-ish)
            if restarts <= 5:
                lam = min(180, int(lam * 1.35) + 2)
            else:
                lam = max(lam0, int(lam * 0.95))

            mu = max(4, lam // 3)
            weights = make_weights(mu)

            # re-center to best or an elite point, and widen sigmas to escape basin
            if elite_u and random.random() < 0.55:
                mean = elite_u[random.randrange(len(elite_u))][:]
            else:
                mean = best_u[:]

            for i in range(dim):
                si = max(math.exp(logsig[i]), 0.22)
                logsig[i] = math.log(min(sig_max, si))

            # a couple of global injections
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
