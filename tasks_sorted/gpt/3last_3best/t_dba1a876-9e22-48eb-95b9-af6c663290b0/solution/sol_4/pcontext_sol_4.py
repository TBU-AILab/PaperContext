import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Upgrade vs your best JADE-like DE:
      - Multi-armed exploration/exploitation: CMA-ES-like weighted sampling around an elite set
        + JADE/current-to-pbest/1 with archive (global exploration)
      - Time-adaptive mixing: early DE-heavy, later model-heavy + stronger local search
      - Two-stage evaluation:
          * cheap cache + "probe" micro-mutations around improvements
          * periodic elite-resampling refresh
      - Better bounded handling: mirror-wrap then clip (reduces bias near bounds)
      - Local search: coordinate/pattern + (1+1)-ES blended, with step control

    Returns:
      best fitness found within max_time seconds (float).
    """

    t0 = time.time()
    deadline = t0 + float(max_time)
    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    # ---------- RNG helpers ----------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy(mu, gamma=0.1):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling ----------
    def mirror_wrap_clip_inplace(x):
        # mirror-wrap into [lo,hi] with period 2*span, then clip (handles far jumps better than single reflect)
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            s = spans[i]
            if s <= 0.0:
                x[i] = lo
                continue

            v = x[i]
            if v < lo or v > hi:
                # map to [0, 2s)
                t = (v - lo) % (2.0 * s)
                if t < 0.0:
                    t += 2.0 * s
                # reflect second half
                if t > s:
                    t = 2.0 * s - t
                v = lo + t

            # final clip (numerical safety)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x[i] = v
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------- evaluation cache ----------
    cache = {}
    # quantization step: ~1e-12 span but not too small
    q = [max(1e-12, (spans[i] if spans[i] > 0 else 1.0) * 1e-12) for i in range(dim)]

    def key_of(x):
        return tuple(int(round(x[i] / q[i])) for i in range(dim))

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is None:
            v = float(func(list(x)))
            cache[k] = v
        return v

    # ---------- low discrepancy init (scrambled Halton) ----------
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

    def van_der_corput(k, base, scramble=0):
        v = 0.0
        denom = 1.0
        while k > 0:
            k, rem = divmod(k, base)
            rem = (rem + scramble) % base
            denom *= base
            v += rem / denom
        return v

    primes = first_primes(max(1, dim))
    scr = [random.randrange(primes[i]) for i in range(dim)]

    def halton_point(index):  # index>=1
        return [van_der_corput(index, primes[j], scr[j]) for j in range(dim)]

    def from_unit(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    # ---------- Local search: blended coord + (1+1)-ES ----------
    def local_search_blend(x0, f0, time_limit):
        x = x0[:]
        fx = f0

        # per-dim steps for coord search
        steps = [0.10 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = 1e-14

        # isotropic sigma for ES
        sigma = 0.10
        succ = 0
        trials = 0

        order = list(range(dim))

        while time.time() < time_limit:
            trials += 1

            # ES move
            y = x[:]
            for i in range(dim):
                if spans[i] > 0:
                    y[i] += randn() * (sigma * spans[i])
            mirror_wrap_clip_inplace(y)
            fy = evaluate(y)
            if fy < fx:
                x, fx = y, fy
                succ += 1
            else:
                # coordinate/pattern tries
                random.shuffle(order)
                improved = False
                for i in order[: max(1, min(dim, 4))]:  # probe a few coords per loop
                    si = steps[i]
                    if si <= min_step:
                        continue
                    # +/- si
                    xp = x[:]
                    xp[i] += si
                    mirror_wrap_clip_inplace(xp)
                    fp = evaluate(xp)
                    xm = x[:]
                    xm[i] -= si
                    mirror_wrap_clip_inplace(xm)
                    fm = evaluate(xm)

                    if fp < fx or fm < fx:
                        if fp <= fm:
                            x2, f2 = xp, fp
                            direction = +1.0
                        else:
                            x2, f2 = xm, fm
                            direction = -1.0
                        # small pattern step
                        xt = x2[:]
                        xt[i] += direction * 0.6 * si
                        mirror_wrap_clip_inplace(xt)
                        ft = evaluate(xt)
                        if ft < f2:
                            x2, f2 = xt, ft
                        x, fx = x2, f2
                        improved = True
                        break

                if not improved:
                    # shrink coord steps a bit when no coord improvement
                    for i in range(dim):
                        steps[i] *= 0.85

            # adapt sigma (rough 1/5 success)
            if trials % 18 == 0:
                rate = succ / 18.0
                if rate > 0.25:
                    sigma *= 1.22
                elif rate < 0.15:
                    sigma *= 0.72
                sigma = max(1e-6, min(0.5, sigma))
                succ = 0

            # stop if coord steps died
            if max(steps) <= min_step and sigma <= 1e-5:
                break

        return x, fx

    # ---------- Initialization ----------
    pop_size = max(20, min(80, 10 * dim + 20))

    pop = []
    # center and some corners
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop.append(center)
    for _ in range(min(6, pop_size - len(pop))):
        x = [highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)]
        pop.append(x)

    # halton + opposition
    while len(pop) < pop_size:
        k = len(pop) + 1
        x = from_unit(halton_point(k))
        pop.append(x)
        if len(pop) < pop_size:
            pop.append(opposition(x))

    pop = pop[:pop_size]
    fit = [evaluate(ind) for ind in pop]
    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # ---------- DE (JADE-like) state ----------
    archive = []
    archive_max = pop_size
    mu_F = 0.55
    mu_CR = 0.60
    c = 0.1

    def pick_from_union(exclude_ids):
        union = pop + archive
        for _ in range(24):
            cand = random.choice(union)
            if id(cand) not in exclude_ids:
                return cand
        return random.choice(union)

    # ---------- Elite model state (CMA-ES-ish diagonal) ----------
    # We'll sample around a weighted mean of elites with per-dim std from elite spread.
    elite_frac = 0.22
    model_min_std = 1e-9  # absolute (later scaled by span)
    model_max_std_frac = 0.35  # relative to span cap

    def build_elite_model():
        # returns (mean, stds)
        idxs = sorted(range(pop_size), key=lambda i: fit[i])
        m = max(3, int(elite_frac * pop_size))
        elites = idxs[:m]

        # weights (log)
        weights = []
        for r in range(m):
            weights.append(max(0.0, math.log(m + 0.5) - math.log(r + 1.0)))
        sw = sum(weights) if sum(weights) > 0 else 1.0
        weights = [w / sw for w in weights]

        mean = [0.0] * dim
        for w, idx in zip(weights, elites):
            x = pop[idx]
            for j in range(dim):
                mean[j] += w * x[j]

        # weighted std
        var = [0.0] * dim
        for w, idx in zip(weights, elites):
            x = pop[idx]
            for j in range(dim):
                d = x[j] - mean[j]
                var[j] += w * d * d

        std = [0.0] * dim
        for j in range(dim):
            s = math.sqrt(max(0.0, var[j]))
            # ensure non-zero exploration; cap to avoid too wide
            floor = max(model_min_std, 0.01 * spans[j] if spans[j] > 0 else model_min_std)
            cap = model_max_std_frac * spans[j] if spans[j] > 0 else 0.0
            s = max(floor, s)
            if cap > 0.0:
                s = min(cap, s)
            std[j] = s
        return mean, std, elites

    model_mean, model_std, _ = build_elite_model()
    next_model_refresh = t0 + 0.12 * max_time

    # time scheduling
    next_ls = t0 + 0.35 * max_time
    ls_interval = 0.20 * max_time
    stagn = 0
    last_best = best

    # micro-probe around improvements
    def probe_around(x, fx, probes=2):
        nonlocal best, best_x
        curx, curf = x, fx
        for _ in range(probes):
            y = curx[:]
            for j in range(dim):
                if spans[j] > 0:
                    y[j] += (random.random() - random.random()) * 0.015 * spans[j]
            mirror_wrap_clip_inplace(y)
            fy = evaluate(y)
            if fy < curf:
                curx, curf = y, fy
                if fy < best:
                    best, best_x = fy, y[:]
        return curx, curf

    # ---------- main loop ----------
    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best

        gen += 1
        frac = (now - t0) / max(1e-12, (deadline - t0))

        # refresh model sometimes
        if now >= next_model_refresh:
            model_mean, model_std, _ = build_elite_model()
            # refresh faster later
            next_model_refresh = now + (0.10 + 0.12 * (1.0 - frac)) * max_time

        # mix probability: early more DE, later more model-sampling
        p_model = 0.20 + 0.55 * frac  # 0.20 -> 0.75

        # DE settings
        pbest_frac = 0.25 - 0.18 * min(1.0, max(0.0, frac))  # 0.25 -> 0.07
        pcount = max(2, int(math.ceil(pbest_frac * pop_size)))
        ranked = sorted(range(pop_size), key=lambda i: fit[i])

        S_F, S_CR = [], []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # --- choose generator: model sample or DE trial ---
            if random.random() < p_model:
                # sample from diagonal Gaussian around elite mean; occasional heavy-tail
                u = model_mean[:]
                heavy = (random.random() < 0.15)
                for j in range(dim):
                    if spans[j] <= 0:
                        u[j] = lows[j]
                        continue
                    step = randn() * model_std[j]
                    if heavy:
                        step *= (1.0 + abs(cauchy(0.0, 0.7)))  # occasional longer jump
                    u[j] += step
                mirror_wrap_clip_inplace(u)
                fu = evaluate(u)

                if fu <= fit[i]:
                    archive.append(xi)
                    if len(archive) > archive_max:
                        archive.pop(random.randrange(len(archive)))
                    pop[i] = u
                    fit[i] = fu
                    if fu < best:
                        best, best_x = fu, u[:]
                        # cheap probing can help a lot when model hits a good basin
                        _, _ = probe_around(best_x, best, probes=2)
                continue

            # --- JADE DE trial ---
            CR = mu_CR + 0.1 * randn()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = cauchy(mu_F, 0.1)
            tries = 0
            while F <= 0.0 and tries < 10:
                F = cauchy(mu_F, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest_idx = ranked[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            exclude = {id(xi), id(xpbest)}

            # r1 from population
            for _ in range(24):
                r1_idx = random.randrange(pop_size)
                xr1 = pop[r1_idx]
                if r1_idx != i and id(xr1) not in exclude:
                    break
            else:
                xr1 = pop[(i + 1) % pop_size]
            exclude.add(id(xr1))

            xr2 = pick_from_union(exclude)

            v = [xi[j] + F * (xpbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]

            mirror_wrap_clip_inplace(u)
            fu = evaluate(u)

            if fu <= fit[i]:
                archive.append(xi)
                if len(archive) > archive_max:
                    archive.pop(random.randrange(len(archive)))
                pop[i] = u
                fit[i] = fu
                S_F.append(F)
                S_CR.append(CR)
                if fu < best:
                    best, best_x = fu, u[:]
                    _, _ = probe_around(best_x, best, probes=2)

        # adapt mu_F, mu_CR
        if S_F:
            sf = sum(S_F)
            if sf > 0.0:
                mu_F = (1.0 - c) * mu_F + c * (sum(f * f for f in S_F) / sf)
            mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

        # stagnation
        if best < last_best - 1e-12:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # local search scheduled by time (more aggressive later)
        now = time.time()
        if now >= next_ls and now < deadline:
            remaining = deadline - now
            # increase LS share later
            budget = min((0.10 + 0.18 * frac) * max_time, 0.30 * remaining)
            if budget > 0:
                xls, fls = local_search_blend(best_x, best, now + budget)
                if fls < best:
                    best, best_x = fls, xls[:]
                    last_best = best
                    stagn = 0
            next_ls = now + ls_interval

        # diversity injection if stuck
        if stagn >= 6 and time.time() < deadline:
            order = sorted(range(pop_size), key=lambda k: fit[k])
            keep = max(6, int(0.55 * pop_size))

            # rebuild model from current elites to reset direction
            model_mean, model_std, elites = build_elite_model()

            # replace worst with mixture: near-best, model samples, and random
            for idx in order[keep:]:
                if time.time() >= deadline:
                    return best

                r = random.random()
                if r < 0.45:
                    xnew = best_x[:]
                    rad = (0.02 + 0.10 * min(1.0, stagn / 18.0))
                    for j in range(dim):
                        if spans[j] > 0:
                            xnew[j] += (random.random() - random.random()) * (rad * spans[j])
                    mirror_wrap_clip_inplace(xnew)
                elif r < 0.85:
                    xnew = model_mean[:]
                    for j in range(dim):
                        if spans[j] > 0:
                            xnew[j] += randn() * (1.4 * model_std[j])
                    mirror_wrap_clip_inplace(xnew)
                else:
                    xnew = rand_vec()

                fnew = evaluate(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    last_best = best

            # trim archive
            if len(archive) > archive_max:
                archive = archive[-archive_max:]
            stagn = 0
