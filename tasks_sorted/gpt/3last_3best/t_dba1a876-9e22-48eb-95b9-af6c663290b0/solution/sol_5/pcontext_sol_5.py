import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements over your best hybrid (JADE + diagonal-elite model + LS):
      1) Multi-start "islands" (sequential, time-sliced): prevents early lock-in to a mediocre basin.
      2) Stronger model: diagonal elite model with time-adaptive shrink + occasional covariance-free
         "line-sampling" along elite difference vectors (captures rotations a bit without full CMA).
      3) Smarter acceptance for model samples: replaces a *random* non-elite more often, preserving elites.
      4) Dedicated trust-region local search late: (1+1)-ES + coordinate pattern with per-dim sigma,
         plus 2-point quadratic interpolation on a few coordinates (cheap surrogate along axis).
      5) Cache uses span-scaled quantization but also stores exact tuple to reduce harmful collisions.

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
        if spans[i] < 0.0:
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

    def cauchy0(scale=1.0):
        # centered Cauchy with scale
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling ----------
    def mirror_wrap_clip_inplace(x):
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            s = spans[i]
            if s <= 0.0:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                t = (v - lo) % (2.0 * s)
                if t < 0.0:
                    t += 2.0 * s
                if t > s:
                    t = 2.0 * s - t
                v = lo + t
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
    cache_q = {}
    cache_exact = {}

    q = [max(1e-12, (spans[i] if spans[i] > 0 else 1.0) * 1e-12) for i in range(dim)]

    def key_of_q(x):
        return tuple(int(round(x[i] / q[i])) for i in range(dim))

    def key_of_exact(x):
        # exact float tuples can be large; keep only for already-visited points
        return tuple(x)

    def evaluate(x):
        # try exact first (if present), else quantized
        ke = key_of_exact(x)
        v = cache_exact.get(ke)
        if v is not None:
            return v
        kq = key_of_q(x)
        v = cache_q.get(kq)
        if v is None:
            v = float(func(list(x)))
            cache_q[kq] = v
            cache_exact[ke] = v
        else:
            # also populate exact to reduce future quant collisions
            cache_exact[ke] = v
        return v

    # ---------- low-discrepancy init (scrambled Halton) ----------
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

    # ---------- local search (late-stage trust region) ----------
    def local_search_trust(x0, f0, time_limit):
        x = x0[:]
        fx = f0

        # per-dim sigma (normalized), adapted
        sig = [0.08 for _ in range(dim)]
        succ = 0
        trials = 0

        # coord steps
        steps = [0.08 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = 1e-14

        order = list(range(dim))

        def axis_quadratic(i, base_x, base_f, step):
            """
            1D quadratic interpolation along axis i using f(-h), f(0), f(+h).
            Returns candidate xq (or None) and fq.
            """
            if step <= 0.0:
                return None, None
            x_m = base_x[:]
            x_p = base_x[:]
            x_m[i] -= step
            x_p[i] += step
            mirror_wrap_clip_inplace(x_m)
            mirror_wrap_clip_inplace(x_p)
            f_m = evaluate(x_m)
            f_p = evaluate(x_p)
            f_0 = base_f

            # Fit parabola through (-h,fm),(0,f0),(+h,fp); vertex at:
            # x* = h*(fm - fp)/(2*(fm - 2f0 + fp))
            denom = (f_m - 2.0 * f_0 + f_p)
            if denom == 0.0:
                return None, None
            xstar = step * (f_m - f_p) / (2.0 * denom)
            # limit to reasonable range
            if abs(xstar) > 1.8 * step:
                return None, None
            x_q = base_x[:]
            x_q[i] += xstar
            mirror_wrap_clip_inplace(x_q)
            f_q = evaluate(x_q)
            return x_q, f_q

        while time.time() < time_limit:
            trials += 1

            # (1+1)-ES step (diagonal)
            y = x[:]
            for j in range(dim):
                if spans[j] > 0:
                    y[j] += randn() * (sig[j] * spans[j])
            mirror_wrap_clip_inplace(y)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                succ += 1
            else:
                # coordinate + quadratic on a couple axes
                random.shuffle(order)
                improved = False
                for i in order[: max(1, min(dim, 3))]:
                    si = steps[i]
                    if si <= min_step:
                        continue

                    # try simple +/- first
                    xp = x[:]
                    xm = x[:]
                    xp[i] += si
                    xm[i] -= si
                    mirror_wrap_clip_inplace(xp)
                    mirror_wrap_clip_inplace(xm)
                    fp = evaluate(xp)
                    fm = evaluate(xm)

                    best_local_f = fx
                    best_local_x = None
                    if fp < best_local_f:
                        best_local_f, best_local_x = fp, xp
                    if fm < best_local_f:
                        best_local_f, best_local_x = fm, xm

                    # try quadratic interpolation if informative
                    xq, fq = axis_quadratic(i, x, fx, si)
                    if xq is not None and fq < best_local_f:
                        best_local_f, best_local_x = fq, xq

                    if best_local_x is not None:
                        x, fx = best_local_x, best_local_f
                        succ += 1
                        improved = True
                        break
                    else:
                        steps[i] *= 0.82  # shrink only the tried axis

                if not improved:
                    # gentle global shrink to focus
                    for j in range(dim):
                        steps[j] *= 0.92
                        sig[j] *= 0.97

            # adapt sigmas and avoid collapse
            if trials % 22 == 0:
                rate = succ / 22.0
                if rate > 0.25:
                    mul = 1.18
                elif rate < 0.15:
                    mul = 0.78
                else:
                    mul = 1.0
                for j in range(dim):
                    sig[j] = max(1e-6, min(0.35, sig[j] * mul))
                succ = 0

            if max(steps) <= min_step:
                break

        return x, fx

    # ---------- island runner ----------
    def run_island(time_budget, seed_offset):
        nonlocal t0

        island_deadline = time.time() + time_budget

        pop_size = max(18, min(84, 10 * dim + 24))
        archive_max = pop_size

        pop = []
        # center + corners
        center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        pop.append(center)
        for _ in range(min(6, pop_size - len(pop))):
            pop.append([highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)])

        # halton + opposition
        idx = 1 + seed_offset
        while len(pop) < pop_size:
            x = from_unit(halton_point(idx))
            idx += 1
            pop.append(x)
            if len(pop) < pop_size:
                pop.append(opposition(x))

        pop = pop[:pop_size]
        fit = [evaluate(ind) for ind in pop]
        best_i = min(range(pop_size), key=lambda i: fit[i])
        best_x = pop[best_i][:]
        best = fit[best_i]

        archive = []

        # JADE params
        mu_F = 0.55
        mu_CR = 0.60
        c = 0.1

        # model params
        elite_frac = 0.22

        def build_elite_model():
            idxs = sorted(range(pop_size), key=lambda i: fit[i])
            m = max(3, int(elite_frac * pop_size))
            elites = idxs[:m]

            weights = []
            for r in range(m):
                weights.append(max(0.0, math.log(m + 0.5) - math.log(r + 1.0)))
            sw = sum(weights) if sum(weights) > 0 else 1.0
            weights = [w / sw for w in weights]

            mean = [0.0] * dim
            for w, ei in zip(weights, elites):
                x = pop[ei]
                for j in range(dim):
                    mean[j] += w * x[j]

            var = [0.0] * dim
            for w, ei in zip(weights, elites):
                x = pop[ei]
                for j in range(dim):
                    d = x[j] - mean[j]
                    var[j] += w * d * d

            std = [0.0] * dim
            for j in range(dim):
                s = math.sqrt(max(0.0, var[j]))
                floor = max(1e-12, 0.006 * spans[j] if spans[j] > 0 else 1e-12)
                cap = (0.40 * spans[j]) if spans[j] > 0 else 0.0
                s = max(floor, s)
                if cap > 0.0:
                    s = min(cap, s)
                std[j] = s

            return mean, std, elites

        model_mean, model_std, elites = build_elite_model()
        next_model_refresh = time.time() + 0.10 * time_budget

        def pick_from_union(exclude_ids):
            union = pop + archive
            for _ in range(20):
                cand = random.choice(union)
                if id(cand) not in exclude_ids:
                    return cand
            return random.choice(union)

        # for line-sampling (approx rotation)
        def elite_direction_sample():
            if len(elites) < 2:
                return None
            a = pop[random.choice(elites)]
            b = pop[random.choice(elites)]
            if a is b:
                return None
            d = [a[j] - b[j] for j in range(dim)]
            # normalize direction in span-scaled metric
            norm2 = 0.0
            for j in range(dim):
                sj = spans[j] if spans[j] > 0 else 1.0
                z = d[j] / sj
                norm2 += z * z
            if norm2 <= 0.0:
                return None
            inv = 1.0 / math.sqrt(norm2)
            for j in range(dim):
                sj = spans[j] if spans[j] > 0 else 1.0
                d[j] = (d[j] / sj) * inv  # now in normalized space
            return d

        last_best = best
        stagn = 0

        while time.time() < island_deadline:
            now = time.time()
            frac = (now - (island_deadline - time_budget)) / max(1e-12, time_budget)

            if now >= next_model_refresh:
                model_mean, model_std, elites = build_elite_model()
                next_model_refresh = now + (0.06 + 0.10 * (1.0 - frac)) * time_budget

            ranked = sorted(range(pop_size), key=lambda i: fit[i])
            pbest_frac = 0.24 - 0.18 * min(1.0, max(0.0, frac))
            pcount = max(2, int(math.ceil(pbest_frac * pop_size)))

            # time-adaptive mixture: more model later
            p_model = 0.22 + 0.58 * frac

            # keep top-k protected from replacement by model samples
            protect = max(2, int(0.15 * pop_size))
            protected = set(ranked[:protect])

            S_F, S_CR = [], []

            for i in range(pop_size):
                if time.time() >= island_deadline:
                    break

                xi = pop[i]

                # ---------- model / line sampling ----------
                if random.random() < p_model:
                    # choose target to replace: avoid protected most of the time
                    if i in protected and random.random() < 0.85:
                        # pick a random non-protected index instead
                        i2 = ranked[random.randrange(protect, pop_size)]
                    else:
                        i2 = i

                    base = model_mean[:]
                    # global shrink late
                    shrink = 1.0 - 0.55 * frac  # 1.0 -> 0.45
                    heavy = (random.random() < 0.12)

                    # sometimes do line-sampling along elite differences
                    if random.random() < (0.18 + 0.10 * frac):
                        d = elite_direction_sample()
                    else:
                        d = None

                    u = base[:]
                    if d is None:
                        for j in range(dim):
                            if spans[j] <= 0.0:
                                u[j] = lows[j]
                                continue
                            step = randn() * (shrink * model_std[j])
                            if heavy:
                                step *= (1.0 + abs(cauchy0(0.9)))
                            u[j] += step
                    else:
                        # sample scalar along direction in normalized space
                        alpha = randn() * (0.35 * shrink)
                        if heavy:
                            alpha *= (1.0 + abs(cauchy0(0.8)))
                        for j in range(dim):
                            if spans[j] > 0.0:
                                u[j] += alpha * spans[j] * d[j]

                        # add small diagonal noise too
                        for j in range(dim):
                            if spans[j] > 0.0 and random.random() < 0.35:
                                u[j] += randn() * (0.20 * shrink * model_std[j])

                    mirror_wrap_clip_inplace(u)
                    fu = evaluate(u)

                    if fu <= fit[i2]:
                        if i2 != i:
                            # archive replaced solution (i2)
                            archive.append(pop[i2])
                        else:
                            archive.append(xi)
                        if len(archive) > archive_max:
                            archive.pop(random.randrange(len(archive)))
                        pop[i2] = u
                        fit[i2] = fu
                        if fu < best:
                            best, best_x = fu, u[:]
                    continue

                # ---------- JADE/current-to-pbest/1 ----------
                CR = mu_CR + 0.1 * randn()
                if CR < 0.0:
                    CR = 0.0
                elif CR > 1.0:
                    CR = 1.0

                # F from cauchy around mu_F
                F = mu_F + 0.1 * cauchy0(1.0)
                tries = 0
                while F <= 0.0 and tries < 10:
                    F = mu_F + 0.1 * cauchy0(1.0)
                    tries += 1
                if F <= 0.0:
                    F = 0.1
                if F > 1.0:
                    F = 1.0

                pbest_idx = ranked[random.randrange(pcount)]
                xpbest = pop[pbest_idx]

                exclude = {id(xi), id(xpbest)}
                # r1 from pop
                for _ in range(20):
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

            if S_F:
                sf = sum(S_F)
                if sf > 0.0:
                    mu_F = (1.0 - c) * mu_F + c * (sum(f * f for f in S_F) / sf)
                mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

            # stagnation & injection
            if best < last_best - 1e-12:
                last_best = best
                stagn = 0
            else:
                stagn += 1

            if stagn >= 7 and time.time() < island_deadline:
                # replace some worst with mix near-best / random
                order = sorted(range(pop_size), key=lambda k: fit[k])
                keep = max(6, int(0.55 * pop_size))
                for idxr in order[keep:]:
                    if time.time() >= island_deadline:
                        break
                    r = random.random()
                    if r < 0.70:
                        xnew = best_x[:]
                        rad = (0.02 + 0.10 * min(1.0, stagn / 18.0))
                        for j in range(dim):
                            if spans[j] > 0:
                                xnew[j] += (random.random() - random.random()) * (rad * spans[j])
                        mirror_wrap_clip_inplace(xnew)
                    else:
                        xnew = rand_vec()
                    fnew = evaluate(xnew)
                    pop[idxr] = xnew
                    fit[idxr] = fnew
                    if fnew < best:
                        best, best_x = fnew, xnew[:]
                        last_best = best
                stagn = 0

        return best, best_x

    # ---------- main: multi-start islands + final polishing ----------
    best = float("inf")
    best_x = None

    # If time is tiny, just sample
    if max_time <= 0.0:
        return best

    # Reserve last slice for local search polishing
    polish_frac = 0.22 if max_time >= 1.0 else 0.15
    polish_time = max(0.0, polish_frac * max_time)

    # Islands share the remaining time
    remaining = max(0.0, max_time - polish_time)
    # 3 islands typically helps; scale down if little time
    islands = 3 if remaining >= 0.6 else (2 if remaining >= 0.25 else 1)
    per_island = remaining / max(1, islands)

    for s in range(islands):
        if time.time() >= deadline:
            break
        # small jitter to avoid exact equal slices
        jitter = 0.85 + 0.30 * random.random()
        budget = min(per_island * jitter, max(0.0, deadline - time.time() - polish_time))
        if budget <= 0.0:
            break
        b, bx = run_island(budget, seed_offset=1 + 37 * s)
        if b < best:
            best, best_x = b, bx

    # Final polishing on the best found
    now = time.time()
    if best_x is not None and now < deadline:
        time_left = deadline - now
        budget = min(polish_time, time_left)
        if budget > 0.0:
            x2, f2 = local_search_trust(best_x, best, now + budget)
            if f2 < best:
                best = f2
                best_x = x2

    # Safety: if nothing evaluated (shouldn't happen), do a few randoms
    if best == float("inf"):
        for _ in range(8):
            x = rand_vec()
            best = min(best, evaluate(x))

    return best
