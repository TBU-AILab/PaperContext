import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Main changes vs your best (JADE-like DE):
      - Better initialization: scrambled Halton + opposition + small Latin-ish shuffle + a few boundary/corners probes
      - Safer/faster constraint handling: reflection + clip (reduces boundary sticking)
      - JADE/current-to-pbest/1 with archive (kept), but:
          * uses proper normal sampling for CR (Box-Muller)
          * adaptive p (p-best fraction) shrinks over time (explore -> exploit)
          * "evaluation budget" aware: local search scheduled by time, not generations
      - Stronger local search: (1+1)-ES / hillclimb with adaptive isotropic sigma + occasional coordinate steps
      - Stagnation response: multi-radius restarts around best + random immigrants
      - Lightweight cache with quantization proportional to span (less harmful than fixed decimals)

    Returns:
        best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)
    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Guard against degenerate bounds
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    # ---------------- helpers ----------------
    def reflect_clip_inplace(x):
        # reflect once then clip (cheap and effective)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo:
                x[i] = lo + (lo - x[i])
                if x[i] > hi:
                    x[i] = lo
            elif x[i] > hi:
                x[i] = hi - (x[i] - hi)
                if x[i] < lo:
                    x[i] = hi
            # final clip
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Box-Muller normal
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

    # Cache quantization: scale-dependent rounding to preserve meaningful differences
    cache = {}
    # quant step: ~1e-11 of span, but not too tiny
    q = [max(1e-12, (spans[i] if spans[i] > 0 else 1.0) * 1e-11) for i in range(dim)]

    def key_of(x):
        # quantize each coordinate
        return tuple(int(round(x[i] / q[i])) for i in range(dim))

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is None:
            v = float(func(list(x)))
            cache[k] = v
        return v

    # ---------------- low-discrepancy init (scrambled Halton) ----------------
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
        # digit permutation by XOR scramble (cheap "scramble")
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

    # ---------------- local search: adaptive (1+1)-ES + coord nudges ----------------
    def local_search_es(x0, f0, time_limit):
        x = x0[:]
        fx = f0

        # sigma relative to span; adapt with 1/5 success rule-ish
        sigma = 0.12
        success = 0
        trials = 0

        # coordinate step fallback
        coord_step = [0.06 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_coord = 1e-14

        while time.time() < time_limit:
            trials += 1

            # propose isotropic mutation in normalized space
            y = x[:]
            for i in range(dim):
                if spans[i] > 0:
                    y[i] += randn() * (sigma * spans[i])
            reflect_clip_inplace(y)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                success += 1
            else:
                # occasional coordinate probe when ES stalls
                if dim > 0 and (trials % 7 == 0):
                    i = random.randrange(dim)
                    si = coord_step[i]
                    if si > min_coord:
                        y2 = x[:]
                        y2[i] += si if random.random() < 0.5 else -si
                        reflect_clip_inplace(y2)
                        fy2 = evaluate(y2)
                        if fy2 < fx:
                            x, fx = y2, fy2
                            success += 1
                        else:
                            coord_step[i] *= 0.7

            # adapt sigma occasionally
            if trials % 20 == 0:
                rate = success / 20.0
                # target success ~0.2
                if rate > 0.25:
                    sigma *= 1.25
                elif rate < 0.15:
                    sigma *= 0.7
                sigma = min(0.5, max(1e-6, sigma))
                success = 0

        return x, fx

    # ----------------------- init population -----------------------
    # Choose pop size to balance evaluation count and dimensionality
    pop_size = max(18, min(72, 12 * dim + 12))

    pop = []

    # A few explicit probes: center and some corners-ish (random sign)
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop.append(center)
    for _ in range(min(6, pop_size - len(pop))):
        x = []
        for i in range(dim):
            if spans[i] == 0:
                x.append(lows[i])
            else:
                x.append(highs[i] if random.random() < 0.5 else lows[i])
        pop.append(x)

    # Halton + opposition pairs
    need = pop_size - len(pop)
    hcount = max(0, need // 2)
    for k in range(1, hcount + 1):
        x = from_unit(halton_point(k))
        pop.append(x)
        if len(pop) < pop_size:
            pop.append(opposition(x))

    # Fill remaining with random + a light "shuffle" per dimension (latin-ish)
    while len(pop) < pop_size:
        pop.append(rand_vec())

    # Evaluate
    fit = [evaluate(ind) for ind in pop]
    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # ----------------------- JADE-like DE -----------------------
    archive = []
    archive_max = pop_size

    mu_F = 0.55
    mu_CR = 0.6
    c = 0.1

    last_best = best
    stagnation_gens = 0
    gen = 0

    # time-based scheduling for local search
    next_ls = t0 + 0.30 * float(max_time)  # first LS after some exploration
    ls_interval = 0.22 * float(max_time)   # then periodically

    def pick_from_union(exclude_ids):
        union = pop + archive
        for _ in range(16):
            cand = random.choice(union)
            if id(cand) not in exclude_ids:
                return cand
        return random.choice(union)

    while True:
        now = time.time()
        if now >= deadline:
            return best

        gen += 1
        # p-best fraction shrinks with time (more exploitation later)
        frac = (now - t0) / max(1e-12, (deadline - t0))
        p = 0.25 - 0.18 * min(1.0, max(0.0, frac))  # from 0.25 -> 0.07
        pcount = max(2, int(math.ceil(p * pop_size)))

        ranked = sorted(range(pop_size), key=lambda i: fit[i])

        S_F = []
        S_CR = []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # CR ~ N(mu_CR, 0.1) clipped
            CR = mu_CR + 0.1 * randn()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # F ~ Cauchy(mu_F, 0.1), resample if <=0, then clip to 1
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
            for _ in range(20):
                r1_idx = random.randrange(pop_size)
                xr1 = pop[r1_idx]
                if r1_idx != i and id(xr1) not in exclude:
                    break
            else:
                xr1 = pop[(i + 1) % pop_size]
            exclude.add(id(xr1))

            # r2 from union
            xr2 = pick_from_union(exclude)

            # Mutation
            v = [xi[j] + F * (xpbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]

            # Crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]

            reflect_clip_inplace(u)
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
                    best = fu
                    best_x = u[:]

        # Adapt mu_F, mu_CR
        if S_F:
            sf = sum(S_F)
            if sf > 0.0:
                mu_F = (1.0 - c) * mu_F + c * (sum(f * f for f in S_F) / sf)  # Lehmer mean
            mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

        # Stagnation
        if best < last_best - 1e-12:
            last_best = best
            stagnation_gens = 0
        else:
            stagnation_gens += 1

        # Time-scheduled local search (budgeted)
        now = time.time()
        if now >= next_ls and now < deadline:
            remaining = deadline - now
            budget = min(0.16 * float(max_time), 0.28 * remaining)
            if budget > 0:
                xls, fls = local_search_es(best_x, best, now + budget)
                if fls < best:
                    best, best_x = fls, xls[:]
                    last_best = best
                    stagnation_gens = 0
            next_ls = now + ls_interval

        # Diversity injection if stuck
        if stagnation_gens >= 7 and time.time() < deadline:
            order = sorted(range(pop_size), key=lambda k: fit[k])
            keep = max(6, int(0.55 * pop_size))

            # multi-radius around best
            radii = [0.02, 0.06, 0.14]
            for idx in order[keep:]:
                if time.time() >= deadline:
                    return best

                if random.random() < 0.75:
                    r = radii[random.randrange(len(radii))]
                    xnew = best_x[:]
                    for j in range(dim):
                        if spans[j] > 0:
                            # triangular noise gives fewer far outliers than gaussian
                            noise = (random.random() - random.random())
                            xnew[j] += noise * (r * spans[j])
                    reflect_clip_inplace(xnew)
                else:
                    xnew = rand_vec()

                fnew = evaluate(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    last_best = best

            # prune archive a bit (avoid bloat of old regions)
            if len(archive) > archive_max:
                archive = archive[-archive_max:]

            # reset stagnation
            stagnation_gens = 0
