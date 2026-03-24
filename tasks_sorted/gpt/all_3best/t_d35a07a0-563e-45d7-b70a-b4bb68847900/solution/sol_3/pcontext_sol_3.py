import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no numpy).

    Core ideas (vs previous best):
      - Keep an elite set + full population (CMA-ES-inspired weighted mean)
      - Use diagonal covariance adaptation (per-dimension sigma) + global sigma
      - Mix 3 move types: (A) DE/current-to-best, (B) elite-mean Gaussian, (C) trust-region coordinate pattern
      - Stronger restart logic: stagnation + sigma-collapse triggers
      - Batch-like evaluation pacing: avoid expensive sorts too often
      - Robust evaluation (exceptions/NaN/inf)

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- helpers --------------------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Box-Muller Gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Halton sequence for seeding (low discrepancy)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    def weighted_mean(elites):
        # elites: list of (f,x) sorted best first
        m = len(elites)
        if m == 1:
            return list(elites[0][1])
        # log weights (CMA-ES-like), positive
        ws = []
        for i in range(m):
            ws.append(max(0.0, math.log(m + 0.5) - math.log(i + 1.0)))
        s = sum(ws)
        if s <= 0:
            ws = [1.0] * m
            s = float(m)
        mean = [0.0] * dim
        for i in range(m):
            w = ws[i] / s
            xi = elites[i][1]
            for j in range(dim):
                mean[j] += w * xi[j]
        return mean

    def pick3(n, exclude):
        # pick 3 distinct indices in [0,n) != exclude
        a = random.randrange(n - 1)
        if a >= exclude:
            a += 1
        b = random.randrange(n - 2)
        # map b to skip exclude and a
        # build via rejection quickly (n is small)
        while True:
            b = random.randrange(n)
            if b != exclude and b != a:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != a and c != b:
                break
        return a, b, c

    # -------------------- setup --------------------
    start = time.time()
    max_time = float(max_time) if max_time is not None else 0.0
    deadline = start + max(0.0, max_time)

    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # Population sizing
    pop_size = max(10, min(40, 5 * dim))
    elite_size = max(4, min(12, pop_size // 2))

    # Diagonal "covariance": per-dim step; plus global scalar
    sigma_d = [0.22 * s for s in span]  # per-dimension
    sigma_g = 1.0

    min_sigma_d = [1e-15 * s for s in span]
    max_sigma_d = [0.80 * s for s in span]

    # DE parameters
    F_base = 0.55
    CR_base = 0.85

    # bookkeeping for adaptation
    accepted = 0
    attempted = 0
    adapt_window = 35

    # restart logic
    best = float("inf")
    best_x = None
    last_improve_t = start
    stagnation = max(0.30, 0.12 * max_time)

    # -------------------- init population (Halton + random + opposition) --------------------
    shift = [random.random() for _ in range(dim)]
    pop = []
    k = 1
    init_budget = max(pop_size, 12 * dim)

    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1

        fx = eval_f(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, list(x)
            last_improve_t = time.time()

        # opposition point often helps for bounded problems
        if time.time() >= deadline:
            break
        if random.random() < 0.55:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, list(xo)
                last_improve_t = time.time()

        i += 1

    if not pop:
        x = rand_vec()
        return eval_f(x)

    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_size]
    best, best_x = pop[0][0], list(pop[0][1])

    # -------------------- main loop --------------------
    # coordinate pattern trust steps (relative to sigma_d)
    def coord_pattern(x, fx, step_scale):
        nonlocal best, best_x, last_improve_t
        # try a handful of coordinates; accept first improvement (greedy)
        m = 1 if dim == 1 else min(dim, 3)
        idxs = random.sample(range(dim), m)
        for j in idxs:
            lo, hi = bounds[j]
            step = step_scale * sigma_d[j]
            if step <= 0:
                continue
            xp = list(x)
            xp[j] = clamp(xp[j] + step, lo, hi)
            fp = eval_f(xp)
            if fp < fx:
                if fp < best:
                    best, best_x = fp, list(xp)
                    last_improve_t = time.time()
                return xp, fp

            xm = list(x)
            xm[j] = clamp(xm[j] - step, lo, hi)
            fm = eval_f(xm)
            if fm < fx:
                if fm < best:
                    best, best_x = fm, list(xm)
                    last_improve_t = time.time()
                return xm, fm
        return x, fx

    # cheap sigma collapse indicator
    def sigma_too_small():
        # if most sigmas are extremely small, we're stuck
        cnt = 0
        for j in range(dim):
            if sigma_d[j] <= (5e-14 * span[j]):
                cnt += 1
        return cnt >= max(1, int(0.75 * dim))

    while time.time() < deadline:
        now = time.time()
        # time fraction for schedule
        tfrac = 0.0
        if deadline > start:
            tfrac = (now - start) / (deadline - start)
            if tfrac < 0.0:
                tfrac = 0.0
            elif tfrac > 1.0:
                tfrac = 1.0

        # maintain elites + mean
        # sort occasionally only (pop is small anyway)
        if random.random() < 0.18:
            pop.sort(key=lambda t: t[0])
            pop = pop[:pop_size]
        else:
            # ensure best is tracked
            # (pop[0] may not be best if unsorted)
            for fx, x in pop:
                if fx < best:
                    best, best_x = fx, list(x)
                    last_improve_t = time.time()

        pop.sort(key=lambda t: t[0])
        elites = pop[:elite_size]
        mean_x = weighted_mean(elites)

        # restart if stagnating or sigma collapse
        if (now - last_improve_t) > stagnation or sigma_too_small():
            inject = max(2, pop_size // 3)
            new_pop = elites[:]  # keep elites
            # reset sigmas moderately (not too large)
            sigma_d = [0.25 * s for s in span]
            sigma_g = 1.0
            accepted = 0
            attempted = 0
            last_improve_t = now

            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if random.random() < 0.65 and best_x is not None:
                    xr = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        xr.append(clamp(best_x[j] + gauss() * (0.35 * span[j]), lo, hi))
                else:
                    xr = rand_vec()
                fr = eval_f(xr)
                new_pop.append((fr, xr))
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

            # fill remaining with halton/random mix
            while len(new_pop) < pop_size and time.time() < deadline:
                if random.random() < 0.5:
                    xr = rand_vec()
                else:
                    xr = halton_point(k, shift)
                    k += 1
                fr = eval_f(xr)
                new_pop.append((fr, xr))
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()
            new_pop.sort(key=lambda t: t[0])
            pop = new_pop[:pop_size]
            continue

        # choose a target
        idx = random.randrange(len(pop))
        fx, x = pop[idx]

        # choose operator mix (more exploitation late)
        # A: DE current-to-best, B: elite-mean Gaussian, C: coord trust region
        r = random.random()
        if r < (0.40 * (1.0 - tfrac) + 0.15):
            op = "A"
        elif r < (0.88 - 0.25 * (1.0 - tfrac)):
            op = "B"
        else:
            op = "C"

        candidate = None

        if op == "A" and len(pop) >= 4 and best_x is not None:
            # DE/current-to-best/1
            a, b, c = pick3(len(pop), idx)
            xa = pop[a][1]
            xb = pop[b][1]
            xc = pop[c][1]

            F = F_base + 0.25 * (random.random() - 0.5)
            if F < 0.25:
                F = 0.25
            elif F > 0.95:
                F = 0.95

            CR = CR_base + 0.25 * (random.random() - 0.5)
            if CR < 0.15:
                CR = 0.15
            elif CR > 0.98:
                CR = 0.98

            v = [0.0] * dim
            for j in range(dim):
                lo, hi = bounds[j]
                vj = x[j] + F * (best_x[j] - x[j]) + F * (xb[j] - xc[j])
                # mild Gaussian dither early
                if random.random() < (0.10 * (1.0 - tfrac)):
                    vj += gauss() * (0.015 * span[j])
                v[j] = clamp(vj, lo, hi)

            u = [0.0] * dim
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]
                else:
                    u[j] = x[j]
            candidate = u

        elif op == "B":
            # sample around mean of elites with diagonal sigma
            # also add a small pull to current best late in time
            pull = 0.15 + 0.35 * tfrac
            y = [0.0] * dim
            for j in range(dim):
                lo, hi = bounds[j]
                base = mean_x[j]
                if best_x is not None and random.random() < pull:
                    base = 0.6 * base + 0.4 * best_x[j]
                step = gauss() * sigma_g * sigma_d[j]
                y[j] = clamp(base + step, lo, hi)
            candidate = y

        else:
            # coordinate pattern trust-region probe
            # larger steps earlier, smaller steps later
            step_scale = 0.9 - 0.5 * tfrac
            xn, fn = coord_pattern(x, fx, step_scale)
            attempted += 1
            if fn <= fx:
                pop[idx] = (fn, xn)
                accepted += 1
                if fn < best:
                    best, best_x = fn, list(xn)
                    last_improve_t = time.time()
            # continue main loop; adaptation below
            candidate = None

        if candidate is not None:
            fn = eval_f(candidate)
            attempted += 1
            if fn <= fx:
                pop[idx] = (fn, candidate)
                accepted += 1
                if fn < best:
                    best, best_x = fn, list(candidate)
                    last_improve_t = time.time()

                # quick extra greedy coord improvement late (cheap)
                if tfrac > 0.60 and time.time() < deadline:
                    xn, fn2 = coord_pattern(candidate, fn, 0.55)
                    if fn2 < fn:
                        pop[idx] = (fn2, xn)
                        accepted += 1
                        if fn2 < best:
                            best, best_x = fn2, list(xn)
                            last_improve_t = time.time()

        # ---- adapt sigma (diagonal + global) using success rate ----
        if attempted >= adapt_window:
            rate = accepted / float(attempted) if attempted > 0 else 0.0

            # target ~0.2 success (ES rule of thumb)
            if rate > 0.24:
                sigma_g *= 1.18
                per = 1.08
            elif rate < 0.14:
                sigma_g *= 0.82
                per = 0.92
            else:
                per = 1.0

            # additionally shrink slowly over time to focus
            sigma_g *= (0.999 ** (1 + int(6 * tfrac)))

            # clamp sigma_g to sensible range
            if sigma_g < 0.08:
                sigma_g = 0.08
            elif sigma_g > 2.5:
                sigma_g = 2.5

            if per != 1.0:
                for j in range(dim):
                    sj = sigma_d[j] * per
                    if sj < min_sigma_d[j]:
                        sj = min_sigma_d[j]
                    elif sj > max_sigma_d[j]:
                        sj = max_sigma_d[j]
                    sigma_d[j] = sj

            accepted = 0
            attempted = 0

        # small random injection early to avoid collapse
        if random.random() < (0.012 * (1.0 - tfrac) + 0.002):
            xr = rand_vec()
            fr = eval_f(xr)
            # replace a random non-elite / or worst
            pop.sort(key=lambda t: t[0])
            replace_idx = random.randrange(max(elite_size, 1), len(pop))
            if fr < pop[replace_idx][0] or random.random() < 0.25:
                pop[replace_idx] = (fr, xr)
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

    return best
