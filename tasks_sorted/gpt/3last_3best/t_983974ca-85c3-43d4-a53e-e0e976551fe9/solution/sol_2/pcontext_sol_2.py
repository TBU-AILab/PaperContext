import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs your hybrid ES+pattern-search:
      - Adds a *true* population with recombination (DE-style "current-to-best")
      - Keeps an elite archive and uses it for restarts + diversity
      - Adds opportunistic coordinate/pattern polishing only on elites
      - Adaptive mutation scale based on recent success rate
      - Uses inexpensive low-discrepancy (Halton) seeding (better than random; simpler than full LHS)

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # ---------------- helpers ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---- Halton sequence (for better seeding coverage than pure random) ----
    def _primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(x)
            x += 1
        return ps

    def _halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = _primes(min(dim, 50))  # enough for most dims
    if dim > len(primes):
        # fallback: reuse last prime bases cyclically (still ok for seeding)
        primes = primes + [primes[-1]] * (dim - len(primes))

    def halton_vec(k):
        # k >= 1
        return [lows[i] + _halton_value(k, primes[i]) * spans[i] for i in range(dim)]

    # ---------------- seeding ----------------
    best = float("inf")
    best_x = None

    # modest time slice for seeding; keep it small so we can iterate with population
    seed_until = min(deadline, t0 + 0.18 * max_time)

    # initial population size
    pop_size = max(12, min(60, 10 + 5 * int(math.log(dim + 1) + 1)))
    pop = []
    fits = []

    # mix: Halton + random + corners jitter
    k = 1
    while time.time() < seed_until and len(pop) < pop_size:
        if len(pop) % 3 == 0:
            x = halton_vec(k)
            k += 1
        elif len(pop) % 3 == 1:
            x = rand_vec()
        else:
            # jitter around a random corner to catch boundary optima
            x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
            for i in range(dim):
                x[i] += random.gauss(0.0, 0.03 * spans[i])
            clip_inplace(x)

        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best, best_x = f, list(x)

    if best_x is None:
        best_x = rand_vec()
        best = safe_eval(best_x)
        pop = [best_x]
        fits = [best]

    # elite archive (best few points)
    ELITE_MAX = max(6, min(16, 4 + int(2 * math.log(dim + 1))))
    elite = []

    def elite_add(x, f):
        nonlocal elite
        elite.append((f, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > ELITE_MAX:
            elite = elite[:ELITE_MAX]

    for x, f in zip(pop, fits):
        elite_add(x, f)

    # ---------------- local polish (cheap coordinate pattern) ----------------
    min_step = 1e-15

    def polish(x, fx, step_scale):
        """
        A few coordinate tries around x. step_scale in (0,1] relative to span.
        Very budget-conscious: tries only a subset of coordinates.
        """
        nonlocal best, best_x
        # choose subset: more in low-d, fewer in high-d
        m = min(dim, max(4, int(0.35 * dim)))
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:m]

        # per-coordinate step
        for i in idxs:
            if time.time() >= deadline:
                break
            si = max(min_step, step_scale * spans[i])

            base = x[i]

            # + step
            x[i] = base + si
            if x[i] > highs[i]:
                x[i] = highs[i]
            f1 = safe_eval(x)

            # - step
            x[i] = base - si
            if x[i] < lows[i]:
                x[i] = lows[i]
            f2 = safe_eval(x)

            # restore
            x[i] = base

            if f1 < fx or f2 < fx:
                if f1 <= f2:
                    x[i] = base + si
                    if x[i] > highs[i]:
                        x[i] = highs[i]
                    fx = f1
                else:
                    x[i] = base - si
                    if x[i] < lows[i]:
                        x[i] = lows[i]
                    fx = f2

                if fx < best:
                    best = fx
                    best_x = list(x)
        return x, fx

    # ---------------- main loop: DE-style with current-to-best + archive restarts ----------------
    # Adaptive mutation parameters
    F = 0.6     # differential weight
    CR = 0.9    # crossover rate
    jitter = 0.02  # noise added to mutation (relative to span)

    # success tracking for adaptation
    trials = 0
    succ = 0

    # ensure population filled if seeding ended early
    while len(pop) < pop_size and time.time() < deadline:
        x = rand_vec()
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        elite_add(x, f)
        if f < best:
            best, best_x = f, list(x)

    # generation loop
    gen = 0
    last_improve_time = time.time()

    while time.time() < deadline:
        gen += 1

        # Occasionally polish the current global best / top elite
        if gen % 8 == 0 and time.time() < deadline:
            # polish best and also the best elite member (often the same)
            bx = list(best_x)
            bf = safe_eval(bx)
            bx, bf = polish(bx, bf, step_scale=0.01)
            if bf < best:
                best, best_x = bf, list(bx)
                elite_add(best_x, best)
                last_improve_time = time.time()

        # Adapt F/CR every so often based on success rate
        if trials >= 40:
            rate = succ / max(1, trials)
            # if too few improvements, increase exploration; else slightly exploit
            if rate < 0.18:
                F = min(0.95, F * 1.10 + 0.03)
                CR = min(0.98, CR + 0.03)
                jitter = min(0.08, jitter * 1.15 + 0.002)
            elif rate > 0.35:
                F = max(0.35, F * 0.95)
                CR = max(0.60, CR - 0.02)
                jitter = max(0.005, jitter * 0.90)
            trials = 0
            succ = 0

        # Stagnation: if no global improvement for a while, inject diversity
        if time.time() - last_improve_time > max(0.25 * max_time, 0.75):
            # Replace worst fraction with random/elite-jittered points
            # (keeps algorithm from collapsing)
            order = list(range(len(pop)))
            order.sort(key=lambda i: fits[i], reverse=True)
            repl = max(2, len(pop) // 6)

            for t in range(repl):
                if time.time() >= deadline:
                    return best
                i = order[t]
                if elite and random.random() < 0.7:
                    ef, ex = elite[random.randrange(len(elite))]
                    x = list(ex)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.12 * spans[d])
                    clip_inplace(x)
                else:
                    x = rand_vec()
                f = safe_eval(x)
                pop[i] = x
                fits[i] = f
                elite_add(x, f)
                if f < best:
                    best, best_x = f, list(x)
                    last_improve_time = time.time()
            last_improve_time = time.time()

        # Identify best index in population
        ibest = min(range(len(pop)), key=lambda i: fits[i])
        xbest = pop[ibest]

        # iterate individuals
        for i in range(len(pop)):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # choose r1, r2 distinct and not i
            r1 = i
            while r1 == i:
                r1 = random.randrange(len(pop))
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(len(pop))

            x1 = pop[r1]
            x2 = pop[r2]

            # Mutation: current-to-best/1: v = xi + F*(xbest-xi) + F*(x1-x2)
            # plus small gaussian jitter
            v = [0.0] * dim
            for d in range(dim):
                v[d] = (xi[d]
                        + F * (xbest[d] - xi[d])
                        + F * (x1[d] - x2[d])
                        + random.gauss(0.0, jitter * spans[d]))
            clip_inplace(v)

            # Binomial crossover to produce trial u
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]

            fu = safe_eval(u)
            trials += 1

            # Selection
            if fu <= fi:
                pop[i] = u
                fits[i] = fu
                succ += 1
                elite_add(u, fu)

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve_time = time.time()

        # occasional micro-polish of a random elite member
        if gen % 12 == 0 and elite and time.time() < deadline:
            ef, ex = elite[random.randrange(len(elite))]
            xx = list(ex)
            ff = safe_eval(xx)
            xx, ff = polish(xx, ff, step_scale=0.006)
            elite_add(xx, ff)
            if ff < best:
                best, best_x = ff, list(xx)
                last_improve_time = time.time()

    return best
