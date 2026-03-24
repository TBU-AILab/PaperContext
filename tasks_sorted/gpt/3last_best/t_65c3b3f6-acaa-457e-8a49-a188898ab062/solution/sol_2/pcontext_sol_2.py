import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libraries).

    Key upgrades vs previous ES:
      - Better initialization: scrambled Halton + opposition + a few uniform points
      - DE (Differential Evolution) style variation (often stronger globally than plain ES)
      - Self-adaptive parameters (F, CR) per trial + "current-to-best/1" for fast convergence
      - Lightweight local pattern search around incumbent best (adaptive step shrink)
      - Stagnation-triggered restart that keeps best and re-diversifies population
      - Strict time checks to respect max_time

    Returns:
      best (float): best (minimum) objective value found within time limit.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0 else 1.0 for s in span]

    def now():
        return time.time()

    def clip(v, i):
        if v < lo[i]:
            return lo[i]
        if v > hi[i]:
            return hi[i]
        return v

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- Halton (scrambled by random digit permutations per base) ----------
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

    primes = first_primes(max(1, dim))

    # For each dimension/base, create a random permutation of digits [0..base-1]
    # to "scramble" the radical inverse, improving early uniformity.
    digit_perm = []
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm.append(perm)

    def halton_scrambled_value(index, base, perm):
        # scrambled radical inverse
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            digit = i % base
            r += f * perm[digit]
            i //= base
        # If index == 0, r == 0.0; we will use k>=1 anyway.
        return r / float(base - 1) if base > 2 else r  # mild normalization for tiny bases

    def halton_point(k):  # k >= 1
        x = []
        for d in range(dim):
            u = halton_scrambled_value(k, primes[d], digit_perm[d])
            # keep u in [0,1] robustly
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            x.append(lo[d] + u * span_safe[d])
        return x

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------- population sizing ----------
    # DE needs a bit of population; keep bounded for time.
    pop_size = int(10 + 3.0 * math.sqrt(max(1, dim)))
    pop_size = max(12, min(60, pop_size))
    if pop_size < 4:
        pop_size = 4

    # ---------- initialize population ----------
    pop = []
    best = float("inf")
    best_x = None

    # Build initial set with: Halton, opposition, and a few random points
    k = 1
    while len(pop) < pop_size and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x

        if len(pop) < pop_size and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo

    while len(pop) < pop_size and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x

    if not pop:
        return float("inf")

    # ---------- DE operators ----------
    def rand_index_excluding(n, excluded):
        # excluded is a set
        j = random.randrange(n)
        while j in excluded:
            j = random.randrange(n)
        return j

    def ensure_distinct_indices(n, banned, count):
        chosen = []
        banned_set = set(banned)
        while len(chosen) < count:
            j = random.randrange(n)
            if j in banned_set:
                continue
            banned_set.add(j)
            chosen.append(j)
        return chosen

    # Self-adaptive parameter ranges (jDE-like)
    Fmin, Fmax = 0.35, 0.95
    CRmin, CRmax = 0.05, 0.95

    # Stagnation / restart controls
    last_improve_time = now()
    restart_cooldown = max(0.3, max_time / 12.0) if max_time > 0 else 0.3
    no_improve_window = max(0.6, max_time / 6.0) if max_time > 0 else 0.6

    # Local search step sizes (start at ~10% range)
    ls_step = [0.10 * span_safe[i] for i in range(dim)]
    ls_min = [1e-12 * span_safe[i] for i in range(dim)]

    # ---------- lightweight local pattern search around best ----------
    def local_search(best_x, best_f):
        x = best_x[:]
        fx = best_f
        # one or two short passes; keep time-bounded
        for _pass in range(2):
            improved = False
            for i in range(dim):
                if now() >= deadline:
                    return x, fx
                step = ls_step[i]
                if step <= ls_min[i]:
                    continue

                # try both directions
                xi = x[i]
                for sgn in (-1.0, 1.0):
                    y = x[:]
                    y[i] = clip(xi + sgn * step, i)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break
            if not improved:
                break

        # adapt local steps: shrink if not improving, expand slightly if improving well
        # (simple but tends to help)
        return x, fx

    # ---------- main loop ----------
    # Store per-individual F/CR (optional, but helps stabilize across dims)
    Fvals = [random.uniform(Fmin, Fmax) for _ in range(len(pop))]
    CRvals = [random.uniform(CRmin, CRmax) for _ in range(len(pop))]

    it = 0
    while True:
        if now() >= deadline:
            return best

        n = len(pop)
        # Sort occasionally so best is near front (helps sampling best quickly)
        if (it & 15) == 0:
            pop.sort(key=lambda t: t[1])

        # Iterate through population
        for i in range(n):
            if now() >= deadline:
                return best

            xi, fxi = pop[i][0], pop[i][1]

            # jDE-like parameter adaptation
            # with small probability, resample F/CR
            if random.random() < 0.12:
                Fvals[i] = random.uniform(Fmin, Fmax)
            if random.random() < 0.12:
                CRvals[i] = random.uniform(CRmin, CRmax)

            F = Fvals[i]
            CR = CRvals[i]

            # choose best index (after occasional sort, pop[0] is best)
            best_idx = 0
            xbest = pop[best_idx][0]

            # pick r1, r2 distinct from i (and each other)
            r1, r2 = ensure_distinct_indices(n, banned=[i, best_idx], count=2)

            x_r1 = pop[r1][0]
            x_r2 = pop[r2][0]

            # current-to-best/1 mutation:
            # v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xbest[d] - xi[d]) + F * (x_r1[d] - x_r2[d])

            # binomial crossover with at least one dimension from v
            jrand = random.randrange(dim)
            u = xi[:]  # trial
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    # reflect + clip to bounds (reflection handles big steps better than hard clip alone)
                    val = v[d]
                    if val < lo[d] or val > hi[d]:
                        # reflect once
                        if val < lo[d]:
                            val = lo[d] + (lo[d] - val)
                        else:
                            val = hi[d] - (val - hi[d])
                    u[d] = clip(val, d)

            fu = evaluate(u)

            # selection
            if fu <= fxi:
                pop[i][0], pop[i][1] = u, fu
                # small success-based nudges
                Fvals[i] = min(Fmax, max(Fmin, Fvals[i] * 1.02))
                CRvals[i] = min(CRmax, max(CRmin, CRvals[i] * 1.01))
                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_time = now()
            else:
                # slight contraction if failing
                Fvals[i] = min(Fmax, max(Fmin, Fvals[i] * 0.99))
                CRvals[i] = min(CRmax, max(CRmin, CRvals[i] * 0.995))

        it += 1

        # Occasional local search on current best
        if best_x is not None and (it % 8 == 0):
            if now() < deadline:
                xb2, fb2 = local_search(best_x, best)
                if fb2 < best:
                    best, best_x = fb2, xb2[:]
                    last_improve_time = now()
                    # encourage larger LS steps when it works
                    for d in range(dim):
                        ls_step[d] = min(0.25 * span_safe[d], ls_step[d] * 1.15)
                else:
                    # shrink LS steps when it doesn't
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.80)

        # Restart on stagnation (time-based, so it adapts to varying evaluation costs)
        t = now()
        if (t - last_improve_time) > no_improve_window and (t - t0) > restart_cooldown:
            last_improve_time = t  # avoid repeated immediate restarts

            # keep a small elite set, reinitialize the rest around best + global points
            pop.sort(key=lambda z: z[1])
            elite_n = max(2, min(6, pop_size // 6))
            elites = pop[:elite_n]

            new_pop = [ [e[0][:], e[1]] for e in elites ]

            # radius shrinks mildly with time spent (more exploit later)
            frac = (t - t0) / max(1e-9, max_time)
            radius = (0.35 * (1.0 - 0.6 * min(1.0, frac)))  # in fraction of span

            # refill population
            while len(new_pop) < pop_size and now() < deadline:
                if best_x is not None and random.random() < 0.75:
                    x = []
                    for d in range(dim):
                        sd = radius * span_safe[d]
                        val = random.gauss(best_x[d], sd)
                        x.append(clip(val, d))
                else:
                    x = rand_uniform_point()

                fx = evaluate(x)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]

            pop = new_pop
            # reset F/CR arrays
            Fvals = [random.uniform(Fmin, Fmax) for _ in range(len(pop))]
            CRvals = [random.uniform(CRmin, CRmax) for _ in range(len(pop))]
            # slightly reset local search steps upward
            for d in range(dim):
                ls_step[d] = max(ls_step[d], 0.06 * span_safe[d])

    # unreachable
