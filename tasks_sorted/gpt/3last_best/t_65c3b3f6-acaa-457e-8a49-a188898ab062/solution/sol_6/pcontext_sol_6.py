import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libraries).

    Improved over your best (SHADE-ish DE):
      - Stronger initialization: scrambled Halton + opposition + a few "anchor" points
      - L-SHADE mechanics: linear population reduction + archive
      - Success-history adaptation of F/CR (SHADE), plus per-individual jitter to avoid stagnation
      - Hybrid exploitation:
          * periodic best-centered "adaptive Gaussian" sampling (CMA-ES-ish diagonal, lightweight)
          * periodic coordinate pattern search with multi-step tries
      - Stagnation handling: soft-restart that preserves elites and rebuilds diversity around (best + archive + uniform)
      - Robust bounds: repeated reflection + final clip + tiny jitter

    Returns:
      best (float): best objective value found within time limit.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def now():
        return time.time()

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def clip(v, d):
        a, b = lo[d], hi[d]
        if v < a:
            return a
        if v > b:
            return b
        return v

    def reflect_repair(v, d):
        """Repeated reflection into [lo,hi] for far out-of-bounds values."""
        a, b = lo[d], hi[d]
        if a == b:
            return a
        # multiple bounces
        for _ in range(10):
            if v < a:
                v = a + (a - v)
            elif v > b:
                v = b - (v - b)
            else:
                break
        # final clip
        if v < a:
            v = a
        elif v > b:
            v = b
        return v

    def jitter_inside(x, rel=1e-15):
        y = x[:]
        for d in range(dim):
            if hi[d] > lo[d]:
                eps = rel * span_safe[d]
                if eps > 0.0:
                    y[d] = clip(y[d] + random.uniform(-eps, eps), d)
        return y

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------------- Halton init (scrambled) ----------------
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
    digit_perm = []
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm.append(perm)

    def halton_scrambled_value(index, base, perm):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            digit = i % base
            r += f * perm[digit]
            i //= base
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        return r

    def halton_point(k):
        x = []
        for d in range(dim):
            u = halton_scrambled_value(k, primes[d], digit_perm[d])
            x.append(lo[d] + u * span_safe[d])
        return x

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def mid_point():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    # ---------------- DE / SHADE / L-SHADE settings ----------------
    init_pop = int(22 + 6.0 * math.sqrt(max(1, dim)))
    init_pop = max(26, min(110, init_pop))
    min_pop = max(10, min(28, 7 + int(2.0 * math.sqrt(max(1, dim)))))

    arc_factor = 1.2

    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    pmin, pmax = 0.05, 0.25

    def rand_cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def pick_distinct_index(n, banned):
        j = random.randrange(n)
        while j in banned:
            j = random.randrange(n)
        return j

    # ---------------- Initialization: Halton + opposition + anchors ----------------
    pop = []  # list of [x, fx]
    best = float("inf")
    best_x = None

    # anchors: center + a couple of random (cheap robustness)
    anchors = [mid_point()]
    for _ in range(min(3, max(1, dim // 10))):
        anchors.append(rand_uniform_point())

    for x in anchors:
        if now() >= deadline:
            return best
        fx = evaluate(x)
        pop.append([x[:], fx])
        if fx < best:
            best, best_x = fx, x[:]
        if now() >= deadline:
            return best
        xo = opposite_point(x)
        fxo = evaluate(xo)
        pop.append([xo[:], fxo])
        if fxo < best:
            best, best_x = fxo, xo[:]

    k = 1
    while len(pop) < init_pop and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

        if len(pop) < init_pop and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo[:]

    while len(pop) < init_pop and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

    if not pop:
        return float("inf")

    # downselect
    pop.sort(key=lambda z: z[1])
    pop = pop[:init_pop]

    archive = []
    last_improve_t = now()

    # ---------------- Local search: coordinate multi-try ----------------
    ls_step = [0.12 * span_safe[i] for i in range(dim)]
    ls_min = [1e-14 * span_safe[i] for i in range(dim)]

    def local_refine(x0, f0, passes=2):
        x = x0[:]
        fx = f0

        rem = max(0.0, deadline - now())
        frac_rem = rem / max(1e-12, float(max_time))
        step_scale = 0.30 + 0.70 * frac_rem  # smaller late

        for _ in range(passes):
            improved_any = False
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if now() >= deadline:
                    return x, fx
                step = ls_step[d] * step_scale
                if step <= ls_min[d]:
                    continue

                xd = x[d]
                # multi-try: +/- step, +/- 2 step, +/- 0.5 step
                for mult in (1.0, 2.0, 0.5):
                    for sgn in (-1.0, 1.0):
                        y = x[:]
                        y[d] = reflect_repair(xd + sgn * mult * step, d)
                        y = jitter_inside(y, rel=1e-15)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved_any = True
                            break
                    if improved_any:
                        break
            if not improved_any:
                break
        return x, fx

    # ---------------- Lightweight diagonal-cov sampling around best ----------------
    # Keep an adaptive diagonal "sigma" vector; updated from successful samples.
    sigma = [0.20 * span_safe[i] for i in range(dim)]
    sigma_min = [1e-16 * span_safe[i] for i in range(dim)]
    sigma_max = [0.50 * span_safe[i] for i in range(dim)]

    def best_centered_samples(num):
        nonlocal best, best_x, last_improve_t, sigma
        if best_x is None:
            return
        for _ in range(num):
            if now() >= deadline:
                return
            # sample around best (diagonal)
            y = [0.0] * dim
            for d in range(dim):
                if hi[d] == lo[d]:
                    y[d] = lo[d]
                else:
                    y[d] = clip(random.gauss(best_x[d], sigma[d]), d)
            y = jitter_inside(y, rel=1e-15)
            fy = evaluate(y)
            if fy < best:
                best, best_x = fy, y[:]
                last_improve_t = now()
                # mild contraction on success (exploit more)
                for d in range(dim):
                    sigma[d] = max(sigma_min[d], sigma[d] * 0.92)
            else:
                # mild expansion on failure (avoid over-tightening)
                for d in range(dim):
                    sigma[d] = min(sigma_max[d], sigma[d] * 1.01)

    # ---------------- Main loop ----------------
    it = 0
    no_improve_window = max(0.9, float(max_time) / 8.0)

    while True:
        t = now()
        if t >= deadline:
            return best

        frac = clamp01((t - t0) / max(1e-12, float(max_time)))

        # L-SHADE population size reduction
        desired_n = int(round(init_pop - (init_pop - min_pop) * frac))
        desired_n = max(min_pop, min(init_pop, desired_n))

        pop.sort(key=lambda z: z[1])
        if len(pop) > desired_n:
            pop = pop[:desired_n]

        # archive maintenance
        arc_max = int(math.ceil(arc_factor * len(pop)))
        if len(archive) > arc_max:
            while len(archive) > arc_max:
                del archive[random.randrange(len(archive))]

        n = len(pop)
        if n < 4:
            while len(pop) < 4 and now() < deadline:
                x = rand_uniform_point()
                fx = evaluate(x)
                pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]
            pop.sort(key=lambda z: z[1])
            n = len(pop)

        # p-best schedule (smaller later => stronger exploitation)
        p = pmax - (pmax - pmin) * frac
        pbest_count = max(2, int(math.ceil(p * n)))

        union = [ind[0] for ind in pop] + archive
        union_n = len(union)

        S_F, S_CR, S_w = [], [], []

        # core DE generation
        for i in range(n):
            if now() >= deadline:
                return best

            xi, fxi = pop[i][0], pop[i][1]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = clamp01(mu_cr + 0.10 * random.gauss(0.0, 1.0))

            F = rand_cauchy(mu_f, 0.10)
            tries = 0
            while F <= 0.0 and tries < 10:
                F = rand_cauchy(mu_f, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            # choose pbest from top pbest_count
            pbest_idx = random.randrange(pbest_count)
            xpbest = pop[pbest_idx][0]

            banned = {i, pbest_idx}
            r1 = pick_distinct_index(n, banned)
            banned.add(r1)

            banned_union = set(idx for idx in (i, pbest_idx, r1) if 0 <= idx < n)
            r2u = random.randrange(union_n)
            tries2 = 0
            while r2u in banned_union and tries2 < 25:
                r2u = random.randrange(union_n)
                tries2 += 1

            xr1 = pop[r1][0]
            xr2 = union[r2u]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # binomial crossover + repair
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = reflect_repair(v[d], d)

            # small time-dependent jitter helps break ties/stagnation
            if random.random() < (0.15 + 0.20 * frac):
                u = jitter_inside(u, rel=1e-14)
            else:
                u = jitter_inside(u, rel=1e-15)

            fu = evaluate(u)

            if fu <= fxi:
                archive.append(xi[:])
                if len(archive) > arc_max:
                    del archive[random.randrange(len(archive))]

                pop[i][0], pop[i][1] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = now()

                df = (fxi - fu)
                if df <= 0.0:
                    df = 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(df)

        # SHADE memory update
        if S_F:
            w_sum = sum(S_w)
            if w_sum <= 0.0:
                w_sum = float(len(S_w))

            mcr_new = 0.0
            for cr, w in zip(S_CR, S_w):
                mcr_new += (w / w_sum) * cr

            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_w):
                num += w * f * f
                den += w * f
            mf_new = (num / den) if den > 0.0 else MF[mem_idx]

            # react a bit faster later
            alpha = 0.10 + 0.15 * frac
            MCR[mem_idx] = (1.0 - alpha) * MCR[mem_idx] + alpha * mcr_new
            MF[mem_idx]  = (1.0 - alpha) * MF[mem_idx]  + alpha * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # Hybrid exploitation schedule:
        # - best-centered sampling becomes more frequent later
        # - local refine every so often (also more frequent later)
        if best_x is not None:
            if (it % (14 if frac < 0.45 else 7)) == 0:
                # number of samples scales mildly with dimension and late-run
                ns = 2 + int(0.25 * math.sqrt(max(1, dim))) + (1 if frac > 0.70 else 0)
                best_centered_samples(ns)

            if (it % (16 if frac < 0.55 else 8)) == 0 and now() < deadline:
                xb, fb = local_refine(best_x, best, passes=2 if frac < 0.70 else 3)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()
                    for d in range(dim):
                        ls_step[d] = min(0.30 * span_safe[d], ls_step[d] * 1.15)
                        sigma[d] = max(sigma_min[d], sigma[d] * 0.95)
                else:
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.83)

        # Stagnation: soft restart with mixture (keep elites)
        if (now() - last_improve_t) > no_improve_window:
            last_improve_t = now()
            pop.sort(key=lambda z: z[1])

            elite = max(2, min(8, len(pop) // 5))
            kept = pop[:elite]
            new_pop = [[kept[i][0][:], kept[i][1]] for i in range(elite)]

            # restart radius decreases with time, but never too small
            rad = 0.35 * (1.0 - 0.60 * frac)
            rad = max(0.06, rad)

            # trim archive but keep some
            if archive:
                random.shuffle(archive)
                archive = archive[:max(0, int(0.40 * arc_max))]

            while len(new_pop) < len(pop) and now() < deadline:
                r = random.random()
                if best_x is not None and r < 0.62:
                    x = [clip(random.gauss(best_x[d], rad * span_safe[d]), d) for d in range(dim)]
                elif archive and r < 0.87:
                    a = archive[random.randrange(len(archive))]
                    x = [clip(random.gauss(a[d], 0.95 * rad * span_safe[d]), d) for d in range(dim)]
                else:
                    x = rand_uniform_point()

                x = jitter_inside(x, rel=1e-15)
                fx = evaluate(x)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]

            pop = new_pop

            # post-restart: broaden DE memories slightly + reset local radii upward a bit
            for h in range(H):
                MF[h] = min(0.95, max(0.20, MF[h] * 1.05))
                MCR[h] = min(0.95, max(0.05, MCR[h] * 1.03))
            for d in range(dim):
                ls_step[d] = max(ls_step[d], 0.08 * span_safe[d])
                sigma[d] = min(sigma_max[d], max(sigma[d], 0.12 * span_safe[d]))
