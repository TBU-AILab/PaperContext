import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded continuous minimizer (no external libs).

    What’s improved vs your current best:
      1) Adds an explicit *intensification loop* around the current best that combines:
         - pairwise-coordinate quadratic interpolation (very cheap surrogate, robust)
         - small-batch SPSA-style gradient sign steps (good in high dimension)
         - short adaptive pattern search (already present, but tightened)
      2) Uses *dual mutation* DE: current-to-pbest/1 (as before) + occasional rand-to-pbest/2
         to increase exploration without full restart.
      3) Better “near-best” memory: keeps a compact elite pool and uses it for interpolation.
      4) More reliable stagnation handling: radius-controlled restart with elite preservation.

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

    def clamp01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def clip(v, d):
        a, b = lo[d], hi[d]
        if v < a:
            return a
        if v > b:
            return b
        return v

    def reflect_repair(v, d):
        a, b = lo[d], hi[d]
        if a == b:
            return a
        # repeated reflection
        for _ in range(14):
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
                if eps:
                    y[d] = clip(y[d] + random.uniform(-eps, eps), d)
        return y

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------------- Scrambled Halton init ----------------
    def first_primes(n):
        primes = []
        v = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(v))
            for p in primes:
                if p > r:
                    break
                if v % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(v)
            v += 1
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
        x = [0.0] * dim
        for d in range(dim):
            u = halton_scrambled_value(k, primes[d], digit_perm[d])
            x[d] = lo[d] + u * span_safe[d]
        return x

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def pick_distinct_index(n, banned):
        j = random.randrange(n)
        while j in banned:
            j = random.randrange(n)
        return j

    def rand_cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- DE / L-SHADE-ish settings ----------------
    init_pop = int(24 + 6.0 * math.sqrt(max(1, dim)))
    init_pop = max(28, min(120, init_pop))
    min_pop = max(10, min(30, 7 + int(2.0 * math.sqrt(max(1, dim)))))

    arc_factor = 1.25

    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    pmin, pmax = 0.05, 0.25

    # ---------------- Initialization (anchors + Halton + opposition) ----------------
    pop = []  # [x, fx]
    best = float("inf")
    best_x = None

    mid = [(lo[i] + hi[i]) * 0.5 for i in range(dim)]
    anchors = [mid]
    for _ in range(min(4, max(1, dim // 10))):
        anchors.append(rand_uniform_point())

    # elite pool near best (kept sorted)
    elite = []   # list of (f, x)
    ELITE_MAX = 40 if dim <= 30 else 55

    def elite_add(x, fx):
        nonlocal elite
        elite.append((fx, x[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > ELITE_MAX:
            elite = elite[:ELITE_MAX]

    for x in anchors:
        if now() >= deadline:
            return best
        fx = evaluate(x)
        pop.append([x[:], fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(x, fx)

        if now() >= deadline:
            return best
        xo = opposite_point(x)
        fxo = evaluate(xo)
        pop.append([xo[:], fxo])
        if fxo < best:
            best, best_x = fxo, xo[:]
        elite_add(xo, fxo)

    k = 1
    while len(pop) < init_pop and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(x, fx)

        if len(pop) < init_pop and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo[:]
            elite_add(xo, fxo)

    while len(pop) < init_pop and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(x, fx)

    if not pop:
        return float("inf")

    pop.sort(key=lambda z: z[1])
    pop = pop[:init_pop]

    archive = []
    last_improve_t = now()

    # ---------------- Local search (adaptive pattern) ----------------
    ls_step = [0.10 * span_safe[i] for i in range(dim)]
    ls_min = [1e-14 * span_safe[i] for i in range(dim)]

    def local_refine(x0, f0, passes=2):
        x = x0[:]
        fx = f0
        rem = max(0.0, deadline - now())
        frac_rem = rem / max(1e-12, float(max_time))
        step_scale = 0.20 + 0.80 * frac_rem  # smaller late

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
                for mult in (1.0, 2.0, 0.5):
                    for sgn in (-1.0, 1.0):
                        y = x[:]
                        y[d] = reflect_repair(xd + sgn * mult * step, d)
                        y = jitter_inside(y, rel=1e-15)
                        fy = evaluate(y)
                        elite_add(y, fy)
                        if fy < fx:
                            x, fx = y, fy
                            improved_any = True
                            break
                    if improved_any:
                        break
            if not improved_any:
                break
        return x, fx

    # ---------------- Intensification around best (pairwise quad + SPSA) ----------------
    def intensify(best_x, best_f, budget_evals):
        nonlocal last_improve_t
        x = best_x[:]
        fx = best_f

        if budget_evals <= 0 or now() >= deadline:
            return x, fx, 0

        used = 0

        # time-scaled radius
        frac = clamp01((now() - t0) / max(1e-12, float(max_time)))
        base_rad = max(0.02, 0.18 * (1.0 - 0.75 * frac))

        # helper: coordinate quadratic interpolation using 3 points (x-h, x, x+h)
        def coord_quad_step(xc, fc, d, h):
            # Evaluate at two sides
            if used + 2 > budget_evals or now() >= deadline:
                return xc, fc, 0
            xm = xc[:]
            xp = xc[:]
            xm[d] = reflect_repair(xc[d] - h, d)
            xp[d] = reflect_repair(xc[d] + h, d)

            fm = evaluate(xm); elite_add(xm, fm)
            fp = evaluate(xp); elite_add(xp, fp)

            evals = 2

            # Fit parabola through (-h,fm), (0,fc), (+h,fp): minimum at t* = h*(fm-fp)/(2*(fm-2fc+fp))
            denom = (fm - 2.0 * fc + fp)
            if abs(denom) < 1e-20:
                # fallback: pick best among three
                if fm < fc and fm <= fp:
                    return xm, fm, evals
                if fp < fc and fp < fm:
                    return xp, fp, evals
                return xc, fc, evals

            tstar = 0.5 * h * (fm - fp) / denom
            # clamp within [-h, h]
            if tstar < -h:
                tstar = -h
            elif tstar > h:
                tstar = h

            xs = xc[:]
            xs[d] = reflect_repair(xc[d] + tstar, d)
            xs = jitter_inside(xs, rel=1e-15)
            fs = evaluate(xs); elite_add(xs, fs)
            evals += 1

            # return best
            bx, bf = xc, fc
            if fm < bf:
                bx, bf = xm, fm
            if fp < bf:
                bx, bf = xp, fp
            if fs < bf:
                bx, bf = xs, fs
            return bx, bf, evals

        # Phase A: a few coordinate quadratic steps on random subset
        if used < budget_evals and now() < deadline:
            m = min(dim, 1 + int(math.sqrt(max(1, dim))))
            coords = random.sample(range(dim), m)
            for d in coords:
                if used >= budget_evals or now() >= deadline:
                    break
                h = base_rad * span_safe[d] * random.uniform(0.7, 1.3)
                if h <= 1e-18:
                    continue
                nx, nf, ev = coord_quad_step(x, fx, d, h)
                used += ev
                if nf < fx:
                    x, fx = nx, nf
                    last_improve_t = now()

        # Phase B: SPSA-style step (2 evaluations per trial)
        # Good for high dim: approximate gradient with random +/- perturbation
        for _ in range(2 if dim <= 25 else 3):
            if used + 2 > budget_evals or now() >= deadline:
                break
            c = base_rad * random.uniform(0.6, 1.2)
            a = base_rad * random.uniform(0.35, 0.9)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            x_plus = x[:]
            x_minus = x[:]
            for d in range(dim):
                step = c * span_safe[d] * delta[d]
                x_plus[d] = reflect_repair(x[d] + step, d)
                x_minus[d] = reflect_repair(x[d] - step, d)

            f_plus = evaluate(x_plus); elite_add(x_plus, f_plus)
            f_minus = evaluate(x_minus); elite_add(x_minus, f_minus)
            used += 2

            # gradient sign estimate and take step
            # g_d ~ (f_plus - f_minus)/(2c*delta_d) => sign via (f_plus - f_minus)*delta_d
            x_new = x[:]
            diff = (f_plus - f_minus)
            for d in range(dim):
                sgn = 1.0 if diff * delta[d] > 0.0 else (-1.0 if diff * delta[d] < 0.0 else 0.0)
                x_new[d] = reflect_repair(x[d] - sgn * a * span_safe[d], d)
            x_new = jitter_inside(x_new, rel=1e-15)
            f_new = evaluate(x_new); elite_add(x_new, f_new)
            used += 1
            if f_new < fx:
                x, fx = x_new, f_new
                last_improve_t = now()

        return x, fx, used

    # ---------------- Main loop ----------------
    it = 0
    no_improve_window = max(0.9, float(max_time) / 8.0)

    while True:
        t = now()
        if t >= deadline:
            return best

        frac = clamp01((t - t0) / max(1e-12, float(max_time)))

        # population reduction
        desired_n = int(round(init_pop - (init_pop - min_pop) * frac))
        desired_n = max(min_pop, min(init_pop, desired_n))

        pop.sort(key=lambda z: z[1])
        if len(pop) > desired_n:
            pop = pop[:desired_n]

        arc_max = int(math.ceil(arc_factor * len(pop)))
        if len(archive) > arc_max:
            while len(archive) > arc_max:
                del archive[random.randrange(len(archive))]

        n = len(pop)
        if n < 6:
            while len(pop) < 6 and now() < deadline:
                x = rand_uniform_point()
                fx = evaluate(x)
                pop.append([x, fx])
                elite_add(x, fx)
                if fx < best:
                    best, best_x = fx, x[:]
            pop.sort(key=lambda z: z[1])
            n = len(pop)

        p = pmax - (pmax - pmin) * frac
        pbest_count = max(2, int(math.ceil(p * n)))

        union = [ind[0] for ind in pop] + archive
        union_n = len(union)

        S_F, S_CR, S_w = [], [], []

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

            pbest_idx = random.randrange(pbest_count)
            xpbest = pop[pbest_idx][0]

            # dual mutation choice
            use_rand2 = (random.random() < (0.18 + 0.12 * (1.0 - frac))) and (n >= 6)

            if not use_rand2:
                # current-to-pbest/1 + archive
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

                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
            else:
                # rand-to-pbest/2
                # pick base from population, then add two differences
                base_idx = random.randrange(n)
                xb = pop[base_idx][0]
                banned = {base_idx, pbest_idx}
                r1 = pick_distinct_index(n, banned); banned.add(r1)
                r2 = pick_distinct_index(n, banned); banned.add(r2)
                r3 = pick_distinct_index(n, banned); banned.add(r3)
                r4 = pick_distinct_index(n, banned)

                x1, x2, x3, x4 = pop[r1][0], pop[r2][0], pop[r3][0], pop[r4][0]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xb[d] + F * (xpbest[d] - xb[d]) + F * ((x1[d] - x2[d]) + 0.5 * (x3[d] - x4[d]))

            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = reflect_repair(v[d], d)

            if random.random() < (0.12 + 0.25 * frac):
                u = jitter_inside(u, rel=1e-14)
            else:
                u = jitter_inside(u, rel=1e-15)

            fu = evaluate(u)
            elite_add(u, fu)

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

            alpha = 0.12 + 0.18 * frac
            MCR[mem_idx] = (1.0 - alpha) * MCR[mem_idx] + alpha * mcr_new
            MF[mem_idx] = (1.0 - alpha) * MF[mem_idx] + alpha * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # Intensification schedule (mid/late)
        if best_x is not None:
            if (it % (12 if frac < 0.40 else 7)) == 0 and now() < deadline:
                # small controlled budget so we don't starve DE
                bud = 6 if dim <= 20 else (7 if dim <= 60 else 8)
                xb, fb, _ = intensify(best_x, best, bud)
                if fb < best:
                    best, best_x = fb, xb[:]

            # brief local refine (still useful)
            if (it % (16 if frac < 0.55 else 8)) == 0 and now() < deadline:
                xb, fb = local_refine(best_x, best, passes=2 if frac < 0.70 else 3)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()
                    for d in range(dim):
                        ls_step[d] = min(0.30 * span_safe[d], ls_step[d] * 1.12)
                else:
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.85)

        # Stagnation: soft restart
        if (now() - last_improve_t) > no_improve_window:
            last_improve_t = now()
            pop.sort(key=lambda z: z[1])

            elite_n = max(2, min(10, len(pop) // 5))
            kept = pop[:elite_n]
            new_pop = [[kept[i][0][:], kept[i][1]] for i in range(elite_n)]

            rad = 0.42 * (1.0 - 0.70 * frac)
            rad = max(0.06, rad)

            if archive:
                random.shuffle(archive)
                archive = archive[:max(0, int(0.45 * arc_max))]

            # also seed from elite pool occasionally
            while len(new_pop) < len(pop) and now() < deadline:
                r = random.random()
                if best_x is not None and r < 0.50:
                    x = [clip(random.gauss(best_x[d], rad * span_safe[d]), d) for d in range(dim)]
                elif elite and r < 0.72:
                    _, xe = elite[random.randrange(min(len(elite), 12))]
                    x = [clip(random.gauss(xe[d], 0.95 * rad * span_safe[d]), d) for d in range(dim)]
                elif archive and r < 0.88:
                    a = archive[random.randrange(len(archive))]
                    x = [clip(random.gauss(a[d], 0.95 * rad * span_safe[d]), d) for d in range(dim)]
                else:
                    x = rand_uniform_point()

                x = jitter_inside(x, rel=1e-15)
                fx = evaluate(x)
                elite_add(x, fx)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]

            pop = new_pop

            # broaden memories slightly and reset steps a bit
            for h in range(H):
                MF[h] = min(0.95, max(0.20, MF[h] * 1.05))
                MCR[h] = min(0.95, max(0.05, MCR[h] * 1.03))
            for d in range(dim):
                ls_step[d] = max(ls_step[d], 0.085 * span_safe[d])
