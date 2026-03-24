import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded continuous minimizer (no external libs).

    Improvements vs the previous best (SHADE-ish DE):
      - Adds an explicit *surrogate-assisted trust region* around the current best:
          * fits a diagonal quadratic model using samples collected near best
          * proposes candidates by minimizing the surrogate (cheap) + verifying with func
      - Stronger local search: adaptive coordinate pattern + occasional random subspace steps
      - Better boundary handling: repeated reflection + final clip + tiny jitter
      - Robust restarts: multi-source rebuild (best/elite/archive/uniform) with radius schedule
      - Keeps DE (current-to-pbest/1 + archive + SHADE memories) as global engine

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
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

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
        # repeated reflection (handles far out of bounds)
        for _ in range(12):
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

    arc_factor = 1.2

    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    pmin, pmax = 0.05, 0.25

    # ---------------- Initialization (anchors + Halton + opposition) ----------------
    pop = []  # [x, fx]
    best = float("inf")
    best_x = None

    # anchors: mid + a couple random
    mid = [(lo[i] + hi[i]) * 0.5 for i in range(dim)]
    anchors = [mid]
    for _ in range(min(3, max(1, dim // 12))):
        anchors.append(rand_uniform_point())

    # evaluation log near best for surrogate
    # store tuples (x, f); we keep a bounded size
    near_log = []

    def maybe_log(x, fx):
        # log points near the current best (in normalized distance)
        if best_x is None:
            return
        # normalized L2 distance
        s = 0.0
        for d in range(dim):
            sd = span_safe[d]
            if sd > 0.0:
                z = (x[d] - best_x[d]) / sd
                s += z * z
        dist = math.sqrt(s)
        # keep points in a moderate neighborhood; threshold adapts loosely over time
        # (we still keep some diversity by random chance)
        if dist < 0.45 or random.random() < 0.05:
            near_log.append((x[:], fx))
            if len(near_log) > 260:
                # remove a random older one
                del near_log[random.randrange(len(near_log))]

    for x in anchors:
        if now() >= deadline:
            return best
        fx = evaluate(x)
        pop.append([x[:], fx])
        if fx < best:
            best, best_x = fx, x[:]
        maybe_log(x, fx)

        if now() >= deadline:
            return best
        xo = opposite_point(x)
        fxo = evaluate(xo)
        pop.append([xo[:], fxo])
        if fxo < best:
            best, best_x = fxo, xo[:]
        maybe_log(xo, fxo)

    k = 1
    while len(pop) < init_pop and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        maybe_log(x, fx)

        if len(pop) < init_pop and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo[:]
            maybe_log(xo, fxo)

    while len(pop) < init_pop and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        maybe_log(x, fx)

    if not pop:
        return float("inf")

    pop.sort(key=lambda z: z[1])
    pop = pop[:init_pop]

    archive = []
    last_improve_t = now()

    # ---------------- Local search (adaptive coordinate + random subspace) ----------------
    ls_step = [0.10 * span_safe[i] for i in range(dim)]
    ls_min = [1e-14 * span_safe[i] for i in range(dim)]

    def local_refine(x0, f0, passes=2):
        x = x0[:]
        fx = f0

        rem = max(0.0, deadline - now())
        frac_rem = rem / max(1e-12, float(max_time))
        step_scale = 0.25 + 0.75 * frac_rem  # smaller late

        for _ in range(passes):
            improved_any = False

            # coordinate pattern
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if now() >= deadline:
                    return x, fx
                step = ls_step[d] * step_scale
                if step <= ls_min[d]:
                    continue

                xd = x[d]
                # multi-try: +/- step, +/-2 step, +/-0.5 step
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

            # occasional random subspace move (helps nonseparable objectives)
            if now() >= deadline:
                return x, fx
            if random.random() < 0.40:
                # pick a small subset
                m = 1 + int(math.sqrt(max(1, dim)) / 2)
                idxs = random.sample(range(dim), min(dim, m))
                y = x[:]
                for d in idxs:
                    step = ls_step[d] * step_scale
                    if step > ls_min[d]:
                        y[d] = reflect_repair(y[d] + random.choice([-1.0, 1.0]) * random.uniform(0.3, 1.5) * step, d)
                y = jitter_inside(y, rel=1e-15)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True

            if not improved_any:
                break
        return x, fx

    # ---------------- Surrogate-assisted trust region (diagonal quadratic) ----------------
    # Model per dimension: f(x) ≈ c + sum_i (a_i*(x_i-b_i)^2 + g_i*(x_i-b_i))
    # Fit (a_i, g_i) by weighted least squares on near_log (no matrix libs; solve 2x2 per dim).
    def surrogate_propose(num_props=6):
        nonlocal best, best_x, last_improve_t
        if best_x is None:
            return
        # need enough points
        if len(near_log) < max(18, 4 * dim):
            return

        # trust radius shrinks over time
        frac = clamp01((now() - t0) / max(1e-12, float(max_time)))
        rad = (0.35 * (1.0 - 0.70 * frac))
        rad = max(0.04, rad)

        # pick a subset biased to best values
        # sort a small random sample by f, keep top
        m = min(len(near_log), 120)
        sample = random.sample(near_log, m)
        sample.sort(key=lambda t: t[1])
        keep = sample[:max(20, m // 2)]

        # compute weights based on distance to best
        # w = 1/(eps + dist^2)
        eps = 1e-12

        # fit per dimension independent (2 params): a and g
        a = [0.0] * dim
        g = [0.0] * dim

        fbest = best
        xb = best_x[:]

        for d in range(dim):
            # regression on features: t = (x_d - xb_d), y = f - fbest
            S_w = 0.0
            S_t = 0.0
            S_t2 = 0.0
            S_t3 = 0.0
            S_t4 = 0.0
            S_y_t = 0.0
            S_y_t2 = 0.0

            sd = span_safe[d]
            if sd <= 0.0:
                a[d] = 0.0
                g[d] = 0.0
                continue

            for x, fx in keep:
                t = (x[d] - xb[d])
                # normalized distance for weight
                # (use cheap approx: only this dimension + small random subset of others)
                dn = (t / sd)
                w = 1.0 / (eps + dn * dn)
                y = (fx - fbest)
                tt = t
                tt2 = tt * tt

                S_w += w
                S_t += w * tt
                S_t2 += w * tt2
                S_t3 += w * tt2 * tt
                S_t4 += w * tt2 * tt2
                S_y_t += w * y * tt
                S_y_t2 += w * y * tt2

            # Solve normal equations for y ≈ g*t + a*t^2
            # [S_t2  S_t3] [g] = [S_y_t ]
            # [S_t3  S_t4] [a]   [S_y_t2]
            det = (S_t2 * S_t4 - S_t3 * S_t3)
            if abs(det) < 1e-20:
                a[d] = 0.0
                g[d] = 0.0
                continue
            inv00 = S_t4 / det
            inv01 = -S_t3 / det
            inv10 = -S_t3 / det
            inv11 = S_t2 / det

            gd = inv00 * S_y_t + inv01 * S_y_t2
            ad = inv10 * S_y_t + inv11 * S_y_t2

            # mild regularization: avoid huge curvature
            # scale bound based on typical f scale (unknown); clamp to a safe range
            # (using span to normalize)
            ad = max(-1e6 / (sd * sd + 1e-30), min(1e6 / (sd * sd + 1e-30), ad))
            gd = max(-1e6 / (sd + 1e-30), min(1e6 / (sd + 1e-30), gd))
            a[d] = ad
            g[d] = gd

        # propose by approximate minimizer per dim within trust box:
        # minimize ad*t^2 + gd*t; if ad>0 => t* = -gd/(2ad); else step opposite sign of g
        props = []
        for _ in range(num_props):
            y = xb[:]
            for d in range(dim):
                sd = span_safe[d]
                if sd <= 0.0:
                    y[d] = lo[d]
                    continue

                # randomize trust radius a bit per proposal
                rloc = rad * random.uniform(0.6, 1.2)
                max_step = rloc * sd

                ad = a[d]
                gd = g[d]

                if ad > 1e-18:
                    tstar = -gd / (2.0 * ad)
                else:
                    # if curvature not convex, just move opposite gradient
                    if gd > 0.0:
                        tstar = -0.6 * max_step
                    elif gd < 0.0:
                        tstar = 0.6 * max_step
                    else:
                        tstar = 0.0

                # clamp to trust box and add a little noise
                tstar = max(-max_step, min(max_step, tstar))
                tstar += random.uniform(-0.15, 0.15) * max_step

                y[d] = reflect_repair(xb[d] + tstar, d)

            y = jitter_inside(y, rel=1e-15)
            props.append(y)

        # evaluate proposals
        for y in props:
            if now() >= deadline:
                return
            fy = evaluate(y)
            maybe_log(y, fy)
            if fy < best:
                best, best_x = fy, y[:]
                last_improve_t = now()

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
        if n < 4:
            while len(pop) < 4 and now() < deadline:
                x = rand_uniform_point()
                fx = evaluate(x)
                pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]
                maybe_log(x, fx)
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

            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = reflect_repair(v[d], d)

            # time-dependent tiny jitter
            if random.random() < (0.10 + 0.25 * frac):
                u = jitter_inside(u, rel=1e-14)
            else:
                u = jitter_inside(u, rel=1e-15)

            fu = evaluate(u)
            maybe_log(u, fu)

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

            alpha = 0.10 + 0.18 * frac
            MCR[mem_idx] = (1.0 - alpha) * MCR[mem_idx] + alpha * mcr_new
            MF[mem_idx] = (1.0 - alpha) * MF[mem_idx] + alpha * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # ---- hybrid exploitation schedule ----
        if best_x is not None:
            # surrogate trust-region more valuable mid/late
            if (it % (18 if frac < 0.45 else 9)) == 0:
                surrogate_propose(num_props=4 if dim > 30 else 6)

            # local refine more frequently late
            if (it % (16 if frac < 0.55 else 8)) == 0 and now() < deadline:
                xb, fb = local_refine(best_x, best, passes=2 if frac < 0.70 else 3)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()
                    for d in range(dim):
                        ls_step[d] = min(0.30 * span_safe[d], ls_step[d] * 1.15)
                else:
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.83)

        # ---- stagnation: soft restart ----
        if (now() - last_improve_t) > no_improve_window:
            last_improve_t = now()
            pop.sort(key=lambda z: z[1])

            elite = max(2, min(10, len(pop) // 5))
            kept = pop[:elite]
            new_pop = [[kept[i][0][:], kept[i][1]] for i in range(elite)]

            rad = 0.40 * (1.0 - 0.65 * frac)
            rad = max(0.06, rad)

            # trim archive but keep some
            if archive:
                random.shuffle(archive)
                archive = archive[:max(0, int(0.45 * arc_max))]

            while len(new_pop) < len(pop) and now() < deadline:
                r = random.random()
                if best_x is not None and r < 0.58:
                    x = [clip(random.gauss(best_x[d], rad * span_safe[d]), d) for d in range(dim)]
                elif r < 0.78:
                    # around an elite (not only best)
                    e = kept[random.randrange(len(kept))][0]
                    x = [clip(random.gauss(e[d], 0.95 * rad * span_safe[d]), d) for d in range(dim)]
                elif archive and r < 0.90:
                    a = archive[random.randrange(len(archive))]
                    x = [clip(random.gauss(a[d], 0.95 * rad * span_safe[d]), d) for d in range(dim)]
                else:
                    x = rand_uniform_point()

                x = jitter_inside(x, rel=1e-15)
                fx = evaluate(x)
                maybe_log(x, fx)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]

            pop = new_pop

            # broaden memories slightly
            for h in range(H):
                MF[h] = min(0.95, max(0.20, MF[h] * 1.05))
                MCR[h] = min(0.95, max(0.05, MCR[h] * 1.03))
            # reset local steps upward a bit
            for d in range(dim):
                ls_step[d] = max(ls_step[d], 0.08 * span_safe[d])
