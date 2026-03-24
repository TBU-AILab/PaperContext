import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded continuous minimizer (no external libs).

    Key improvements vs your best (SHADE + diagonal quadratic TR):
      1) Adds a *true* local second-order trust-region step using a small random subspace:
         - builds a finite-difference quadratic model (gradient + Hessian in subspace)
         - takes a damped Newton / LM step inside a trust radius
         This captures interactions (non-separable structure) better than diagonal surrogate.

      2) Keeps the strong global engine (L-SHADE current-to-pbest/1 + archive),
         but improves exploitation by:
         - periodic subspace TR (as above)
         - short coordinate quadratic (1D parabolic) refinements on best

      3) Better stagnation response: restart radius uses both time and recent progress;
         elite preservation remains.

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
        # repeated reflection handles far OOB
        for _ in range(12):
            if v < a:
                v = a + (a - v)
            elif v > b:
                v = b - (v - b)
            else:
                break
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
    init_pop = max(28, min(130, init_pop))
    min_pop = max(10, min(34, 7 + int(2.0 * math.sqrt(max(1, dim)))))

    arc_factor = 1.25

    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    pmin, pmax = 0.05, 0.25

    # ---------------- Initialization (anchors + Halton + opposition) ----------------
    pop = []
    best = float("inf")
    best_x = None

    mid = [(lo[i] + hi[i]) * 0.5 for i in range(dim)]
    anchors = [mid]
    for _ in range(min(3, max(1, dim // 12))):
        anchors.append(rand_uniform_point())

    # Keep a modest log of points near best (for TR step acceptance heuristics / reuse)
    near_log = []  # (x, f)

    def maybe_log(x, fx):
        if best_x is None:
            return
        s = 0.0
        for d in range(dim):
            sd = span_safe[d]
            if sd > 0.0:
                z = (x[d] - best_x[d]) / sd
                s += z * z
        dist = math.sqrt(s)
        if dist < 0.55 or random.random() < 0.03:
            near_log.append((x[:], fx))
            if len(near_log) > 320:
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
    last_best_val = best

    # ---------------- Local: 1D quadratic refinement on best ----------------
    def coord_quad_refine(x0, f0, coord_count):
        x = x0[:]
        fx = f0
        # time-scaled radius
        frac = clamp01((now() - t0) / max(1e-12, float(max_time)))
        base = max(0.02, 0.14 * (1.0 - 0.75 * frac))

        if dim <= 0:
            return x, fx

        coords = random.sample(range(dim), min(dim, max(1, coord_count)))
        for d in coords:
            if now() >= deadline:
                break
            sd = span_safe[d]
            h = base * sd * random.uniform(0.8, 1.25)
            if h <= 1e-20:
                continue

            xm = x[:]
            xp = x[:]
            xm[d] = reflect_repair(x[d] - h, d)
            xp[d] = reflect_repair(x[d] + h, d)

            fm = evaluate(xm); maybe_log(xm, fm)
            if now() >= deadline:
                break
            fp = evaluate(xp); maybe_log(xp, fp)

            denom = (fm - 2.0 * fx + fp)
            # candidate along parabola (clamped)
            if abs(denom) > 1e-22:
                tstar = 0.5 * h * (fm - fp) / denom
                if tstar < -h:
                    tstar = -h
                elif tstar > h:
                    tstar = h
                xs = x[:]
                xs[d] = reflect_repair(x[d] + tstar, d)
                xs = jitter_inside(xs, rel=1e-15)
                fs = evaluate(xs); maybe_log(xs, fs)
            else:
                xs, fs = x, fx

            # take best among {xm, x, xp, xs}
            bx, bf = x, fx
            if fm < bf:
                bx, bf = xm, fm
            if fp < bf:
                bx, bf = xp, fp
            if fs < bf:
                bx, bf = xs, fs
            if bf < fx:
                x, fx = bx[:], bf
        return x, fx

    # ---------------- Subspace Trust-Region (finite-diff quadratic) ----------------
    # Build model in a small subspace S: indices idxs (k dims):
    #   g ~ (f(x+he_i)-f(x-he_i))/(2h)
    #   H_ii ~ (f(x+he_i)-2f(x)+f(x-he_i))/h^2
    #   H_ij via 4-point formula using f(x±he_i±he_j)
    # Take LM step: (H + lam I) p = -g, clamp to trust radius, accept if improves.
    def subspace_trust_region(x0, f0, idxs, rad_norm):
        if now() >= deadline:
            return x0, f0
        ksub = len(idxs)
        if ksub == 0:
            return x0, f0

        # choose step sizes per chosen coordinate
        h = [0.0] * ksub
        for t, d in enumerate(idxs):
            h[t] = max(1e-12 * span_safe[d], 0.18 * rad_norm * span_safe[d])

        f_center = f0
        # cache evaluations at +/- ei
        f_plus = [None] * ksub
        f_minus = [None] * ksub

        # evaluate +/- for each axis
        for i in range(ksub):
            if now() >= deadline:
                return x0, f0
            d = idxs[i]
            xi = x0[d]
            hp = h[i]

            x_p = x0[:]
            x_m = x0[:]
            x_p[d] = reflect_repair(xi + hp, d)
            x_m[d] = reflect_repair(xi - hp, d)

            fp = evaluate(x_p); maybe_log(x_p, fp)
            if now() >= deadline:
                return x0, f0
            fm = evaluate(x_m); maybe_log(x_m, fm)

            f_plus[i] = fp
            f_minus[i] = fm

        # gradient and diagonal Hessian
        g = [0.0] * ksub
        H = [[0.0] * ksub for _ in range(ksub)]
        for i in range(ksub):
            hp = h[i]
            g[i] = (f_plus[i] - f_minus[i]) / (2.0 * hp)
            H[i][i] = (f_plus[i] - 2.0 * f_center + f_minus[i]) / (hp * hp + 1e-300)

        # off-diagonals
        for i in range(ksub):
            for j in range(i + 1, ksub):
                if now() >= deadline:
                    return x0, f0
                di, dj = idxs[i], idxs[j]
                hi_, hj_ = h[i], h[j]

                # 4 corners around center
                x_pp = x0[:]; x_pp[di] = reflect_repair(x_pp[di] + hi_, di); x_pp[dj] = reflect_repair(x_pp[dj] + hj_, dj)
                x_pm = x0[:]; x_pm[di] = reflect_repair(x_pm[di] + hi_, di); x_pm[dj] = reflect_repair(x_pm[dj] - hj_, dj)
                x_mp = x0[:]; x_mp[di] = reflect_repair(x_mp[di] - hi_, di); x_mp[dj] = reflect_repair(x_mp[dj] + hj_, dj)
                x_mm = x0[:]; x_mm[di] = reflect_repair(x_mm[di] - hi_, di); x_mm[dj] = reflect_repair(x_mm[dj] - hj_, dj)

                fpp = evaluate(x_pp); maybe_log(x_pp, fpp)
                if now() >= deadline:
                    return x0, f0
                fpm = evaluate(x_pm); maybe_log(x_pm, fpm)
                if now() >= deadline:
                    return x0, f0
                fmp = evaluate(x_mp); maybe_log(x_mp, fmp)
                if now() >= deadline:
                    return x0, f0
                fmm = evaluate(x_mm); maybe_log(x_mm, fmm)

                hij = (fpp - fpm - fmp + fmm) / (4.0 * hi_ * hj_ + 1e-300)
                H[i][j] = hij
                H[j][i] = hij

        # Solve (H + lam I) p = -g via Gaussian elimination (small k)
        def solve_linear(A, b):
            n = len(b)
            M = [A[i][:] + [b[i]] for i in range(n)]
            for col in range(n):
                # pivot
                piv = col
                best_abs = abs(M[col][col])
                for r in range(col + 1, n):
                    av = abs(M[r][col])
                    if av > best_abs:
                        best_abs = av
                        piv = r
                if best_abs < 1e-18:
                    return None
                if piv != col:
                    M[col], M[piv] = M[piv], M[col]
                # normalize
                div = M[col][col]
                inv = 1.0 / div
                for c in range(col, n + 1):
                    M[col][c] *= inv
                # eliminate
                for r in range(n):
                    if r == col:
                        continue
                    factor = M[r][col]
                    if factor != 0.0:
                        for c in range(col, n + 1):
                            M[r][c] -= factor * M[col][c]
            return [M[i][n] for i in range(n)]

        # LM loop: increase damping until step improves (or time runs out)
        lam = 1e-6
        best_local_x = x0[:]
        best_local_f = f0

        # trust radius in actual units: rad_norm * span
        # We'll enforce normalized radius in chosen subspace:
        #   ||p_norm||2 = sqrt(sum((p_i/span_i)^2)) <= rad_norm
        for _ in range(6):
            if now() >= deadline:
                break
            A = [row[:] for row in H]
            for i in range(ksub):
                A[i][i] += lam

            b = [-gi for gi in g]
            p = solve_linear(A, b)
            if p is None:
                lam *= 20.0
                continue

            # clamp to trust region (normalized)
            norm2 = 0.0
            for i in range(ksub):
                d = idxs[i]
                norm2 += (p[i] / (span_safe[d] + 1e-300)) ** 2
            norm = math.sqrt(norm2)
            if norm > rad_norm and norm > 0.0:
                scale = rad_norm / norm
                for i in range(ksub):
                    p[i] *= scale

            x_try = x0[:]
            for i in range(ksub):
                d = idxs[i]
                x_try[d] = reflect_repair(x_try[d] + p[i], d)
            x_try = jitter_inside(x_try, rel=1e-15)

            f_try = evaluate(x_try); maybe_log(x_try, f_try)
            if f_try < best_local_f:
                best_local_x, best_local_f = x_try[:], f_try
                # success: slightly reduce damping and try once more for extra gain
                lam *= 0.35
            else:
                lam *= 10.0

        return best_local_x, best_local_f

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
                maybe_log(x, fx)
                if fx < best:
                    best, best_x = fx, x[:]
            pop.sort(key=lambda z: z[1])
            n = len(pop)

        p = pmax - (pmax - pmin) * frac
        pbest_count = max(2, int(math.ceil(p * n)))

        union = [ind[0] for ind in pop] + archive
        union_n = len(union)

        S_F, S_CR, S_w = [], [], []

        # --- DE generation ---
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

            # time-dependent jitter
            u = jitter_inside(u, rel=(1e-14 if random.random() < (0.10 + 0.25 * frac) else 1e-15))

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

            alpha = 0.10 + 0.20 * frac
            MCR[mem_idx] = (1.0 - alpha) * MCR[mem_idx] + alpha * mcr_new
            MF[mem_idx] = (1.0 - alpha) * MF[mem_idx] + alpha * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # --- Hybrid exploitation ---
        if best_x is not None and now() < deadline:
            # (A) 1D quadratic refine
            if (it % (16 if frac < 0.55 else 8)) == 0:
                xb, fb = coord_quad_refine(best_x, best, coord_count=1 + int(math.sqrt(max(1, dim)) / 2))
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()

            # (B) Subspace trust-region (more valuable mid/late)
            if (it % (22 if frac < 0.45 else 11)) == 0:
                # choose a small subspace size
                ksub = 2 if dim <= 6 else (3 if dim <= 20 else (4 if dim <= 60 else 5))
                ksub = min(dim, ksub)

                # normalized trust radius shrinks over time
                rad = 0.22 * (1.0 - 0.70 * frac)
                rad = max(0.04, rad)

                idxs = random.sample(range(dim), ksub)
                xb, fb = subspace_trust_region(best_x, best, idxs, rad_norm=rad)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()

        # --- stagnation: soft restart ---
        if (now() - last_improve_t) > no_improve_window:
            # estimate "progress"; if none, restart more aggressively
            progress = abs(last_best_val - best)
            last_best_val = best

            pop.sort(key=lambda z: z[1])
            elite = max(2, min(10, len(pop) // 5))
            kept = pop[:elite]
            new_pop = [[kept[i][0][:], kept[i][1]] for i in range(elite)]

            # radius depends on time + progress
            base_rad = 0.42 * (1.0 - 0.65 * frac)
            base_rad = max(0.06, base_rad)
            if progress < 1e-10:
                base_rad = min(0.60, base_rad * 1.35)

            if archive:
                random.shuffle(archive)
                archive = archive[:max(0, int(0.45 * arc_max))]

            while len(new_pop) < len(pop) and now() < deadline:
                r = random.random()
                if best_x is not None and r < 0.58:
                    x = [clip(random.gauss(best_x[d], base_rad * span_safe[d]), d) for d in range(dim)]
                elif r < 0.80:
                    e = kept[random.randrange(len(kept))][0]
                    x = [clip(random.gauss(e[d], 0.95 * base_rad * span_safe[d]), d) for d in range(dim)]
                elif archive and r < 0.92:
                    a = archive[random.randrange(len(archive))]
                    x = [clip(random.gauss(a[d], 0.95 * base_rad * span_safe[d]), d) for d in range(dim)]
                else:
                    x = rand_uniform_point()

                x = jitter_inside(x, rel=1e-15)
                fx = evaluate(x)
                maybe_log(x, fx)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = now()

            pop = new_pop

            # broaden memories slightly after restart
            for h in range(H):
                MF[h] = min(0.95, max(0.18, MF[h] * 1.06))
                MCR[h] = min(0.95, max(0.05, MCR[h] * 1.03))

            last_improve_t = now()
