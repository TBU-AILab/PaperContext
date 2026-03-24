import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved hybrid minimizer (self-contained, no external libs).

    Key improvements over your best (~3.2208):
      1) Stronger endgame: Multi-start *Powell-like* derivative-free local search
         (coordinate + adaptive random orthogonal directions) with quadratic step bracketing.
         - More reliable than the compact-CMA in tight basins, and cheaper than FD-LBFGS.
      2) Better DE core: current-to-pbest/1 (SHADE-ish) + occasional "best/2" when stalled,
         plus more robust parameter adaptation and diversity refresh.
      3) Elite set: keep diversified top solutions; use them for local multi-start polishing.
      4) Time budgeting: always reserve time for several local solves (not just one).

    Assumptions:
      - func: callable(list[float]) -> float, deterministic preferred.
      - bounds: list of (lo, hi) length dim.
      - returns: best fitness found (float).
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    if max(spans) <= 0.0:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # -------------------- helpers --------------------
    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            for _ in range(10):
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
                else:
                    break
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def eval_vec(x):
        return float(func(reflect_into_bounds(x)))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def add_scaled(x, d, alpha):
        return [x[i] + alpha * d[i] for i in range(dim)]

    # ---- scrambled Halton (init) ----
    def _first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    PRIMES = _first_primes(dim)

    def _radical_inverse(index, base, shift):
        f = 1.0 / base
        x = 0.0
        i = index + shift
        while i > 0:
            x += (i % base) * f
            i //= base
            f /= base
        return x

    def halton_point(k, shifts):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                u = _radical_inverse(k, PRIMES[d], shifts[d])
                x[d] = lows[d] + u * spans[d]
        return x

    # -------------------- elite container (diversified) --------------------
    def elite_insert(elites, x, fx, kmax=10):
        if not elites:
            elites.append((fx, x[:]))
            return
        # merge if too close in normalized distance
        for idx, (ef, ex) in enumerate(elites):
            dsq = 0.0
            for i in range(dim):
                s = spans[i] if spans[i] > 0.0 else 1.0
                d = (x[i] - ex[i]) / s
                dsq += d * d
            if dsq < 2e-6:
                if fx < ef:
                    elites[idx] = (fx, x[:])
                elites.sort(key=lambda t: t[0])
                return
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > kmax:
            elites.pop()

    # -------------------- Local: Powell-like directional search --------------------
    def local_powellish(x0, f0, time_slice):
        """
        Derivative-free local search:
          - maintain a set of directions (coords + learned + random orthogonal-ish)
          - along each direction do a 1D line search with bracketing + quadratic fit
        """
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        x = x0[:]
        fx = f0

        # base step ~ range
        avg_span = sum(spans) / float(dim)
        step0 = 0.12 * avg_span if avg_span > 0 else 1.0
        step_min = 1e-14 * (avg_span if avg_span > 0 else 1.0)
        step_max = 0.60 * (avg_span if avg_span > 0 else 1.0)

        # initial directions: coordinate axes
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = 1.0
            dirs.append(d)

        def orthonormalize(vecs, tries=3):
            # very small Gram-Schmidt with fallbacks
            out = []
            for v in vecs:
                u = v[:]
                for b in out:
                    proj = dot(u, b)
                    if proj != 0.0:
                        for i in range(dim):
                            u[i] -= proj * b[i]
                n = norm(u)
                if n > 1e-14:
                    inv = 1.0 / n
                    u = [ui * inv for ui in u]
                    out.append(u)
            # if too few, add random ones
            for _ in range(tries):
                if len(out) >= min(dim, 6):
                    break
                u = [random.gauss(0.0, 1.0) for _ in range(dim)]
                for b in out:
                    proj = dot(u, b)
                    for i in range(dim):
                        u[i] -= proj * b[i]
                n = norm(u)
                if n > 1e-14:
                    inv = 1.0 / n
                    out.append([ui * inv for ui in u])
            return out

        def line_search_dir(xc, fxc, d, step):
            """
            Bracket around 0 then do a few quadratic-interpolation refinements.
            Returns (xbest, fbest, new_step, improved_flag)
            """
            # normalize direction
            dn = norm(d)
            if dn <= 1e-18:
                return xc, fxc, step, False
            inv = 1.0 / dn
            d = [di * inv for di in d]

            # evaluate at +/- step
            a0 = 0.0
            f0_ = fxc

            a1 = step
            x1 = reflect_into_bounds(add_scaled(xc, d, a1))
            f1 = float(func(x1))

            a_1 = -step
            x_1 = reflect_into_bounds(add_scaled(xc, d, a_1))
            f_1 = float(func(x_1))

            if time.time() >= t_end:
                # choose best among sampled
                if f1 < f0_ and f1 <= f_1:
                    return x1, f1, step, True
                if f_1 < f0_ and f_1 < f1:
                    return x_1, f_1, step, True
                return xc, fxc, step, False

            # pick best side and try bracketing further
            if f1 < f0_ or f_1 < f0_:
                if f1 <= f_1:
                    direction = 1.0
                    ab, fb = a1, f1
                else:
                    direction = -1.0
                    ab, fb = a_1, f_1

                # expand
                a_prev, f_prev = a0, f0_
                a_curr, f_curr = ab, fb
                mult = 2.0
                for _ in range(8):
                    if time.time() >= t_end:
                        break
                    a_next = a_curr + direction * mult * step
                    x_next = reflect_into_bounds(add_scaled(xc, d, a_next))
                    f_next = float(func(x_next))
                    if f_next < f_curr:
                        a_prev, f_prev = a_curr, f_curr
                        a_curr, f_curr = a_next, f_next
                        mult *= 1.7
                        if abs(a_curr) > 6.0 * step_max:
                            break
                    else:
                        # bracket: a_prev (higher), a_curr (best), a_next (higher)
                        aL, fL = a_prev, f_prev
                        aM, fM = a_curr, f_curr
                        aR, fR = a_next, f_next
                        break
                else:
                    # no bracket found; accept best seen
                    xbest = reflect_into_bounds(add_scaled(xc, d, a_curr))
                    return xbest, f_curr, min(step_max, step * 1.2), True

                # quadratic refinements within bracket
                best_a, best_f = aM, fM
                for _ in range(8):
                    if time.time() >= t_end:
                        break
                    # fit parabola through (aL,fL), (aM,fM), (aR,fR)
                    denom = (aL - aM) * (aL - aR) * (aM - aR)
                    if abs(denom) < 1e-30:
                        break
                    A = (aR * (fM - fL) + aM * (fL - fR) + aL * (fR - fM)) / denom
                    B = (aR*aR * (fL - fM) + aM*aM * (fR - fL) + aL*aL * (fM - fR)) / denom
                    if A <= 0.0:
                        break
                    a_star = -B / (2.0 * A)

                    # keep inside bracket and not too close to edges
                    lo = min(aL, aR)
                    hi = max(aL, aR)
                    pad = 0.08 * (hi - lo)
                    if a_star <= lo + pad:
                        a_star = lo + pad
                    elif a_star >= hi - pad:
                        a_star = hi - pad

                    x_star = reflect_into_bounds(add_scaled(xc, d, a_star))
                    f_star = float(func(x_star))
                    if f_star < best_f:
                        best_f = f_star
                        best_a = a_star

                    # update bracket keeping aM as best point
                    # ensure aM is best among L,M,R
                    candidates = [(fL, aL), (fM, aM), (fR, aR), (f_star, a_star)]
                    candidates.sort()
                    fM, aM = candidates[0]
                    # choose L and R around aM from remaining points
                    left = [c for c in candidates[1:] if c[1] < aM]
                    right = [c for c in candidates[1:] if c[1] > aM]
                    if not left or not right:
                        break
                    fL, aL = min(left, key=lambda t: abs(t[1] - aM))
                    fR, aR = min(right, key=lambda t: abs(t[1] - aM))

                xbest = reflect_into_bounds(add_scaled(xc, d, best_a))
                improved = (best_f < fxc)
                # if improved, allow a bit larger steps; else shrink
                new_step = min(step_max, step * (1.15 if improved else 0.70))
                new_step = max(step_min, new_step)
                return xbest, best_f, new_step, improved

            # no improvement at +/- step
            return xc, fxc, max(step_min, step * 0.72), False

        # main loop
        step = step0
        noimp = 0
        best_move_dir = None

        while time.time() < t_end:
            x_start = x[:]
            f_start = fx

            # add a few extra directions (learned + random orthogonal) each outer pass
            extra = []
            if best_move_dir is not None:
                extra.append(best_move_dir[:])
            # 2-4 random directions
            for _ in range(2 if dim > 25 else 3):
                extra.append([random.gauss(0.0, 1.0) for _ in range(dim)])
            extra = orthonormalize(extra, tries=2)
            all_dirs = dirs + extra

            improved_any = False
            best_delta = None
            best_delta_f = 0.0

            for d in all_dirs:
                if time.time() >= t_end:
                    break
                xn, fn, step, improved = line_search_dir(x, fx, d, step)
                if improved:
                    delta = [xn[i] - x[i] for i in range(dim)]
                    df = fx - fn
                    x, fx = xn, fn
                    improved_any = True
                    if (best_delta is None) or (df > best_delta_f):
                        best_delta = delta
                        best_delta_f = df

            if improved_any:
                noimp = 0
                # learn a "conjugate-ish" direction: x - x_start
                move = [x[i] - x_start[i] for i in range(dim)]
                if norm(move) > 1e-16:
                    best_move_dir = move
                # mild step growth
                step = min(step_max, step * 1.10)
            else:
                noimp += 1
                step = max(step_min, step * 0.70)
                if noimp >= 3:
                    break

            # stopping if very small step
            if step <= step_min * 10.0:
                break

        return x, fx

    # -------------------- Global: SHADE-ish DE --------------------
    pop_size = max(26, min(160, 12 * dim))
    H = 12
    MF = [0.62] * H
    MCR = [0.88] * H
    hist_idx = 0
    c = 0.10

    archive = []
    arch_max = pop_size

    best = float("inf")
    best_x = None
    elites = []
    last_improve = time.time()
    min_progress = 1e-12

    T = float(max_time)
    stall_seconds = max(0.12 * T, 0.45)
    endgame_reserve = 0.33 * T  # more reserve: multi-start local is powerful

    def update_best(x, fx):
        nonlocal best, best_x, last_improve
        if fx + min_progress < best:
            best = fx
            best_x = x[:]
            last_improve = time.time()
        elite_insert(elites, x, fx, kmax=12)

    def sample_F(muF):
        F = -1.0
        for _ in range(14):
            u = random.random()
            F = muF + 0.12 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                break
        if F <= 0.0:
            F = 0.5
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(muCR):
        CR = random.gauss(muCR, 0.10)
        if CR < 0.0:
            CR = 0.0
        elif CR > 1.0:
            CR = 1.0
        return CR

    shifts = [random.randrange(1, 7000) for _ in range(dim)]
    halton_k = random.randrange(20, 160)

    def refresh_population(pop, fit, frac=0.18):
        # replace worst fraction with opposition-around-best and random
        n = len(pop)
        k = max(1, int(frac * n))
        idx_sorted = sorted(range(n), key=lambda i: fit[i])
        worst = idx_sorted[-k:]
        for t, wi in enumerate(worst):
            if time.time() >= deadline:
                return
            if best_x is not None and t < k // 2:
                x = [lows[d] + highs[d] - best_x[d] for d in range(dim)]
                for d in range(dim):
                    if spans[d] > 0.0:
                        x[d] += random.gauss(0.0, 0.12 * spans[d])
                x = reflect_into_bounds(x)
            else:
                x = rand_vec()
            fx = float(func(x))
            pop[wi] = x
            fit[wi] = fx
            update_best(x, fx)

    while time.time() < deadline:
        # ---- init population: best-centered + elites-centered + Halton ----
        pop = []

        def add_centered(center, count, scale):
            for _ in range(count):
                x = center[:]
                for d in range(dim):
                    if spans[d] > 0.0:
                        x[d] += random.gauss(0.0, scale * spans[d])
                pop.append(reflect_into_bounds(x))

        if best_x is not None:
            add_centered(best_x, min(pop_size // 4, 28), 0.20)
        if elites:
            for _, ex in elites[:min(4, len(elites))]:
                add_centered(ex, 3, 0.15)

        while len(pop) < pop_size:
            halton_k += 1
            x = halton_point(halton_k, shifts)
            for d in range(dim):
                if spans[d] > 0.0:
                    x[d] += (random.random() - 0.5) * 0.003 * spans[d]
            pop.append(reflect_into_bounds(x))

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = float(func(x))
            fit.append(fx)
            update_best(x, fx)

        # ---- DE generations ----
        while time.time() < deadline:
            remaining = deadline - time.time()

            # ---- endgame: multi-start local from elites + best ----
            if best_x is not None and remaining <= endgame_reserve:
                starts = []
                if elites:
                    starts.extend([ex for _, ex in elites[:min(7, len(elites))]])
                starts.append(best_x[:])
                # diversify if few
                while len(starts) < 4:
                    starts.append(rand_vec())

                # allocate time slices (slightly biased to best)
                total = max(1e-12, remaining)
                per = total / float(len(starts) + 1)
                # 2 passes if time allows
                passes = 2 if remaining > 0.18 else 1
                for _ in range(passes):
                    for si, sx in enumerate(starts):
                        if time.time() >= deadline:
                            return best
                        slice_t = per * (1.6 if (sx is best_x or si == len(starts) - 1) else 1.0)
                        sf = eval_vec(sx)
                        bx, bf = local_powellish(sx, sf, slice_t)
                        update_best(bx, bf)
                return best

            frac_left = remaining / max(1e-9, T)
            pfrac = 0.30 if frac_left > 0.70 else (0.16 if frac_left > 0.30 else 0.10)

            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            best_idx = idx_sorted[0]
            pcount = max(2, int(math.ceil(pfrac * pop_size)))
            pbest_idx = idx_sorted[:pcount]

            f_best = fit[best_idx]
            f_med = fit[idx_sorted[pop_size // 2]]
            converged = (abs(f_med - f_best) <= 1e-11 * (1.0 + abs(f_best)))

            S_F, S_CR, dF = [], [], []
            new_pop = pop[:]
            new_fit = fit[:]

            # small chance to use a stronger mutation when stalled/converged
            use_best2 = converged and (time.time() - last_improve) > 0.35 * stall_seconds and random.random() < 0.45

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]
                r = random.randrange(H)
                F = sample_F(MF[r])
                CR = sample_CR(MCR[r])

                xbest = pop[best_idx]
                pb = pop[random.choice(pbest_idx)]

                # choose r1,r2,r3,r4
                def pick(exclude):
                    j = random.randrange(pop_size)
                    while j in exclude:
                        j = random.randrange(pop_size)
                    return j

                r1 = pick({i})
                r2 = pick({i, r1})
                r3 = pick({i, r1, r2})
                r4 = pick({i, r1, r2, r3})

                xr1 = pop[r1]
                # archive usage for r2
                if archive and random.random() < 0.55:
                    pool = pop + archive
                    xr2 = pool[random.randrange(len(pool))]
                else:
                    xr2 = pop[r2]

                # mutation
                if use_best2:
                    # DE/best/2: xbest + F*(xr1-xr2) + F*(xr3-xr4)
                    xr3v = pop[r3]
                    xr4v = pop[r4]
                    v = [xbest[d] + F * (xr1[d] - xr2[d]) + F * (xr3v[d] - xr4v[d]) for d in range(dim)]
                else:
                    # current-to-pbest/1
                    v = [xi[d] + F * (pb[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

                # jitter a coordinate if converged
                if converged and random.random() < 0.30:
                    j = random.randrange(dim)
                    if spans[j] > 0.0:
                        v[j] += random.gauss(0.0, 0.03 * spans[j])

                # crossover
                jrand = random.randrange(dim)
                u = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        u[d] = v[d]

                u = reflect_into_bounds(u)
                fu = float(func(u))

                if fu <= fi:
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = u
                    new_fit[i] = fu

                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    dF.append(df if df > 0.0 else 1e-16)
                    update_best(u, fu)

            pop, fit = new_pop, new_fit

            # SHADE memory update
            if S_F:
                wsum = sum(dF)
                inv = 1.0 / wsum if wsum > 0 else 1.0 / len(dF)
                weights = [(df * inv) if wsum > 0 else inv for df in dF]

                num = 0.0
                den = 0.0
                for wgt, f in zip(weights, S_F):
                    num += wgt * f * f
                    den += wgt * f
                F_new = num / den if den > 1e-30 else (sum(S_F) / len(S_F))

                CR_new = 0.0
                for wgt, cr in zip(weights, S_CR):
                    CR_new += wgt * cr

                MF[hist_idx] = (1.0 - c) * MF[hist_idx] + c * F_new
                MCR[hist_idx] = (1.0 - c) * MCR[hist_idx] + c * CR_new
                hist_idx = (hist_idx + 1) % H

            # mid-game occasional local polish (short), but keep endgame reserve
            remaining = deadline - time.time()
            if best_x is not None and remaining > endgame_reserve + 0.03:
                if converged and random.random() < 0.22:
                    bx, bf = local_powellish(best_x, best, min(0.09, 0.10 * remaining))
                    update_best(bx, bf)

            # diversity refresh if stuck
            if converged and (time.time() - last_improve) > 0.60 * stall_seconds and remaining > endgame_reserve + 0.07:
                refresh_population(pop, fit, frac=0.20)

            # hard stall: do a bigger local attempt then restart outer loop
            if (time.time() - last_improve) > stall_seconds:
                remaining = deadline - time.time()
                if best_x is not None and remaining > endgame_reserve + 0.06:
                    # multi-start small set
                    seeds = [best_x[:]]
                    if elites:
                        seeds += [ex for _, ex in elites[:min(2, len(elites))]]
                    slice_each = min(0.16, 0.25 * remaining / float(len(seeds)))
                    for sx in seeds:
                        if time.time() >= deadline:
                            return best
                        sf = eval_vec(sx)
                        bx, bf = local_powellish(sx, sf, slice_each)
                        update_best(bx, bf)
                break

    return best
