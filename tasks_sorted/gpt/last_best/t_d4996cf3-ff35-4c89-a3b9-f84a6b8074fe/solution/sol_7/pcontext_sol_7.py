import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (no external libs).

    Upgrade over prior code:
      - Adds a proper trust-region local optimizer (Powell-like pattern search / GPS)
        with adaptive step + occasional 1D parabolic line search along promising dirs.
      - Uses a two-phase scheme:
          (1) Global: "DE/current-to-pbest/1 + archive" with SHADE-lite parameter memory.
          (2) Intensification: trust-region pattern search on the current best, triggered
              periodically and on stagnation.
      - Adds cheap "speculative restart" sampling around best with decreasing radii.
      - Improves time-awareness by estimating evaluation cost and scaling NP / local budgets.

    Returns best fitness (float).
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    var_idx = [i for i in range(dim) if spans[i] > 0.0]

    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # ---------------- utilities ----------------
    def now():
        return time.time()

    def reflect_inplace(x):
        # robust fold+reflect into bounds
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            w = hi - lo
            xi = x[i]
            if xi < lo or xi > hi:
                xi = lo + (xi - lo) % (2.0 * w)
                if xi > hi:
                    xi = hi - (xi - hi)
                x[i] = xi
        return x

    eval_count = 0
    eval_time = 0.0

    def eval_f(x):
        nonlocal eval_count, eval_time
        t = time.time()
        fx = float(func(x))
        eval_time += (time.time() - t)
        eval_count += 1
        return fx

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] > 0.0:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    # Halton init for coverage
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def clamp01(z):
        if z < 0.0:
            return 0.0
        if z > 1.0:
            return 1.0
        return z

    def cauchy(mu, gamma=0.12):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- Trust-region local optimizer (Pattern search + line search) ----------------
    def _dir_order_by_span():
        d = var_idx[:]
        d.sort(key=lambda j: spans[j], reverse=True)
        return d

    def _try_line_search(x, fx, direction, step, budget):
        """
        Very small 1D search along 'direction' (sparse vector: list of (j, dj)).
        Uses bracket + parabolic step if possible; otherwise exponential + shrink.
        """
        if budget <= 0 or now() >= deadline:
            return fx, x, 0

        used = 0

        def make_point(alpha):
            y = x[:]
            for j, dj in direction:
                y[j] = y[j] + alpha * dj
            reflect_inplace(y)
            return y

        # start with alpha = +/- step
        a0 = 0.0
        f0 = fx
        a1 = step
        y1 = make_point(a1)
        f1 = eval_f(y1); used += 1
        if now() >= deadline or used >= budget:
            if f1 < fx:
                return f1, y1, used
            return fx, x, used

        a2 = -step
        y2 = make_point(a2)
        f2 = eval_f(y2); used += 1

        # pick best among {-step,0,+step}
        best_a, best_f, best_x = a0, f0, x
        if f1 < best_f:
            best_a, best_f, best_x = a1, f1, y1
        if f2 < best_f:
            best_a, best_f, best_x = a2, f2, y2

        # if no improvement, quit
        if best_f >= fx:
            return fx, x, used

        # bracket further in best direction
        a = best_a
        fa = best_f
        grow = 2.0
        for _ in range(3):
            if used >= budget or now() >= deadline:
                break
            a_next = a * grow
            y = make_point(a_next)
            fy = eval_f(y); used += 1
            if fy < fa:
                a, fa, best_x = a_next, fy, y
            else:
                break

        return fa, best_x, used

    def local_trust_region(x_best, f_best, budget, init_frac=0.12):
        """
        Deterministic-ish GPS/pattern search:
          - coordinate +/- steps, adaptive step size (trust region)
          - occasionally do a sparse line search along a successful move direction
        """
        if budget <= 0 or now() >= deadline:
            return f_best, x_best

        x = x_best[:]
        fx = f_best
        used = 0

        # initial step per coordinate
        steps = [0.0] * dim
        for j in var_idx:
            steps[j] = max(1e-12, init_frac * spans[j])

        dims = _dir_order_by_span()

        no_improve_rounds = 0
        while used < budget and now() < deadline:
            improved = False
            best_move = None  # list of (j, dj)
            best_after = fx
            best_point = None

            # poll directions (+/- coordinate)
            for j in dims:
                if used >= budget or now() >= deadline:
                    break
                s = steps[j]
                if s <= 0.0:
                    continue

                xj = x[j]

                # +s
                y = x[:]
                y[j] = xj + s
                reflect_inplace(y)
                fy = eval_f(y); used += 1
                if fy < best_after:
                    best_after = fy
                    best_point = y
                    best_move = [(j, +s)]

                if used >= budget or now() >= deadline:
                    break

                # -s
                y = x[:]
                y[j] = xj - s
                reflect_inplace(y)
                fy = eval_f(y); used += 1
                if fy < best_after:
                    best_after = fy
                    best_point = y
                    best_move = [(j, -s)]

            if best_point is not None and best_after < fx:
                x, fx = best_point, best_after
                improved = True

                # expand successful step(s)
                for j, dj in best_move:
                    steps[j] *= 1.35

                # cheap line search along that direction
                if used < budget and now() < deadline and random.random() < 0.6:
                    f2, x2, u2 = _try_line_search(x, fx, best_move, step=1.0, budget=max(2, (budget - used) // 6))
                    used += u2
                    if f2 < fx:
                        x, fx = x2, f2

                no_improve_rounds = 0
            else:
                # shrink trust region
                for j in var_idx:
                    steps[j] *= 0.55
                no_improve_rounds += 1
                if no_improve_rounds >= 2:
                    break

            # stop if step sizes are tiny
            tiny = True
            for j in var_idx:
                if steps[j] > 1e-10 * spans[j]:
                    tiny = False
                    break
            if tiny:
                break

        return fx, x

    # ---------------- Initialization with time-aware NP ----------------
    # quick probe to estimate eval cost
    probe_x = rand_point()
    reflect_inplace(probe_x)
    f_probe = eval_f(probe_x)

    # estimate evaluations per second (rough)
    spent = max(1e-6, eval_time)
    eps = eval_count / spent

    # choose NP based on dim and speed; keep moderate
    NP = 10 + 6 * dim
    if eps < 800:   # expensive objective -> smaller pop
        NP = 8 + 4 * dim
    if eps < 200:
        NP = 6 + 3 * dim
    NP = int(max(18, min(140, NP)))
    if max_time <= 1.0:
        NP = min(NP, 60)
    if max_time <= 0.5:
        NP = min(NP, 36)

    pop = []
    fit = []

    x_best = probe_x[:]
    f_best = f_probe

    mid = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    # fill population
    for i in range(NP):
        if now() >= deadline:
            return f_best

        r = i % 6
        if r == 0:
            x = halton_point()
        elif r == 1:
            x = rand_point()
        elif r == 2:
            x = mid[:]
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.22 * spans[j])
        elif r == 3:
            x = halton_point()
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.16 * spans[j])
        elif r == 4:
            x = rand_point()
            xo = opposite_point(x)
            reflect_inplace(x); reflect_inplace(xo)
            fx = eval_f(x)
            if now() >= deadline:
                return min(f_best, fx)
            fo = eval_f(xo)
            if fo < fx:
                x, fx = xo, fo
            pop.append(x); fit.append(fx)
            if fx < f_best:
                f_best, x_best = fx, x[:]
            continue
        else:
            # around best (once we have it)
            x = x_best[:]
            rad = 0.35
            for j in var_idx:
                x[j] += random.gauss(0.0, rad * spans[j])

        reflect_inplace(x)
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # ---------------- Global optimizer: JADE/SHADE-lite DE ----------------
    archive = []
    archive_max = NP

    H = max(6, min(24, dim + 6))
    M_F = [0.55] * H
    M_CR = [0.75] * H
    mem_idx = 0

    p_s = [0.62, 0.26, 0.12]  # current-to-pbest, rand/1, best/1

    last_best = f_best
    stagn = 0
    gen = 0

    def pick_distinct(exclude_set):
        while True:
            r = random.randrange(NP)
            if r not in exclude_set:
                exclude_set.add(r)
                return r

    # intensification schedule
    next_local = 0

    while now() < deadline:
        gen += 1

        # sort indices by fitness
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda k: fit[k])
        if fit[idx_sorted[0]] < f_best:
            f_best = fit[idx_sorted[0]]
            x_best = pop[idx_sorted[0]][:]

        # decide local search moments
        # run more often near end or on stagnation
        tleft = max(0.0, deadline - now())
        frac_left = tleft / max(1e-9, float(max_time))
        want_local = (gen >= next_local) or (stagn >= 5) or (frac_left < 0.35 and gen % 3 == 0)

        if want_local and now() < deadline:
            # budget depends on eval speed and dim
            # keep small if objective expensive
            spent = max(1e-6, eval_time)
            eps = eval_count / spent
            base = 6 * len(var_idx) + 12
            if eps < 200:
                base = 3 * len(var_idx) + 8
            if eps < 80:
                base = 2 * len(var_idx) + 6

            fb2, xb2 = local_trust_region(x_best, f_best, budget=base, init_frac=0.10 if frac_left > 0.4 else 0.06)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                last_best = f_best
                stagn = 0
            next_local = gen + (5 if frac_left > 0.5 else 3)

        # p-best fraction: more exploitation later
        p_top = 0.07 + 0.23 * frac_left
        p_top = max(0.05, min(0.35, p_top))
        p_num = max(2, int(p_top * NP))

        S_F, S_CR, dF = [], [], []

        for i in range(NP):
            if now() >= deadline:
                return f_best

            r_mem = random.randrange(H)
            Fi = cauchy(M_F[r_mem], gamma=0.12)
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(M_F[r_mem], gamma=0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            CRi = clamp01(random.gauss(M_CR[r_mem], 0.12))

            u = random.random()
            if u < p_s[0]:
                strat = 0
            elif u < p_s[0] + p_s[1]:
                strat = 1
            else:
                strat = 2

            xi = pop[i]
            pbest = pop[idx_sorted[random.randrange(p_num)]]

            excl = {i}
            r1 = pick_distinct(excl)
            x_r1 = pop[r1]

            # pick r2 from pop or archive
            use_arch = (archive and random.random() < 0.55)
            if use_arch and random.random() < (len(archive) / (len(archive) + NP)):
                x_r2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_distinct(excl)
                x_r2 = pop[r2]

            v = xi[:]
            if strat == 0:
                for j in var_idx:
                    v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])
            elif strat == 1:
                excl2 = {i, r1}
                r2p = pick_distinct(excl2)
                x2 = pop[r2p]
                if archive and random.random() < 0.5:
                    x3 = archive[random.randrange(len(archive))]
                else:
                    r3p = pick_distinct(excl2)
                    x3 = pop[r3p]
                for j in var_idx:
                    v[j] = x_r1[j] + Fi * (x2[j] - x3[j])
            else:
                xb = x_best
                for j in var_idx:
                    v[j] = xb[j] + Fi * (x_r1[j] - x_r2[j])

            # binomial crossover
            uvec = xi[:]
            jrand = var_idx[random.randrange(len(var_idx))]
            for j in var_idx:
                if random.random() < CRi or j == jrand:
                    uvec[j] = v[j]

            reflect_inplace(uvec)
            fu = eval_f(uvec)

            if fu <= fit[i]:
                # archive parent
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                old = fit[i]
                pop[i] = uvec
                fit[i] = fu

                if fu < f_best:
                    f_best = fu
                    x_best = uvec[:]

                imp = old - fu
                if imp <= 0.0:
                    imp = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(imp)

        # update memories
        if dF:
            wsum = sum(dF)
            if wsum <= 0.0:
                wsum = 1e-12
            numF = denF = numCR = 0.0
            for w, Fi, CRi in zip(dF, S_F, S_CR):
                ww = w / wsum
                numF += ww * (Fi * Fi)
                denF += ww * Fi
                numCR += ww * CRi
            new_MF = (numF / denF) if denF > 1e-12 else M_F[mem_idx]
            M_F[mem_idx] = min(1.0, max(0.05, new_MF))
            M_CR[mem_idx] = min(1.0, max(0.0, numCR))
            mem_idx = (mem_idx + 1) % H

        # stagnation logic + speculative restart around best
        if f_best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = f_best
            stagn = 0
            p_s[0] = min(0.75, p_s[0] + 0.02)
            p_s[1] = max(0.16, p_s[1] - 0.015)
            p_s[2] = 1.0 - p_s[0] - p_s[1]
        else:
            stagn += 1
            if stagn >= 6:
                stagn = 0
                # more exploration
                p_s[0] = max(0.42, p_s[0] - 0.08)
                p_s[1] = min(0.46, p_s[1] + 0.07)
                p_s[2] = 1.0 - p_s[0] - p_s[1]

                # reinit worst ~20%: half around best with shrinking radius, half halton/random
                worst = list(range(NP))
                worst.sort(key=lambda k: fit[k], reverse=True)
                k_re = max(2, NP // 5)

                # radius shrinks with time
                tleft = max(0.0, deadline - now())
                frac_left = tleft / max(1e-9, float(max_time))
                rad = 0.35 * frac_left + 0.06

                for t in range(k_re):
                    if now() >= deadline:
                        return f_best
                    k = worst[t]
                    rr = random.random()
                    if rr < 0.55:
                        x = x_best[:]
                        for j in var_idx:
                            x[j] += random.gauss(0.0, rad * spans[j])
                    elif rr < 0.8:
                        x = halton_point()
                    else:
                        x = rand_point()
                    reflect_inplace(x)
                    fx = eval_f(x)
                    pop[k] = x
                    fit[k] = fx
                    if fx < f_best:
                        f_best = fx
                        x_best = x[:]

    return f_best
