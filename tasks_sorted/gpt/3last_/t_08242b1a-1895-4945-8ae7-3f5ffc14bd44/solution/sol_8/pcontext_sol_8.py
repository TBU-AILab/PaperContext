import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (self-contained, no external libs).

    Main upgrades over your current best (DE + compact-CMA-ish):
      1) Much stronger *local* engine: small L-BFGS using finite-difference gradients
         (with safe fallback to derivative-free trust-region random directions).
         - When objective is smooth-ish, this is a big jump in endgame quality.
      2) Better global-to-local handoff: run SHADE-like DE, but maintain a TOP-K elite
         set and repeatedly launch short local solves from diversified elites.
      3) Smarter time budgeting and restart triggers, fewer wasted evaluations.

    Notes:
      - func is assumed deterministic; if noisy, FD gradients may be less effective.
      - Bounds are enforced by reflection.
      - Returns best fitness found (float).
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
            # reflect a few times then clip
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

    def norm2(a):
        return math.sqrt(max(0.0, dot(a, a)))

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

    # -------------------- elite container --------------------
    # keep small list of best solutions found (diversified by distance)
    def elite_insert(elites, x, fx, kmax=8):
        # elites: list of (fx, x)
        if not elites:
            elites.append((fx, x[:]))
            return
        # if very close to an existing elite, only keep best
        for idx, (ef, ex) in enumerate(elites):
            # normalized distance
            dsq = 0.0
            for i in range(dim):
                s = spans[i] if spans[i] > 0.0 else 1.0
                d = (x[i] - ex[i]) / s
                dsq += d * d
            if dsq < 1e-6:
                if fx < ef:
                    elites[idx] = (fx, x[:])
                elites.sort(key=lambda t: t[0])
                return
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > kmax:
            elites.pop()

    # -------------------- Local: L-BFGS (finite-diff) + fallback trust-region --------------------
    def local_lbfgs(x0, f0, time_slice):
        """
        Small L-BFGS with backtracking line search.
        Uses finite-difference gradients with adaptive step sizes.
        Falls back to random-direction trust region if gradients fail to help.
        """
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        x = x0[:]
        fx = f0

        # memory size
        m = 7 if dim <= 40 else 5
        S = []  # s vectors
        Y = []  # y vectors
        RHO = []  # 1/(y^T s)

        # finite-diff base step
        base_eps = 1e-7

        # trust radius fallback
        trust = 0.15 * (sum(spans) / float(dim))
        trust_min = 1e-12 * (sum(spans) / float(dim))
        trust_max = 0.65 * (sum(spans) / float(dim))

        def fd_grad(xc, fxc):
            # central/forward mixed: use forward differences (cheap)
            g = [0.0] * dim
            # scale eps per coordinate
            for i in range(dim):
                if spans[i] <= 0.0:
                    g[i] = 0.0
                    continue
                h = max(base_eps * (1.0 + abs(xc[i])), 1e-12 * spans[i])
                # keep h inside bounds scale
                if h > 1e-2 * spans[i]:
                    h = 1e-2 * spans[i]
                xp = xc[:]
                xp[i] += h
                fp = eval_vec(xp)
                g[i] = (fp - fxc) / h
                if time.time() >= t_end:
                    break
            return g

        def two_loop(g):
            # returns approx H*g (actually -H*g for descent direction later)
            q = g[:]
            alpha = [0.0] * len(S)
            for i in range(len(S) - 1, -1, -1):
                a = RHO[i] * dot(S[i], q)
                alpha[i] = a
                yi = Y[i]
                for j in range(dim):
                    q[j] -= a * yi[j]
            # scaling by gamma = (s^T y)/(y^T y)
            if S:
                sy = dot(S[-1], Y[-1])
                yy = dot(Y[-1], Y[-1])
                gamma = sy / yy if yy > 1e-30 else 1.0
            else:
                gamma = 1.0
            for j in range(dim):
                q[j] *= gamma
            for i in range(len(S)):
                b = RHO[i] * dot(Y[i], q)
                si = S[i]
                coeff = alpha[i] - b
                for j in range(dim):
                    q[j] += coeff * si[j]
            return q

        no_improve = 0
        while time.time() < t_end:
            # compute gradient
            g = fd_grad(x, fx)
            if time.time() >= t_end:
                break

            gnorm = norm2(g)
            if not (gnorm == gnorm) or gnorm > 1e100:  # NaN/inf guard
                break

            # if gradient tiny, try small trust-region random search to still improve
            if gnorm < 1e-12:
                improved = False
                for _ in range(8):
                    if time.time() >= t_end:
                        break
                    v = [random.gauss(0.0, 1.0) for _ in range(dim)]
                    vn = norm2(v) + 1e-18
                    v = [vi / vn for vi in v]
                    step = (0.35 + 0.65 * random.random()) * trust
                    xt = [x[i] - step * v[i] for i in range(dim)]
                    ft = eval_vec(xt)
                    if ft < fx:
                        x, fx = reflect_into_bounds(xt), ft
                        improved = True
                        break
                if improved:
                    no_improve = 0
                    trust = min(trust_max, trust * 1.2)
                    continue
                else:
                    trust = max(trust_min, trust * 0.6)
                    no_improve += 1
                    if no_improve >= 3:
                        break
                    continue

            # L-BFGS direction: p = -H*g
            Hg = two_loop(g)
            p = [-Hg[i] for i in range(dim)]
            # ensure descent
            if dot(p, g) >= 0.0:
                # fallback to steepest descent
                p = [-gi for gi in g]

            # line search (Armijo)
            f0_ls = fx
            gtp = dot(g, p)
            if gtp >= 0.0:
                gtp = -abs(gtp)

            # step length based on trust and gradient scale
            alpha0 = 1.0
            # scale alpha0 so typical move ~ trust
            pn = norm2(p) + 1e-18
            alpha0 = min(alpha0, trust / pn)

            c1 = 1e-4
            alpha = alpha0
            best_local_fx = fx
            best_local_x = x[:]
            accepted = False

            for _ in range(18):
                if time.time() >= t_end:
                    break
                xt = [x[i] + alpha * p[i] for i in range(dim)]
                ft = eval_vec(xt)
                if ft < best_local_fx:
                    best_local_fx = ft
                    best_local_x = reflect_into_bounds(xt)
                if ft <= f0_ls + c1 * alpha * gtp:
                    accepted = True
                    break
                alpha *= 0.5
                if alpha < 1e-16:
                    break

            if accepted:
                x_new = reflect_into_bounds([x[i] + alpha * p[i] for i in range(dim)])
                fx_new = eval_vec(x_new)

                # update memory
                s = [x_new[i] - x[i] for i in range(dim)]
                y = [0.0] * dim
                g_new = fd_grad(x_new, fx_new)
                for i in range(dim):
                    y[i] = g_new[i] - g[i]
                ys = dot(y, s)
                if ys > 1e-30 and all(v == v for v in y):  # positive curvature + non-NaN
                    rho = 1.0 / ys
                    S.append(s)
                    Y.append(y)
                    RHO.append(rho)
                    if len(S) > m:
                        S.pop(0); Y.pop(0); RHO.pop(0)

                if fx_new < fx:
                    x, fx = x_new, fx_new
                    trust = min(trust_max, trust * 1.15)
                    no_improve = 0
                else:
                    # if line search accepted but no improvement, keep best seen in LS window
                    x, fx = best_local_x, best_local_fx
                    trust = max(trust_min, trust * 0.85)
                    no_improve += 1
            else:
                # fallback: try best point from failed line search, else trust-region random
                if best_local_fx < fx:
                    x, fx = best_local_x, best_local_fx
                    trust = min(trust_max, trust * 1.05)
                    no_improve = 0
                else:
                    # random trust region probe
                    improved = False
                    for _ in range(10):
                        if time.time() >= t_end:
                            break
                        v = [random.gauss(0.0, 1.0) for _ in range(dim)]
                        vn = norm2(v) + 1e-18
                        v = [vi / vn for vi in v]
                        step = (0.25 + 0.75 * random.random()) * trust
                        xt = [x[i] - step * v[i] for i in range(dim)]
                        ft = eval_vec(xt)
                        if ft < fx:
                            x, fx = reflect_into_bounds(xt), ft
                            improved = True
                            break
                    trust = min(trust_max, trust * 1.05) if improved else max(trust_min, trust * 0.65)
                    no_improve = 0 if improved else (no_improve + 1)

            if no_improve >= 4:
                break

        return x, fx

    # -------------------- Global: SHADE-ish DE --------------------
    pop_size = max(24, min(160, 12 * dim))
    H = 10
    MF = [0.6] * H
    MCR = [0.85] * H
    hist_idx = 0
    c_mem = 0.12

    archive = []
    arch_max = pop_size

    best = float("inf")
    best_x = None
    elites = []

    last_improve = time.time()
    min_progress = 1e-12

    T = float(max_time)
    stall_seconds = max(0.10 * T, 0.40)
    endgame_reserve = 0.34 * T  # more time for L-BFGS endgame

    def update_best(x, fx):
        nonlocal best, best_x, last_improve
        if fx + min_progress < best:
            best = fx
            best_x = x[:]
            last_improve = time.time()
        elite_insert(elites, x, fx, kmax=10)

    def sample_F(muF):
        # Cauchy-like around muF
        F = -1.0
        for _ in range(14):
            u = random.random()
            F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                break
        if F <= 0.0:
            F = 0.5
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(muCR):
        CR = random.gauss(muCR, 0.1)
        if CR < 0.0:
            CR = 0.0
        elif CR > 1.0:
            CR = 1.0
        return CR

    shifts = [random.randrange(1, 6000) for _ in range(dim)]
    halton_k = random.randrange(20, 140)

    # -------------------- main loop --------------------
    while time.time() < deadline:
        # init population: mix best-centered, elites-centered, halton
        pop = []

        def add_centered(center, count, scale):
            for _ in range(count):
                x = center[:]
                for d in range(dim):
                    if spans[d] > 0.0:
                        x[d] += random.gauss(0.0, scale * spans[d])
                pop.append(reflect_into_bounds(x))

        if best_x is not None:
            add_centered(best_x, min(pop_size // 4, 26), 0.20)

        if elites:
            ecount = min(len(elites), 4)
            for ei in range(ecount):
                add_centered(elites[ei][1], 3, 0.14)

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

        # DE generations
        while time.time() < deadline:
            now = time.time()
            remaining = deadline - now

            # endgame: run multiple short L-BFGS starts from elites (robust)
            if best_x is not None and remaining <= endgame_reserve:
                # allocate remaining among a few starts
                starts = []
                if elites:
                    starts.extend([ex for _, ex in elites[:min(6, len(elites))]])
                else:
                    starts.append(best_x[:])

                # ensure some diversification
                if len(starts) < 3:
                    for _ in range(3 - len(starts)):
                        starts.append(rand_vec())

                slice_each = max(0.0, remaining / float(len(starts)))
                for sx in starts:
                    if time.time() >= deadline:
                        break
                    sf = eval_vec(sx)
                    bx, bf = local_lbfgs(sx, sf, slice_each)
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

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]
                r = random.randrange(H)
                F = sample_F(MF[r])
                CR = sample_CR(MCR[r])

                pb = pop[random.choice(pbest_idx)]
                xbest = pop[best_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)
                xr1 = pop[r1]

                use_arch = (archive and random.random() < 0.55)
                if use_arch:
                    pool = pop + archive
                    xr2 = pool[random.randrange(len(pool))]
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    xr2 = pop[r2]

                # mutation mix (slightly more exploit late)
                if frac_left > 0.55:
                    prand = 0.20
                    pbestguide = 0.72
                elif frac_left > 0.22:
                    prand = 0.12
                    pbestguide = 0.80
                else:
                    prand = 0.06
                    pbestguide = 0.74

                u = random.random()
                if u < pbestguide:
                    v = [xi[d] + F * (pb[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
                elif u < pbestguide + prand:
                    a = random.randrange(pop_size)
                    b = random.randrange(pop_size)
                    cidx = random.randrange(pop_size)
                    xa, xb, xc = pop[a], pop[b], pop[cidx]
                    v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
                else:
                    v = [xi[d] + F * (xbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

                if converged and random.random() < 0.35:
                    j = random.randrange(dim)
                    if spans[j] > 0.0:
                        v[j] += random.gauss(0.0, 0.03 * spans[j])

                jrand = random.randrange(dim)
                uvec = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

                uvec = reflect_into_bounds(uvec)
                fu = float(func(uvec))

                if fu <= fi:
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = uvec
                    new_fit[i] = fu

                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    dF.append(df if df > 0.0 else 1e-16)
                    update_best(uvec, fu)

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
                F_new = num / den if den > 0 else (sum(S_F) / len(S_F))

                CR_new = 0.0
                for wgt, cr in zip(weights, S_CR):
                    CR_new += wgt * cr

                MF[hist_idx] = (1.0 - c_mem) * MF[hist_idx] + c_mem * F_new
                MCR[hist_idx] = (1.0 - c_mem) * MCR[hist_idx] + c_mem * CR_new
                hist_idx = (hist_idx + 1) % H

            remaining = deadline - time.time()

            # mid-game: occasional short L-BFGS from current best / elite when converged
            if best_x is not None and remaining > endgame_reserve + 0.05:
                if converged and random.random() < 0.22:
                    startx = best_x[:]
                    # sometimes start from another elite to escape local traps
                    if elites and random.random() < 0.35:
                        startx = elites[random.randrange(min(5, len(elites)))][1][:]
                    sf = eval_vec(startx)
                    bx, bf = local_lbfgs(startx, sf, min(0.10, 0.10 * remaining))
                    update_best(bx, bf)

            # diversity injection if stuck
            if converged and (time.time() - last_improve) > 0.55 * stall_seconds and remaining > endgame_reserve + 0.08:
                wcount = max(1, pop_size // 10)
                worst = idx_sorted[-wcount:]
                for t, wi in enumerate(worst):
                    if time.time() >= deadline:
                        return best
                    if best_x is not None and t < (wcount // 2):
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

            # stall: do a heavier local attempt then restart population
            if (time.time() - last_improve) > stall_seconds:
                remaining = deadline - time.time()
                if best_x is not None and remaining > endgame_reserve + 0.10:
                    # run a couple of short local optimizations from diversified elites
                    seeds = []
                    if elites:
                        seeds = [ex for _, ex in elites[:min(3, len(elites))]]
                    else:
                        seeds = [best_x[:]]
                    slice_each = min(0.18, 0.20 * remaining / float(len(seeds)))
                    for sx in seeds:
                        if time.time() >= deadline:
                            break
                        sf = eval_vec(sx)
                        bx, bf = local_lbfgs(sx, sf, slice_each)
                        update_best(bx, bf)
                break

    return best
