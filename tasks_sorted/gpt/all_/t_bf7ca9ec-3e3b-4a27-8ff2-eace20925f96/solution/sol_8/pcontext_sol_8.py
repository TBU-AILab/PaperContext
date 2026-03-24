import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (standard library only).

    Improved hybrid (aimed at better results under tight budgets):
      - Global phase: L-SHADE-style DE/current-to-pbest/1 with
          * linear population size reduction
          * success-history memories (F, CR)
          * archive diversity
      - Mid/late phase: CMA-ES-like diagonal sampling around best (fast local exploitation)
      - Always-on polish: occasional 1D quadratic step along a random coordinate
      - Stagnation: partial refresh of worst + temporary radius bump

    Returns: best fitness found (float).
    """

    t0 = time.time()

    def time_up():
        return (time.time() - t0) >= max_time

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def eval_f(x):
        return float(func(x))

    # reflection mapping into [lo, hi]
    def reflect(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            w = hi - lo
            v = y[i] - lo
            m = v % (2.0 * w)
            y[i] = lo + (m if m <= w else (2.0 * w - m))
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return reflect([lows[i] + highs[i] - x[i] for i in range(dim)])

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # --- Population schedule ---
    NP_init = max(28, min(140, 18 + 7 * dim))
    NP_min = max(10, min(40, 7 + 2 * dim))

    # --- Initialize population (with mild seeding around center) ---
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop, fit = [], []
    best = float("inf")
    best_x = None

    # create some points near center + some uniform
    n_center = max(0, min(NP_init // 3, 10 + dim))
    for k in range(NP_init):
        if time_up():
            return best
        if k < n_center:
            x = center[:]
            for i in range(dim):
                x[i] += random.gauss(0.0, 0.20 * spans[i])
            x = reflect(x)
        else:
            x = rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # opposition injection (small)
    for _ in range(max(2, NP_init // 8)):
        if time_up():
            return best
        i = random.randrange(len(pop))
        xo = opposite_point(pop[i])
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
            if fo < best:
                best, best_x = fo, xo[:]

    # --- SHADE memories ---
    H = 14
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    # Archive
    archive = []
    arch_max = len(pop)

    # Stagnation control
    last_improve_t = time.time()
    patience = max(0.10 * max_time, 0.9)

    # --- Local sampling (diagonal "CMA-like") state ---
    # per-dimension sigma in absolute units
    sig = [0.18 * spans[i] for i in range(dim)]
    sig_min = [1e-12 * spans[i] + 1e-15 for i in range(dim)]
    sig_max = [0.50 * spans[i] for i in range(dim)]
    # success stats to adapt local sigmas
    loc_succ = [0] * dim
    loc_try = [0] * dim
    loc_window = 18  # update frequency

    next_local = time.time() + max(0.18, 0.10 * max_time)
    local_batch_base = max(10, 3 * dim)

    # --- cheap 1D quadratic step around best on one coordinate ---
    def quad_1d_step(x, fx, j, step):
        if step <= 0.0:
            return x, fx, False
        lo, hi = lows[j], highs[j]
        if lo == hi:
            return x, fx, False

        x0 = x[:]
        xm = x0[:]
        xp = x0[:]
        xm[j] -= step
        xp[j] += step
        xm = reflect(xm)
        xp = reflect(xp)

        fm = eval_f(xm)
        if time_up():
            return x, fx, False
        fp = eval_f(xp)
        if time_up():
            return x, fx, False

        denom = (fm - 2.0 * fx + fp)
        if abs(denom) < 1e-18:
            if fm < fx and fm <= fp:
                return xm, fm, True
            if fp < fx and fp <= fm:
                return xp, fp, True
            return x, fx, False

        tstar = 0.5 * (fm - fp) / denom
        if tstar < -1.5:
            tstar = -1.5
        elif tstar > 1.5:
            tstar = 1.5

        xq = x0[:]
        xq[j] += tstar * step
        xq = reflect(xq)
        fq = eval_f(xq)
        if time_up():
            return x, fx, False

        bx, bf = x, fx
        improved = False
        for cand, fc in ((xm, fm), (xp, fp), (xq, fq)):
            if fc < bf:
                bx, bf = cand, fc
                improved = True
        return bx, bf, improved

    def adapt_local_sigmas():
        # simple per-dim 1/5-ish adaptation using windowed success rates
        for j in range(dim):
            if loc_try[j] >= loc_window:
                rate = loc_succ[j] / float(loc_try[j]) if loc_try[j] else 0.0
                if rate > 0.22:
                    sig[j] = min(sig_max[j], sig[j] * 1.15)
                elif rate < 0.12:
                    sig[j] = max(sig_min[j], sig[j] * 0.78)
                else:
                    # mild drift
                    sig[j] = min(sig_max[j], max(sig_min[j], sig[j] * 0.98))
                loc_succ[j] = 0
                loc_try[j] = 0

    def local_phase():
        nonlocal best, best_x, last_improve_t
        if best_x is None:
            return

        elapsed = time.time() - t0
        frac = elapsed / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))

        # allocate more near the end
        batch = int(local_batch_base * (1.0 + 1.6 * frac))
        if batch < 8:
            batch = 8

        for _ in range(batch):
            if time_up():
                break

            r = random.random()
            if r < (0.35 + 0.25 * frac):
                # 1D quadratic polish
                j = random.randrange(dim)
                step = max(1e-15, 0.9 * sig[j])
                bx, bf, imp = quad_1d_step(best_x, best, j, step)
                if imp and bf < best:
                    best, best_x = bf, bx[:]
                    last_improve_t = time.time()
                    loc_succ[j] += 1
                loc_try[j] += 1
            else:
                # diagonal sampling around best (gaussian, occasional heavy-tailed)
                y = best_x[:]
                if random.random() < 0.85:
                    for j in range(dim):
                        if sig[j] > 0.0:
                            y[j] += random.gauss(0.0, sig[j])
                else:
                    j = random.randrange(dim)
                    y[j] += sig[j] * math.tan(math.pi * (random.random() - 0.5))
                y = reflect(y)

                fy = eval_f(y)
                if fy < best:
                    # measure which dims changed most to assign credit
                    for j in range(dim):
                        if abs(y[j] - best_x[j]) > 0.5 * sig[j]:
                            loc_succ[j] += 1
                        loc_try[j] += 1
                    best, best_x = fy, y[:]
                    last_improve_t = time.time()
                else:
                    # still count tries lightly
                    j = random.randrange(dim)
                    loc_try[j] += 1

            adapt_local_sigmas()

    def top_p_indices(pop_fit):
        order = sorted(range(len(pop_fit)), key=lambda i: pop_fit[i])
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        # more exploitative later
        p = max(0.06, min(0.30, 0.24 - 0.18 * frac))
        pnum = int(math.ceil(p * len(pop_fit)))
        if pnum < 2:
            pnum = 2
        return order, pnum

    # --- Main loop ---
    while not time_up():
        # schedule local phase (more frequent later)
        if time.time() >= next_local and not time_up():
            local_phase()
            rem = max(0.0, max_time - (time.time() - t0))
            next_local = time.time() + max(0.06, 0.035 * rem)

        # stagnation -> partial refresh of worst + sigma bump
        if (time.time() - last_improve_t) >= patience and not time_up():
            n = len(pop)
            k = max(2, n // 3)
            worst = sorted(range(n), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if time_up():
                    break
                if random.random() < 0.40:
                    x = rand_point()
                else:
                    x = best_x[:]
                    for j in range(dim):
                        x[j] += 0.22 * spans[j] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx], fit[idx] = x, fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()
            # trim archive and expand local radius a bit to escape
            if archive:
                archive = archive[len(archive) // 2 :]
            for j in range(dim):
                sig[j] = min(sig_max[j], max(sig[j], 0.16 * spans[j]))
            last_improve_t = time.time()

        # linear pop reduction by time fraction
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        target_NP = int(round(NP_init - (NP_init - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])  # best->worst
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            arch_max = len(pop)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # DE generation
        order, pnum = top_p_indices(fit)

        S_F, S_CR, S_w = [], [], []
        NP = len(pop)

        for i in range(NP):
            if time_up():
                break

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CR = random.gauss(muCR, 0.09)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = cauchy(muF, 0.10)
            tries = 0
            while F <= 0.0 and tries < 10:
                F = cauchy(muF, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.12
            if F > 1.0:
                F = 1.0

            pbest_idx = order[random.randrange(pnum)]
            xpbest = pop[pbest_idx]

            r1 = i
            while r1 == i or r1 == pbest_idx:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            use_arch = (len(archive) > 0 and random.random() < (len(archive) / float(len(archive) + NP)))
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == pbest_idx or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]

            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            u = reflect(u)
            fu = eval_f(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > arch_max:
                    j = random.randrange(len(archive))
                    archive[j] = archive[-1]
                    archive.pop()

                pop[i], fit[i] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = time.time()

                w = fi - fu
                if w <= 0.0:
                    w = 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

        # Update SHADE memories
        if S_F:
            wsum = sum(S_w)
            if wsum <= 0.0:
                wsum = float(len(S_w))

            meanCR = 0.0
            for cr, w in zip(S_CR, S_w):
                meanCR += cr * (w / wsum)

            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_w):
                wf = w / wsum
                num += wf * f * f
                den += wf * f
            meanF = (num / den) if den > 1e-18 else M_F[k_mem]

            M_CR[k_mem] = meanCR
            M_F[k_mem] = meanF
            k_mem = (k_mem + 1) % H

    return best
