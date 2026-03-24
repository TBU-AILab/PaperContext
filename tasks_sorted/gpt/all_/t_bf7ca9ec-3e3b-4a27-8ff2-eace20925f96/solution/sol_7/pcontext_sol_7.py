import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (standard library only).

    Hybrid strategy (robust under unknown landscapes, tight budgets):
      1) L-SHADE-like DE/current-to-pbest/1 with:
         - linear population size reduction
         - SHADE success-history memories for F and CR
         - external archive for diversity
         - p-best selection pressure scheduled over time
      2) End-game intensification:
         - budget-aware coordinate/pattern + quadratic 1D fit (3-point) around best
         - adaptive trust-region radius, shrinks on failures, expands on successes
      3) Stagnation handling:
         - soft restart of worst individuals with mix of random + heavy-tailed around best
         - archive refresh

    Returns
    -------
    best : float
        Best (minimum) fitness found within the time budget.
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

    # Reflection mapping into [lo, hi]
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

    # --- L-SHADE population schedule ---
    NP_init = max(24, min(120, 16 + 6 * dim))
    NP_min = max(8, min(32, 6 + 2 * dim))

    # --- Initialize population ---
    pop, fit = [], []
    best = float("inf")
    best_x = None

    for _ in range(NP_init):
        x = rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        if time_up():
            return best

    # Opposition injection (early diversity boost)
    for _ in range(max(2, NP_init // 7)):
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
    H = 12
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    # Archive
    archive = []
    arch_max = len(pop)

    # Stagnation control
    last_improve_t = time.time()
    patience = max(0.10 * max_time, 0.8)

    # Intensification / trust-region
    tr_sigma = 0.12          # fraction of span
    tr_sigma_min = 1e-7
    tr_sigma_max = 0.30
    next_intense = time.time() + max(0.20, 0.10 * max_time)

    def pbest_fraction(frac_time):
        # Start explorative, end more exploitative
        # (higher p => more exploration; lower p => stronger selection)
        # Keep in [0.08, 0.30]
        return max(0.08, min(0.30, 0.26 - 0.18 * frac_time))

    def top_p_indices():
        order = sorted(range(len(pop)), key=lambda i: fit[i])
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        p = pbest_fraction(max(0.0, min(1.0, frac)))
        pnum = int(math.ceil(p * len(pop)))
        if pnum < 2:
            pnum = 2
        return order, pnum

    # --- 1D quadratic step along a coordinate (cheap polish) ---
    def quad_1d_step(x, fx, j, step):
        # sample at x-step, x, x+step and fit parabola; take minimizer if stable
        if step <= 0.0:
            return x, fx, False

        x0 = x[:]
        lo, hi = lows[j], highs[j]
        if lo == hi:
            return x, fx, False

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

        # Fit parabola through (-1,fm), (0,fx), (+1,fp)
        # Minimizer at t* = (fm - fp) / (2*(fm - 2f0 + fp))
        denom = (fm - 2.0 * fx + fp)
        if abs(denom) < 1e-18:
            # fallback: best among three
            if fm <= fx and fm <= fp:
                return xm, fm, True
            if fp <= fx and fp <= fm:
                return xp, fp, True
            return x, fx, False

        tstar = 0.5 * (fm - fp) / denom  # in "step units"
        # Only trust if within a reasonable bracket
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

        # pick best
        bx, bf = x, fx
        improved = False
        for cand, fc in ((xm, fm), (xp, fp), (xq, fq)):
            if fc < bf:
                bx, bf = cand, fc
                improved = True
        return bx, bf, improved

    def intensify():
        nonlocal best, best_x, last_improve_t, tr_sigma
        if best_x is None:
            return

        # Allocate more effort later
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        batch = max(10, int((3 + 5 * frac) * dim))

        improved_any = False

        # Mix: TR random + coordinate quadratic polish
        for _ in range(batch):
            if time_up():
                break

            if random.random() < (0.55 + 0.25 * frac):
                # coordinate quadratic step
                j = random.randrange(dim)
                step = max(1e-15, tr_sigma * spans[j])
                bx, bf, imp = quad_1d_step(best_x, best, j, step)
                if imp and bf < best:
                    best, best_x = bf, bx[:]
                    last_improve_t = time.time()
                    improved_any = True
            else:
                # TR gaussian/heavy-tail proposal
                y = best_x[:]
                if random.random() < 0.75:
                    for d in range(dim):
                        y[d] += random.gauss(0.0, tr_sigma * spans[d])
                else:
                    j = random.randrange(dim)
                    y[j] += (tr_sigma * spans[j]) * math.tan(math.pi * (random.random() - 0.5))
                y = reflect(y)
                fy = eval_f(y)
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve_t = time.time()
                    improved_any = True

        # Adapt TR radius
        if improved_any:
            tr_sigma = min(tr_sigma_max, tr_sigma * 1.10)
        else:
            tr_sigma = max(tr_sigma_min, tr_sigma * 0.75)

    # --- Main loop ---
    while not time_up():
        # Intensification scheduling: more frequent near end
        elapsed = time.time() - t0
        rem = max(0.0, max_time - elapsed)
        frac = elapsed / float(max_time) if max_time > 0 else 1.0
        if time.time() >= next_intense and not time_up():
            intensify()
            # more frequent later
            next_intense = time.time() + max(0.08, 0.04 * rem)

        # Stagnation -> soft restart of worst + archive refresh + TR reset a bit
        if (time.time() - last_improve_t) >= patience and not time_up():
            n = len(pop)
            k = max(2, n // 3)
            worst = sorted(range(n), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if time_up():
                    break
                if random.random() < 0.45:
                    x = rand_point()
                else:
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += 0.14 * spans[d] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx], fit[idx] = x, fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()
            if len(archive) > 0:
                archive = archive[len(archive) // 2 :]
            tr_sigma = min(tr_sigma_max, max(tr_sigma, 0.12))
            last_improve_t = time.time()

        # Linear population size reduction by time fraction
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

        # Prepare p-best indices
        order, pnum = top_p_indices()

        # SHADE success lists
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

            CR = random.gauss(muCR, 0.1)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = cauchy(muF, 0.1)
            tries = 0
            while F <= 0.0 and tries < 10:
                F = cauchy(muF, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.1
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

            # current-to-pbest/1
            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            # binomial crossover
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
