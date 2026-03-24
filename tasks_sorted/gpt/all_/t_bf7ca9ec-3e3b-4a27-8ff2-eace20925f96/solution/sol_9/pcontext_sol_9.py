import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (standard library only).

    Improved vs previous DE-only hybrids by adding:
      - Two-track search that shares one global best:
          (A) L-SHADE style DE/current-to-pbest/1 + archive + linear pop reduction
          (B) Powerful local optimizer: Powell-like pattern/coordinate search with
              adaptive step per-dimension + occasional quadratic 1D refinement
      - Better seeding: center + Latin-ish stratified samples + opposition
      - Budget-aware scheduling: local search becomes dominant near the end
      - Stagnation response: radius bump + partial refresh of worst + archive trim

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

    # Reflect mapping into [lo, hi] (robust for far-out values)
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
        return loc + scale * math.tan(math.pi * (random.random() - 0.5))

    # ----------- Initialization (better coverage than pure uniform) -----------
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    NP_init = max(28, min(150, 18 + 7 * dim))
    NP_min = max(10, min(40, 7 + 2 * dim))

    pop, fit = [], []
    best = float("inf")
    best_x = None

    # Stratified-ish: for each dimension, cycle through bins, shuffle per-dim
    bins = max(4, int(round(math.sqrt(NP_init))))
    per_dim_perm = [list(range(bins)) for _ in range(dim)]
    for d in range(dim):
        random.shuffle(per_dim_perm[d])

    for k in range(NP_init):
        if time_up():
            return best

        # Mix: some around center, some stratified, some uniform
        r = random.random()
        if r < 0.25:
            x = center[:]
            for i in range(dim):
                x[i] += random.gauss(0.0, 0.18 * spans[i])
            x = reflect(x)
        elif r < 0.75:
            x = [0.0] * dim
            for i in range(dim):
                b = per_dim_perm[i][k % bins]
                # jitter within bin
                u = (b + random.random()) / float(bins)
                x[i] = lows[i] + u * spans[i]
        else:
            x = rand_point()

        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # Opposition injection (small, early)
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

    # ----------- L-SHADE memories + archive -----------
    H = 16
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    archive = []
    arch_max = len(pop)

    # ----------- Local search state (Powell/pattern-like) -----------
    # Per-dimension step sizes (absolute)
    step = [0.12 * spans[i] for i in range(dim)]
    step_min = [1e-12 * spans[i] + 1e-15 for i in range(dim)]
    step_max = [0.50 * spans[i] for i in range(dim)]
    # Track local progress for adaptive rules
    loc_success = 0
    loc_trials = 0
    loc_window = 24

    # Stagnation management
    last_improve_t = time.time()
    patience = max(0.10 * max_time, 0.9)

    # Scheduling: local search frequency ramps up
    next_local = time.time() + max(0.18, 0.10 * max_time)

    def adapt_steps(success_event):
        nonlocal loc_success, loc_trials
        loc_trials += 1
        if success_event:
            loc_success += 1
        if loc_trials >= loc_window:
            rate = loc_success / float(loc_trials) if loc_trials else 0.0
            # more aggressive than 1/5, but bounded
            if rate > 0.22:
                for j in range(dim):
                    step[j] = min(step_max[j], step[j] * 1.12)
            elif rate < 0.10:
                for j in range(dim):
                    step[j] = max(step_min[j], step[j] * 0.72)
            else:
                for j in range(dim):
                    step[j] = min(step_max[j], max(step_min[j], step[j] * 0.97))
            loc_success = 0
            loc_trials = 0

    def quad_1d_refine(x, fx, j, s):
        # 3-point quadratic fit at x-s, x, x+s, take bounded minimizer
        if s <= 0.0 or spans[j] <= 0.0:
            return x, fx, False

        xm = x[:]
        xp = x[:]
        xm[j] -= s
        xp[j] += s
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
            # just pick best among sampled
            if fm < fx and fm <= fp:
                return xm, fm, True
            if fp < fx and fp <= fm:
                return xp, fp, True
            return x, fx, False

        tstar = 0.5 * (fm - fp) / denom  # in step units
        if tstar < -1.5:
            tstar = -1.5
        elif tstar > 1.5:
            tstar = 1.5

        xq = x[:]
        xq[j] += tstar * s
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

    def local_phase():
        nonlocal best, best_x, last_improve_t

        if best_x is None:
            return

        elapsed = time.time() - t0
        frac = elapsed / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))

        # Budget: local becomes stronger near end
        batch = int(max(12, (2.5 + 6.0 * frac) * dim))

        x = best_x[:]
        fx = best

        # Randomized coordinate order per call
        coords = list(range(dim))
        random.shuffle(coords)

        for _ in range(batch):
            if time_up():
                break

            improved_any = False

            # Pattern/coordinate search sweep
            for j in coords:
                if time_up():
                    break
                s = step[j]
                if s < step_min[j]:
                    continue

                # try +s and -s
                xp = x[:]
                xp[j] += s
                xp = reflect(xp)
                fp = eval_f(xp)
                if time_up():
                    break

                xm = x[:]
                xm[j] -= s
                xm = reflect(xm)
                fm = eval_f(xm)
                if time_up():
                    break

                if fp < fx or fm < fx:
                    if fp <= fm:
                        x, fx = xp, fp
                    else:
                        x, fx = xm, fm
                    improved_any = True

                    if fx < best:
                        best, best_x = fx, x[:]
                        last_improve_t = time.time()
                else:
                    # occasional quadratic refine even on failure (cheap and stabilizing)
                    if random.random() < (0.10 + 0.25 * frac):
                        bx, bf, imp = quad_1d_refine(x, fx, j, 0.8 * s)
                        if imp and bf < fx:
                            x, fx = bx[:], bf
                            improved_any = True
                            if fx < best:
                                best, best_x = fx, x[:]
                                last_improve_t = time.time()

            # TR-ish random move around current x (helps avoid axis-only limitations)
            if not time_up() and random.random() < (0.35 + 0.25 * frac):
                y = x[:]
                if random.random() < 0.85:
                    for j in range(dim):
                        y[j] += random.gauss(0.0, 0.65 * step[j])
                else:
                    j = random.randrange(dim)
                    y[j] += (0.65 * step[j]) * math.tan(math.pi * (random.random() - 0.5))
                y = reflect(y)
                fy = eval_f(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True
                    if fx < best:
                        best, best_x = fx, x[:]
                        last_improve_t = time.time()

            adapt_steps(improved_any)

            # If no improvement, mildly shrink a few steps to focus
            if not improved_any:
                j = random.randrange(dim)
                step[j] = max(step_min[j], step[j] * 0.85)

    def top_p_indices(pop_fit):
        order = sorted(range(len(pop_fit)), key=lambda i: pop_fit[i])
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        # stronger selection later (smaller p)
        p = max(0.05, min(0.30, 0.24 - 0.19 * frac))
        pnum = int(math.ceil(p * len(pop_fit)))
        if pnum < 2:
            pnum = 2
        return order, pnum

    # ----------- Main loop -----------
    while not time_up():
        # schedule local phase (more frequent near end)
        if time.time() >= next_local and not time_up():
            local_phase()
            rem = max(0.0, max_time - (time.time() - t0))
            next_local = time.time() + max(0.05, 0.030 * rem)

        # stagnation: refresh and bump steps
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
                        x[j] += 0.18 * spans[j] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx], fit[idx] = x, fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()
            # trim archive and expand local steps
            if archive:
                archive = archive[len(archive) // 2 :]
            for j in range(dim):
                step[j] = min(step_max[j], max(step[j], 0.10 * spans[j]))
            last_improve_t = time.time()

        # linear population reduction with time
        frac = (time.time() - t0) / float(max_time) if max_time > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        target_NP = int(round(NP_init - (NP_init - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        if len(pop) > target_NP:
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = set(order[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            arch_max = len(pop)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # DE generation (L-SHADE current-to-pbest/1)
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

        # Update memories (SHADE)
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
