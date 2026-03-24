import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (stdlib-only) with stronger exploitation.

    Changes vs previous version (high impact):
      - Adds explicit trust-region local search with *quadratic interpolation*
        along coordinates (uses 3-point parabolic fit), which is often much
        more effective than simple pattern search.
      - Replaces the weak kNN surrogate with a *rank-based local model*:
        propose candidates by sampling around several elite points with an
        adaptive radius + occasional coordinate-wise 1D parabola steps.
      - Keeps a robust DE (current-to-pbest/1 with archive, SHADE memories)
        but biases mutation to use elite anchors (more intensification late).
      - Better stagnation handling: quick trust-region shrink/expand + restart.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ---------- helpers ----------
    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def repair_reflect(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                y[i] = lo
                continue
            v = y[i]
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            w = hi - lo
            if w > 0 and (v < lo or v > hi):
                v = lo + (v - lo) % w
            y[i] = clamp(v, lo, hi)
        return y

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            sp = spans[i]
            x[i] = lows[i] if sp <= 0.0 else (lows[i] + random.random() * sp)
        return x

    def center_vec():
        return [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    if time.time() >= deadline:
        return evaluate(center_vec())

    # ---------- estimate eval time ----------
    probe_n = 4
    probe_best = float("inf")
    st = time.time()
    for _ in range(probe_n):
        if time.time() >= deadline:
            return probe_best
        f = evaluate(rand_vec())
        if f < probe_best:
            probe_best = f
    eval_time = max(1e-6, (time.time() - st) / float(probe_n))

    def remaining_evals():
        rem = deadline - time.time()
        if rem <= 0:
            return 0
        return max(0, int(rem / max(eval_time, 1e-12)))

    # ---------- history (bounded) ----------
    HIST_MAX = 1200
    hist_x, hist_f = [], []

    def hist_add(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        n = len(hist_x)
        if n > HIST_MAX:
            # keep best 40% + newest 40% (cheap thinning)
            idx = list(range(n))
            idx.sort(key=lambda i: hist_f[i])
            keep_best = idx[: max(80, int(0.4 * HIST_MAX))]
            keep_new = list(range(max(0, n - int(0.4 * HIST_MAX)), n))
            keep = sorted(set(keep_best + keep_new))
            hist_x[:] = [hist_x[i] for i in keep]
            hist_f[:] = [hist_f[i] for i in keep]

    # ---------- init population (LHS-ish) ----------
    rem0 = max(60, remaining_evals())
    NP0 = max(24, 10 + 6 * dim)
    NPmin = max(12, 6 + 2 * dim)
    NP0 = min(NP0, max(NPmin, rem0 // 5 if rem0 > 0 else NP0))

    perms = []
    for d in range(dim):
        p = list(range(NP0))
        random.shuffle(p)
        perms.append(p)

    pop = []
    for i in range(NP0):
        x = [0.0] * dim
        for d in range(dim):
            sp = spans[d]
            if sp <= 0.0:
                x[d] = lows[d]
            else:
                u = (perms[d][i] + random.random()) / float(NP0)
                x[d] = lows[d] + sp * u
        pop.append(x)

    pop[0] = center_vec()
    for k in range(1, min(NP0, 1 + max(3, NP0 // 10))):
        pop[k] = rand_vec()

    fits = [float("inf")] * NP0
    best = float("inf")
    best_x = None

    for i in range(NP0):
        if time.time() >= deadline:
            return best if best < float("inf") else probe_best
        f = evaluate(pop[i])
        fits[i] = f
        hist_add(pop[i], f)
        if f < best:
            best = f
            best_x = pop[i][:]

    # ---------- DE memories (SHADE-like) ----------
    H = 12
    MCR = [0.8] * H
    MF = [0.55] * H
    k_mem = 0

    archive = []
    arch_max = NP0

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def weighted_lehmer(values, weights):
        num = 0.0
        den = 0.0
        for v, w in zip(values, weights):
            num += w * v * v
            den += w * v
        if den <= 1e-18:
            return None
        return num / den

    def weighted_mean(values, weights):
        s = 0.0
        wsum = 0.0
        for v, w in zip(values, weights):
            s += w * v
            wsum += w
        if wsum <= 1e-18:
            return None
        return s / wsum

    def pick_excluding(n, exclude):
        for _ in range(32):
            r = random.randrange(n)
            if r not in exclude:
                return r
        for r in range(n):
            if r not in exclude:
                return r
        return 0

    # ---------- 1D parabolic (quadratic) coordinate step ----------
    def parabola_minimizer(a, fa, b, fb, c, fc):
        # return argmin x of parabola through (a,fa),(b,fb),(c,fc); or None if degenerate
        # Using formula with denominator 2*( (b-a)(fb-fc) - (b-c)(fb-fa) )
        denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
        if abs(denom) <= 1e-18:
            return None
        num = (b - a) * (b - a) * (fb - fc) - (b - c) * (b - c) * (fb - fa)
        x = b - 0.5 * (num / denom)
        return x

    # ---------- trust-region local search around incumbent ----------
    def trust_region_search(x0, f0, budget, rad_frac):
        if budget <= 0 or time.time() >= deadline:
            return x0, f0, 0
        x = x0[:]
        fx = f0
        evals = 0

        # radius per dimension
        rad = [rad_frac * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        rad_min = [1e-12 + 1e-7 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        rad_max = [0.5 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]

        # prioritize wider dimensions
        dims = list(range(dim))
        dims.sort(key=lambda i: spans[i], reverse=True)

        while evals < budget and time.time() < deadline:
            improved = False

            if dim <= 20:
                sub = dims
            else:
                sub = dims[: max(10, dim // 3)]
                random.shuffle(sub)

            for d in sub:
                if evals >= budget or time.time() >= deadline:
                    break
                if spans[d] <= 0.0 or rad[d] <= rad_min[d]:
                    continue

                base = x[d]
                step = rad[d]

                # sample three points along coordinate: base-step, base, base+step
                a = clamp(base - step, lows[d], highs[d])
                c = clamp(base + step, lows[d], highs[d])
                b = base

                # if clamped collapses, skip
                if abs(c - a) <= 1e-15:
                    rad[d] *= 0.5
                    continue

                # Evaluate endpoints (avoid re-evaluating base)
                y = x[:]
                y[d] = a
                fa = evaluate(y); evals += 1; hist_add(y, fa)
                if fa < fx:
                    x, fx = y, fa
                    improved = True
                    continue

                y = x[:]
                y[d] = c
                fc = evaluate(y); evals += 1; hist_add(y, fc)
                if fc < fx:
                    x, fx = y, fc
                    improved = True
                    continue

                # Parabolic interpolation using (a,fa),(b,fx),(c,fc)
                xm = parabola_minimizer(a, fa, b, fx, c, fc)
                if xm is None:
                    # no curvature info; shrink a bit
                    rad[d] *= 0.7
                    continue
                xm = clamp(xm, lows[d], highs[d])
                if abs(xm - b) <= 1e-15:
                    rad[d] *= 0.7
                    continue

                y = x[:]
                y[d] = xm
                fm = evaluate(y); evals += 1; hist_add(y, fm)
                if fm < fx:
                    x, fx = y, fm
                    improved = True
                    # expand slightly on success
                    rad[d] = min(rad_max[d], rad[d] * 1.25)
                else:
                    rad[d] *= 0.6

            # global radius update
            if not improved:
                # if nothing improved in this sweep, shrink all a bit
                for d in sub:
                    rad[d] *= 0.85

            # termination if radii tiny
            tiny = 0
            for d in sub:
                if rad[d] <= rad_min[d]:
                    tiny += 1
            if tiny >= max(1, len(sub) - 1):
                break

        return x, fx, evals

    # ---------- elite-biased sampler (rank-based) ----------
    def elite_propose(ratio):
        n = len(hist_x)
        if n < max(40, 4 * dim):
            return rand_vec()

        # choose among top-K to avoid always picking the best
        idx = list(range(n))
        idx.sort(key=lambda i: hist_f[i])
        K = max(10, min(80, n // 6))
        base = hist_x[random.choice(idx[:K])][:]

        # radius shrinks with time
        rad = (0.30 * (1.0 - ratio) + 0.03)
        x = base[:]
        for d in range(dim):
            if spans[d] > 0.0:
                x[d] = clamp(x[d] + random.gauss(0.0, rad) * spans[d], lows[d], highs[d])

        # occasionally do a single coordinate parabolic nudge around this point
        if random.random() < (0.25 + 0.25 * ratio) and best_x is not None:
            d = random.randrange(dim)
            if spans[d] > 0.0:
                step = (0.10 * (1.0 - ratio) + 0.015) * spans[d]
                b = x[d]
                a = clamp(b - step, lows[d], highs[d])
                c = clamp(b + step, lows[d], highs[d])
                if abs(c - a) > 1e-15:
                    y = x[:]; y[d] = a
                    fa = evaluate(y); hist_add(y, fa)
                    y = x[:]; y[d] = c
                    fc = evaluate(y); hist_add(y, fc)
                    # approximate fb by evaluating current x (cheap if used rarely)
                    fb = evaluate(x); hist_add(x, fb)
                    xm = parabola_minimizer(a, fa, b, fb, c, fc)
                    if xm is not None:
                        x[d] = clamp(xm, lows[d], highs[d])
        return x

    # ---------- main loop ----------
    no_improve_gens = 0
    last_best = best
    gen = 0
    NP = NP0

    while time.time() < deadline and NP >= NPmin:
        gen += 1

        order = list(range(NP))
        order.sort(key=lambda i: fits[i])

        elapsed = time.time() - t0
        T = max(1e-9, deadline - t0)
        ratio = elapsed / T
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0

        # p-best schedule (slightly more exploitative late)
        p = 0.34 - 0.26 * ratio  # 0.34 -> 0.08
        pbest_n = max(2, int(math.ceil(p * NP)))
        top = order[:pbest_n]

        SF, SCR, dF = [], [], []
        new_pop = [None] * NP
        new_fits = [None] * NP
        arch_max = max(NP0, NP)

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # more frequent exploitation late: elite proposal stream
            if random.random() < (0.10 + 0.25 * ratio):
                ui = elite_propose(ratio)
                ui = repair_reflect(ui)
                fui = evaluate(ui)
                hist_add(ui, fui)
                if fui <= fi:
                    new_pop[i] = ui
                    new_fits[i] = fui
                    if fui < best:
                        best, best_x = fui, ui[:]
                else:
                    new_pop[i] = xi
                    new_fits[i] = fi
                continue

            r = random.randrange(H)
            CRm = MCR[r]
            Fm = MF[r]

            CR = CRm + random.gauss(0.0, 0.06)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = Fm
            for _ in range(10):
                Fc = cauchy(Fm, 0.06)
                if Fc > 0.0:
                    F = Fc
                    break
            if F < 0.04:
                F = 0.04
            elif F > 1.0:
                F = 1.0

            pbest = random.choice(top)

            excl = {i, pbest}
            r1 = pick_excluding(NP, excl)
            excl.add(r1)

            use_arch = (archive and random.random() < 0.70)
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_excluding(NP, excl)
                xr2 = pop[r2]

            xr1 = pop[r1]
            xp = pop[pbest]

            # mutation: current-to-pbest/1 (more often), with late elite bias
            if random.random() < (0.92 - 0.45 * ratio):
                vi = [xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
            else:
                # rand/1 from population (diversification)
                a = pick_excluding(NP, {i})
                b = pick_excluding(NP, {i, a})
                c = pick_excluding(NP, {i, a, b})
                xa, xb, xc = pop[a], pop[b], pop[c]
                vi = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # binomial crossover
            ui = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    ui[d] = vi[d]
            ui = repair_reflect(ui)

            fui = evaluate(ui)
            hist_add(ui, fui)

            if fui <= fi:
                new_pop[i] = ui
                new_fits[i] = fui

                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                imp = fi - fui
                if imp <= 0.0:
                    imp = 1e-12
                SF.append(F)
                SCR.append(CR)
                dF.append(imp)

                if fui < best:
                    best, best_x = fui, ui[:]
            else:
                new_pop[i] = xi
                new_fits[i] = fi

        pop, fits = new_pop, new_fits

        # update parameter memories
        if SF:
            wsum = sum(dF)
            weights = [w / wsum for w in dF] if wsum > 1e-18 else [1.0 / len(dF)] * len(dF)
            mcr = weighted_mean(SCR, weights)
            mf = weighted_lehmer(SF, weights)
            if mcr is not None:
                if mcr < 0.0:
                    mcr = 0.0
                elif mcr > 1.0:
                    mcr = 1.0
                MCR[k_mem] = mcr
            if mf is not None:
                if mf < 0.04:
                    mf = 0.04
                elif mf > 0.98:
                    mf = 0.98
                MF[k_mem] = mf
            k_mem = (k_mem + 1) % H

        # stagnation tracking
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # intensification: trust-region local search around best
        if best_x is not None and time.time() < deadline:
            rem = remaining_evals()
            if rem > 0 and (gen % 4 == 0 or no_improve_gens >= 6):
                budget = min(max(18, 3 * dim), max(18, rem // 10))
                rad_frac = (0.18 * (1.0 - ratio) + 0.02)
                x2, f2, _ = trust_region_search(best_x, best, budget, rad_frac)
                if f2 < best:
                    best, best_x = f2, x2[:]
                    # inject best into population by replacing worst
                    worst_i = max(range(NP), key=lambda j: fits[j])
                    pop[worst_i] = best_x[:]
                    fits[worst_i] = best
                    no_improve_gens = 0

        # restart if stuck: reinit worst half around best with moderate radius
        if best_x is not None and no_improve_gens >= 10 and time.time() < deadline:
            no_improve_gens = 0
            order_worst = list(range(NP))
            order_worst.sort(key=lambda i: fits[i], reverse=True)
            cnt = max(1, NP // 2)
            rad = (0.28 * (1.0 - ratio) + 0.05)
            for k in range(cnt):
                i = order_worst[k]
                x = best_x[:]
                for d in range(dim):
                    if spans[d] > 0.0:
                        x[d] = clamp(x[d] + random.gauss(0.0, rad) * spans[d], lows[d], highs[d])
                    else:
                        x[d] = lows[d]
                x = repair_reflect(x)
                fi = evaluate(x)
                pop[i] = x
                fits[i] = fi
                hist_add(x, fi)
                if fi < best:
                    best, best_x = fi, x[:]

        # linear population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * ratio))
        if target_NP < NPmin:
            target_NP = NPmin
        if target_NP > NP0:
            target_NP = NP0

        if target_NP < NP:
            order_keep = list(range(NP))
            order_keep.sort(key=lambda i: fits[i])
            keep = order_keep[:target_NP]
            pop = [pop[i] for i in keep]
            fits = [fits[i] for i in keep]
            NP = target_NP
            if len(archive) > arch_max:
                random.shuffle(archive)
                del archive[arch_max:]

    return best
