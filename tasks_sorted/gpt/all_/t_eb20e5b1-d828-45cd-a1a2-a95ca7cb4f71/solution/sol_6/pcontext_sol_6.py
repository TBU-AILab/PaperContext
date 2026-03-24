import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (stdlib only).

    Core idea: Hybridize three components with a strict time scheduler:
      (A) Global exploration: SHADE-like DE/current-to-pbest/1 + archive
      (B) Structured local search: adaptive coordinate + random-subspace pattern search
      (C) End-game polish: bounded Powell-style line search on a small direction set

    Key improvements vs your previous set:
      - Stronger, more deterministic end-game (Powell-ish) than Nelder–Mead under bounds.
      - Better use of evaluations: local search only when promising (based on progress/stall),
        and it reuses adaptive step sizes per dimension.
      - Lightweight surrogate-free "quadratic" step on 1D line-search (3-point fit) to
        accelerate along good directions.
      - More robust parameter adaptation and restart policy; avoids over-spending time on seeding.

    Returns: best fitness (float)
    """

    t0 = time.time()
    T = float(max_time) if max_time is not None else 0.0
    deadline = t0 + max(0.0, T)

    # ----------------- guards -----------------
    if dim <= 0:
        try:
            v = float(func([]))
            return v if (not math.isnan(v) and not math.isinf(v)) else float("inf")
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def ensure_bounds(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect repeatedly
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                if v > hi:
                    v = hi - (v - hi)
            if v < lo: v = lo
            if v > hi: v = hi
            y[i] = v
        return y

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def rand_dir_unit():
        s2 = 0.0
        d = [0.0] * dim
        for i in range(dim):
            r = random.gauss(0.0, 1.0)
            d[i] = r
            s2 += r * r
        if s2 <= 1e-30:
            j = random.randrange(dim)
            d = [0.0] * dim
            d[j] = 1.0
            return d
        inv = 1.0 / math.sqrt(s2)
        for i in range(dim):
            d[i] *= inv
        return d

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # ----------------- elites -----------------
    elite_cap = max(10, min(60, 3 * dim + 12))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # ----------------- initialization (fast LHS-ish + opposition, but time-capped) -----------------
    pop_size = max(18, min(80, 10 + 2 * dim + 4 * int(math.sqrt(dim))))
    if T <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif T <= 3.0:
        pop_size = max(16, min(pop_size, 48))

    # LHS bins
    bins = []
    for d in range(dim):
        n = pop_size
        if spans[d] <= 0:
            bins.append([lows[d]] * n)
        else:
            pts = [(k + random.random()) / n for k in range(n)]
            random.shuffle(pts)
            bins.append([lows[d] + p * spans[d] for p in pts])

    def opposition(x):
        return ensure_bounds([(lows[i] + highs[i]) - x[i] for i in range(dim)])

    pop, fit = [], []
    best = float("inf")
    best_x = rand_vec()

    seed_deadline = t0 + min(0.12, 0.05 + 0.002 * dim) * max(1e-12, T)
    i = 0
    while i < pop_size and time.time() < seed_deadline:
        x = ensure_bounds([bins[d][i] for d in range(dim)])
        fx = safe_eval(x)
        pop.append(x); fit.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        xo = opposition(x)
        fo = safe_eval(xo)
        pop.append(xo); fit.append(fo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]
        i += 1

    while len(pop) < pop_size and time.time() < deadline:
        x = rand_vec()
        fx = safe_eval(x)
        pop.append(x); fit.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

    if len(pop) > pop_size:
        order = sorted(range(len(pop)), key=lambda k: fit[k])[:pop_size]
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]

    # ----------------- DE/SHADE-lite global engine -----------------
    H = max(6, min(24, dim + 6))
    M_F = [0.65] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []
    archive_max = pop_size

    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.22 * T)

    # Local step sizes per dimension (adapted)
    local_step = [0.20 * s if s > 0 else 0.0 for s in spans]
    min_step = [max(1e-15, 0.0015 * s) if s > 0 else 0.0 for s in spans]

    # ----------------- local search primitives -----------------
    def line_search_1d(x0, f0, dvec, step0, frac, max_trials=10):
        """
        Bounded line-search along direction dvec from x0.
        Uses: try +/- step, then backtracking; if 3 points available -> quadratic proposal.
        Returns (xbest, fbest).
        """
        if time.time() >= deadline:
            return x0, f0

        # make a working step
        step = step0
        if step <= 0.0:
            return x0, f0

        bestl_x, bestl_f = x0[:], f0

        # helper to evaluate point along direction
        def eval_at(alpha):
            x = [x0[i] + alpha * dvec[i] for i in range(dim)]
            x = ensure_bounds(x)
            return x, safe_eval(x)

        # initial probes
        x1, f1 = eval_at(+step)
        push_elite(f1, x1)
        x2, f2 = eval_at(-step)
        push_elite(f2, x2)

        # choose side
        if f1 < bestl_f or f2 < bestl_f:
            if f1 <= f2:
                a_best, xb, fb = +step, x1, f1
            else:
                a_best, xb, fb = -step, x2, f2
            bestl_x, bestl_f = xb[:], fb
        else:
            # no immediate improvement: backtrack quickly
            for _ in range(3):
                if time.time() >= deadline:
                    break
                step *= 0.5
                if step <= 1e-18:
                    break
                x1, f1 = eval_at(+step); push_elite(f1, x1)
                if f1 < bestl_f:
                    bestl_x, bestl_f = x1[:], f1
                    break
                x2, f2 = eval_at(-step); push_elite(f2, x2)
                if f2 < bestl_f:
                    bestl_x, bestl_f = x2[:], f2
                    break
            return bestl_x, bestl_f

        # if we have improvement, try to accelerate (a couple expansions + optional quadratic)
        # track three points for quadratic in chosen direction
        a0, f0a = 0.0, f0
        a1, f1a = a_best, bestl_f
        a2, f2a = 2.0 * a_best, None

        # one expansion
        if time.time() < deadline:
            x3, f3 = eval_at(a2); push_elite(f3, x3)
            f2a = f3
            if f3 < bestl_f:
                bestl_x, bestl_f = x3[:], f3
                a1, f1a = a2, f3
                a2 = 3.2 * a_best
                # second expansion attempt
                if time.time() < deadline:
                    x4, f4 = eval_at(a2); push_elite(f4, x4)
                    if f4 < bestl_f:
                        bestl_x, bestl_f = x4[:], f4
                        a0, f0a = a1, f1a
                        a1, f1a = a2, f4

        # quadratic proposal if we have 3 distinct alphas
        if f2a is not None and time.time() < deadline:
            # use points (0,f0), (a_best, f_best), (2*a_best, f2a)
            # fit parabola and take minimizer: alpha* = -b/(2c)
            # Compute using finite differences:
            # f(0)=A, f(h)=B, f(2h)=C
            A = f0a
            h = a_best
            B = f1a
            C = f2a
            denom = (2.0 * (A - 2.0 * B + C))
            if abs(denom) > 1e-18:
                alpha_star = h * (A - C) / denom
                # keep it within a reasonable bracket
                alpha_star = max(-3.0 * abs(h), min(3.0 * abs(h), alpha_star))
                if abs(alpha_star) > 1e-18:
                    xq, fq = eval_at(alpha_star); push_elite(fq, xq)
                    if fq < bestl_f:
                        bestl_x, bestl_f = xq[:], fq

        # mild backtracking around best found (2 trials)
        step2 = abs(a_best) * 0.5
        for _ in range(2):
            if time.time() >= deadline:
                break
            step2 *= 0.5
            if step2 <= 1e-18:
                break
            # try halfway between 0 and best alpha
            alpha_mid = 0.5 * a_best
            xm, fm = eval_at(alpha_mid); push_elite(fm, xm)
            if fm < bestl_f:
                bestl_x, bestl_f = xm[:], fm

        return bestl_x, bestl_f

    def pattern_local_search(x0, f0, frac, budget_evals=12):
        """
        Adaptive pattern search: coordinate +/- plus one random direction line-search.
        Uses per-dimension step sizes (local_step) and updates them on success/failure.
        """
        xbest, fbest = x0[:], f0
        evals = 0

        # coordinate pokes
        trials = min(dim, max(3, dim // 3))
        for _ in range(trials):
            if time.time() >= deadline or evals >= budget_evals:
                break
            d = random.randrange(dim)
            if spans[d] <= 0:
                continue
            step = local_step[d]
            if step <= 0.0:
                continue

            for sign in (1.0, -1.0):
                if time.time() >= deadline or evals >= budget_evals:
                    break
                cand = xbest[:]
                cand[d] += sign * step
                cand = ensure_bounds(cand)
                fc = safe_eval(cand); evals += 1
                push_elite(fc, cand)
                if fc < fbest:
                    xbest, fbest = cand[:], fc
                    # success => slightly increase step (but cap)
                    local_step[d] = min(0.45 * spans[d], local_step[d] * 1.25)
                else:
                    # failure => shrink a bit but not below min_step
                    local_step[d] = max(min_step[d], local_step[d] * 0.85)

        # one (sometimes two) directional line-search in random subspace
        if time.time() < deadline and evals < budget_evals:
            ndirs = 1 if dim > 25 else 2
            for _ in range(ndirs):
                if time.time() >= deadline or evals >= budget_evals:
                    break
                dvec = rand_dir_unit()
                # step tied to median step size
                med = sorted(local_step)[len(local_step) // 2]
                step0 = max(1e-15, (0.20 * (1.0 - frac) + 0.02) * (med if med > 0 else (sum(spans) / max(1, dim))))
                x2, f2 = line_search_1d(xbest, fbest, dvec, step0, frac, max_trials=8)
                # line_search already spent evals internally; we can't count exactly without instrumenting,
                # but keep budget small by calling it few times.
                if f2 < fbest:
                    xbest, fbest = x2[:], f2

        return xbest, fbest

    # ----------------- end-game polish (Powell-ish) -----------------
    def powell_polish(x0, f0, frac):
        """
        Small-direction-set Powell-like improvement under bounds.
        Keeps a set of directions (axes + a few random) and does line searches.
        """
        x = x0[:]
        fx = f0

        # build direction set (limit size for speed)
        dirs = []
        # a few axes (random subset if high dim)
        if dim <= 14:
            for k in range(dim):
                v = [0.0] * dim
                v[k] = 1.0
                dirs.append(v)
        else:
            # sample ~10 axes
            for k in random.sample(range(dim), 10):
                v = [0.0] * dim
                v[k] = 1.0
                dirs.append(v)

        # plus random directions
        rcount = 3 if dim > 25 else 5
        for _ in range(rcount):
            dirs.append(rand_dir_unit())

        # base step: small late
        base_span = (sum(spans) / max(1, dim)) if dim > 0 else 1.0
        step0 = (0.08 * (1.0 - frac) + 0.006) * (base_span if base_span > 0 else 1.0)

        # do one pass; if improvement, do a second short pass
        for _pass in range(2):
            improved = False
            for dvec in dirs:
                if time.time() >= deadline:
                    return x0, f0
                x2, f2 = line_search_1d(x, fx, dvec, step0, frac, max_trials=8)
                if f2 < fx:
                    improved = True
                    x, fx = x2[:], f2
            if not improved:
                break
            step0 *= 0.55

        return x, fx

    # ----------------- main loop -----------------
    gen = 0
    nm_like_end_frac = 0.18  # reserve for polish
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1
        frac = min(1.0, (now - t0) / max(1e-12, T))

        # end-game: aggressive polish and return
        if frac >= 1.0 - nm_like_end_frac:
            # try best + a few elites
            starts = [(best, best_x[:])]
            for k in range(min(6, len(elites))):
                starts.append(elites[k])
            for fs, xs in starts:
                if time.time() >= deadline:
                    break
                xp, fp = powell_polish(xs, fs, frac)
                if fp < best:
                    best, best_x = fp, xp[:]
                    last_improve_t = time.time()
            return best

        # pbest pressure
        p_min, p_max = 0.05, 0.35
        p_frac = p_min + (p_max - p_min) * (0.15 + 0.85 * frac)
        p_cnt = max(2, int(math.ceil(p_frac * pop_size)))
        order = sorted(range(pop_size), key=lambda k: fit[k])

        S_F, S_CR, dF = [], [], []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            # Fi ~ cauchy(M_F[r], 0.1)
            Fi = None
            for _t in range(10):
                u = random.random()
                val = M_F[r] + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < val <= 1.0:
                    Fi = val
                    break
            if Fi is None:
                Fi = max(0.05, min(1.0, M_F[r]))
            # CRi ~ normal(M_CR[r], 0.1)
            CRi = clip01(random.gauss(M_CR[r], 0.1))

            # later: smaller Fi
            Fi *= (0.98 - 0.25 * frac)
            Fi = max(0.05, min(1.0, Fi))

            pbest_idx = order[random.randrange(p_cnt)]
            xpb = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            union = pop + archive
            union_n = len(union)
            x2 = None
            for _t in range(25):
                j = random.randrange(union_n)
                if j < pop_size and (j == i or j == r1):
                    continue
                x2 = union[j]
                break
            if x2 is None:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x2 = pop[r2]

            xr1 = pop[r1]
            xr2 = x2

            v = [xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]
            v = ensure_bounds(v)

            jrand = random.randrange(dim)
            uvec = [v[d] if (random.random() < CRi or d == jrand) else xi[d] for d in range(dim)]
            uvec = ensure_bounds(uvec)

            fu = safe_eval(uvec)
            push_elite(fu, uvec)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]
                pop[i] = uvec
                fit[i] = fu

                if fu < fi:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    dF.append(fi - fu)

                if fu < best:
                    best, best_x = fu, uvec[:]
                    last_improve_t = time.time()

        # update memories
        if dF:
            s = sum(dF)
            wts = [di / s for di in dF] if s > 1e-18 else [1.0 / len(dF)] * len(dF)

            mcr = 0.0
            for w, cr in zip(wts, S_CR):
                mcr += w * cr

            num = 0.0
            den = 0.0
            for w, f in zip(wts, S_F):
                num += w * f * f
                den += w * f
            mf = M_F[h_idx]
            if den > 1e-12:
                mf = num / den

            M_CR[h_idx] = clip01(mcr)
            M_F[h_idx] = max(0.05, min(1.0, mf))
            h_idx = (h_idx + 1) % H

        # interleaved local search when it helps (mid/late or if stalled)
        now2 = time.time()
        frac2 = min(1.0, (now2 - t0) / max(1e-12, T))
        stalled = (now2 - last_improve_t) > (0.45 * stall_restart_seconds)
        if time.time() < deadline and (stalled or (gen % 2 == 0 and frac2 > 0.30)):
            budget = 10 if dim > 25 else 14
            x2, f2 = pattern_local_search(best_x, best, frac2, budget_evals=budget)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()

        # restart on stall
        if time.time() - last_improve_t > stall_restart_seconds:
            keep = max(3, pop_size // 6)
            base = elites[:min(len(elites), keep)]
            if not base:
                base = [(best, best_x[:])]

            new_pop = [x[:] for (f, x) in base]
            new_fit = [safe_eval(x) for x in new_pop]

            archive = []
            M_F = [0.68] * H
            M_CR = [0.5] * H
            h_idx = 0

            # reset local steps a bit wider
            for d in range(dim):
                if spans[d] > 0:
                    local_step[d] = max(local_step[d], 0.12 * spans[d])

            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.50:
                    x = rand_vec()
                else:
                    # around best/elite
                    if r < 0.80:
                        c = best_x
                        s = 0.32 * (1.0 - frac) + 0.10
                    else:
                        c = base[random.randrange(len(base))][1]
                        s = 0.26 * (1.0 - frac) + 0.08
                    x = c[:]
                    for d in range(dim):
                        if spans[d] > 0:
                            x[d] += random.gauss(0.0, s * spans[d])
                    x = ensure_bounds(x)

                fx = safe_eval(x)
                new_pop.append(x); new_fit.append(fx)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()

            pop, fit = new_pop, new_fit
            last_improve_t = time.time()
