import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (stdlib only).

    Key upgrades vs the provided DE/SHADE-ish variants:
      - Uses a more reliable local optimizer: bounded Nelder–Mead (NM) with shrink/expand,
        restarted from best + a few elite-based starts (very strong "polish" under time limit).
      - Keeps global exploration via a lightweight JADE current-to-pbest/1 + archive loop.
      - Adds an explicit time scheduler: early global search, mid hybrid, late NM intensification.
      - Better restart logic: multi-start NM + scatter population when stalled.
      - Very robust bound handling (reflect) + safe eval.

    Returns best fitness (float).
    """

    t0 = time.time()
    T = float(max_time) if max_time is not None else 0.0
    deadline = t0 + max(0.0, T)

    # ---------- guards ----------
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
            # reflect repeatedly
            v = y[i]
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                if v > hi:
                    v = hi - (v - hi)
            # final clamp for numeric safety
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

    # ---------- elite store ----------
    elite_cap = max(10, min(60, 4 * dim + 10))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # ---------- bounded Nelder–Mead (local optimizer) ----------
    def nelder_mead_local(x_start, f_start, frac, max_evals):
        """
        Bounded Nelder–Mead with reflection to bounds.
        Returns (xbest, fbest, evals_used).
        """
        if max_evals <= 0 or time.time() >= deadline:
            return x_start, f_start, 0

        # NM coefficients (slightly conservative for noisy/rough functions)
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        # initial simplex step size: ties to span and time fraction
        base = (0.12 * (1.0 - frac) + 0.01)
        steps = [base * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
        # avoid zero steps
        for i in range(dim):
            if steps[i] <= 0.0:
                steps[i] = 1e-12

        # build simplex: x0 plus one step in each coordinate (with a touch of random rotation sometimes)
        simplex = [ensure_bounds(x_start)]
        fvals = [f_start]
        evals = 0

        # slight randomization to avoid axis issues in high-dim
        use_rand_dirs = (dim >= 20 and random.random() < 0.35)
        for i in range(dim):
            if time.time() >= deadline or evals >= max_evals:
                break
            x = x_start[:]
            if use_rand_dirs:
                d = rand_dir_unit()
                for k in range(dim):
                    x[k] += d[k] * steps[i]
            else:
                x[i] += steps[i]
            x = ensure_bounds(x)
            fx = safe_eval(x)
            evals += 1
            simplex.append(x); fvals.append(fx)
            push_elite(fx, x)

        n = len(simplex) - 1
        if n <= 0:
            return x_start, f_start, evals

        # helper: sort simplex by fitness
        def sort_simplex():
            nonlocal simplex, fvals
            order = sorted(range(len(fvals)), key=lambda k: fvals[k])
            simplex = [simplex[k] for k in order]
            fvals = [fvals[k] for k in order]

        sort_simplex()
        bestx, bestf = simplex[0][:], fvals[0]

        it_no_improve = 0
        while time.time() < deadline and evals < max_evals:
            sort_simplex()
            if fvals[0] < bestf:
                bestf = fvals[0]
                bestx = simplex[0][:]
                it_no_improve = 0
            else:
                it_no_improve += 1

            # stop if simplex is tiny or stalled
            if it_no_improve > 18:
                break

            # centroid of best n points (excluding worst)
            centroid = [0.0] * dim
            for j in range(n):
                xj = simplex[j]
                for d in range(dim):
                    centroid[d] += xj[d]
            inv = 1.0 / n
            for d in range(dim):
                centroid[d] *= inv

            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_second = simplex[-2]
            f_second = fvals[-2]

            # reflection
            xr = [centroid[d] + alpha * (centroid[d] - x_worst[d]) for d in range(dim)]
            xr = ensure_bounds(xr)
            fr = safe_eval(xr); evals += 1
            push_elite(fr, xr)

            if fr < fvals[0]:
                # expansion
                if evals >= max_evals or time.time() >= deadline:
                    simplex[-1], fvals[-1] = xr, fr
                    continue
                xe = [centroid[d] + gamma * (xr[d] - centroid[d]) for d in range(dim)]
                xe = ensure_bounds(xe)
                fe = safe_eval(xe); evals += 1
                push_elite(fe, xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < f_second:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contraction
                if evals >= max_evals or time.time() >= deadline:
                    # cannot evaluate more, accept reflection if better else stop
                    if fr < f_worst:
                        simplex[-1], fvals[-1] = xr, fr
                    break

                if fr < f_worst:
                    # outside contraction
                    xc = [centroid[d] + rho * (xr[d] - centroid[d]) for d in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[d] - rho * (centroid[d] - x_worst[d]) for d in range(dim)]
                xc = ensure_bounds(xc)
                fc = safe_eval(xc); evals += 1
                push_elite(fc, xc)

                if fc < f_worst:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink
                    x_best = simplex[0]
                    for j in range(1, len(simplex)):
                        if evals >= max_evals or time.time() >= deadline:
                            break
                        xs = [x_best[d] + sigma * (simplex[j][d] - x_best[d]) for d in range(dim)]
                        xs = ensure_bounds(xs)
                        fs = safe_eval(xs); evals += 1
                        simplex[j], fvals[j] = xs, fs
                        push_elite(fs, xs)

        sort_simplex()
        if fvals[0] < bestf:
            bestf = fvals[0]
            bestx = simplex[0][:]
        return bestx, bestf, evals

    # ---------- initialization (LHS-ish + opposition) ----------
    # pop size moderate
    pop_size = max(18, min(80, 10 + 2 * dim + 4 * int(math.sqrt(dim))))
    if T <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif T <= 3.0:
        pop_size = max(16, min(pop_size, 48))

    # LHS bins per dimension
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

    seed_deadline = t0 + min(0.18, 0.07 + 0.0025 * dim) * max(1e-12, T)
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

    # ---------- global engine: JADE/SHADE-lite ----------
    H = max(6, min(24, dim + 6))
    M_F = [0.6] * H
    M_CR = [0.5] * H
    h_idx = 0

    archive = []
    archive_max = pop_size

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.20 * T)

    # a small local-NM budget reserved for the end
    nm_reserved_frac = 0.28 if T >= 2.0 else 0.22

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1
        frac = min(1.0, (now - t0) / max(1e-12, T))

        # late intensification: run a few NM restarts (best + top elites)
        if frac >= (1.0 - nm_reserved_frac):
            # allocate remaining evals approximately by time (since func cost unknown, use loop guards)
            # do multiple short NM runs rather than one long run
            starts = []
            starts.append((best, best_x[:]))
            for k in range(min(4, len(elites))):
                starts.append(elites[k])
            # add one random start if time
            if time.time() < deadline:
                xr = rand_vec()
                fr = safe_eval(xr)
                push_elite(fr, xr)
                starts.append((fr, xr))

            for (fs, xs) in starts:
                if time.time() >= deadline:
                    break
                # short, time-aware eval budget
                # (dim+1 simplex; we give a few iterations worth)
                max_evals = max(10, min(60 + 6 * dim, 140 + 2 * dim))
                xnm, fnm, _ = nelder_mead_local(xs, fs, frac, max_evals)
                if fnm < best:
                    best, best_x = fnm, xnm[:]
                    last_improve_t = time.time()
            return best

        # pbest pressure
        p_min, p_max = 0.05, 0.35
        p_frac = p_min + (p_max - p_min) * (0.20 + 0.80 * frac)
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

            # late: smaller Fi for refinement
            Fi *= (0.98 - 0.20 * frac)
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

        # update memories (SHADE style)
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

        # occasional cheap NM micro-polish mid-run (helps a lot)
        if (gen % 3 == 0) and time.time() < deadline and frac > 0.35:
            max_evals = max(10, min(30 + 3 * dim, 70))
            xnm, fnm, _ = nelder_mead_local(best_x, best, frac, max_evals)
            if fnm < best:
                best, best_x = fnm, xnm[:]
                last_improve_t = time.time()

        # restart on stall: scatter most of pop, keep a few elites
        if time.time() - last_improve_t > stall_restart_seconds:
            keep = max(3, pop_size // 6)
            base = elites[:min(len(elites), keep)]
            if not base:
                base = [(best, best_x[:])]

            new_pop = [x[:] for (f, x) in base]
            new_fit = [safe_eval(x) for x in new_pop]

            archive = []
            M_F = [0.65] * H
            M_CR = [0.5] * H
            h_idx = 0

            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.55:
                    x = rand_vec()
                else:
                    # around best / around a random elite
                    if r < 0.82:
                        c = best_x
                        s = 0.35 * (1.0 - frac) + 0.10
                    else:
                        c = base[random.randrange(len(base))][1]
                        s = 0.28 * (1.0 - frac) + 0.08
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
