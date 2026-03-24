import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (stdlib only).

    Improvement focus vs your best (#4):
      - Stronger exploitation: (1+1)-ES with 1/5th success rule around incumbent
      - Better global search: adaptive DE/current-to-pbest/1 (JADE/SHADE-lite) + archive
      - Better endgame: deterministic coordinate line-search + random-direction line-search
      - More reliable restarts: elite-biased, radius resets, and diversity injection
      - Strict time scheduling and low overhead

    Returns best fitness (float).
    """

    t0 = time.time()
    T = float(max_time) if max_time is not None else 0.0
    deadline = t0 + max(0.0, T)

    # ---------------- guards ----------------
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
            # reflect repeatedly (robust for far-out values)
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

    # ---------------- elite store ----------------
    elite_cap = max(10, min(70, 4 * dim + 14))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # ---------------- seeding (fast LHS-ish + opposition) ----------------
    # population size tuned for time-bounded operation
    pop_size = max(18, min(90, 10 + 2 * dim + 4 * int(math.sqrt(dim))))
    if T <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif T <= 3.0:
        pop_size = max(16, min(pop_size, 52))

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

    # seed budget: small fraction of time
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

    # ---------------- global engine: SHADE-lite current-to-pbest/1 ----------------
    H = max(6, min(24, dim + 6))
    M_F = [0.65] * H
    M_CR = [0.50] * H
    h_idx = 0

    archive = []
    archive_max = pop_size

    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.22 * T)

    # ---------------- local engine: (1+1)-ES + line searches ----------------
    # per-dimension step size for ES / local searches
    base_sigma = [0.18 * s if s > 0 else 0.0 for s in spans]
    min_sigma = [max(1e-15, 0.001 * s) if s > 0 else 0.0 for s in spans]
    max_sigma = [0.60 * s if s > 0 else 0.0 for s in spans]
    sigma = base_sigma[:]

    succ_rate = 0.2  # EMA success probability
    succ_ema = 0.12

    def local_es_step(x0, f0, frac, n_tries):
        """(1+1)-ES around incumbent with 1/5th success rule; cheap and effective."""
        nonlocal sigma, succ_rate, best, best_x, last_improve_t
        if time.time() >= deadline:
            return

        # time-aware global scaling: later smaller
        scale = (0.85 * (1.0 - frac) + 0.12)
        successes = 0
        trials = 0

        for _ in range(n_tries):
            if time.time() >= deadline:
                break
            trials += 1
            cand = x0[:]
            # perturb subset to reduce wasted moves in high dim
            if dim <= 10:
                k = dim
            else:
                k = max(2, dim // 5)
            # choose coordinates
            for d in random.sample(range(dim), k):
                if spans[d] > 0:
                    cand[d] += random.gauss(0.0, sigma[d] * scale)
            cand = ensure_bounds(cand)
            fc = safe_eval(cand)
            push_elite(fc, cand)
            if fc < best:
                best, best_x = fc, cand[:]
                last_improve_t = time.time()
            if fc < f0:
                x0, f0 = cand[:], fc
                successes += 1

        if trials > 0:
            p = successes / float(trials)
            succ_rate = (1.0 - succ_ema) * succ_rate + succ_ema * p
            # 1/5th success rule: target ~0.2
            if succ_rate > 0.22:
                mult = 1.12
            elif succ_rate < 0.18:
                mult = 0.90
            else:
                mult = 1.0
            for d in range(dim):
                if spans[d] > 0:
                    sigma[d] = max(min_sigma[d], min(max_sigma[d], sigma[d] * mult))

    def line_search_direction(x0, f0, dvec, step, max_steps):
        """Bounded directional search with geometric steps (very low overhead)."""
        if step <= 0.0:
            return x0, f0
        bestl_x, bestl_f = x0[:], f0

        # try both signs at initial step
        for sign in (1.0, -1.0):
            if time.time() >= deadline:
                break
            alpha = step * sign
            cand = ensure_bounds([x0[i] + alpha * dvec[i] for i in range(dim)])
            fc = safe_eval(cand)
            push_elite(fc, cand)
            if fc < bestl_f:
                bestl_x, bestl_f = cand[:], fc

        # extend from best direction if improved
        if bestl_f < f0:
            # determine direction from x0 -> bestl_x
            dir2 = [bestl_x[i] - x0[i] for i in range(dim)]
            norm = math.sqrt(sum(v*v for v in dir2))
            if norm > 1e-30:
                inv = 1.0 / norm
                dir2 = [v * inv for v in dir2]
            else:
                dir2 = dvec

            alpha = step
            for _ in range(max_steps):
                if time.time() >= deadline:
                    break
                alpha *= 1.6
                cand = ensure_bounds([x0[i] + alpha * dir2[i] for i in range(dim)])
                fc = safe_eval(cand)
                push_elite(fc, cand)
                if fc < bestl_f:
                    bestl_x, bestl_f = cand[:], fc
                else:
                    break

        return bestl_x, bestl_f

    def endgame_polish(frac):
        """Coordinate + random-direction line searches around best."""
        nonlocal best, best_x, last_improve_t
        if time.time() >= deadline:
            return

        # coordinate search
        coord_trials = min(dim, 10) if dim <= 30 else 12
        idxs = list(range(dim))
        random.shuffle(idxs)
        for j in idxs[:coord_trials]:
            if time.time() >= deadline:
                return
            if spans[j] <= 0:
                continue
            step = max(min_sigma[j], (0.06 * (1.0 - frac) + 0.006) * spans[j])
            # direction vector = axis j
            dvec = [0.0] * dim
            dvec[j] = 1.0
            x2, f2 = line_search_direction(best_x, best, dvec, step, max_steps=2)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()

        # random directions
        rdirs = 2 if dim > 25 else 3
        base = (0.08 * (1.0 - frac) + 0.006) * ((sum(spans) / max(1, dim)) if dim else 1.0)
        base = max(1e-15, base)
        for _ in range(rdirs):
            if time.time() >= deadline:
                return
            dvec = rand_dir_unit()
            x2, f2 = line_search_direction(best_x, best, dvec, base, max_steps=3)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()

    # ---------------- restart ----------------
    def restart(frac):
        nonlocal pop, fit, archive, M_F, M_CR, h_idx, sigma, last_improve_t
        keep = max(3, pop_size // 6)
        base = elites[:min(len(elites), keep)]
        if not base:
            base = [(best, best_x[:])]

        new_pop = [x[:] for (f, x) in base]
        new_fit = [safe_eval(x) for x in new_pop]

        # reset DE memories mildly
        M_F = [0.68] * H
        M_CR = [0.5] * H
        h_idx = 0
        archive = []

        # reset local sigmas wider
        for d in range(dim):
            if spans[d] > 0:
                sigma[d] = max(sigma[d], 0.16 * spans[d])

        # choose a center: best, random elite, or random
        r = random.random()
        if r < 0.55:
            center = best_x[:]
        elif r < 0.85 and base:
            center = base[random.randrange(len(base))][1][:]
        else:
            center = rand_vec()

        while len(new_pop) < pop_size and time.time() < deadline:
            u = random.random()
            if u < 0.45:
                x = rand_vec()
            else:
                x = center[:]
                # spread decreases with time
                s = (0.40 * (1.0 - frac) + 0.10)
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, s * spans[d])
                x = ensure_bounds(x)

            fx = safe_eval(x)
            new_pop.append(x); new_fit.append(fx)
            push_elite(fx, x)
            if fx < best:
                # (best, best_x) updated by outer scope; keep consistent here
                pass

        pop, fit = new_pop, new_fit
        # update best from current structures
        bi = min(range(len(pop)), key=lambda i: fit[i])
        if fit[bi] < best:
            # update globals
            # (best is outer variable; assign via closure technique)
            pass
        last_improve_t = time.time()

    # ---------------- main loop ----------------
    gen = 0
    # reserve time for endgame polish
    endgame_frac = 0.22 if T >= 2.0 else 0.18

    # Make sure best is valid
    bi = min(range(pop_size), key=lambda i: fit[i])
    best = fit[bi]
    best_x = pop[bi][:]
    push_elite(best, best_x)

    while True:
        now = time.time()
        if now >= deadline:
            return best

        frac = min(1.0, (now - t0) / max(1e-12, T))

        # endgame
        if frac >= 1.0 - endgame_frac:
            # run multiple quick polishes until time runs out
            while time.time() < deadline:
                # alternate ES and line-search polish
                local_es_step(best_x, best, frac, n_tries=(3 if dim > 25 else 5))
                endgame_polish(frac)
            return best

        gen += 1

        # DE settings
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
            for _ in range(10):
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
            for _ in range(25):
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

            # mutation
            v = [xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]
            v = ensure_bounds(v)

            # crossover
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

        # update SHADE memories
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

        # interleave local ES often (this is a key improvement)
        if time.time() < deadline and (gen % 2 == 0 or frac > 0.45):
            local_es_step(best_x, best, frac, n_tries=(4 if dim > 25 else 7))

        # occasional quick directional polish mid-run
        if time.time() < deadline and (gen % 3 == 0) and frac > 0.30:
            endgame_polish(frac)

        # restart on stall
        if time.time() - last_improve_t > stall_restart_seconds:
            # perform restart but keep best & elites; then refresh best explicitly
            restart(frac)
            bi = min(range(len(pop)), key=lambda i: fit[i])
            if fit[bi] < best:
                best, best_x = fit[bi], pop[bi][:]
                last_improve_t = time.time()
            else:
                # still keep time stamp to avoid restart loops
                last_improve_t = time.time()
