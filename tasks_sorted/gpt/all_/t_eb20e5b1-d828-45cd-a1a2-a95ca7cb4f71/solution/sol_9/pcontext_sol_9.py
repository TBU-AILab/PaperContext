import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (stdlib only).

    Design goals vs your latest batch:
      - Keep the strong global engine: SHADE-like DE/current-to-pbest/1 + archive.
      - Fix common time-waste: too many evals in weak regions -> add "best/elite centered"
        sampling bursts and a cheaper, higher-signal local optimizer.
      - Stronger exploitation than (1+1)-ES alone: bounded multi-direction line-search
        + adaptive coordinate steps + lightweight diagonal-ES refresh.
      - More reliable restarts: use elites + radius schedule; restart earlier when truly stalled.
      - Lower overhead per eval, and strict time guards.

    Returns best fitness (float).
    """

    t0 = time.time()
    T = float(max_time) if max_time is not None else 0.0
    if T <= 0.0:
        # best effort with no time
        try:
            v = float(func([0.0] * max(0, dim)))
            return v if (not math.isnan(v) and not math.isinf(v)) else float("inf")
        except Exception:
            return float("inf")
    deadline = t0 + T

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

    # ---------- utilities ----------
    def ensure_bounds(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect (robust even for far-out)
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

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

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
    elite_cap = max(12, min(80, 4 * dim + 20))
    elites = []  # (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # ---------- initialization: LHS-ish + opposition + a few best-centered ----------
    pop_size = max(18, min(90, 10 + 2 * dim + 4 * int(math.sqrt(dim))))
    if T <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif T <= 3.0:
        pop_size = max(16, min(pop_size, 52))

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

    # seed time budget
    seed_deadline = t0 + min(0.16, 0.06 + 0.0022 * dim) * T
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

    # truncate
    if len(pop) > pop_size:
        order = sorted(range(len(pop)), key=lambda k: fit[k])[:pop_size]
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]

    # set best from pop
    bi = min(range(len(pop)), key=lambda k: fit[k])
    best, best_x = fit[bi], pop[bi][:]
    push_elite(best, best_x)

    # best-centered probes (very cheap, often large gain)
    for _ in range(3 if dim <= 20 else 2):
        if time.time() >= deadline:
            return best
        cand = best_x[:]
        s = 0.22
        for d in range(dim):
            if spans[d] > 0:
                cand[d] += random.gauss(0.0, s * spans[d])
        cand = ensure_bounds(cand)
        fc = safe_eval(cand)
        push_elite(fc, cand)
        if fc < best:
            best, best_x = fc, cand[:]

    # ---------- DE (SHADE-lite) state ----------
    H = max(6, min(24, dim + 6))
    M_F = [0.65] * H
    M_CR = [0.50] * H
    h_idx = 0
    archive = []
    archive_max = pop_size

    # ---------- local search state: adaptive coordinate steps ----------
    coord_step = [0.10 * s if s > 0 else 0.0 for s in spans]
    coord_min = [max(1e-15, 0.0010 * s) if s > 0 else 0.0 for s in spans]
    coord_max = [0.45 * s if s > 0 else 0.0 for s in spans]

    # ---------- diagonal ES state (cheap basin exploitation) ----------
    es_mean = best_x[:]
    es_sig = [max(1e-12, 0.28 * spans[i]) if spans[i] > 0 else 0.0 for i in range(dim)]
    es_g = 1.0
    es_succ = 0.20
    es_ema = 0.12

    last_improve_t = time.time()
    stall_restart_seconds = max(0.22, 0.20 * T)

    # reserve time for endgame polishing
    endgame_frac = 0.26 if T >= 2.0 else 0.20

    # ---------- local primitives ----------
    def line_search_dir(x0, f0, dvec, step, max_steps):
        if step <= 0.0 or time.time() >= deadline:
            return x0, f0
        bestl_x, bestl_f = x0[:], f0

        # probe both signs
        for sign in (1.0, -1.0):
            if time.time() >= deadline:
                break
            a = sign * step
            cand = ensure_bounds([x0[i] + a * dvec[i] for i in range(dim)])
            fc = safe_eval(cand)
            push_elite(fc, cand)
            if fc < bestl_f:
                bestl_x, bestl_f = cand[:], fc

        if bestl_f >= f0:
            return bestl_x, bestl_f

        # extend along improving direction
        dir2 = [bestl_x[i] - x0[i] for i in range(dim)]
        nrm = math.sqrt(sum(v * v for v in dir2))
        if nrm > 1e-30:
            inv = 1.0 / nrm
            dir2 = [v * inv for v in dir2]
        else:
            dir2 = dvec

        a = step
        for _ in range(max_steps):
            if time.time() >= deadline:
                break
            a *= 1.7
            cand = ensure_bounds([x0[i] + a * dir2[i] for i in range(dim)])
            fc = safe_eval(cand)
            push_elite(fc, cand)
            if fc < bestl_f:
                bestl_x, bestl_f = cand[:], fc
            else:
                break
        return bestl_x, bestl_f

    def local_polish(frac, rounds):
        nonlocal best, best_x, last_improve_t, coord_step
        if time.time() >= deadline:
            return

        # coordinate line-search on a subset
        idxs = list(range(dim))
        random.shuffle(idxs)
        ncoords = min(dim, 12 if dim > 25 else 10)
        for j in idxs[:ncoords]:
            if time.time() >= deadline:
                return
            if spans[j] <= 0:
                continue
            # time-aware base step
            base = (0.06 * (1.0 - frac) + 0.004) * spans[j]
            step = max(coord_min[j], min(coord_max[j], max(base, coord_step[j])))
            dvec = [0.0] * dim
            dvec[j] = 1.0
            x2, f2 = line_search_dir(best_x, best, dvec, step, max_steps=2)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()
                coord_step[j] = min(coord_max[j], coord_step[j] * 1.18)
            else:
                coord_step[j] = max(coord_min[j], coord_step[j] * 0.88)

        # random directions
        base_span = (sum(spans) / max(1, dim))
        step0 = max(1e-15, (0.08 * (1.0 - frac) + 0.004) * (base_span if base_span > 0 else 1.0))
        nd = 2 if dim > 25 else 3
        for _ in range(nd * max(1, rounds)):
            if time.time() >= deadline:
                return
            dvec = rand_dir_unit()
            x2, f2 = line_search_dir(best_x, best, dvec, step0, max_steps=3)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()

    def restart(frac):
        nonlocal pop, fit, archive, M_F, M_CR, h_idx, es_mean, es_sig, es_g, coord_step
        keep = max(3, pop_size // 6)
        base = elites[:min(len(elites), keep)]
        if not base:
            base = [(best, best_x[:])]

        # choose center
        r = random.random()
        if r < 0.55:
            center = best_x[:]
        elif r < 0.85:
            center = base[random.randrange(len(base))][1][:]
        else:
            center = rand_vec()

        # reset DE memories moderately
        M_F = [0.68] * H
        M_CR = [0.50] * H
        h_idx = 0
        archive = []

        # reset ES around center
        es_mean = center[:]
        for d in range(dim):
            if spans[d] > 0:
                es_sig[d] = max(es_sig[d], (0.22 * (1.0 - frac) + 0.14) * spans[d])
        es_g = 1.0

        # widen coord steps slightly
        for d in range(dim):
            if spans[d] > 0:
                coord_step[d] = max(coord_step[d], 0.06 * spans[d])

        new_pop = [x[:] for (_, x) in base]
        new_fit = [safe_eval(x) for x in new_pop]
        for f, x in zip(new_fit, new_pop):
            push_elite(f, x)

        while len(new_pop) < pop_size and time.time() < deadline:
            u = random.random()
            if u < 0.45:
                x = rand_vec()
            else:
                x = center[:]
                s = (0.40 * (1.0 - frac) + 0.10)
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, s * spans[d])
                x = ensure_bounds(x)
            fx = safe_eval(x)
            new_pop.append(x); new_fit.append(fx)
            push_elite(fx, x)

        # truncate
        if len(new_pop) > pop_size:
            order = sorted(range(len(new_pop)), key=lambda k: new_fit[k])[:pop_size]
            new_pop = [new_pop[k] for k in order]
            new_fit = [new_fit[k] for k in order]

        pop, fit = new_pop, new_fit

    # ---------- main loop ----------
    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        frac = min(1.0, (now - t0) / T)

        # endgame: aggressive polish + tiny ES bursts
        if frac >= 1.0 - endgame_frac:
            while time.time() < deadline:
                # tiny diagonal-ES burst (cheap and effective)
                lam = 6 if dim > 25 else 8
                mu = max(3, lam // 2)
                if es_succ > 0.22:
                    es_g *= 1.04
                elif es_succ < 0.18:
                    es_g *= 0.96
                es_g = max(0.10, min(2.5, es_g))
                step_scale = (0.30 + 0.70 * (1.0 - frac)) * es_g

                off = []
                for _ in range(lam):
                    if time.time() >= deadline:
                        return best
                    x = [es_mean[d] + random.gauss(0.0, es_sig[d] * step_scale) if spans[d] > 0 else lows[d]
                         for d in range(dim)]
                    x = ensure_bounds(x)
                    fx = safe_eval(x)
                    off.append((fx, x))
                    push_elite(fx, x)
                    if fx < best:
                        best, best_x = fx, x[:]
                        last_improve_t = time.time()

                off.sort(key=lambda t: t[0])
                parents = off[:mu]

                # update es_mean
                w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
                ws = sum(w)
                w = [wi / ws for wi in w]
                nm = [0.0] * dim
                for wi, (_, xi) in zip(w, parents):
                    for d in range(dim):
                        nm[d] += wi * xi[d]
                es_mean = ensure_bounds(nm)

                # success estimate vs median offspring
                med = off[lam // 2][0]
                succ = sum(1 for f, _ in off if f < med) / float(lam)
                es_succ = (1.0 - es_ema) * es_succ + es_ema * succ

                # polish around best
                local_polish(frac, rounds=1)

            return best

        gen += 1

        # choose between DE and ES (more ES later)
        r = random.random()
        use_es = (r < (0.18 if frac < 0.35 else (0.38 if frac < 0.70 else 0.55)))

        if use_es:
            # diagonal ES generation
            lam = 8 if dim > 25 else 10
            mu = max(3, lam // 2)

            if es_succ > 0.22:
                es_g *= 1.05
            elif es_succ < 0.18:
                es_g *= 0.95
            es_g = max(0.08, min(2.5, es_g))
            step_scale = (0.40 + 0.60 * (1.0 - frac)) * es_g

            off = []
            for _ in range(lam):
                if time.time() >= deadline:
                    return best
                x = [es_mean[d] + random.gauss(0.0, es_sig[d] * step_scale) if spans[d] > 0 else lows[d]
                     for d in range(dim)]
                x = ensure_bounds(x)
                fx = safe_eval(x)
                off.append((fx, x))
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()

            off.sort(key=lambda t: t[0])
            parents = off[:mu]

            w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            ws = sum(w)
            w = [wi / ws for wi in w]

            nm = [0.0] * dim
            for wi, (_, xi) in zip(w, parents):
                for d in range(dim):
                    nm[d] += wi * xi[d]
            es_mean = ensure_bounds(nm)

            # adapt diagonal sigmas from parent spread
            cs = min(0.35, (mu + 2.0) / (dim + mu + 6.0))
            for d in range(dim):
                if spans[d] <= 0:
                    es_sig[d] = 0.0
                    continue
                md = es_mean[d]
                var = 0.0
                for wi, (_, xi) in zip(w, parents):
                    diff = xi[d] - md
                    var += wi * diff * diff
                sd = math.sqrt(max(1e-30, var))
                min_sd = 1e-12 + 1e-8 * spans[d]
                max_sd = 2.0 * spans[d] + 1e-12
                es_sig[d] = (1.0 - cs) * es_sig[d] + cs * max(min_sd, min(max_sd, sd))

            # update es success
            med = off[lam // 2][0]
            succ = sum(1 for f, _ in off if f < med) / float(lam)
            es_succ = (1.0 - es_ema) * es_succ + es_ema * succ

            # light polish sometimes
            if gen % 2 == 0:
                local_polish(frac, rounds=1)

        else:
            # DE/current-to-pbest/1 (SHADE-lite)
            p_min, p_max = 0.05, 0.35
            p_frac = p_min + (p_max - p_min) * (0.18 + 0.82 * frac)
            p_cnt = max(2, int(math.ceil(p_frac * pop_size)))
            order = sorted(range(pop_size), key=lambda k: fit[k])

            S_F, S_CR, dF = [], [], []

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]
                rmem = random.randrange(H)

                # Fi ~ cauchy(M_F,0.1)
                Fi = None
                for _ in range(10):
                    u = random.random()
                    val = M_F[rmem] + 0.1 * math.tan(math.pi * (u - 0.5))
                    if 0.0 < val <= 1.0:
                        Fi = val
                        break
                if Fi is None:
                    Fi = max(0.05, min(1.0, M_F[rmem]))

                # CRi ~ normal(M_CR,0.1)
                CRi = clip01(random.gauss(M_CR[rmem], 0.1))

                # later: smaller F
                Fi *= (0.98 - 0.22 * frac)
                Fi = max(0.05, min(1.0, Fi))

                pbest_idx = order[random.randrange(p_cnt)]
                xpb = pop[pbest_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                union = pop + archive
                union_n = len(union)
                xr2 = None
                for _ in range(25):
                    j = random.randrange(union_n)
                    if j < pop_size and (j == i or j == r1):
                        continue
                    xr2 = union[j]
                    break
                if xr2 is None:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    xr2 = pop[r2]

                xr1 = pop[r1]

                v = [xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]
                v = ensure_bounds(v)

                jrand = random.randrange(dim)
                uvec = [v[d] if (random.random() < CRi or d == jrand) else xi[d] for d in range(dim)]
                uvec = ensure_bounds(uvec)

                fu = safe_eval(uvec)
                push_elite(fu, uvec)

                # occasional elite blend (cheap robustness)
                if i == 0 and elites and (gen % 2 == 0) and time.time() < deadline:
                    e = elites[random.randrange(min(len(elites), 10))][1]
                    mix = 0.70 + 0.25 * random.random()
                    cand = [mix * best_x[d] + (1.0 - mix) * e[d] for d in range(dim)]
                    for d in range(dim):
                        if spans[d] > 0:
                            cand[d] += random.gauss(0.0, 0.010 * spans[d] * (1.0 - frac))
                    cand = ensure_bounds(cand)
                    fc = safe_eval(cand)
                    push_elite(fc, cand)
                    if fc < fu:
                        uvec, fu = cand, fc

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

            # mid-run polish sometimes (but keep light)
            if gen % 3 == 0 and frac > 0.30:
                local_polish(frac, rounds=1)

        # stall restart
        if time.time() - last_improve_t > stall_restart_seconds:
            restart(frac)
            # refresh best from pop
            if pop:
                bi = min(range(len(pop)), key=lambda k: fit[k])
                if fit[bi] < best:
                    best, best_x = fit[bi], pop[bi][:]
                    last_improve_t = time.time()
            last_improve_t = time.time()
