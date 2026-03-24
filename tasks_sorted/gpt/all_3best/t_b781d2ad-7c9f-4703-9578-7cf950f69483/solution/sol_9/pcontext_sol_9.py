import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    What is improved vs the previous best (two-phase SHADE+Halton+kNN+simplex+1+1):
      1) Proper L-SHADE style population-size reduction (PSR):
         - start larger for exploration, shrink to a small pop for exploitation.
      2) Better surrogate usage without skipping evaluations (safer):
         - kNN is used to *rank extra proposals* (from several generators),
           but we still evaluate at least one trial per individual per gen when time allows.
      3) Eigenvector-free "rotated" local search (random orthonormal-ish directions):
         - a cheap direction-set pattern search around best that often beats simplex in
           medium/high dimensions.
      4) Smarter restarts:
         - if stuck, do a small "multi-start around elites" instead of full rebuild.
      5) Cleaner / faster inner loops and stricter duplicate suppression.

    Returns: best fitness (float).
    """

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---------------- helpers ----------------
    def clamp(x, a, b):
        return a if x < a else b if x > b else x

    def reflect(x, a, b):
        if a == b:
            return a
        # reflect repeatedly until inside
        while x < a or x > b:
            if x < a:
                x = a + (a - x)
            if x > b:
                x = b - (x - b)
        return a if x < a else b if x > b else x

    def randn():
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    inv_span = [(1.0 / s) if s > 1e-18 else 0.0 for s in span]

    def in_bounds_jitter(x, eps_scale=1e-12):
        y = x[:]
        for i in range(dim):
            if span[i] <= 0:
                y[i] = lo[i]
            else:
                eps = eps_scale * span[i]
                if y[i] <= lo[i]:
                    y[i] = lo[i] + eps
                elif y[i] >= hi[i]:
                    y[i] = hi[i] - eps
        return y

    def random_point():
        return [lo[i] if span[i] <= 0 else (lo[i] + span[i] * random.random()) for i in range(dim)]

    def opposite_point(x):
        return [lo[i] if span[i] <= 0 else (lo[i] + hi[i] - x[i]) for i in range(dim)]

    def sqdist_white(a, b):
        s = 0.0
        for i in range(dim):
            if span[i] <= 0:
                continue
            d = (a[i] - b[i]) * inv_span[i]
            s += d * d
        return s

    # Deb polynomial mutation (bounded)
    def poly_mutate(x, eta=16.0, pm=0.08):
        y = x[:]
        for i in range(dim):
            if span[i] <= 0:
                y[i] = lo[i]
                continue
            if random.random() > pm:
                continue
            xl, xu = lo[i], hi[i]
            xi = y[i]
            if xl == xu:
                y[i] = xl
                continue
            d1 = (xi - xl) / (xu - xl)
            d2 = (xu - xi) / (xu - xl)
            r = random.random()
            mp = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - d1
                val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta + 1.0))
                dq = (val ** mp) - 1.0
            else:
                xy = 1.0 - d2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta + 1.0))
                dq = 1.0 - (val ** mp)
            xi = xi + dq * (xu - xl)
            y[i] = reflect(xi, xl, xu)
        return in_bounds_jitter(y)

    # ---------------- Halton sequence ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(max(1, dim))

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    halton_index = 1
    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= 0:
                x[i] = lo[i]
            else:
                u = halton_value(idx, primes[i])
                x[i] = lo[i] + u * span[i]
        return x

    # ---------------- history + kNN predictor (whitened) ----------------
    HCAP = 40 + 7 * dim
    hist_x, hist_f = [], []

    def add_hist(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        if len(hist_f) > HCAP:
            # keep best ~1/3, drop one from the rest
            idx = sorted(range(len(hist_f)), key=lambda i: hist_f[i])
            keep_n = max(12, len(idx) // 3)
            keep = set(idx[:keep_n])
            drop = [i for i in range(len(hist_f)) if i not in keep]
            j = random.choice(drop) if drop else random.randrange(len(hist_f))
            hist_x.pop(j)
            hist_f.pop(j)

    def knn_predict(x, k=14):
        n = len(hist_f)
        if n == 0:
            return float("inf")
        k = min(k, n)
        d = [(sqdist_white(x, hist_x[i]), hist_f[i]) for i in range(n)]
        d.sort(key=lambda t: t[0])
        eps = 1e-12
        num = 0.0
        den = 0.0
        for j in range(k):
            d2, fj = d[j]
            w = 1.0 / (eps + d2)
            num += w * fj
            den += w
        return num / den if den > 0.0 else float("inf")

    # ---------------- duplicate suppression (quantized) ----------------
    seen_cap = 4000
    seen = {}
    seen_q = []

    def key_quant(x):
        parts = []
        for i in range(dim):
            if span[i] <= 0:
                parts.append(0)
            else:
                q = int((x[i] - lo[i]) / (span[i] + 1e-300) * 1e9)
                parts.append(q)
        return tuple(parts)

    def mark_seen(x):
        k = key_quant(x)
        if k in seen:
            return False
        seen[k] = 1
        seen_q.append(k)
        if len(seen_q) > seen_cap:
            old = seen_q.pop(0)
            seen.pop(old, None)
        return True

    def too_close(a, b, thr=1e-14):
        return sqdist_white(a, b) <= thr

    # ---------------- local search: rotated direction pattern search ----------------
    def make_random_dirs(m=8):
        # Build a small set of random directions; normalize in whitened scale
        dirs = []
        for _ in range(m):
            v = [0.0] * dim
            s2 = 0.0
            for i in range(dim):
                if span[i] <= 0:
                    v[i] = 0.0
                else:
                    r = randn()
                    v[i] = r
                    s2 += r * r
            if s2 <= 1e-18:
                continue
            invn = 1.0 / math.sqrt(s2)
            for i in range(dim):
                v[i] *= invn
            dirs.append(v)
        if not dirs:
            # fallback: coordinate axes
            for i in range(dim):
                v = [0.0] * dim
                if span[i] > 0:
                    v[i] = 1.0
                    dirs.append(v)
        return dirs

    def dir_pattern_refine(x0, f0, rad, iters=10, ndirs=10):
        x = x0[:]
        f = f0
        # ensure steps nonzero
        step = [0.0] * dim
        for i in range(dim):
            step[i] = (max(rad[i], 1e-15 * span[i]) if span[i] > 0 else 0.0)

        dirs = make_random_dirs(ndirs)

        for _ in range(iters):
            if time.time() >= deadline:
                break
            improved = False
            # shuffle directions each pass
            random.shuffle(dirs)
            for dvec in dirs:
                if time.time() >= deadline:
                    break
                # try +/- along dvec with anisotropic step per coordinate
                for sgn in (1.0, -1.0):
                    cand = x[:]
                    changed = False
                    for i in range(dim):
                        if span[i] <= 0:
                            cand[i] = lo[i]
                        else:
                            di = dvec[i]
                            if di != 0.0:
                                cand[i] = reflect(cand[i] + sgn * di * step[i], lo[i], hi[i])
                                changed = True
                    if not changed:
                        continue
                    cand = in_bounds_jitter(cand)
                    if not mark_seen(cand):
                        continue
                    fc = safe_eval(cand)
                    add_hist(cand, fc)
                    if fc < f:
                        x, f = cand, fc
                        improved = True
                        break
                if improved and random.random() < 0.35:
                    # occasional second step in same direction (acceleration)
                    cand = x[:]
                    for i in range(dim):
                        if span[i] > 0:
                            cand[i] = reflect(cand[i] + 0.7 * (x[i] - x0[i]), lo[i], hi[i])
                    cand = in_bounds_jitter(cand)
                    if mark_seen(cand):
                        fc = safe_eval(cand)
                        add_hist(cand, fc)
                        if fc < f:
                            x, f = cand, fc
                if improved:
                    break

            # step update
            if improved:
                for i in range(dim):
                    if span[i] > 0:
                        step[i] *= 1.10
            else:
                for i in range(dim):
                    if span[i] > 0:
                        step[i] *= 0.55
                # stop when tiny
                small = True
                for i in range(dim):
                    if span[i] > 0 and step[i] > 1e-10 * span[i]:
                        small = False
                        break
                if small:
                    break
        return x, f

    # ---------------- init: stratified + halton + opposition ----------------
    pop_init = max(28, min(110, 14 * dim))
    pop_min = max(8, min(26, 3 * dim))

    pop, fit = [], []
    best = float("inf")
    best_x = None

    strata = []
    for i in range(dim):
        perm = list(range(pop_init))
        random.shuffle(perm)
        strata.append(perm)

    for k in range(pop_init):
        if time.time() >= deadline:
            return best
        if k % 2 == 0:
            x = [0.0] * dim
            for i in range(dim):
                if span[i] <= 0:
                    x[i] = lo[i]
                else:
                    u = (strata[i][k] + random.random()) / pop_init
                    x[i] = lo[i] + u * span[i]
        else:
            x = halton_point()

        fx = safe_eval(x)
        xo = opposite_point(x)
        fxo = safe_eval(xo)
        if fxo < fx:
            x, fx = xo, fxo

        x = in_bounds_jitter(x)
        pop.append(x)
        fit.append(fx)
        add_hist(x, fx)
        mark_seen(x)

        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)
        add_hist(best_x, best)
        mark_seen(best_x)

    # ---------------- SHADE memories ----------------
    Hm = 12
    MCR = [0.5] * Hm
    MF = [0.7] * Hm
    mem_k = 0

    archive = []
    archive_max = pop_init

    base_rad = [0.10 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0

    # ---------------- main loop ----------------
    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # L-SHADE population size reduction
        cur_n = len(pop)
        target_n = int(round(pop_init - (pop_init - pop_min) * tfrac))
        if target_n < pop_min:
            target_n = pop_min
        if target_n < cur_n:
            # remove worst individuals
            order = sorted(range(cur_n), key=lambda i: fit[i])
            keep = order[:target_n]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            cur_n = target_n
            archive_max = cur_n

        # pbest fraction: larger early, smaller late
        pfrac = clamp(0.30 - 0.20 * tfrac, 2.0 / cur_n, 0.40)
        order = sorted(range(cur_n), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(pfrac * cur_n)))
        pbest_set = order[:pcount]

        # periodic local refinement (time-aware)
        if (gen % (7 if tfrac < 0.6 else 4) == 0) and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_rad]

            # 1 eval: kNN-ranked probe near best (safe)
            if len(hist_f) >= max(12, 2 * dim) and time.time() < deadline:
                best_c = None
                best_p = float("inf")
                tries = 18 if tfrac > 0.5 else 12
                for _ in range(tries):
                    x = best_x[:]
                    for i in range(dim):
                        if span[i] > 0 and random.random() < 0.6:
                            x[i] = reflect(x[i] + randn() * rad[i], lo[i], hi[i])
                    x = in_bounds_jitter(x)
                    pv = knn_predict(x) + 0.0008 * sqdist_white(x, best_x)
                    if pv < best_p:
                        best_p = pv
                        best_c = x
                if best_c is not None and mark_seen(best_c):
                    fc = safe_eval(best_c)
                    add_hist(best_c, fc)
                    if fc < best:
                        best, best_x = fc, best_c[:]
                        last_best = best
                        last_improve_time = time.time()
                        no_improve_gens = 0

            # rotated direction search refine
            if time.time() < deadline:
                xr, fr = dir_pattern_refine(best_x, best, rad, iters=(8 if dim > 30 else 10), ndirs=(10 if dim <= 30 else 8))
                if fr < best:
                    best, best_x = fr, xr[:]
                    last_best = best
                    last_improve_time = time.time()
                    no_improve_gens = 0

        # success lists for SHADE update
        SCR, SF, dF = [], [], []

        # build extra proposals around best to occasionally replace a trial if it looks better
        have_sur = (len(hist_f) >= max(15, 2 * dim))

        for i in range(cur_n):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(Hm)
            mu_cr = MCR[r]
            mu_f = MF[r]

            cr = clamp(mu_cr + 0.10 * randn(), 0.0, 1.0)

            # Cauchy for F (resample)
            F = -1.0
            for _ in range(10):
                u = random.random()
                F = mu_f + 0.12 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            if no_improve_gens >= 6:
                F = max(F, 0.62)
                cr = max(cr, 0.55)
            F = clamp(F, 0.05, 1.0)

            # choose strategy: current-to-pbest/1 mostly, rand/1 sometimes (esp early)
            use_pb = (random.random() < (0.70 + 0.20 * tfrac))

            if use_pb:
                pbest_idx = random.choice(pbest_set)
                xpbest = pop[pbest_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(cur_n)

                pool_n = cur_n + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_n)

                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < cur_n else archive[r2 - cur_n]

                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])
            else:
                r0 = i
                while r0 == i:
                    r0 = random.randrange(cur_n)
                r1 = r0
                while r1 == r0 or r1 == i:
                    r1 = random.randrange(cur_n)

                pool_n = cur_n + len(archive)
                r2 = r1
                while r2 == r0 or r2 == r1 or r2 == i:
                    r2 = random.randrange(pool_n)

                xr0 = pop[r0]
                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < cur_n else archive[r2 - cur_n]

                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = xr0[d] + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])

            # crossover
            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if span[d] <= 0:
                    uvec[d] = lo[d]
                else:
                    if random.random() < cr or d == jrand:
                        uvec[d] = v[d]
            uvec = in_bounds_jitter(uvec)

            # avoid near-duplicates
            if too_close(uvec, xi, 1e-14) or too_close(uvec, best_x, 1e-14):
                for d in range(dim):
                    if span[d] > 0 and random.random() < 0.15:
                        uvec[d] = reflect(uvec[d] + randn() * (1e-6 * span[d]), lo[d], hi[d])
                uvec = in_bounds_jitter(uvec)

            # occasional kick
            if random.random() < (0.05 + (0.06 if no_improve_gens >= 6 else 0.0) + (0.03 if tfrac > 0.55 else 0.0)):
                pm = 1.0 / max(1, dim)
                uvec = poly_mutate(uvec, eta=16.0, pm=min(0.35, 6.0 * pm))

            # optional: replace uvec by a surrogate-ranked alternative probe (still evaluate only one)
            if have_sur and random.random() < 0.22 and time.time() < deadline:
                rad = base_rad
                cand_list = [uvec]
                for _ in range(3):
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0 and random.random() < 0.55:
                            x[d] = reflect(x[d] + randn() * rad[d], lo[d], hi[d])
                    x = in_bounds_jitter(x)
                    cand_list.append(x)
                # choose best predicted
                best_pred = float("inf")
                best_alt = uvec
                for c in cand_list:
                    p = knn_predict(c) + 0.0007 * sqdist_white(c, best_x)
                    if p < best_pred:
                        best_pred = p
                        best_alt = c
                uvec = best_alt

            if not mark_seen(uvec):
                # ensure we still evaluate something (tiny jitter)
                uvec = uvec[:]
                for d in range(dim):
                    if span[d] > 0 and random.random() < 0.10:
                        uvec[d] = reflect(uvec[d] + randn() * (1e-7 * span[d]), lo[d], hi[d])
                uvec = in_bounds_jitter(uvec)
                mark_seen(uvec)

            fu = safe_eval(uvec)
            add_hist(uvec, fu)

            # selection
            if fu <= fi:
                # archive update
                if len(archive) < archive_max:
                    archive.append(xi[:])
                elif archive_max > 0:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                imp = (fi - fu) if (fi < float("inf") and fu < float("inf")) else 1.0
                if imp < 0.0:
                    imp = 0.0
                SCR.append(cr)
                SF.append(F)
                dF.append(imp)

                if fu < best:
                    best, best_x = fu, uvec[:]

        # update SHADE memories
        if SCR:
            s_imp = sum(dF)
            if s_imp <= 0.0:
                wts = [1.0 / len(dF)] * len(dF)
            else:
                wts = [di / s_imp for di in dF]

            mean_cr = 0.0
            for wi, cri in zip(wts, SCR):
                mean_cr += wi * cri

            num = 0.0
            den = 0.0
            for wi, Fi_ in zip(wts, SF):
                num += wi * Fi_ * Fi_
                den += wi * Fi_
            mean_f = (num / den) if den > 0.0 else MF[mem_k]

            MCR[mem_k] = clamp(mean_cr, 0.0, 1.0)
            MF[mem_k] = clamp(mean_f, 0.05, 1.0)
            mem_k = (mem_k + 1) % Hm

        # stagnation / radius schedule
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = max(1e-15, base_rad[d] * (0.92 if tfrac < 0.6 else 0.88))
        else:
            no_improve_gens += 1
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min((0.30 if tfrac < 0.6 else 0.22) * span[d], base_rad[d] * 1.03)

        # multi-start around elites if stuck
        stuck_time = time.time() - last_improve_time
        if (no_improve_gens >= 14) or (stuck_time > 0.50 * max_time):
            if time.time() >= deadline:
                return best

            cur_n = len(pop)
            order = sorted(range(cur_n), key=lambda i: fit[i])
            keep = max(3, cur_n // 4)
            elites = order[:keep]

            # rebuild remainder from: (a) best kicks, (b) halton, (c) random
            new_pop = [pop[j][:] for j in elites]
            new_fit = [fit[j] for j in elites]

            archive = []
            # diversify seen a bit
            # (do not clear seen entirely; keep duplicate guard effective)

            while len(new_pop) < cur_n and time.time() < deadline:
                r = random.random()
                if r < 0.45:
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0:
                            g1 = randn()
                            g2 = abs(randn()) + 1e-12
                            step = (g1 / g2) * base_rad[d] * (1.5 + 0.7 * random.random())
                            x[d] = reflect(x[d] + step, lo[d], hi[d])
                    x = in_bounds_jitter(x)
                elif r < 0.75:
                    x = halton_point()
                    x = in_bounds_jitter(x)
                else:
                    x = random_point()
                    x = in_bounds_jitter(x)

                if not mark_seen(x):
                    continue
                fx = safe_eval(x)
                add_hist(x, fx)
                new_pop.append(x)
                new_fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()
                    no_improve_gens = 0

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            last_improve_time = time.time()
            last_best = best

    return best
