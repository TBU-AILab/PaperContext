import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    Main upgrade vs your current best (#2):
      - Keeps the same strong core (SHADE/ensemble DE + Halton + kNN + simplex),
        but adds an explicit *two-phase schedule*:
          Phase A (early): exploration-heavy DE with larger pbest-set + more rand/1.
          Phase B (late): exploitation-heavy around best with smaller pbest-set,
                          more current-to-pbest/1, and more frequent local search.
      - Adds *whitened distance* for kNN (dimension scaling by span) -> better proposals.
      - Adds *duplicate/too-close suppression* (cheap) to avoid wasting evaluations.
      - Adds *1+1 success-based step search* (very cheap) around best as a second local
        tool alongside simplex; helps on narrow curved valleys where simplex can stall.
      - Tightens restart logic: staged "reheat" first, then soft restart, keeping elites.

    Returns: best fitness found (float).
    """

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---------------- helpers ----------------
    def clamp(x, a, b):
        return a if x < a else b if x > b else x

    def reflect(x, a, b):
        if a == b:
            return a
        while x < a or x > b:
            if x < a:
                x = a + (a - x)
            if x > b:
                x = b - (x - b)
        return clamp(x, a, b)

    def randu(a, b):
        return a + (b - a) * random.random()

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
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
    inv_span = [0.0] * dim
    for i in range(dim):
        inv_span[i] = (1.0 / span[i]) if span[i] > 1e-18 else 0.0

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
        return [randu(lo[i], hi[i]) if span[i] > 0 else lo[i] for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] if span[i] > 0 else lo[i] for i in range(dim)]

    # Deb polynomial mutation (bounded)
    def poly_mutate(x, eta=18.0, pm=0.1):
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
            delta1 = (xi - xl) / (xu - xl)
            delta2 = (xu - xi) / (xu - xl)
            r = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - (val ** mut_pow)
            xi = xi + deltaq * (xu - xl)
            y[i] = reflect(xi, xl, xu)
        return in_bounds_jitter(y)

    # ---------------- Halton sequence ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
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

    # ---------------- history + kNN (whitened) ----------------
    HCAP = 30 + 6 * dim
    hist_x = []
    hist_f = []

    def sqdist_white(a, b):
        s = 0.0
        for i in range(dim):
            if span[i] <= 0:
                continue
            d = (a[i] - b[i]) * inv_span[i]
            s += d * d
        return s

    def add_hist(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        if len(hist_f) > HCAP:
            idx = sorted(range(len(hist_f)), key=lambda i: hist_f[i])
            keep = set(idx[:max(10, len(idx)//3)])
            drop = [i for i in range(len(hist_f)) if i not in keep]
            j = random.choice(drop) if drop else random.randrange(len(hist_f))
            hist_x.pop(j); hist_f.pop(j)

    def knn_predict(x, k=12):
        n = len(hist_f)
        if n == 0:
            return float("inf")
        k = min(k, n)
        dists = []
        for i in range(n):
            d2 = sqdist_white(x, hist_x[i])
            dists.append((d2, hist_f[i]))
        dists.sort(key=lambda t: t[0])
        eps = 1e-12
        num = 0.0
        den = 0.0
        for j in range(k):
            d2, fj = dists[j]
            w = 1.0 / (eps + d2)
            num += w * fj
            den += w
        return num / den if den > 0.0 else float("inf")

    # cheap "too-close" suppression to avoid duplicated evals
    def too_close(x, ref, thr=1e-10):
        # whitened squared distance
        return sqdist_white(x, ref) <= thr

    def propose_from_best(best_x, rad, tries=18):
        best_c = None
        best_p = float("inf")
        for _ in range(tries):
            x = best_x[:]
            for i in range(dim):
                if span[i] > 0 and random.random() < 0.60:
                    x[i] = reflect(x[i] + randn() * rad[i], lo[i], hi[i])
            x = in_bounds_jitter(x)
            pv = knn_predict(x)
            # mild exploration bias: prefer being not too close to already-evaluated best
            pv += 0.001 * sqdist_white(x, best_x)
            if pv < best_p:
                best_p = pv
                best_c = x
        return best_c

    # ---------------- simplex refine (as in your best, small budget) ----------------
    def simplex_refine(x0, f0, rad, max_steps=16):
        n = dim
        if n == 1:
            x = x0[:]
            f = f0
            step = rad[0] if span[0] > 0 else 0.0
            if step <= 0:
                return x, f
            for s in (step, -step, 2*step, -2*step):
                if time.time() >= deadline:
                    break
                c = x0[:]
                c[0] = reflect(c[0] + s, lo[0], hi[0])
                c = in_bounds_jitter(c)
                fc = safe_eval(c)
                add_hist(c, fc)
                if fc < f:
                    x, f = c, fc
            return x, f

        simplex = [x0[:]]
        fvals = [f0]
        for i in range(n):
            if time.time() >= deadline:
                return x0, f0
            x = x0[:]
            if span[i] > 0:
                x[i] = reflect(x[i] + (0.8 * rad[i] + 1e-15), lo[i], hi[i])
            x = in_bounds_jitter(x)
            fx = safe_eval(x)
            add_hist(x, fx)
            simplex.append(x); fvals.append(fx)

        alpha, gamma, rho, sig = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_steps):
            if time.time() >= deadline:
                break
            order = sorted(range(len(fvals)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]
            second_worstf = fvals[-2]

            centroid = [0.0] * n
            inv = 1.0 / n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += simplex[j][i]
                centroid[i] = s * inv

            xr = [0.0] * n
            for i in range(n):
                xr[i] = lo[i] if span[i] <= 0 else reflect(centroid[i] + alpha * (centroid[i] - worstx[i]), lo[i], hi[i])
            xr = in_bounds_jitter(xr)
            fr = safe_eval(xr)
            add_hist(xr, fr)

            if fr < bestf:
                xe = [0.0] * n
                for i in range(n):
                    xe[i] = lo[i] if span[i] <= 0 else reflect(centroid[i] + gamma * (xr[i] - centroid[i]), lo[i], hi[i])
                xe = in_bounds_jitter(xe)
                fe = safe_eval(xe)
                add_hist(xe, fe)
                simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < second_worstf:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = [0.0] * n
                for i in range(n):
                    xc[i] = lo[i] if span[i] <= 0 else reflect(centroid[i] + rho * (worstx[i] - centroid[i]), lo[i], hi[i])
                xc = in_bounds_jitter(xc)
                fc = safe_eval(xc)
                add_hist(xc, fc)
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    bx = simplex[0]
                    for j in range(1, len(simplex)):
                        if time.time() >= deadline:
                            break
                        xs = simplex[j][:]
                        for i in range(n):
                            xs[i] = lo[i] if span[i] <= 0 else reflect(bx[i] + sig * (xs[i] - bx[i]), lo[i], hi[i])
                        xs = in_bounds_jitter(xs)
                        fs = safe_eval(xs)
                        add_hist(xs, fs)
                        simplex[j], fvals[j] = xs, fs

        idx = min(range(len(fvals)), key=lambda i: fvals[i])
        return simplex[idx], fvals[idx]

    # ---------------- ultra-cheap 1+1 step search around best ----------------
    def one_plus_one(best_x, best_f, rad, iters=10):
        x = best_x[:]
        f = best_f
        # success-based scalar multiplier
        ss = 1.0
        for _ in range(iters):
            if time.time() >= deadline:
                break
            cand = x[:]
            # gaussian step in random subspace
            for i in range(dim):
                if span[i] > 0 and random.random() < 0.5:
                    cand[i] = reflect(cand[i] + randn() * rad[i] * ss, lo[i], hi[i])
            cand = in_bounds_jitter(cand)
            if too_close(cand, x, 1e-14):
                continue
            fc = safe_eval(cand)
            add_hist(cand, fc)
            if fc < f:
                x, f = cand, fc
                ss *= 1.25
            else:
                ss *= 0.82
            ss = clamp(ss, 0.05, 5.0)
        return x, f

    # ---------------- init: stratified + opposition + Halton ----------------
    pop_size = max(22, min(90, 12 * dim))
    elite_n = max(3, pop_size // 8)

    pop = []
    fit = []
    best = float("inf")
    best_x = None

    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    for k in range(pop_size):
        if time.time() >= deadline:
            return best
        if k % 2 == 0:
            x = [0.0] * dim
            for i in range(dim):
                if span[i] <= 0:
                    x[i] = lo[i]
                else:
                    u = (strata[i][k] + random.random()) / pop_size
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
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)
        add_hist(best_x, best)

    # ---------------- SHADE memories ----------------
    Hm = 12
    MCR = [0.5] * Hm
    MF = [0.7] * Hm
    mem_k = 0

    archive = []
    archive_max = pop_size

    succ_pb = 1.0
    succ_r1 = 1.0

    base_rad = [0.12 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0
    reheats = 0

    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # ---- 2-phase schedule ----
        # early -> more exploration; late -> more exploitation
        late = 1.0 if tfrac > 0.55 else 0.0
        # dynamic pbest fraction
        # larger early, smaller late
        pfrac = clamp(0.28 - 0.16 * tfrac, 2.0 / pop_size, 0.35)

        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(pfrac * pop_size)))
        pbest_set = order[:pcount]

        # periodic exploitation: (late -> more frequent)
        exploit_period = 4 if late else 6
        if gen % exploit_period == 0 and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_rad]

            if len(hist_f) >= max(12, 2 * dim):
                ps = propose_from_best(best_x, rad, tries=(16 if late else 12))
                if ps is not None and time.time() < deadline and not too_close(ps, best_x, 1e-14):
                    fps = safe_eval(ps)
                    add_hist(ps, fps)
                    if fps < best:
                        best, best_x = fps, ps[:]
                        last_best = best
                        last_improve_time = time.time()
                        no_improve_gens = 0

            # add 1+1 step search (super cheap)
            if time.time() < deadline:
                x11, f11 = one_plus_one(best_x, best, rad, iters=(12 if late else 7))
                if f11 < best:
                    best, best_x = f11, x11[:]
                    last_best = best
                    last_improve_time = time.time()
                    no_improve_gens = 0

            # simplex refine
            if time.time() < deadline:
                xr, fr = simplex_refine(best_x, best, rad, max_steps=(14 if dim <= 25 else 10))
                if fr < best:
                    best, best_x = fr, xr[:]
                    last_best = best
                    last_improve_time = time.time()
                    no_improve_gens = 0

        SCR, SF, dF = [], [], []

        # DE generation
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(Hm)
            mu_cr = MCR[r]
            mu_f = MF[r]

            cr = clamp(mu_cr + 0.10 * randn(), 0.0, 1.0)

            # Cauchy for F with resampling
            F = -1.0
            for _ in range(10):
                u = random.random()
                F = mu_f + 0.12 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            # stagnation -> stronger params
            if no_improve_gens >= 6:
                F = max(F, 0.62)
                cr = max(cr, 0.55)
            F = clamp(F, 0.05, 1.0)

            # strategy mixture:
            # early: more rand/1; late: more current-to-pbest/1
            prob_pb_adapt = succ_pb / (succ_pb + succ_r1 + 1e-12)
            prob_pb = clamp(0.25 + 0.55 * tfrac + 0.25 * (prob_pb_adapt - 0.5), 0.10, 0.95)
            use_pb = (random.random() < prob_pb)

            if use_pb:
                pbest_idx = random.choice(pbest_set)
                xpbest = pop[pbest_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                pool_n = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_n)

                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

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
                    r0 = random.randrange(pop_size)
                r1 = r0
                while r1 == r0 or r1 == i:
                    r1 = random.randrange(pop_size)
                pool_n = pop_size + len(archive)
                r2 = r1
                while r2 == r0 or r2 == r1 or r2 == i:
                    r2 = random.randrange(pool_n)

                xr0 = pop[r0]
                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = xr0[d] + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])

            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if span[d] <= 0:
                    uvec[d] = lo[d]
                else:
                    if random.random() < cr or d == jrand:
                        uvec[d] = v[d]
            uvec = in_bounds_jitter(uvec)

            # avoid wasting eval on near-duplicates of best/parent
            if too_close(uvec, xi, 1e-14) or too_close(uvec, best_x, 1e-14):
                # tiny jitter in a few dims
                for d in range(dim):
                    if span[d] > 0 and random.random() < 0.15:
                        uvec[d] = reflect(uvec[d] + randn() * (1e-6 * span[d]), lo[d], hi[d])
                uvec = in_bounds_jitter(uvec)

            # mutation kick sometimes
            kick_p = 0.05 + (0.06 if no_improve_gens >= 6 else 0.0) + (0.04 if late else 0.0)
            if random.random() < kick_p:
                pm = 1.0 / max(1, dim)
                uvec = poly_mutate(uvec, eta=16.0, pm=min(0.35, 6.0 * pm))

            fu = safe_eval(uvec)
            add_hist(uvec, fu)

            if fu <= fi:
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                if use_pb:
                    succ_pb += 1.0
                else:
                    succ_r1 += 1.0

                imp = (fi - fu) if (fi < float("inf") and fu < float("inf")) else 1.0
                if imp < 0.0:
                    imp = 0.0
                SCR.append(cr)
                SF.append(F)
                dF.append(imp)

                if fu < best:
                    best, best_x = fu, uvec[:]

        succ_pb *= 0.97
        succ_r1 *= 0.97
        succ_pb = max(0.2, succ_pb)
        succ_r1 = max(0.2, succ_r1)

        # update SHADE memories
        if SCR:
            s = sum(dF)
            wts = ([1.0 / len(dF)] * len(dF)) if s <= 0.0 else [di / s for di in dF]

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

        # stagnation control and trust radius schedule
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            reheats = 0
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = max(1e-15, base_rad[d] * (0.90 if late else 0.94))
        else:
            no_improve_gens += 1
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min((0.35 if not late else 0.22) * span[d], base_rad[d] * 1.04)

        # staged stagnation handling:
        #  (1) reheat once: inflate radius + clear archive, keep pop
        #  (2) then soft restart: keep elites and refill
        stuck_time = (time.time() - last_improve_time)
        if (no_improve_gens >= 10 and reheats == 0) and time.time() < deadline:
            reheats = 1
            archive = []
            # inflate local radii a bit
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min(0.35 * span[d], max(base_rad[d], (0.18 if not late else 0.12) * span[d]))
            no_improve_gens = 6  # don't instantly restart

        if (no_improve_gens >= 16) or (stuck_time > 0.48 * max_time):
            if time.time() >= deadline:
                return best

            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(elite_n, pop_size // 4)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]
            archive = []

            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.32:
                    x = halton_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                elif r < 0.62:
                    x = random_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                else:
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0:
                            g1 = randn()
                            g2 = abs(randn()) + 1e-12
                            step = (g1 / g2) * base_rad[d] * (1.2 + 0.8 * random.random())
                            x[d] = reflect(x[d] + step, lo[d], hi[d])
                    x = in_bounds_jitter(x)
                    fx = safe_eval(x)

                new_pop.append(x); new_fit.append(fx)
                add_hist(x, fx)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            reheats = 0
            last_improve_time = time.time()
            last_best = best

    return best
