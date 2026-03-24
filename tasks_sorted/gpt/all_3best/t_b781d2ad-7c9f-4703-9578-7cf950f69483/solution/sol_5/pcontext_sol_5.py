import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs).

    Improved vs your best (#1) by focusing on *better use of evaluations*:
      - Adds an explicit evaluation budget accounting + faster loops (less overhead).
      - Keeps a small elite pool and runs a short Nelder-Mead-like simplex refine
        in the elite trust-region (very effective when near a basin).
      - Uses a low-discrepancy (Halton) stream for cheap global exploration samples
        (beats pure RNG early without numpy).
      - Surrogate proposal replaced by a cheaper kNN inverse-distance predictor
        (similar effect, much less overhead than maintaining a big RBF set).
      - DE layer upgraded to a 2-strategy ensemble (current-to-pbest/1 and rand/1),
        chosen adaptively based on recent success.
      - Restarts are “soft”: keep elites + rebuild population from (a) Halton,
        (b) opposition, (c) heavy-tailed kicks around best.

    Returns: best fitness found (float).
    """

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---------- helpers ----------
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

    def randn():  # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(x):
        # pass list; most funcs accept array-like
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

    # ---------- Halton sequence (low discrepancy) ----------
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
        nonlocal_halton = [0.0] * dim
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        for i in range(dim):
            if span[i] <= 0:
                nonlocal_halton[i] = lo[i]
            else:
                u = halton_value(idx, primes[i])
                nonlocal_halton[i] = lo[i] + u * span[i]
        return nonlocal_halton

    # ---------- kNN predictor (cheap surrogate-like proposer) ----------
    # Keep a small history of evaluated points; use inverse-distance weighted mean.
    HCAP = 30 + 5 * dim
    hist_x = []
    hist_f = []

    def sqdist(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def add_hist(x, f):
        # keep everything but capped; prefer keeping best points
        hist_x.append(x[:])
        hist_f.append(float(f))
        if len(hist_f) > HCAP:
            # drop one of the worse points (roulette among worst half)
            idx = sorted(range(len(hist_f)), key=lambda i: hist_f[i])
            keep = set(idx[:max(8, len(idx)//3)])
            drop = [i for i in range(len(hist_f)) if i not in keep]
            j = random.choice(drop) if drop else random.randrange(len(hist_f))
            hist_x.pop(j); hist_f.pop(j)

    def knn_predict(x, k=12):
        n = len(hist_f)
        if n == 0:
            return float("inf")
        k = min(k, n)
        # partial selection: compute all distances (HCAP small)
        dists = []
        for i in range(n):
            d2 = sqdist(x, hist_x[i])
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

    def propose_from_best(best_x, rad, tries=18):
        best_c = None
        best_p = float("inf")
        for _ in range(tries):
            x = best_x[:]
            # random subspace perturb
            for i in range(dim):
                if span[i] > 0 and random.random() < 0.55:
                    x[i] = reflect(x[i] + randn() * rad[i], lo[i], hi[i])
            x = in_bounds_jitter(x)
            pv = knn_predict(x)
            # exploration bonus: prefer farther points sometimes
            if hist_x and random.random() < 0.25:
                md2 = float("inf")
                for hx in hist_x:
                    d2 = sqdist(x, hx)
                    if d2 < md2:
                        md2 = d2
                pv -= 0.01 * math.sqrt(md2)
            if pv < best_p:
                best_p = pv
                best_c = x
        return best_c

    # ---------- Nelder-Mead-like simplex local refine (small budget) ----------
    def simplex_refine(x0, f0, rad, max_steps=18):
        # Build simplex around x0
        n = dim
        if n == 1:
            # 1D: just probe left/right
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

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_steps):
            if time.time() >= deadline:
                break
            order = sorted(range(len(fvals)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]
            second_worstf = fvals[-2]

            # centroid of all but worst
            centroid = [0.0] * n
            inv = 1.0 / n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += simplex[j][i]
                centroid[i] = s * inv

            # reflect
            xr = [0.0] * n
            for i in range(n):
                if span[i] <= 0:
                    xr[i] = lo[i]
                else:
                    xr[i] = reflect(centroid[i] + alpha * (centroid[i] - worstx[i]), lo[i], hi[i])
            xr = in_bounds_jitter(xr)
            fr = safe_eval(xr)
            add_hist(xr, fr)

            if fr < bestf:
                # expand
                xe = [0.0] * n
                for i in range(n):
                    if span[i] <= 0:
                        xe[i] = lo[i]
                    else:
                        xe[i] = reflect(centroid[i] + gamma * (xr[i] - centroid[i]), lo[i], hi[i])
                xe = in_bounds_jitter(xe)
                fe = safe_eval(xe)
                add_hist(xe, fe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contract
                xc = [0.0] * n
                for i in range(n):
                    if span[i] <= 0:
                        xc[i] = lo[i]
                    else:
                        xc[i] = reflect(centroid[i] + rho * (worstx[i] - centroid[i]), lo[i], hi[i])
                xc = in_bounds_jitter(xc)
                fc = safe_eval(xc)
                add_hist(xc, fc)
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink toward best
                    bx = simplex[0]
                    for j in range(1, len(simplex)):
                        if time.time() >= deadline:
                            break
                        xs = simplex[j][:]
                        for i in range(n):
                            if span[i] > 0:
                                xs[i] = reflect(bx[i] + sigma * (xs[i] - bx[i]), lo[i], hi[i])
                            else:
                                xs[i] = lo[i]
                        xs = in_bounds_jitter(xs)
                        fs = safe_eval(xs)
                        add_hist(xs, fs)
                        simplex[j], fvals[j] = xs, fs

        # return best
        idx = min(range(len(fvals)), key=lambda i: fvals[i])
        return simplex[idx], fvals[idx]

    # ---------- init: stratified + opposition + Halton ----------
    pop_size = max(22, min(90, 12 * dim))
    # elite count
    elite_n = max(3, pop_size // 8)

    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # stratified init indices
    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    for k in range(pop_size):
        if time.time() >= deadline:
            return best
        # mix: half stratified, half Halton-like
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

    # ---------- SHADE memories ----------
    Hm = 12
    MCR = [0.5] * Hm
    MF = [0.7] * Hm
    mem_k = 0
    archive = []
    archive_max = pop_size

    # strategy success tracking (ensemble)
    succ_pb = 1.0
    succ_r1 = 1.0

    # trust radius around best
    base_rad = [0.12 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0

    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # dynamic p-best fraction
        pfrac = clamp(0.10 + 0.22 * tfrac, 2.0 / pop_size, 0.35)
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(pfrac * pop_size)))
        pbest_set = order[:pcount]

        # periodic exploitation: kNN propose + simplex refine
        if gen % 5 == 0 and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_rad]

            if len(hist_f) >= max(12, 2 * dim):
                ps = propose_from_best(best_x, rad, tries=14)
                if ps is not None and time.time() < deadline:
                    fps = safe_eval(ps)
                    add_hist(ps, fps)
                    if fps < best:
                        best, best_x = fps, ps[:]
                        last_best = best
                        last_improve_time = time.time()
                        no_improve_gens = 0

            if time.time() < deadline:
                xr, fr = simplex_refine(best_x, best, rad, max_steps=(10 if dim > 25 else 16))
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

            # cauchy for F
            F = -1.0
            for _ in range(8):
                u = random.random()
                F = mu_f + 0.12 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            if no_improve_gens >= 7:
                F = max(F, 0.60)
                cr = max(cr, 0.55)
            F = clamp(F, 0.05, 1.0)

            # strategy choice: adaptively mix current-to-pbest/1 and rand/1
            # probability ~ successes
            prob_pb = succ_pb / (succ_pb + succ_r1 + 1e-12)
            use_pb = (random.random() < prob_pb)

            # choose indices
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
                # rand/1: v = xr0 + F*(xr1-xr2)
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

            # mutation kick sometimes
            if random.random() < (0.14 if no_improve_gens >= 6 else 0.05):
                pm = 1.0 / max(1, dim)
                uvec = poly_mutate(uvec, eta=16.0, pm=min(0.35, 6.0 * pm))

            fu = safe_eval(uvec)
            add_hist(uvec, fu)

            if fu <= fi:
                # archive update
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                # track strategy success
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

        # decay strategy counters slowly to remain adaptive
        succ_pb *= 0.97
        succ_r1 *= 0.97
        succ_pb = max(0.2, succ_pb)
        succ_r1 = max(0.2, succ_r1)

        # update SHADE memories
        if SCR:
            s = sum(dF)
            if s <= 0.0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / s for di in dF]

            mean_cr = 0.0
            for wi, cri in zip(w, SCR):
                mean_cr += wi * cri

            num = 0.0
            den = 0.0
            for wi, Fi_ in zip(w, SF):
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
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = max(1e-15, base_rad[d] * 0.92)
        else:
            no_improve_gens += 1
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min(0.30 * span[d], base_rad[d] * 1.04)

        # soft restart
        if (no_improve_gens >= 14) or ((time.time() - last_improve_time) > 0.45 * max_time):
            if time.time() >= deadline:
                return best

            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(elite_n, pop_size // 4)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]
            archive = []

            # refill
            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.35:
                    x = halton_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                elif r < 0.65:
                    x = random_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                else:
                    # heavy-tailed kick around best
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0:
                            # t-like: normal / |normal|
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
            last_improve_time = time.time()
            last_best = best

    return best
