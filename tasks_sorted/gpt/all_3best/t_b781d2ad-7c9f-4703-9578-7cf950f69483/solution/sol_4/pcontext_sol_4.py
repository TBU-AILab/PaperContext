import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs).

    Improvement over your best SHADE/JADE-DE hybrid:
      - Adds a lightweight surrogate-assisted layer (RBF on best-so-far region)
        to propose candidates cheaply when evaluations are expensive.
      - Uses a dual-population schedule: global LS/opposition exploration early,
        then focuses on a shrinking "elite trust region".
      - Upgrades local refinement to a small-budget Powell-like directional search
        (derivative-free) with adaptive step + occasional random subspace search.
      - More robust constraint handling via reflection + small in-bound jitter.
      - Stagnation handling: staged reheating (increase diversity) before restart.

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
        # reflect repeatedly until inside
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

    # avoid exact boundary stickiness
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

    # polynomial mutation (Deb)
    def poly_mutate(x, eta=20.0, pm=0.1):
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

    # ---------------- init: stratified + opposition ----------------
    pop_size = max(20, min(80, 12 * dim))

    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    pop, fit = [], []
    best = float("inf")
    best_x = None

    for k in range(pop_size):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= 0:
                x[i] = lo[i]
            else:
                u = (strata[i][k] + random.random()) / pop_size
                x[i] = lo[i] + u * span[i]

        fx = safe_eval(x)
        xo = opposite_point(x)
        fxo = safe_eval(xo)
        if fxo < fx:
            x, fx = xo, fxo

        x = in_bounds_jitter(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # ---------------- SHADE memories ----------------
    H = 12
    MCR = [0.5] * H
    MF = [0.7] * H
    k_mem = 0

    archive = []
    archive_max = pop_size

    # p-best fraction
    p = max(2.0 / pop_size, min(0.30, 0.12))

    # ---------------- tiny RBF surrogate (for proposing points) ----------------
    # We keep a small dataset of best points; surrogate used only to *propose*.
    Xs = []  # list of vectors
    Ys = []  # list of fitness

    def add_data(x, y):
        # keep unique-ish and bounded size
        Xs.append(x[:])
        Ys.append(float(y))
        # cap
        cap = 40 + 6 * dim
        if len(Xs) > cap:
            # drop a random non-elite; keep best few always
            idx = sorted(range(len(Ys)), key=lambda i: Ys[i])
            keep = set(idx[:max(6, cap // 6)])
            drop_candidates = [i for i in range(len(Ys)) if i not in keep]
            if drop_candidates:
                j = random.choice(drop_candidates)
            else:
                j = random.randrange(len(Ys))
            Xs.pop(j)
            Ys.pop(j)

    # seed surrogate data from initial population (top few)
    order0 = sorted(range(pop_size), key=lambda i: fit[i])
    for j in order0[:min(pop_size, max(10, pop_size // 2))]:
        if fit[j] < float("inf"):
            add_data(pop[j], fit[j])

    def sqdist(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def rbf_predict(x):
        # inverse-quadratic RBF: w_i = 1/(eps + ||x-xi||^2)
        # return weighted average. If no data, return +inf.
        n = len(Xs)
        if n == 0:
            return float("inf")
        eps = 1e-12
        num = 0.0
        den = 0.0
        for i in range(n):
            d2 = sqdist(x, Xs[i])
            w = 1.0 / (eps + d2)
            num += w * Ys[i]
            den += w
        return num / den if den > 0.0 else float("inf")

    def propose_surrogate(elite_center, rad, tries=24):
        # sample a handful of candidates in trust region; pick best predicted
        best_c = None
        best_p = float("inf")
        for _ in range(tries):
            x = elite_center[:]
            # random subspace
            for i in range(dim):
                if span[i] <= 0:
                    x[i] = lo[i]
                else:
                    if random.random() < 0.5:
                        x[i] = reflect(x[i] + randn() * rad[i], lo[i], hi[i])
            x = in_bounds_jitter(x)
            pval = rbf_predict(x)
            # add exploration bonus: prefer far from known points sometimes
            if Xs and random.random() < 0.35:
                # distance to nearest data
                md2 = float("inf")
                for xi in Xs:
                    d2 = sqdist(x, xi)
                    if d2 < md2:
                        md2 = d2
                pval -= 0.02 * math.sqrt(md2)  # tiny
            if pval < best_p:
                best_p = pval
                best_c = x
        return best_c

    # ---------------- local refine: Powell-like small budget ----------------
    def local_refine(x0, f0, rad):
        x = x0[:]
        f = f0

        # initial directions: coordinate basis
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = 1.0
            dirs.append(d)

        def line_search(x_base, f_base, dvec, step0):
            xb = x_base[:]
            fb = f_base
            step = step0
            # try +/- and then expand/shrink a bit (very small budget)
            bestx = xb
            bestf = fb

            for sgn in (1.0, -1.0):
                if time.time() >= deadline:
                    break
                cand = xb[:]
                for i in range(dim):
                    if span[i] <= 0:
                        cand[i] = lo[i]
                    else:
                        cand[i] = reflect(cand[i] + sgn * step * dvec[i], lo[i], hi[i])
                cand = in_bounds_jitter(cand)
                fc = safe_eval(cand)
                if fc < bestf:
                    bestx, bestf = cand, fc

            # if improved, try a slightly larger step in same direction
            if bestf < fb and time.time() < deadline:
                for mul in (1.6, 2.2):
                    cand = bestx[:]
                    for i in range(dim):
                        if span[i] > 0:
                            cand[i] = reflect(cand[i] + (mul - 1.0) * (bestx[i] - xb[i]), lo[i], hi[i])
                    cand = in_bounds_jitter(cand)
                    fc = safe_eval(cand)
                    if fc < bestf:
                        bestx, bestf = cand, fc
                    else:
                        break
            return bestx, bestf

        # budgeted passes
        for _pass in range(2):
            if time.time() >= deadline:
                break
            x_start = x[:]
            f_start = f

            for dvec in dirs:
                if time.time() >= deadline:
                    break
                # step size from rad projected on direction
                step0 = 0.0
                for i in range(dim):
                    step0 += abs(dvec[i]) * rad[i]
                step0 = max(1e-15, step0)
                x, f = line_search(x, f, dvec, step0)

            # new direction: overall displacement
            disp = [x[i] - x_start[i] for i in range(dim)]
            disp_norm = math.sqrt(sum(v * v for v in disp))
            if disp_norm > 0:
                for i in range(dim):
                    disp[i] /= disp_norm
                dirs = dirs[1:] + [disp]

            # adjust radius
            if f < f_start:
                for i in range(dim):
                    rad[i] = max(1e-15, rad[i] * 1.15)
            else:
                for i in range(dim):
                    rad[i] = max(1e-15, rad[i] * 0.6)

            # occasional random subspace poke
            if time.time() < deadline and random.random() < 0.35:
                cand = x[:]
                for i in range(dim):
                    if span[i] > 0 and random.random() < 0.35:
                        cand[i] = reflect(cand[i] + randn() * rad[i], lo[i], hi[i])
                cand = in_bounds_jitter(cand)
                fc = safe_eval(cand)
                if fc < f:
                    x, f = cand, fc

        return x, f

    # trust region radius around best (shrinks over time)
    base_rad = [0.10 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0

    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # dynamic p: more exploitation later
        p_dyn = clamp(p + 0.20 * tfrac, 2.0 / pop_size, 0.35)

        # periodic local refine + surrogate proposal (cheap guiding)
        if gen % 5 == 0 and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_rad]

            # surrogate proposal first (1 eval)
            if len(Xs) >= max(10, 2 * dim) and time.time() < deadline:
                ps = propose_surrogate(best_x, rad, tries=18)
                if ps is not None:
                    fps = safe_eval(ps)
                    add_data(ps, fps)
                    if fps < best:
                        best, best_x = fps, ps[:]
                        last_best = best
                        last_improve_time = time.time()
                        no_improve_gens = 0

            # local refine (few evals)
            x_lr, f_lr = local_refine(best_x, best, rad)
            if f_lr < best:
                best, best_x = f_lr, x_lr[:]
                last_best = best
                last_improve_time = time.time()
                no_improve_gens = 0
                add_data(best_x, best)

        # p-best set
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p_dyn * pop_size)))
        pbest_set = order[:pcount]

        SCR, SF, dF = [], [], []

        # DE generation
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # memory sample
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            cr = clamp(mu_cr + 0.10 * randn(), 0.0, 1.0)

            # cauchy F with resampling
            F = -1.0
            for _ in range(10):
                u = random.random()
                F = mu_f + 0.12 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            # slight "reheating" if stagnating
            if no_improve_gens >= 8:
                F = max(F, 0.65)
                cr = max(cr, 0.55)
            F = clamp(F, 0.05, 1.0)

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

            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if span[d] <= 0:
                    uvec[d] = lo[d]
                else:
                    if random.random() < cr or d == jrand:
                        uvec[d] = v[d]
            uvec = in_bounds_jitter(uvec)

            # occasional polynomial mutation on trial (small prob; helps separable + rugged)
            if random.random() < (0.15 if no_improve_gens >= 6 else 0.06):
                pm = 1.0 / max(1, dim)
                uvec = poly_mutate(uvec, eta=18.0, pm=min(0.35, 5.0 * pm))

            fu = safe_eval(uvec)
            add_data(uvec, fu)

            if fu <= fi:
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
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

        # SHADE memory update
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
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            mean_f = (num / den) if den > 0.0 else MF[k_mem]

            MCR[k_mem] = clamp(mean_cr, 0.0, 1.0)
            MF[k_mem] = clamp(mean_f, 0.05, 1.0)
            k_mem = (k_mem + 1) % H

        # stagnation / radius schedule
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            # focus
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = max(1e-15, base_rad[d] * 0.93)
        else:
            no_improve_gens += 1
            # broaden a bit
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min(0.25 * span[d], base_rad[d] * 1.03)

        # staged restart if stuck: first "reheat", then partial restart
        if (no_improve_gens >= 14) or ((time.time() - last_improve_time) > 0.42 * max_time):
            if time.time() >= deadline:
                return best

            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(5, pop_size // 4)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]

            archive = []

            # refill mix: random/opposition, mutated best, and trust-region samples
            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.40:
                    x = random_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                elif r < 0.70:
                    pm = 1.0 / max(1, dim)
                    x = poly_mutate(best_x, eta=14.0, pm=min(0.55, 7.0 * pm))
                    fx = safe_eval(x)
                else:
                    # trust region around best
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0:
                            x[d] = reflect(x[d] + randn() * base_rad[d], lo[d], hi[d])
                    x = in_bounds_jitter(x)
                    fx = safe_eval(x)

                new_pop.append(x)
                new_fit.append(fx)
                add_data(x, fx)

                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            last_improve_time = time.time()
            last_best = best

    return best
