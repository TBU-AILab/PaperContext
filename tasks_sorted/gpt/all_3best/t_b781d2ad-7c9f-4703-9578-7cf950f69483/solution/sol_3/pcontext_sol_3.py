import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs).

    Improved vs your best (DE/current-to-best + coord refine) by adding:
      - SHADE/JADE-style self-adaptation with success-history memories (F, CR)
      - current-to-pbest/1 mutation (less greedy than current-to-best; better on multimodal)
      - external archive (diversity)
      - strong but cheap bounded-polynomial mutation "kicks" when stagnating
      - small multi-start local refinement around elite (coordinate + parabolic probe)

    Returns: best fitness found (float).
    """

    # ---------------- time ----------------
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

    # Polynomial mutation (Deb) for bounded continuous variables
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
        return y

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    def random_point():
        return [randu(lo[i], hi[i]) if span[i] > 0 else lo[i] for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] if span[i] > 0 else lo[i] for i in range(dim)]

    # ---------------- initialization (stratified + opposition) ----------------
    # Keep moderate but slightly larger for adaptation stability
    pop_size = max(18, min(70, 10 * dim))

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

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # ---------------- SHADE-like success-history memories ----------------
    H = 10
    MCR = [0.5] * H
    MF = [0.7] * H
    k_mem = 0

    archive = []
    archive_max = pop_size

    # p-best fraction (dynamic-ish)
    p_min, p_max = 2.0 / pop_size, 0.25
    p = max(p_min, min(p_max, 0.15))

    # ---------------- local refinement (coordinate + parabolic probe) ----------------
    def local_refine(x0, f0, base_steps):
        x = x0[:]
        f = f0
        steps = base_steps[:]
        # very small budget
        for _pass in range(2):
            if time.time() >= deadline:
                break
            improved_any = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                if span[i] <= 0:
                    continue
                step = steps[i]
                if step <= 0:
                    continue

                # probe -step, +step
                x_m = x[:]
                x_p = x[:]
                x_m[i] = reflect(x[i] - step, lo[i], hi[i])
                x_p[i] = reflect(x[i] + step, lo[i], hi[i])
                f_m = safe_eval(x_m)
                f_p = safe_eval(x_p)

                # take best of three
                if f_m < f or f_p < f:
                    if f_m <= f_p:
                        x, f = x_m, f_m
                    else:
                        x, f = x_p, f_p
                    improved_any = True

                    # optional parabolic step using (x-step, x, x+step) if both sides are valid
                    # (only if it looks unimodal locally)
                    if time.time() < deadline and f_m < float("inf") and f_p < float("inf"):
                        # If current x is at center, approximate using the old center value:
                        # We'll estimate f_center by re-evaluating only sometimes.
                        if random.random() < 0.35:
                            f_c = safe_eval(x0 if _pass == 0 else x)  # cheap-ish heuristic
                        else:
                            f_c = f  # use current best as center surrogate

                        denom = (f_m - 2.0 * f_c + f_p)
                        if abs(denom) > 1e-18:
                            # vertex offset in units of step
                            t = 0.5 * (f_m - f_p) / denom
                            if abs(t) <= 2.5:
                                x_q = x[:]
                                # move from current coordinate value (not necessarily center) is tricky;
                                # use original center coordinate (x0[i]) as reference:
                                ref = x0[i]
                                x_q[i] = reflect(ref + t * step, lo[i], hi[i])
                                f_q = safe_eval(x_q)
                                if f_q < f:
                                    x, f = x_q, f_q
                                    improved_any = True

            # trust region update
            mul = 0.65 if not improved_any else 1.10
            for j in range(dim):
                steps[j] *= mul
        return x, f

    base_steps = [0.06 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    # ---------------- main loop ----------------
    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0

    while time.time() < deadline:
        gen += 1

        # occasional local refine
        if gen % 6 == 0 and time.time() < deadline:
            steps = [max(1e-15, s) for s in base_steps]
            x_lr, f_lr = local_refine(best_x, best, steps)
            if f_lr < best:
                best, best_x = f_lr, x_lr[:]
                last_best = best
                last_improve_time = time.time()
                no_improve_gens = 0

        # p-best set
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p * pop_size)))
        pbest_set = order[:pcount]

        SCR, SF, dF = [], [], []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # sample memory index
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # CR ~ N(mu, 0.1)
            cr = clamp(mu_cr + 0.1 * randn(), 0.0, 1.0)

            # F ~ Cauchy(mu, 0.1) with resampling
            F = -1.0
            for _ in range(8):
                u = random.random()
                F = mu_f + 0.1 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            F = clamp(F, 0.05, 1.0)

            # choose pbest
            pbest_idx = random.choice(pbest_set)
            xpbest = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from pop U archive distinct from i and r1
            pool_n = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pool_n)

            xr1 = pop[r1]
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if span[d] <= 0:
                    v[d] = lo[d]
                else:
                    vd = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
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

            fu = safe_eval(uvec)

            # selection
            if fu <= fi:
                # archive update
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                # success history
                imp = (fi - fu) if (fi < float("inf") and fu < float("inf")) else 1.0
                if imp < 0.0:
                    imp = 0.0
                SCR.append(cr)
                SF.append(F)
                dF.append(imp)

                if fu < best:
                    best, best_x = fu, uvec[:]

        # update memories (SHADE)
        if SCR:
            s = sum(dF)
            if s <= 0.0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / s for di in dF]

            # weighted mean for CR
            mean_cr = 0.0
            for wi, cri in zip(w, SCR):
                mean_cr += wi * cri

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            mean_f = (num / den) if den > 0.0 else MF[k_mem]

            MCR[k_mem] = clamp(mean_cr, 0.0, 1.0)
            MF[k_mem] = clamp(mean_f, 0.05, 1.0)
            k_mem = (k_mem + 1) % H

        # stagnation logic + step schedule
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            # more exploitation over time
            for d in range(dim):
                base_steps[d] = max(1e-15, base_steps[d] * 0.96) if span[d] > 0 else 0.0
        else:
            no_improve_gens += 1
            for d in range(dim):
                if span[d] > 0:
                    base_steps[d] = min(0.20 * span[d], base_steps[d] * 1.02)

        # "kick" or partial restart if stuck
        if (no_improve_gens >= 10) or ((time.time() - last_improve_time) > 0.35 * max_time):
            if time.time() >= deadline:
                return best

            # keep elites
            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(4, pop_size // 4)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]

            # clear archive (often helps)
            archive = []

            # refill with a mix of random/opposition and mutated-best samples
            while len(new_pop) < pop_size and time.time() < deadline:
                if random.random() < 0.55:
                    x = random_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                else:
                    # exploit: mutate best (bounded polynomial mutation)
                    pm = 1.0 / max(1, dim)
                    x = poly_mutate(best_x, eta=15.0, pm=min(0.5, 6.0 * pm))
                    fx = safe_eval(x)

                new_pop.append(x)
                new_fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            last_improve_time = time.time()
            last_best = best

    return best
