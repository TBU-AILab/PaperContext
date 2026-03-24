import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs).

    Hybrid optimizer:
      1) LHS-like stratified init + oppositional sampling
      2) JADE-like Differential Evolution (current-to-pbest/1 with archive)
         - per-individual self-adaptation of F and CR
         - p-best selection for balanced exploration/exploitation
      3) Lightweight local search around the best (SPSA-style 2-point gradient + line tries)
      4) Stagnation-triggered partial restarts

    Returns: best fitness found (float)
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
        # reflect until inside
        while x < a or x > b:
            if x < a:
                x = a + (a - x)
            if x > b:
                x = b - (x - b)
        return clamp(x, a, b)

    def randu(a, b):
        return a + (b - a) * random.random()

    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-12)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---------------- degenerate ----------------
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
    # Keep pop moderate for speed; DE still needs diversity.
    pop_size = max(16, min(60, 10 * dim))

    # LHS-like: per-dimension random permutation of strata
    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    pop = []
    fit = []

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
            best = fx
            best_x = x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # ---------------- JADE-like DE state ----------------
    # Per-individual memories
    CRi = [0.5] * pop_size
    Fi = [0.7] * pop_size

    # Global parameter means (adapted from successful trials)
    mu_CR = 0.5
    mu_F = 0.7

    # External archive for diversity (stores replaced individuals)
    archive = []
    archive_max = pop_size

    # p-best fraction
    pmin = 2.0 / pop_size  # at least 2 individuals when pop is small
    pmax = 0.2
    p = max(pmin, min(pmax, 0.15))

    # Restart / stagnation tracking
    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0

    # ---------------- local search (cheap SPSA-like) ----------------
    def local_refine(best_x, best_f):
        if time.time() >= deadline:
            return best_x, best_f

        x = best_x[:]
        f = best_f

        # base step sizes scale with span
        a = 0.08
        c = 0.03
        # a few tiny iterations only
        iters = 5 if dim <= 25 else 3

        for t in range(1, iters + 1):
            if time.time() >= deadline:
                break

            # diminishing steps
            at = a / (t ** 0.6)
            ct = c / (t ** 0.2)

            # Rademacher perturbation
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

            xp = x[:]
            xm = x[:]
            for i in range(dim):
                if span[i] <= 0:
                    xp[i] = lo[i]
                    xm[i] = lo[i]
                else:
                    step = ct * span[i]
                    xp[i] = reflect(x[i] + step * delta[i], lo[i], hi[i])
                    xm[i] = reflect(x[i] - step * delta[i], lo[i], hi[i])

            fp = safe_eval(xp)
            fm = safe_eval(xm)

            # gradient estimate and update
            # g_i ~ (fp - fm) / (2*step) * delta_i
            cand = x[:]
            for i in range(dim):
                if span[i] <= 0:
                    cand[i] = lo[i]
                else:
                    step = max(1e-15, ct * span[i])
                    gi = (fp - fm) / (2.0 * step) * delta[i]
                    cand[i] = reflect(x[i] - at * gi, lo[i], hi[i])

            fc = safe_eval(cand)
            if fc < f:
                x, f = cand, fc
                # quick line-ish attempt toward the improvement direction
                if time.time() < deadline:
                    cand2 = x[:]
                    for i in range(dim):
                        if span[i] > 0:
                            cand2[i] = reflect(x[i] + 0.5 * (x[i] - best_x[i]), lo[i], hi[i])
                    f2 = safe_eval(cand2)
                    if f2 < f:
                        x, f = cand2, f2

        return x, f

    # ---------------- main loop ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # periodic local refinement
        if gen % 8 == 0 and time.time() < deadline:
            xlr, flr = local_refine(best_x, best)
            if flr < best:
                best, best_x = flr, xlr[:]
                last_improve_time = time.time()
                last_best = best
                no_improve_gens = 0

        # Sort indices by fitness to define p-best set
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p * pop_size)))
        pbest_set = order[:pcount]

        # Successful parameters for adapting mu_F and mu_CR
        S_CR = []
        S_F = []
        dF = []  # fitness improvements for weighting

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # sample CR from normal around mu_CR
            cr = mu_CR + 0.1 * randn()
            cr = clamp(cr, 0.0, 1.0)

            # sample F from cauchy-like around mu_F (approx)
            # simple heavy-tail: mu + scale * tan(pi*(u-0.5))
            u = random.random()
            fval = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
            # resample if nonpositive
            tries = 0
            while fval <= 0.0 and tries < 5:
                u = random.random()
                fval = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            fval = clamp(fval, 0.05, 1.0)

            CRi[i] = cr
            Fi[i] = fval

            # pick pbest
            pbest_idx = random.choice(pbest_set)
            xpbest = pop[pbest_idx]

            # choose r1 from population (not i)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from pop U archive (not i, not r1)
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pool_size)

            xr1 = pop[r1]
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if span[d] <= 0:
                    v[d] = lo[d]
                else:
                    vd = xi[d] + fval * (xpbest[d] - xi[d]) + fval * (xr1[d] - xr2[d])
                    v[d] = reflect(vd, lo[d], hi[d])

            # binomial crossover
            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if span[d] <= 0:
                    uvec[d] = lo[d]
                else:
                    if random.random() < cr or d == jrand:
                        uvec[d] = v[d]

            fu = safe_eval(uvec)

            # selection + archive update
            if fu <= fit[i]:
                # push replaced solution to archive (if it was valid)
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                # record success for parameter adaptation
                if fit[i] < float("inf"):
                    improvement = max(0.0, fit[i] - fu)
                else:
                    improvement = 1.0
                S_CR.append(cr)
                S_F.append(fval)
                dF.append(improvement)

                pop[i] = uvec
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = uvec[:]

        # adapt mu_CR and mu_F (JADE style)
        if S_CR:
            # weight by improvement; if all zeros, use uniform
            s = sum(dF)
            if s <= 0.0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / s for di in dF]

            # mu_CR update (weighted mean)
            mcr = 0.0
            for wi, cri in zip(w, S_CR):
                mcr += wi * cri
            mu_CR = 0.9 * mu_CR + 0.1 * mcr

            # mu_F update (weighted Lehmer mean)
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, S_F):
                num += wi * fi * fi
                den += wi * fi
            if den > 0.0:
                mf = num / den
                mu_F = 0.9 * mu_F + 0.1 * mf

            mu_CR = clamp(mu_CR, 0.0, 1.0)
            mu_F = clamp(mu_F, 0.05, 1.0)

        # stagnation / restart logic
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # partial restart if stagnating (keep elites)
        if (no_improve_gens >= 10 and time.time() < deadline) or ((time.time() - last_improve_time) > 0.40 * max_time):
            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(3, pop_size // 5)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]
            new_CRi = [CRi[j] for j in keep_idx]
            new_Fi = [Fi[j] for j in keep_idx]

            # refresh archive as well (often helps)
            archive = []

            while len(new_pop) < pop_size and time.time() < deadline:
                x = random_point()
                fx = safe_eval(x)
                xo = opposite_point(x)
                fxo = safe_eval(xo)
                if fxo < fx:
                    x, fx = xo, fxo

                new_pop.append(x)
                new_fit.append(fx)
                new_CRi.append(mu_CR)
                new_Fi.append(mu_F)

                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best
                    last_improve_time = time.time()

            pop, fit, CRi, Fi = new_pop, new_fit, new_CRi, new_Fi
            no_improve_gens = 0
            last_improve_time = time.time()

    return best
