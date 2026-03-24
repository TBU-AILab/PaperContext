import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs):
    Hybrid of:
      1) Oppositional + stratified init (broad coverage quickly)
      2) Differential Evolution (DE/current-to-best/1) for robust global search
      3) Lightweight trust-region coordinate search around best for fast local refinement
      4) Restarts on stagnation with partial re-randomization

    Returns: best fitness found (float)
    """

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly for large steps
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    def randu(lo, hi):
        return lo + (hi - lo) * random.random()

    def safe_eval(x):
        try:
            v = func(x)
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def random_point():
        return [randu(lo[i], hi[i]) if span[i] > 0 else lo[i] for i in range(dim)]

    def opposite_point(x):
        # oppositional point mirrored around center of bounds
        return [lo[i] + hi[i] - x[i] if span[i] > 0 else lo[i] for i in range(dim)]

    # ---------------- time ----------------
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    # ---------------- initialization ----------------
    # Population size (DE typically wants ~5-10*dim; keep time-safe)
    pop_size = max(12, min(40, 8 * dim))
    # stratified per-dimension
    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    pop = []
    fit = []

    best_x = None
    best = float("inf")

    # Build initial pop with stratified samples + opposition evaluation
    for k in range(pop_size):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            if span[i] <= 0:
                x.append(lo[i])
            else:
                u = (strata[i][k] + random.random()) / pop_size
                x.append(lo[i] + u * span[i])

        fx = safe_eval(x)
        xo = opposite_point(x)
        fxo = safe_eval(xo)

        if fxo < fx:
            x, fx = xo, fxo

        pop.append(x)
        fit.append(fx)

        if fx < best:
            best, best_x = fx, x[:]

    # If everything failed (rare), fallback
    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # ---------------- local search (trust-region coordinate) ----------------
    def local_refine(x0, f0, radius):
        x = x0[:]
        f = f0
        # one or two passes only (time-safe)
        for _ in range(2):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                if span[i] <= 0:
                    continue
                step = radius[i]
                if step <= 0:
                    continue
                # try 3-point stencil: -step, +step, and a larger move sometimes
                for s in (-step, step, (2.0 * step if random.random() < 0.25 else 0.0)):
                    if s == 0.0:
                        continue
                    cand = x[:]
                    cand[i] = reflect(cand[i] + s, lo[i], hi[i])
                    fc = safe_eval(cand)
                    if fc < f:
                        x, f = cand, fc
                        improved = True
            # shrink if no improvement, otherwise slightly expand
            mul = 0.6 if not improved else 1.15
            for i in range(dim):
                radius[i] *= mul
        return x, f

    # ---------------- DE parameters ----------------
    # Self-tuning ranges for F and CR
    F_min, F_max = 0.4, 0.95
    CR_min, CR_max = 0.05, 0.95

    # stagnation / restart control
    last_improve_time = time.time()
    last_best = best
    no_improve_gens = 0

    # For local refine radius
    base_radius = [0.08 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    gen = 0
    while time.time() < deadline:
        gen += 1

        # occasional local refine around best
        if gen % 7 == 0 and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_radius]
            x_lr, f_lr = local_refine(best_x, best, rad)
            if f_lr < best:
                best, best_x = f_lr, x_lr[:]
                last_improve_time = time.time()
                last_best = best
                no_improve_gens = 0

        # DE generation
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose distinct r1,r2,r3 != i
            idxs = list(range(pop_size))
            # lightweight distinct picks without shuffle of all
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)

            # DE/current-to-best/1: v = x_i + F*(best-x_i) + F*(x_r1-x_r2)
            F = randu(F_min, F_max)
            CR = randu(CR_min, CR_max)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            v = [0.0] * dim
            for d in range(dim):
                if span[d] <= 0:
                    v[d] = lo[d]
                else:
                    v[d] = xi[d] + F * (best_x[d] - xi[d]) + F * (xr1[d] - xr2[d])
                    v[d] = reflect(v[d], lo[d], hi[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]  # trial
            for d in range(dim):
                if span[d] <= 0:
                    u[d] = lo[d]
                else:
                    if random.random() < CR or d == jrand:
                        u[d] = v[d]

            fu = safe_eval(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # update stagnation signals
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            # when improving, gently reduce base radius (more exploitation)
            for d in range(dim):
                base_radius[d] = max(1e-15, base_radius[d] * 0.95)
        else:
            no_improve_gens += 1
            # if not improving, allow a bit more exploration locally
            for d in range(dim):
                base_radius[d] = min(0.25 * span[d] if span[d] > 0 else 0.0,
                                     base_radius[d] * 1.03 if span[d] > 0 else 0.0)

        # restart on stagnation (time-aware)
        if (no_improve_gens >= 12) or ((time.time() - last_improve_time) > 0.35 * max_time):
            if time.time() >= deadline:
                return best
            # keep top k, re-randomize the rest (with opposition)
            order = sorted(range(pop_size), key=lambda t: fit[t])
            k_keep = max(3, pop_size // 4)
            keep = order[:k_keep]

            new_pop = [pop[j][:] for j in keep]
            new_fit = [fit[j] for j in keep]

            while len(new_pop) < pop_size and time.time() < deadline:
                x = random_point()
                fx = safe_eval(x)
                xo = opposite_point(x)
                fxo = safe_eval(xo)
                if fxo < fx:
                    x, fx = xo, fxo
                new_pop.append(x)
                new_fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            last_improve_time = time.time()
            last_best = best

    return best
