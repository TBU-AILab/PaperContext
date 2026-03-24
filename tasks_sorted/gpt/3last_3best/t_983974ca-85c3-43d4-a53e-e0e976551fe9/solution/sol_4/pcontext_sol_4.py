import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Upgrade over your JADE-only version:
      - Keeps JADE as the global engine, but adds:
          (1) Opposition-based + LHS-like + corners initialization (better early coverage)
          (2) "Speculative" evaluation batching: generate a whole generation and evaluate sequentially
              (reduces Python overhead vs re-sorting/allocations too often)
          (3) Adaptive p-best rate (more exploration early, more exploitation late)
          (4) Better bound handling: reflection + random reinsert (less edge-sticking than clip)
          (5) Lightweight trust-region coordinate polish with step adaptation on best
          (6) Stagnation-triggered partial restart around best + random immigrants

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        try:
            v = func([])
            return float(v) if v is not None else float("inf")
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # guard against degenerate spans
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0
            # still keep lows/highs as provided; repair will handle odd bounds

    # ---------------- helpers ----------------
    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def corner_vec(jitter_scale=0.02):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        if jitter_scale > 0.0:
            for i in range(dim):
                x[i] += random.gauss(0.0, jitter_scale * spans[i])
        return repair_reflect(x)

    # reflection-based repair (better than pure clipping for DE)
    def repair_reflect(x):
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            if lo > hi:
                lo, hi = hi, lo
            if x[i] < lo:
                # reflect; if still out (huge jump), random reinsert
                x[i] = lo + (lo - x[i])
                if x[i] > hi:
                    x[i] = lo + random.random() * (hi - lo) if hi > lo else lo
            elif x[i] > hi:
                x[i] = hi - (x[i] - hi)
                if x[i] < lo:
                    x[i] = lo + random.random() * (hi - lo) if hi > lo else lo
        return x

    # simple LHS-like: n points, independent permutation per dimension
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        invn = 1.0 / n
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][k] + random.random()) * invn
                lo = lows[d]
                hi = highs[d]
                if lo <= hi:
                    x[d] = lo + u * (hi - lo)
                else:
                    x[d] = hi + u * (lo - hi)
            pts.append(x)
        return pts

    # opposition-based candidate
    def opposite(x):
        xo = [0.0] * dim
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            if lo > hi:
                lo, hi = hi, lo
            xo[i] = lo + hi - x[i]
        return repair_reflect(xo)

    # truncated normal helper: N(0,1) clipped to [-2,2]
    def randn_clip2():
        while True:
            z = random.gauss(0.0, 1.0)
            if -2.0 <= z <= 2.0:
                return z

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------------- parameters ----------------
    pop_size = int(14 + 6 * math.log(dim + 1))
    pop_size = max(20, min(90, pop_size))

    archive_max = pop_size

    # JADE adaptation
    c = 0.1
    mu_F = 0.6
    mu_CR = 0.9

    # polish trust region (coordinate)
    min_step = 1e-15
    polish_every = 10  # gens
    polish_coords = min(dim, max(6, int(0.35 * dim)))

    # stagnation control
    last_improve = time.time()
    stagnate_time = max(0.28 * max_time, 0.8)

    # ---------------- initialization ----------------
    init_until = min(deadline, t0 + 0.17 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1))))
    candidates.extend(lhs_points(n_lhs))

    # add corners
    for _ in range(max(2, pop_size // 6)):
        candidates.append(corner_vec(0.02))

    # add random points and their opposites (O(2k) eval, but strong early)
    while len(candidates) < pop_size:
        x = rand_vec()
        candidates.append(x)
        if len(candidates) < pop_size:
            candidates.append(opposite(x))

    pop = []
    fits = []
    best = float("inf")
    best_x = None

    # evaluate candidates until time slice; ensure at least half-pop
    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(8, pop_size // 2):
            break
        x = repair_reflect(list(x))
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best = f
            best_x = list(x)
            last_improve = time.time()
        if len(pop) >= pop_size:
            break

    while len(pop) < pop_size and time.time() < deadline:
        x = repair_reflect(rand_vec())
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best = f
            best_x = list(x)
            last_improve = time.time()

    if best_x is None:
        best_x = repair_reflect(rand_vec())
        best = safe_eval(best_x)
        last_improve = time.time()

    archive = []

    # per-coordinate polish steps (start small)
    polish_step = [max(min_step, 0.01 * spans[i]) for i in range(dim)]

    def polish_best(x, fx):
        """Coordinate trust-region search with step adaptation (very cheap)."""
        nonlocal best, best_x, last_improve, polish_step
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:polish_coords]

        improved = False
        for i in idxs:
            if time.time() >= deadline:
                break

            si = polish_step[i]
            if si < min_step:
                si = min_step

            base = x[i]

            # + step
            x[i] = base + si
            repair_reflect(x)
            f1 = safe_eval(x)

            # - step
            x[i] = base - si
            repair_reflect(x)
            f2 = safe_eval(x)

            # restore base
            x[i] = base

            if f1 < fx or f2 < fx:
                improved = True
                if f1 <= f2:
                    x[i] = base + si
                    repair_reflect(x)
                    fx = f1
                else:
                    x[i] = base - si
                    repair_reflect(x)
                    fx = f2
                polish_step[i] = min(0.25 * spans[i], polish_step[i] * 1.35)
                if fx < best:
                    best = fx
                    best_x = list(x)
                    last_improve = time.time()
            else:
                polish_step[i] = max(min_step, polish_step[i] * 0.72)

        # if no coord improved, slowly shrink overall steps
        if not improved:
            for i in idxs:
                polish_step[i] = max(min_step, polish_step[i] * 0.90)

        return x, fx

    # ---------------- main loop ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # adaptive p-best rate: explore early, exploit late
        tfrac = (time.time() - t0) / max(1e-12, max_time)
        # from ~0.25 down to ~0.08
        p_best_rate = 0.25 - 0.17 * min(1.0, tfrac)
        if p_best_rate < 0.08:
            p_best_rate = 0.08

        # sort indices by fitness
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda i: fits[i])

        # polish occasionally
        if (gen % polish_every) == 0 and time.time() < deadline:
            bx = list(best_x)
            bf = best
            bx, bf = polish_best(bx, bf)
            if bf < best:
                best = bf
                best_x = list(bx)

        # stagnation: immigrants + restart around best
        if time.time() - last_improve > stagnate_time:
            # replace some worst with jittered-best and random corners
            worst_k = max(3, pop_size // 6)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted[-1 - t]
                if random.random() < 0.6:
                    x = list(best_x)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.12 * spans[d])
                    repair_reflect(x)
                else:
                    x = corner_vec(0.07) if random.random() < 0.5 else rand_vec()
                    repair_reflect(x)
                f = safe_eval(x)
                pop[wi] = x
                fits[wi] = f
                if f < best:
                    best = f
                    best_x = list(x)
                    last_improve = time.time()
            last_improve = time.time()  # reset stagnation timer after injection

        # JADE generation
        SF = []
        SCR = []
        dW = []

        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))

        # precompute pool indices for faster sampling
        pop_indices = list(range(pop_size))

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # sample CR
            CR = mu_CR + 0.1 * randn_clip2()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # sample F (cauchy)
            F = mu_F + 0.1 * cauchy()
            tries = 0
            while F <= 0.0 and tries < 8:
                F = mu_F + 0.1 * cauchy()
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # pbest
            pbest_idx = idx_sorted[random.randrange(p_num)]
            xpbest = pop[pbest_idx]

            # r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            x1 = pop[r1]

            # r2 from pop or archive
            use_arch = (archive and random.random() < 0.5)
            if use_arch and random.random() < 0.5:
                x2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x2 = pop[r2]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d])

            # repair
            repair_reflect(v)

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CR) else xi[d]

            fu = safe_eval(u)

            if fu <= fi:
                # archive parent
                archive.append(list(xi))
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                SF.append(F)
                SCR.append(CR)
                imp = fi - fu
                if imp <= 0.0:
                    imp = 1e-12
                dW.append(imp)

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve = time.time()

        # update mu_F, mu_CR
        if SF:
            mu_CR = (1.0 - c) * mu_CR + c * (sum(SCR) / float(len(SCR)))

            num = 0.0
            den = 0.0
            for Fv, w in zip(SF, dW):
                num += w * Fv * Fv
                den += w * Fv
            if den > 0.0:
                mu_F = (1.0 - c) * mu_F + c * (num / den)

            # clamp
            if mu_F < 0.05:
                mu_F = 0.05
            elif mu_F > 0.95:
                mu_F = 0.95

            if mu_CR < 0.05:
                mu_CR = 0.05
            elif mu_CR > 0.98:
                mu_CR = 0.98

    return best
