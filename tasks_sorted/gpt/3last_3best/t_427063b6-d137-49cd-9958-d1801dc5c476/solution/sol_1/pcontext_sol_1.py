import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization:
      - Differential Evolution "current-to-pbest/1" + archive (JADE-like)
      - Success-history adaptation of F and CR
      - Bound handling via reflection (usually better than simple clipping)
      - Lightweight local refinement around current best (coordinate + random-direction)
      - Occasional diversity injection when stagnating

    No external libraries required.
    Returns: best (float) best fitness found within max_time seconds.
    """

    # ------------------------ helpers ------------------------
    def clip(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def reflect_into_bounds(x, lo, hi):
        # Reflect with wrap-like reflections until inside bounds
        if lo == hi:
            return lo
        # Handle huge excursions robustly
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        # Numerical safety
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return x

    def ensure_bounds_reflect(vec):
        out = vec[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = reflect_into_bounds(out[i], lo, hi)
        return out

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(vec):
        try:
            v = func(vec)
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                v = float(v)
                if v != v or v == float("inf") or v == float("-inf"):
                    return float("inf")
                return v
            return float("inf")
        except Exception:
            return float("inf")

    def l2_dist(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    # ------------------------ setup ------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # Always do at least one evaluation
    best = float("inf")
    best_x = None

    # Population size: slightly larger than before but still time-friendly
    pop_size = max(12, min(50, 15 + 3 * dim))

    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    for i in range(pop_size):
        if fit[i] < best:
            best = fit[i]
            best_x = pop[i][:]

    # External archive (stores replaced solutions)
    archive = []
    archive_max = pop_size

    # JADE-like parameters (adapted)
    mu_F = 0.6
    mu_CR = 0.6
    c_adapt = 0.1

    # pbest rate range
    p_min, p_max = 0.05, 0.25

    # Stagnation / injection control
    no_improve_rounds = 0
    last_best = best

    # Local search control
    last_local = time.time()
    local_interval = 0.15  # seconds between local refinements (adaptive inside loop)

    # Precompute per-dimension scale (for step sizes)
    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # Avoid zero-span issues
    for i in range(dim):
        if span[i] <= 0.0:
            span[i] = 1.0

    # ------------------------ main loop ------------------------
    while time.time() < deadline:
        # Sort indices by fitness ascending for pbest selection
        idx_sorted = sorted(range(pop_size), key=lambda k: fit[k])

        # Draw p for this generation
        p = random.uniform(p_min, p_max)
        p_num = max(2, int(math.ceil(p * pop_size)))

        # Success memories for adaptation
        S_F = []
        S_CR = []
        S_df = []  # fitness improvements used as weights

        # Iterate individuals (one "generation")
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # Choose pbest from top p_num
            pbest_idx = idx_sorted[random.randrange(p_num)]
            xpbest = pop[pbest_idx]

            # Sample CR ~ N(mu_CR, 0.1) clipped
            CRi = random.gauss(mu_CR, 0.1)
            CRi = clip(CRi, 0.0, 1.0)

            # Sample Fi from Cauchy(mu_F, 0.1) clipped to (0, 1]
            # Cauchy: mu + gamma * tan(pi*(u-0.5))
            Fi = None
            for _ in range(10):
                u = random.random()
                Fi = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                if Fi > 0.0:
                    break
            if Fi is None or Fi <= 0.0:
                Fi = 0.5
            Fi = clip(Fi, 0.05, 1.0)

            # Pick r1 != i
            r1 = random.randrange(pop_size)
            while r1 == i:
                r1 = random.randrange(pop_size)

            # Pick r2 from union(pop, archive) and not equal to i or r1 (when applicable)
            union = pop + archive
            uN = len(union)
            r2 = random.randrange(uN)
            # Ensure r2 doesn't refer to same vector as i or r1 in pop part
            # (archive elements are distinct objects anyway)
            tries = 0
            while tries < 20:
                ok = True
                if r2 < pop_size and (r2 == i or r2 == r1):
                    ok = False
                if ok:
                    break
                r2 = random.randrange(uN)
                tries += 1

            xr1 = pop[r1]
            xr2 = union[r2] if uN > 0 else pop[random.randrange(pop_size)]

            # current-to-pbest/1 mutation:
            # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            # Binomial crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]

            uvec = ensure_bounds_reflect(uvec)
            fu = safe_eval(uvec)

            # Selection
            if fu <= fi:
                # push replaced into archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal (fast)
                    del archive[random.randrange(len(archive))]

                pop[i] = uvec
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = uvec[:]

                # store successes for parameter adaptation
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(df)
            # else: keep

        # Adapt mu_F and mu_CR using weighted means of successful parameters
        if S_F:
            # weights proportional to improvement; if all zero, uniform
            sw = sum(S_df)
            if sw <= 0.0:
                weights = [1.0 / len(S_df)] * len(S_df)
            else:
                weights = [d / sw for d in S_df]

            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            F_lehmer = (num / den) if den > 0.0 else mu_F

            # Weighted arithmetic mean for CR
            CR_mean = 0.0
            for w, cr in zip(weights, S_CR):
                CR_mean += w * cr

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * CR_mean

            mu_F = clip(mu_F, 0.05, 0.95)
            mu_CR = clip(mu_CR, 0.0, 1.0)

        # Stagnation tracking
        if best < last_best - 1e-12:
            no_improve_rounds = 0
            last_best = best
        else:
            no_improve_rounds += 1

        # Diversity injection if stagnating: reinitialize a few worst (keep best)
        if no_improve_rounds >= 8:
            no_improve_rounds = 0
            # replace worst 20% with random or around-best samples
            k = max(1, pop_size // 5)
            worst_idx = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for wi in worst_idx:
                if time.time() >= deadline:
                    return best
                # don't overwrite current best individual if it exists in pop
                # (approx check by fitness equality and proximity)
                if best_x is not None and fit[wi] == best and l2_dist(pop[wi], best_x) < 1e-18:
                    continue

                if random.random() < 0.6 and best_x is not None:
                    # sample near best with decreasing radius
                    rad = 0.2
                    x = best_x[:]
                    for j in range(dim):
                        x[j] = x[j] + random.uniform(-rad, rad) * span[j]
                    x = ensure_bounds_reflect(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best

        # Local refinement around best (cheap, periodic)
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now

            x0 = best_x[:]
            f0 = best

            # coordinate steps with shrinking schedule
            # few probes only (time-safe)
            base_step = 0.08
            # use a small number of coordinates each time to reduce cost for large dim
            coords = list(range(dim))
            random.shuffle(coords)
            m = min(dim, 6)  # probe up to 6 dimensions per local call
            coords = coords[:m]

            for j in coords:
                if time.time() >= deadline:
                    return best
                step = base_step * span[j]
                if step <= 0.0:
                    continue

                # try +/- step
                for sgn in (-1.0, 1.0):
                    xt = x0[:]
                    xt[j] = xt[j] + sgn * step
                    xt = ensure_bounds_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft

            # random-direction small step
            if time.time() < deadline:
                xt = x0[:]
                # sparse direction
                for j in range(dim):
                    if random.random() < (2.0 / max(2.0, dim)):
                        xt[j] = xt[j] + random.uniform(-0.05, 0.05) * span[j]
                xt = ensure_bounds_reflect(xt)
                ft = safe_eval(xt)
                if ft < f0:
                    x0, f0 = xt, ft

            if f0 < best:
                best = f0
                best_x = x0[:]
                last_best = best

            # adapt local interval: if improving, do it slightly more often
            local_interval = clip(local_interval * (0.9 if f0 < best + 1e-12 else 1.05), 0.05, 0.35)

    return best
