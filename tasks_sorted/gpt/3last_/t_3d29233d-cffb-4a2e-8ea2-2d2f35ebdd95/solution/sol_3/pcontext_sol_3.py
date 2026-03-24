import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs) using a stronger portfolio:

    1) LHS-like initialization + opposite points
    2) Multi-start "micro-CMA"-like search:
         - maintains diagonal covariance (per-dim sigma) + evolution path
         - 2-point mirrored sampling for variance reduction
         - weighted recombination of the best offspring
         - 1/5-style success control + path-based anisotropic scaling
    3) Occasional cheap coordinate/pattern refinement on the global best
    4) Stagnation-triggered restart with diversified scale

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ---------------- helpers ----------------
    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(x))  # list is acceptable

    def widths():
        return [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    W = widths()

    def rand_uniform_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def lhs_points(n):
        # Stratified sampling per dimension (LHS-like)
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        pts = []
        for j in range(n):
            x = [0.0] * dim
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    x[d] = lo
                else:
                    u = (strata[d][j] + random.random()) / n
                    x[d] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = lo + hi - x[i]
                y[i] = clip(y[i], lo, hi)
        return y

    def pattern_refine(x0, f0, step, sweeps=2):
        # Very cheap coordinate search + pattern move
        x = x0[:]
        fx = f0
        steps = step[:]
        for _ in range(sweeps):
            base = x[:]
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                si = steps[i]
                if si <= 0.0:
                    continue
                lo, hi = bounds[i]
                xi = x[i]

                xp = x[:]
                xp[i] = clip(xi + si, lo, hi)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[i] = clip(xi - si, lo, hi)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if improved:
                d = [x[i] - base[i] for i in range(dim)]
                xt = [clip(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
            else:
                for i in range(dim):
                    steps[i] *= 0.5
        return x, fx

    # ---------------- time ----------------
    start = time.time()
    deadline = start + float(max_time)

    # ---------------- init ----------------
    best = float("inf")
    best_x = None

    # Initial design
    n0 = max(28, 7 * dim)
    pts = lhs_points(n0)
    # Opposite points (often helps in bounded domains)
    pts += [opposite(p) for p in pts[: max(6, n0 // 3)]]
    # A few extra pure randoms
    for _ in range(max(6, dim)):
        pts.append(rand_uniform_vec())

    for x in pts:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_uniform_vec()
        best = eval_f(best_x)

    # ---------------- micro-CMA (diagonal) state ----------------
    # Parent mean
    m = best_x[:]
    fm = best

    # Diagonal stds; start moderately global
    sigma = [w * 0.22 if w > 0 else 0.0 for w in W]
    # Evolution path (directional memory)
    p = [0.0] * dim

    # Offspring settings
    lam = max(16, 10 + int(6 * math.log(dim + 1.0)))
    if lam % 2 == 1:
        lam += 1  # for mirrored pairs
    mu = max(4, lam // 3)

    # log-weights
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    # Adaptation knobs
    # Path update strength (small, stable)
    c_path = 0.20
    # Sigma clamp
    sig_min_frac = 1e-10
    sig_max_frac = 0.60

    # Success rule (batch-based)
    target_succ = 0.20
    succ_ema = 0.0
    succ_decay = 0.90
    sig_adapt = 0.45  # exp factor for sigma on success deviation

    # Stagnation & refinement
    it = 0
    no_global = 0
    restart_after = 55
    refine_every = 20

    # For restarts: keep best but diversify around it
    base_sig0 = sigma[:]

    # ---------------- main loop ----------------
    while time.time() < deadline:
        it += 1

        # --- generate offspring (mirrored sampling) ---
        off = []
        improved_over_parent = False

        # sample in pairs: z and -z
        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]

            # child 1
            x1 = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                if hi <= lo:
                    x1[i] = lo
                else:
                    x1[i] = clip(m[i] + (p[i] * 0.35 + z[i]) * sigma[i], lo, hi)
            f1 = eval_f(x1)
            off.append((f1, x1))

            # child 2 (mirrored)
            x2 = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                if hi <= lo:
                    x2[i] = lo
                else:
                    x2[i] = clip(m[i] + (p[i] * 0.35 - z[i]) * sigma[i], lo, hi)
            f2 = eval_f(x2)
            off.append((f2, x2))

            if f1 < fm or f2 < fm:
                improved_over_parent = True

        off.sort(key=lambda t: t[0])

        # --- recombination of top mu to new mean ---
        top = off[:mu]
        new_m = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * top[k][1][i]
            new_m[i] = clip(s, bounds[i][0], bounds[i][1])
        new_fm = eval_f(new_m)

        # Use the best candidate among (new_m) and (best offspring)
        best_off_f, best_off_x = off[0]
        if best_off_f <= new_fm:
            cand_m, cand_fm = best_off_x[:], best_off_f
        else:
            cand_m, cand_fm = new_m, new_fm

        # Update parent
        old_m = m[:]
        if cand_fm < fm:
            m, fm = cand_m[:], cand_fm
        else:
            # still move to best offspring to keep momentum
            m, fm = best_off_x[:], best_off_f

        # Global best update
        if fm < best:
            best, best_x = fm, m[:]
            no_global = 0
        else:
            no_global += 1

        # --- update evolution path (directional memory) ---
        # p <- (1-c)p + c * (m - old_m)/sigma  (per dimension)
        for i in range(dim):
            if sigma[i] > 0.0:
                step = (m[i] - old_m[i]) / sigma[i]
            else:
                step = 0.0
            p[i] = (1.0 - c_path) * p[i] + c_path * step

        # --- success-based sigma adaptation (batch) ---
        # If we improved parent at least once in the batch, treat as success
        succ = 1.0 if improved_over_parent else 0.0
        succ_ema = succ_decay * succ_ema + (1.0 - succ_decay) * succ
        # drive succ_ema toward target
        mult = math.exp(sig_adapt * (succ_ema - target_succ))
        for i in range(dim):
            w = W[i]
            if w <= 0.0:
                sigma[i] = 0.0
                continue
            sigma[i] *= mult
            smin = w * sig_min_frac
            smax = w * sig_max_frac
            if sigma[i] < smin:
                sigma[i] = smin
            elif sigma[i] > smax:
                sigma[i] = smax

        # --- occasional local refinement on the best ---
        if (it % refine_every) == 0 and time.time() < deadline and best_x is not None:
            step = [max(sigma[i], W[i] * 0.02) for i in range(dim)]
            rx, rf = pattern_refine(best_x, best, step, sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                m, fm = best_x[:], best
                # reset path a bit to align to new best
                for i in range(dim):
                    p[i] *= 0.3
                no_global = 0

        # --- restart if stagnating ---
        if no_global >= restart_after and time.time() < deadline:
            no_global = 0

            # restart mean: mostly best, plus a random point
            r = rand_uniform_vec()
            alpha = 0.78
            m = [clip(alpha * best_x[i] + (1.0 - alpha) * r[i], bounds[i][0], bounds[i][1])
                 for i in range(dim)]
            fm = eval_f(m)

            # diversify sigma: sometimes large, sometimes medium-small
            g = 0.25 + 0.75 * random.random()
            for i in range(dim):
                w = W[i]
                if w <= 0.0:
                    sigma[i] = 0.0
                else:
                    sigma[i] = clip(base_sig0[i] * (0.6 + 1.8 * g), w * 1e-6, w * 0.5)
                p[i] = 0.0

            if fm < best:
                best, best_x = fm, m[:]

    return best
