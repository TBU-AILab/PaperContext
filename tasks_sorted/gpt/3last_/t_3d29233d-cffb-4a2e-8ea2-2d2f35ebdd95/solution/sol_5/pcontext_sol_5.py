import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Changes vs prior versions (aimed at better best-so-far within fixed time):
      - Stronger initialization: LHS-like + corners + opposites + a few axis probes
      - Main engine: Differential Evolution (DE/rand/1/bin) with:
          * adaptive F and CR (jDE-style, per-individual)
          * current-to-best injection sometimes (faster convergence)
          * bound handling via reflection (better than hard clip for DE)
      - Memetic phase: occasional local coordinate/pattern refinement on best
      - Stagnation restarts: keep elite, re-seed part of the population diversified
      - Time-aware evaluation budgeting (checks before expensive loops)

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        return float(func(x))

    def widths():
        return [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    W = widths()

    def is_fixed(i):
        return bounds[i][1] <= bounds[i][0]

    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        # Reflection into [lo,hi] (handles overshoots better for DE)
        if hi <= lo:
            return lo
        w = hi - lo
        t = v - lo
        # map to [0, 2w)
        t = t % (2.0 * w)
        if t < 0.0:
            t += 2.0 * w
        if t <= w:
            return lo + t
        else:
            return hi - (t - w)

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = reflect(lo + hi - x[i], lo, hi)
        return y

    def lhs_points(n):
        # stratified per dimension (LHS-like)
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

    def corners_subset(k):
        pts = []
        if dim <= 14:
            total = 1 << dim
            step = max(1, total // max(1, k))
            for mask in range(0, total, step):
                x = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    x[i] = hi if ((mask >> i) & 1) else lo
                pts.append(x)
                if len(pts) >= k:
                    break
        else:
            for _ in range(k):
                x = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    x[i] = hi if random.getrandbits(1) else lo
                pts.append(x)
        return pts

    def axis_probes(k_each=1):
        # probe around center along axes (helps on separable-ish problems)
        pts = []
        c = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            c[i] = lo if hi <= lo else (lo + hi) * 0.5

        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                continue
            w = hi - lo
            for t in range(1, k_each + 1):
                step = (0.35 / t) * w
                xp = c[:]
                xm = c[:]
                xp[i] = clip(c[i] + step, lo, hi)
                xm[i] = clip(c[i] - step, lo, hi)
                pts.append(xp)
                pts.append(xm)
        return pts

    def pattern_refine(x0, f0, step, sweeps=2):
        # cheap coordinate search + pattern move
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
                if hi <= lo:
                    continue

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
                xt = [reflect(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
            else:
                for i in range(dim):
                    steps[i] *= 0.5
        return x, fx

    # -------------------- time --------------------
    start = time.time()
    deadline = start + float(max_time)

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None

    # Population size: DE usually benefits from 8..20 * dim, but keep it light for time-bounded runs
    NP = max(18, min(80, 10 * dim))
    # initial design points
    pts = []
    pts += lhs_points(max(NP // 2, 12))
    pts += [opposite(p) for p in pts[:max(6, len(pts) // 3)]]
    pts += corners_subset(max(6, 2 * dim))
    pts += axis_probes(k_each=1)
    while len(pts) < NP:
        pts.append(rand_vec())

    pop = []
    fit = []
    F = []
    CR = []

    for x in pts[:NP]:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        # jDE params per individual
        F.append(0.5 + 0.3 * random.random())     # in [0.5,0.8]
        CR.append(0.1 + 0.9 * random.random())    # in [0.1,1.0]
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_vec()
        best = eval_f(best_x)

    # -------------------- DE main loop --------------------
    it = 0
    no_global = 0
    refine_every = 25
    restart_after = 120

    # jDE adaptation probabilities
    tau1 = 0.10  # probability to resample F
    tau2 = 0.10  # probability to resample CR

    while time.time() < deadline:
        it += 1
        improved_any = False

        # Precompute best index occasionally
        try:
            best_idx = min(range(NP), key=lambda i: fit[i])
        except ValueError:
            best_idx = 0

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # --- jDE parameter adaptation ---
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                # sample in [0.1, 0.9]
                Fi = 0.1 + 0.8 * random.random()
            if random.random() < tau2:
                CRi = random.random()

            # pick r1,r2,r3 distinct and != i
            # small loop; NP is >= 18 so this is fine
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(NP)

            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]

            # --- mutation ---
            v = [0.0] * dim

            # Sometimes use current-to-best/1 to accelerate
            use_ctb = (random.random() < 0.25)
            xb = pop[best_idx]
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    v[d] = lo
                    continue
                if use_ctb:
                    vraw = xi[d] + 0.6 * (xb[d] - xi[d]) + Fi * (x2[d] - x3[d])
                else:
                    vraw = x1[d] + Fi * (x2[d] - x3[d])
                v[d] = reflect(vraw, lo, hi)

            # --- crossover (binomial) ---
            u = xi[:]
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if is_fixed(d):
                    u[d] = bounds[d][0]
                else:
                    if random.random() < CRi or d == jrand:
                        u[d] = v[d]

            fu = eval_f(u)

            # --- selection ---
            if fu <= fi:
                pop[i] = u
                fit[i] = fu
                F[i] = Fi
                CR[i] = CRi
                improved_any = True

                if fu < best:
                    best = fu
                    best_x = u[:]
                    no_global = 0
            else:
                # keep params even if not accepted? jDE keeps old if rejected
                no_global += 1

        if not improved_any:
            no_global += 1

        # --- occasional memetic refinement on global best ---
        if (it % refine_every) == 0 and best_x is not None and time.time() < deadline:
            step = [max(W[d] * 0.02, 1e-12) for d in range(dim)]
            rx, rf = pattern_refine(best_x, best, step, sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject refined best into population (replace worst)
                worst = max(range(NP), key=lambda k: fit[k])
                pop[worst] = best_x[:]
                fit[worst] = best
                no_global = 0

        # --- stagnation restart (partial) ---
        if no_global >= restart_after and time.time() < deadline:
            no_global = 0
            # keep a small elite set, reinitialize the rest around best + random
            elite_k = max(2, NP // 8)
            idx_sorted = sorted(range(NP), key=lambda k: fit[k])
            elite = idx_sorted[:elite_k]

            # compute a scale for reseeding
            # mix global width with a shrinking factor based on iterations
            shrink = 0.25 + 0.75 * random.random()
            scale = [max(W[d] * 1e-9, W[d] * 0.35 * shrink) for d in range(dim)]

            for k in range(NP):
                if k in elite:
                    continue
                if time.time() >= deadline:
                    return best

                # sample near best, with occasional full random dimension-wise reset
                x = best_x[:]
                for d in range(dim):
                    lo, hi = bounds[d]
                    if hi <= lo:
                        x[d] = lo
                    else:
                        if random.random() < 0.20:
                            x[d] = random.uniform(lo, hi)
                        else:
                            x[d] = reflect(x[d] + random.gauss(0.0, scale[d]), lo, hi)

                fx = eval_f(x)
                pop[k] = x
                fit[k] = fx
                F[k] = 0.1 + 0.8 * random.random()
                CR[k] = random.random()

                if fx < best:
                    best, best_x = fx, x[:]

    return best
