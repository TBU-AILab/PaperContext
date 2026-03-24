import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs) using:
      - fast RNG + safe evaluation
      - hybrid global/local search:
          * initial Latin-Hypercube-like seeding
          * Differential Evolution style "current-to-best/1" mutation (very strong baseline)
          * adaptive step/temperature with stagnation handling
          * occasional coordinate resets and random immigrants
          * final short coordinate/pattern local refinement around best

    func(x) -> float, x is a sequence length dim
    bounds: list of (lo, hi)
    max_time: seconds
    Returns: best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---- preprocess bounds ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lo, hi = highs[i], lows[i]
            lows[i], highs[i] = lo, hi
            spans[i] = highs[i] - lows[i]
        if spans[i] == 0.0:
            spans[i] = 0.0

    def clip(v, i):
        lo = lows[i]; hi = highs[i]
        if v < lo: return lo
        if v > hi: return hi
        return v

    # ---- safe evaluation ----
    def safe_eval(x):
        try:
            v = func(x)
            v = float(v)
            if v != v or v == float("inf") or v == float("-inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---- sampling helpers ----
    def rand_point():
        return [lows[i] + spans[i] * random.random() for i in range(dim)]

    def lhs_points(n):
        # stratified per-dimension, shuffled; cheap LHS-like
        if n <= 0:
            return []
        perms = []
        for i in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)
        inv = 1.0 / n
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                u = (perms[i][k] + random.random()) * inv
                x[i] = lows[i] + spans[i] * u
            pts.append(x)
        return pts

    # ---- parameters (auto-scaled) ----
    pop = max(18, min(90, 10 + 6 * dim))        # DE population
    elite_keep = max(4, min(20, pop // 4))      # keep best few for sampling bias
    immigrants = max(1, pop // 20)

    # DE parameters (adapted mildly with stagnation)
    F0 = 0.55
    CR0 = 0.85

    # local refinement params
    refine_every = 25
    base_local = 0.08  # fraction of span initial local step

    # ---- init population ----
    X = []
    F = []
    best = float("inf")
    best_x = None

    # LHS seed + random fill
    init_pts = lhs_points(min(pop, max(8, pop // 2)))
    while len(init_pts) < pop:
        init_pts.append(rand_point())

    for x in init_pts:
        if time.time() >= deadline:
            return best if best < float("inf") else float("inf")
        fx = safe_eval(x)
        X.append(x)
        F.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    # utility to pick distinct indices
    def pick3(exclude):
        # exclude is an int
        n = len(X)
        a = exclude
        while a == exclude:
            a = random.randrange(n)
        b = exclude
        while b == exclude or b == a:
            b = random.randrange(n)
        c = exclude
        while c == exclude or c == a or c == b:
            c = random.randrange(n)
        return a, b, c

    # ---- light coordinate/pattern refinement around best ----
    def refine_around(x0, f0, budget_evals=40, step_frac=0.05):
        x = x0[:]
        fx = f0
        # coordinate pattern: try +/- step on each dim, shrink if no progress
        step = [step_frac * spans[i] for i in range(dim)]
        # handle zero spans
        for i in range(dim):
            if spans[i] == 0.0:
                step[i] = 0.0

        evals = 0
        shrink = 0.6
        min_step = 1e-12
        while evals < budget_evals and time.time() < deadline:
            improved = False
            for i in range(dim):
                if evals >= budget_evals or time.time() >= deadline:
                    break
                si = step[i]
                if si <= min_step:
                    continue

                # try + step
                xp = x[:]
                xp[i] = clip(xp[i] + si, i)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= budget_evals or time.time() >= deadline:
                    break

                # try - step
                xm = x[:]
                xm[i] = clip(xm[i] - si, i)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                # shrink steps
                for i in range(dim):
                    step[i] *= shrink
                # stop if all tiny
                if max(step) <= min_step:
                    break
        return x, fx

    # ---- main loop ----
    stagnation = 0
    it = 0

    while time.time() < deadline:
        it += 1

        # get index of current best in population
        bidx = min(range(len(F)), key=F.__getitem__)
        if F[bidx] < best:
            best = F[bidx]
            best_x = X[bidx][:]
            stagnation = max(0, stagnation - 2)
        else:
            stagnation += 1

        # adapt DE controls with stagnation (more exploration when stuck)
        # keep in stable ranges
        Fm = min(0.95, max(0.25, F0 + 0.25 * min(1.0, stagnation / 80.0)))
        CR = min(0.98, max(0.15, CR0 - 0.45 * min(1.0, stagnation / 120.0)))

        # rank bias: pick target order from better to worse sometimes
        order = list(range(pop))
        if random.random() < 0.65:
            order.sort(key=lambda i: F[i])  # exploit more often
        else:
            random.shuffle(order)

        # occasional random immigrants when stuck
        if stagnation > 60:
            for _ in range(immigrants):
                j = random.randrange(pop)
                X[j] = rand_point()
                F[j] = safe_eval(X[j])
            stagnation = 45  # reduce but keep exploration bias

        # DE generation
        for idx in order:
            if time.time() >= deadline:
                break

            # choose r1,r2,r3 distinct
            r1, r2, r3 = pick3(idx)

            xi = X[idx]
            xbest = X[bidx]
            xr1 = X[r1]; xr2 = X[r2]; xr3 = X[r3]

            # current-to-best/1 with an extra difference (robust)
            # v = xi + F*(xbest - xi) + F*(xr2 - xr3)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fm * (xbest[d] - xi[d]) + Fm * (xr2[d] - xr3[d])

            # binomial crossover + forced jrand
            jrand = random.randrange(dim)
            u = xi[:]  # start from parent
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    val = v[d]
                    # bounce/clip
                    if val < lows[d] or val > highs[d]:
                        # bounce once then clip (keeps some gradient)
                        if spans[d] > 0.0:
                            if val < lows[d]:
                                val = lows[d] + (lows[d] - val)
                            else:
                                val = highs[d] - (val - highs[d])
                        val = clip(val, d)
                    u[d] = val

                # tiny chance to reset a coordinate to keep diversity
                if spans[d] > 0.0 and random.random() < 0.0025:
                    u[d] = lows[d] + spans[d] * random.random()

            fu = safe_eval(u)

            # selection
            if fu <= F[idx]:
                X[idx] = u
                F[idx] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]
                    stagnation = max(0, stagnation - 4)

        # periodic refinement around best (cheap local improvement)
        if best_x is not None and (it % refine_every == 0 or stagnation > 90):
            # refinement step shrinks with time and with stagnation (but not too small)
            time_left = max(0.0, deadline - time.time())
            # spend only a tiny fraction of remaining time in refinement
            budget = 20 if time_left < 0.2 else 50
            step_frac = max(0.01, base_local * (0.7 ** (stagnation / 60.0)))
            rx, rf = refine_around(best_x, best, budget_evals=budget, step_frac=step_frac)
            if rf < best:
                best = rf
                best_x = rx[:]
                # inject refined best into population worst
                widx = max(range(len(F)), key=F.__getitem__)
                X[widx] = best_x[:]
                F[widx] = best
                stagnation = 0

    return best
