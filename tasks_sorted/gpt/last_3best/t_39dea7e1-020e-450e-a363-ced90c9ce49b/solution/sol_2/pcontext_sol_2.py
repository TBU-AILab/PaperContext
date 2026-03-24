import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer: multi-strategy, evaluation-efficient hybrid
      1) Opposition-based + random initialization
      2) Differential Evolution "current-to-best/1/bin" (fast convergence)
      3) Occasional heavy-tailed (Cauchy-like) perturbations to escape basins
      4) Tiny SPSA-style local refinement near the end (few evals, scales well in dim)
      5) Stagnation-triggered partial restarts

    Self-contained: uses only Python stdlib.
    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ---------------------------
    # Helpers
    # ---------------------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        # robust to func signature + NaN/inf
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    spans = []
    mids = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        spans.append(s if s > 0.0 else 0.0)
        mids.append((lo + hi) * 0.5)

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi > lo:
                x[i] = random.uniform(lo, hi)
            else:
                x[i] = lo
        return x

    def opposite_vec(x):
        # opposition in normalized space: x' = lo + hi - x
        ox = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if hi > lo:
                ox[i] = (lo + hi) - x[i]
                if ox[i] < lo: ox[i] = lo
                elif ox[i] > hi: ox[i] = hi
            else:
                ox[i] = lo
        return ox

    # Cauchy-ish heavy tail without external libs
    def cauchy_step(scale):
        # tan(pi*(u-0.5)) is Cauchy; clip extremes to avoid numerical blowups
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 50: t = 50
        if t < -50: t = -50
        return scale * t

    # ---------------------------
    # Time bookkeeping
    # ---------------------------
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return deadline - time.time()

    if dim <= 0:
        # Degenerate: just evaluate empty vector if possible
        best = eval_f([])
        return best

    # ---------------------------
    # Population initialization (random + opposition) to improve coverage
    # ---------------------------
    # Keep pop modest but not too small; allow higher dim to still work.
    NP = max(16, min(80, 10 * dim))

    pop = []
    fit = []

    # Seed with some random points and their opposites; take best NP of 2*NP candidates
    candidates = []
    for _ in range(NP):
        x = rand_vec()
        candidates.append(x)
        candidates.append(opposite_vec(x))

    # Evaluate candidates (time-safe)
    for x in candidates:
        if time.time() >= deadline:
            break
        candidates_fx = eval_f(x)
        pop.append(x)
        fit.append(candidates_fx)

    # If time ended very early, return best seen so far
    if not fit:
        return float("inf")

    # If we evaluated fewer than needed, fill remainder randomly
    while len(pop) < NP and time.time() < deadline:
        x = rand_vec()
        pop.append(x)
        fit.append(eval_f(x))

    # If we have more than NP (from candidates), keep best NP
    if len(pop) > NP:
        order = sorted(range(len(pop)), key=lambda i: fit[i])
        order = order[:NP]
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # ---------------------------
    # Tiny local refinement: SPSA (2 evals per iteration), good for higher dim
    # ---------------------------
    def spsa_refine(x, fx, iters, a0, c0):
        # Minimization. Uses simultaneous perturbation gradient estimate.
        # Returns improved (x, fx).
        if dim == 0:
            return x, fx
        x_best = x[:]
        f_best = fx
        # decreasing schedules (standard-ish)
        alpha = 0.602
        gamma = 0.101

        for k in range(1, iters + 1):
            if time.time() >= deadline:
                break
            ak = a0 / (k ** alpha)
            ck = c0 / (k ** gamma)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

            xp = x_best[:]
            xm = x_best[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    xp[i] = bounds[i][0]
                    xm[i] = bounds[i][0]
                else:
                    step = ck * spans[i] * delta[i]
                    xp[i] = clamp(x_best[i] + step, bounds[i][0], bounds[i][1])
                    xm[i] = clamp(x_best[i] - step, bounds[i][0], bounds[i][1])

            fp = eval_f(xp)
            if time.time() >= deadline:
                break
            fm = eval_f(xm)

            # gradient estimate and update
            # g_i ~= (fp - fm)/(2*ck*span_i*delta_i)
            xn = x_best[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    xn[i] = bounds[i][0]
                    continue
                denom = 2.0 * ck * spans[i]
                if denom <= 0.0:
                    continue
                gi = (fp - fm) / denom
                # divide by delta_i is equivalent to multiply since delta is ±1
                gi = gi * delta[i]
                xn[i] = clamp(x_best[i] - ak * gi, bounds[i][0], bounds[i][1])

            fn = eval_f(xn)
            if fn < f_best:
                x_best, f_best = xn, fn

        return x_best, f_best

    # ---------------------------
    # Main loop: DE current-to-best + heavy-tail kicks + restarts
    # ---------------------------
    gen = 0
    stall = 0
    last_best = best
    # restart when no improvement for some generations
    max_stall = max(12, 4 + dim)

    while time.time() < deadline:
        gen += 1

        # Near end: spend a bit more on refinement
        tl = time_left()
        near_end = (tl <= 0.20 * max_time)

        # Adapt DE params by phase
        if near_end:
            F = 0.55
            CR = 0.85
        else:
            # jittered params for robustness
            F = 0.45 + 0.45 * random.random()
            CR = 0.15 + 0.80 * random.random()

        # One generation
        for i in range(NP):
            if time.time() >= deadline:
                break

            # Choose r1, r2 distinct and != i
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2 = random.sample(idxs, 2)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # current-to-best/1 mutation: v = xi + F*(best - xi) + F*(xr1 - xr2)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = bounds[d][0]
                else:
                    v[d] = xi[d] + F * (best_x[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # Binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    u[d] = v[d]

            # Bound handling: reflect + clamp (better than pure clamp for DE)
            for d in range(dim):
                lo, hi = bounds[d]
                if hi <= lo:
                    u[d] = lo
                    continue
                if u[d] < lo:
                    u[d] = lo + (lo - u[d])
                    if u[d] > hi:
                        u[d] = lo + random.random() * (hi - lo)
                elif u[d] > hi:
                    u[d] = hi - (u[d] - hi)
                    if u[d] < lo:
                        u[d] = lo + random.random() * (hi - lo)
                # final clamp safety
                if u[d] < lo: u[d] = lo
                elif u[d] > hi: u[d] = hi

            fu = eval_f(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # Heavy-tail kick sometimes to escape (acts like small basin-hopping)
        if time.time() < deadline and (gen % 7 == 0) and dim > 0:
            # try one kicked candidate around current best
            kick_scale = 0.03 if not near_end else 0.01
            cand = best_x[:]
            for d in range(dim):
                if spans[d] > 0.0:
                    cand[d] = clamp(cand[d] + cauchy_step(kick_scale * spans[d]),
                                    bounds[d][0], bounds[d][1])
            fc = eval_f(cand)
            if fc < best:
                best = fc
                best_x = cand[:]

        # Stagnation detection
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        # Partial restart: keep elites, refresh others (mix random + near-best)
        if stall >= max_stall and time.time() < deadline:
            stall = 0
            order = sorted(range(NP), key=lambda k: fit[k])
            k_keep = max(4, NP // 6)
            elites = order[:k_keep]

            for idx in order[k_keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.70:
                    # re-sample near best
                    x = best_x[:]
                    scale = 0.20
                    for d in range(dim):
                        if spans[d] > 0.0:
                            x[d] = clamp(x[d] + random.uniform(-1.0, 1.0) * spans[d] * scale,
                                         bounds[d][0], bounds[d][1])
                    pop[idx] = x
                else:
                    pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])

            best_i = min(range(NP), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                best_x = pop[best_i][:]

        # End-game refinement (very small budget)
        if near_end and time.time() < deadline:
            # 2 evaluations per SPSA iter; keep tiny to respect time
            iters = 2 + max(1, dim // 8)
            # step sizes relative to spans
            x2, f2 = spsa_refine(best_x, best, iters=iters, a0=0.05, c0=0.02)
            if f2 < best:
                best, best_x = f2, x2

    return best
