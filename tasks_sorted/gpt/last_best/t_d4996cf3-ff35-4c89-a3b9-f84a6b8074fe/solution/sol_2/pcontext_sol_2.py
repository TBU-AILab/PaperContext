import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a compact, robust hybrid:
      - Low-discrepancy Sobol-like fallback via scrambled Halton (global exploration)
      - Adaptive Evolution Strategy (mu,lambda)-ES with diagonal sigma (fast progress on smooth problems)
      - Nelder-Mead style simplex steps around the best (good local refinement without gradients)
      - Occasional coordinate/pattern micro-search (cheap exploitation)

    Returns:
      best (float): best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    nonzero = [spans[i] > 0.0 for i in range(dim)]

    # ---------- helpers ----------
    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    # Fast approx exp for accept? keep simple: use math.exp

    # ---------- Halton (scrambled) ----------
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    halton_index = 1
    # fixed random shifts per dimension (better than per-point)
    halton_shift = [random.random() for _ in range(dim)]

    def halton_point():
        nonlocal halton_index
        x = [0.0] * dim
        idx = halton_index
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        halton_index = idx + 1
        return x

    # ---------- initial best ----------
    def random_point():
        x = [0.0] * dim
        for i in range(dim):
            if nonzero[i]:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    x_best = random_point()
    f_best = eval_f(x_best)

    # ---------- simplex (Nelder-Mead-lite) ----------
    # Keep a small simplex around current best for local improvements.
    # We'll rebuild it occasionally.
    def make_simplex(center):
        # step proportional to spans but not too large
        base = [max(1e-12, 0.08 * spans[i]) if nonzero[i] else 0.0 for i in range(dim)]
        simp = [center[:]]
        for i in range(dim):
            x = center[:]
            if nonzero[i]:
                x[i] = x[i] + base[i]
                if x[i] > highs[i]:
                    x[i] = center[i] - base[i]
                clip(x)
            simp.append(x)
        return simp

    # ---------- ES state ----------
    # Diagonal sigma, adapted by 1/5th success rule style.
    sigma = [0.18 * spans[i] if nonzero[i] else 0.0 for i in range(dim)]
    sigma_min = [1e-12 * (spans[i] if nonzero[i] else 1.0) for i in range(dim)]
    sigma_max = [0.5 * spans[i] if nonzero[i] else 0.0 for i in range(dim)]

    # ES parameters
    lam = max(8, min(60, 8 + 4 * dim))
    mu = max(3, lam // 3)

    # success tracking
    succ = 0
    trials = 0

    # ---------- micro local pattern search around x_best ----------
    def micro_pattern(x0, f0, budget_evals, step_scale):
        x = x0[:]
        fx = f0
        step = [step_scale * spans[i] if nonzero[i] else 0.0 for i in range(dim)]
        order = list(range(dim))
        order.sort(key=lambda i: spans[i], reverse=True)

        evals = 0
        while evals < budget_evals and time.time() < deadline:
            improved = False
            for i in order:
                if not nonzero[i]:
                    continue
                si = step[i]
                if si <= 0.0:
                    continue
                # try +/- si
                best_local = fx
                best_x = None
                for d in (-1.0, 1.0):
                    cand = x[:]
                    cand[i] += d * si
                    if cand[i] < lows[i]: cand[i] = lows[i]
                    if cand[i] > highs[i]: cand[i] = highs[i]
                    fc = eval_f(cand); evals += 1
                    if fc < best_local:
                        best_local = fc
                        best_x = cand
                    if evals >= budget_evals or time.time() >= deadline:
                        break
                if best_x is not None:
                    x, fx = best_x, best_local
                    improved = True
                if evals >= budget_evals or time.time() >= deadline:
                    break
            if not improved:
                break
        return fx, x

    # ---------- warmup global sampling ----------
    # Do a quick batch to get a decent starting point.
    warm = max(40, min(500, 40 + 25 * dim))
    if max_time < 0.2:
        warm = min(warm, 40)

    for _ in range(warm):
        if time.time() >= deadline:
            return f_best
        x = halton_point()
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x

    simplex = make_simplex(x_best)
    simp_f = [eval_f(x) for x in simplex]
    # update best from simplex
    for fx, x in zip(simp_f, simplex):
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # ---------- main loop ----------
    # Mix: ES generations + occasional simplex steps + occasional global injections.
    inject_period = 6
    rebuild_simplex_period = 10
    iter_no = 0

    while time.time() < deadline:
        iter_no += 1

        # ---- ES generation around current best ----
        # Sample offspring from N(x_best, diag(sigma^2)) with occasional Cauchy jumps.
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return f_best
            child = x_best[:]  # center at best
            if random.random() < 0.12:
                # heavy-tailed jump
                for i in range(dim):
                    if nonzero[i]:
                        u = random.random()
                        c = math.tan(math.pi * (u - 0.5))
                        child[i] += c * 0.05 * spans[i]
            else:
                for i in range(dim):
                    if nonzero[i]:
                        child[i] += random.gauss(0.0, sigma[i])
            clip(child)
            fc = eval_f(child)
            pop.append((fc, child))

            trials += 1
            if fc < f_best:
                f_best = fc
                x_best = child[:]
                succ += 1

        pop.sort(key=lambda t: t[0])

        # Recombine best mu (mean)
        # Use rank-based weights (simple, stable)
        weights = []
        ws = 0.0
        for k in range(mu):
            w = (mu - k)
            weights.append(w)
            ws += w
        inv_ws = 1.0 / ws

        new_center = [0.0] * dim
        for i in range(dim):
            if nonzero[i]:
                s = 0.0
                for k in range(mu):
                    s += weights[k] * pop[k][1][i]
                new_center[i] = s * inv_ws
            else:
                new_center[i] = lows[i]
        clip(new_center)
        f_center = eval_f(new_center)
        trials += 1
        if f_center < f_best:
            f_best, x_best = f_center, new_center[:]
            succ += 1

        # Adapt sigma via success rate
        if trials >= 30:
            rate = succ / float(trials)
            # target ~0.2
            if rate > 0.25:
                factor = 1.18
            elif rate < 0.15:
                factor = 0.82
            else:
                factor = 1.0
            if factor != 1.0:
                for i in range(dim):
                    if nonzero[i]:
                        s = sigma[i] * factor
                        if s < sigma_min[i]: s = sigma_min[i]
                        if s > sigma_max[i]: s = sigma_max[i]
                        sigma[i] = s
            succ = 0
            trials = 0

        # ---- occasional Nelder-Mead-lite step on simplex ----
        if time.time() < deadline and (iter_no % 2 == 0):
            # sort simplex
            idx = list(range(len(simplex)))
            idx.sort(key=lambda j: simp_f[j])
            simplex = [simplex[j] for j in idx]
            simp_f = [simp_f[j] for j in idx]

            bestx, bestf = simplex[0], simp_f[0]
            worstx, worstf = simplex[-1], simp_f[-1]
            second_worst_f = simp_f[-2] if len(simp_f) >= 2 else worstf

            if bestf < f_best:
                f_best, x_best = bestf, bestx[:]

            # centroid of all but worst
            centroid = [0.0] * dim
            m = len(simplex) - 1
            for i in range(dim):
                if nonzero[i]:
                    s = 0.0
                    for j in range(m):
                        s += simplex[j][i]
                    centroid[i] = s / m
                else:
                    centroid[i] = lows[i]

            # reflect
            alpha = 1.0
            xr = [centroid[i] + alpha * (centroid[i] - worstx[i]) for i in range(dim)]
            clip(xr)
            fr = eval_f(xr)

            if fr < bestf:
                # expand
                gamma = 2.0
                xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(dim)]
                clip(xe)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], simp_f[-1] = xe, fe
                else:
                    simplex[-1], simp_f[-1] = xr, fr
            elif fr < second_worst_f:
                simplex[-1], simp_f[-1] = xr, fr
            else:
                # contract
                rho = 0.5
                xc = [centroid[i] + rho * (worstx[i] - centroid[i]) for i in range(dim)]
                clip(xc)
                fc = eval_f(xc)
                if fc < worstf:
                    simplex[-1], simp_f[-1] = xc, fc
                else:
                    # shrink towards best
                    sigma_shrink = 0.5
                    for j in range(1, len(simplex)):
                        xj = simplex[j]
                        for i in range(dim):
                            if nonzero[i]:
                                xj[i] = bestx[i] + sigma_shrink * (xj[i] - bestx[i])
                        clip(xj)
                        simp_f[j] = eval_f(xj)

            # Update global best from simplex after step
            for fx, x in zip(simp_f, simplex):
                if fx < f_best:
                    f_best, x_best = fx, x[:]

        # ---- periodic micro pattern around current best ----
        if time.time() < deadline and (iter_no % 5 == 0):
            fb2, xb2 = micro_pattern(x_best, f_best, budget_evals=2*dim + 6, step_scale=0.02)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]

        # ---- periodic global injection + occasional restart around halton best ----
        if time.time() < deadline and (iter_no % inject_period == 0):
            inj = 3 if dim > 25 else 6
            for _ in range(inj):
                if time.time() >= deadline:
                    return f_best
                xg = halton_point()
                fg = eval_f(xg)
                if fg < f_best:
                    f_best, x_best = fg, xg[:]
                    # reset sigma a bit larger to explore around new basin
                    for i in range(dim):
                        if nonzero[i]:
                            sigma[i] = max(sigma[i], 0.10 * spans[i])
            # mild random restart around best (helps if stuck on boundary)
            if random.random() < 0.35:
                xr = x_best[:]
                for i in range(dim):
                    if nonzero[i]:
                        xr[i] += random.gauss(0.0, 0.06 * spans[i])
                clip(xr)
                fr = eval_f(xr)
                if fr < f_best:
                    f_best, x_best = fr, xr[:]

        # ---- rebuild simplex occasionally around current best ----
        if time.time() < deadline and (iter_no % rebuild_simplex_period == 0):
            simplex = make_simplex(x_best)
            simp_f = [eval_f(x) for x in simplex]
            for fx, x in zip(simp_f, simplex):
                if fx < f_best:
                    f_best, x_best = fx, x[:]

    return f_best
