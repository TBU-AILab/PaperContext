import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs provided versions:
      - Strong global coverage: LHS batches + (optional) opposition points.
      - Strong local search: bounded Nelder–Mead + occasional coordinate polish.
      - Adaptive restarts around elites (shrinking Gaussians) + global injections.
      - Robust evaluation (handles exceptions / NaN / inf).
      - Careful time checks; works with list inputs.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # minimal meaningful scale (avoids 0 span issues)
    scales = [s if s > 0.0 else 1.0 for s in spans]

    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect(v, lo, hi):
        # robust reflection/folding into [lo, hi]
        if lo == hi:
            return lo
        w = hi - lo
        if w <= 0.0:
            return lo
        # fold
        z = (v - lo) % (2.0 * w)
        if z > w:
            z = 2.0 * w - z
        return lo + z

    def fix_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect(x[i], lo, hi)
        return y

    def evaluate(x):
        # func expects an array-like; lists are acceptable per prompt examples
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if not math.isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = clamp(lo + hi - x[i], lo, hi)
        return y

    # Latin Hypercube Sampling (space-filling, cheap)
    def lhs_batch(n):
        per_dim = []
        for i in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            per_dim.append(arr)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                u = per_dim[i][k]
                x[i] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    # Normalized L1 distance (for elite diversity)
    inv_scales = [1.0 / s for s in scales]
    def dist_norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            s += abs(a[i] - b[i]) * inv_scales[i]
        return s

    def elite_insert(elites, x, fx, max_elites, dedup_eps):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        for f, p in elites:
            ok = True
            for f2, q in out:
                if dist_norm_l1(p, q) < dedup_eps:
                    ok = False
                    break
            if ok:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # ---------------- local search: bounded Nelder–Mead ----------------
    def nelder_mead(x0, f0, budget_evals, init_step_frac):
        """
        Small-eval Nelder–Mead with bound reflection.
        Good general-purpose local optimizer, no derivatives.
        """
        if budget_evals <= 0:
            return x0, f0

        # Build initial simplex: x0 plus one step in each coordinate
        simplex = [list(x0)]
        fvals = [f0]

        step = [max(1e-12 * scales[i], init_step_frac * scales[i]) for i in range(dim)]
        for i in range(dim):
            if time.time() >= deadline:
                return simplex[0], fvals[0]
            xi = list(x0)
            xi[i] = reflect(xi[i] + step[i], bounds[i][0], bounds[i][1])
            fi = evaluate(xi)
            simplex.append(xi)
            fvals.append(fi)
            budget_evals -= 1
            if budget_evals <= 0:
                break

        # NM coefficients
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        # Ensure we have at least 2 points
        m = len(simplex)
        if m < 2:
            return x0, f0

        while budget_evals > 0 and time.time() < deadline:
            # Order
            idx = sorted(range(m), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals   = [fvals[i] for i in idx]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]
            second_worstf = fvals[-2] if m >= 2 else worstf

            # Centroid of all but worst
            centroid = [0.0] * dim
            for j in range(m - 1):
                xj = simplex[j]
                for d in range(dim):
                    centroid[d] += xj[d]
            inv = 1.0 / (m - 1)
            for d in range(dim):
                centroid[d] *= inv

            # Reflect
            xr = [0.0] * dim
            for d in range(dim):
                xr[d] = centroid[d] + alpha * (centroid[d] - worstx[d])
                xr[d] = reflect(xr[d], bounds[d][0], bounds[d][1])
            fr = evaluate(xr)
            budget_evals -= 1
            if fr < bestf:
                # Expand
                xe = [0.0] * dim
                for d in range(dim):
                    xe[d] = centroid[d] + gamma * (xr[d] - centroid[d])
                    xe[d] = reflect(xe[d], bounds[d][0], bounds[d][1])
                fe = evaluate(xe)
                budget_evals -= 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contract
                if fr < worstf:
                    # outside contraction
                    xc = [0.0] * dim
                    for d in range(dim):
                        xc[d] = centroid[d] + rho * (xr[d] - centroid[d])
                        xc[d] = reflect(xc[d], bounds[d][0], bounds[d][1])
                else:
                    # inside contraction
                    xc = [0.0] * dim
                    for d in range(dim):
                        xc[d] = centroid[d] - rho * (centroid[d] - worstx[d])
                        xc[d] = reflect(xc[d], bounds[d][0], bounds[d][1])

                fc = evaluate(xc)
                budget_evals -= 1
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink towards best
                    for j in range(1, m):
                        if budget_evals <= 0 or time.time() >= deadline:
                            break
                        xs = [0.0] * dim
                        for d in range(dim):
                            xs[d] = bestx[d] + sigma * (simplex[j][d] - bestx[d])
                            xs[d] = reflect(xs[d], bounds[d][0], bounds[d][1])
                        fs = evaluate(xs)
                        budget_evals -= 1
                        simplex[j], fvals[j] = xs, fs

            # update m in case init simplex was truncated by time/budget
            m = len(simplex)

        # Return best in simplex
        bi = min(range(len(fvals)), key=lambda i: fvals[i])
        return simplex[bi], fvals[bi]

    # ---------------- cheap coordinate polish ----------------
    def coord_polish(x0, f0, step_frac, rounds):
        x = list(x0)
        fx = f0
        step = [max(1e-12 * scales[i], step_frac * scales[i]) for i in range(dim)]
        mults = (1.0, 0.5, 0.25)
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                lo, hi = bounds[i]
                base = x[i]
                bestv = base
                bestf = fx
                for m in mults:
                    d = step[i] * m
                    for sgn in (-1.0, 1.0):
                        v = base + sgn * d
                        if v < lo or v > hi:
                            continue
                        xt = list(x)
                        xt[i] = v
                        ft = evaluate(xt)
                        if ft < bestf:
                            bestf, bestv = ft, v
                if bestf < fx:
                    x[i] = bestv
                    fx = bestf
                    improved = True
            if not improved:
                break
        return x, fx

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None
    elites = []
    max_elites = max(10, min(60, 4 * dim))
    dedup_eps = 1e-10

    # initial global LHS (+ opposition) for robust coverage
    n0 = max(24, min(120, 8 * dim))
    pts = lhs_batch(n0)
    for k, x in enumerate(pts):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites, dedup_eps)

        # opposition sometimes helps a lot, cheap extra eval (do for ~half)
        if (k & 1) == 0:
            xo = opposite(x)
            fxo = evaluate(xo)
            if fxo < best:
                best, best_x = fxo, xo
            elites = elite_insert(elites, xo, fxo, max_elites, dedup_eps)

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)
        elites = [(best, best_x)]

    # ---------------- main loop ----------------
    phase = 0
    no_improve = 0

    while time.time() < deadline:
        phase += 1

        # pick elite with quadratic bias to best
        m = len(elites)
        if m == 0:
            elites = [(best, best_x)]
            m = 1
        r = random.random()
        idx = int((r * r) * m)
        fx0, x0 = elites[idx]

        # local NM budget: scale mildly with dimension, but keep time-safe
        nm_budget = 12 + 4 * dim
        if nm_budget > 220:
            nm_budget = 220

        # step size: smaller for better elites, larger for worse
        rank = idx / max(1, m - 1)
        init_step_frac = 0.06 + 0.20 * rank
        # gradually reduce over time, but not too fast
        init_step_frac *= 1.0 / (1.0 + 0.01 * phase)

        x1, f1 = nelder_mead(x0, fx0, nm_budget, init_step_frac)
        if f1 < best:
            best, best_x = f1, x1
            no_improve = 0
        else:
            no_improve += 1
        elites = elite_insert(elites, x1, f1, max_elites, dedup_eps)

        if time.time() >= deadline:
            break

        # occasional coordinate polish on current best
        if phase % 3 == 0:
            x2, f2 = coord_polish(best_x, best, step_frac=0.01 / (1.0 + 0.01 * phase), rounds=1)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0
            elites = elite_insert(elites, x2, f2, max_elites, dedup_eps)

        if time.time() >= deadline:
            break

        # restart/perturbation: Gaussian around best or a top elite
        topk = min(len(elites), max(3, dim))
        pick = random.randrange(topk)
        fp, xp = elites[pick]

        # sigma shrinks over time; expands if stagnating
        stagn = 1.0 + min(3.0, no_improve / 12.0)
        decay = 1.0 / (1.0 + 0.015 * phase)
        base_frac = (0.08 + 0.22 * (pick / max(1, topk - 1))) * decay * stagn

        xg = [0.0] * dim
        for i in range(dim):
            sig = max(1e-12 * scales[i], base_frac * scales[i])
            xg[i] = reflect(xp[i] + random.gauss(0.0, sig), bounds[i][0], bounds[i][1])

        fg = evaluate(xg)
        if fg < best:
            best, best_x = fg, xg
            no_improve = 0
        elites = elite_insert(elites, xg, fg, max_elites, dedup_eps)

        # periodic global injection batch (fresh basins)
        if phase % 9 == 0 and time.time() < deadline:
            nG = max(8, min(50, 3 * dim))
            for x in lhs_batch(nG):
                if time.time() >= deadline:
                    break
                fx = evaluate(x)
                if fx < best:
                    best, best_x = fx, x
                    no_improve = 0
                elites = elite_insert(elites, x, fx, max_elites, dedup_eps)

        # if very stuck, do a full random restart
        if no_improve >= 30 and time.time() < deadline:
            xr = rand_uniform()
            fr = evaluate(xr)
            if fr < best:
                best, best_x = fr, xr
            elites = elite_insert(elites, xr, fr, max_elites, dedup_eps)
            no_improve = 0

    return best
