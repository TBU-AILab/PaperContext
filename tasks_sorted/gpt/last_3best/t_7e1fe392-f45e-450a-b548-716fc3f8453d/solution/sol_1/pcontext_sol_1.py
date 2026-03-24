import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Strategy (hybrid, restart-based):
      1) Global exploration via Latin Hypercube Sampling (cheap space-filling).
      2) Maintain an elite set of best points found.
      3) Local refinement from elites using a coordinate pattern search with
         multi-step probing and adaptive step sizes.
      4) Occasional Gaussian perturbation around elites + random restarts.

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_gauss_around(x, sigmas):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(clamp(x[i] + random.gauss(0.0, sigmas[i]), lo, hi))
        return y

    def evaluate(x):
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

    # Latin Hypercube Sampling (no external libs)
    def lhs_samples(n):
        # For each dimension, create n stratified values in [0,1] then permute
        per_dim = []
        for i in range(dim):
            vals = [(k + random.random()) / n for k in range(n)]
            random.shuffle(vals)
            per_dim.append(vals)
        pts = []
        for k in range(n):
            x = []
            for i in range(dim):
                lo, hi = bounds[i]
                u = per_dim[i][k]
                x.append(lo + u * (hi - lo))
            pts.append(x)
        return pts

    # Insert into elite list (sorted by f), keep unique-ish and bounded size
    def elite_insert(elites, x, fx, max_elites):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        # light dedup (avoid many near-identical points)
        filtered = []
        for f, p in elites:
            ok = True
            for f2, p2 in filtered:
                # relative L1 distance threshold
                d = 0.0
                for i in range(dim):
                    s = spans[i] if spans[i] > 0 else 1.0
                    d += abs(p[i] - p2[i]) / s
                if d < 1e-6:
                    ok = False
                    break
            if ok:
                filtered.append((f, p))
            if len(filtered) >= max_elites:
                break
        return filtered

    # Pattern search / coordinate search from x with adaptive steps
    def local_refine(x_start, f_start, step0, iters_limit):
        x = list(x_start)
        fx = f_start
        steps = step0[:]  # per-dimension
        # success-based adaptation
        shrink = 0.6
        grow = 1.15
        min_step = [max(1e-15, (spans[i] * 1e-12) if spans[i] > 0 else 1e-15) for i in range(dim)]

        # probe multipliers: try big then smaller to get "line-search-ish" behavior
        mults = (1.0, 0.5, 0.25)

        for _ in range(iters_limit):
            if time.time() >= deadline:
                break

            improved_any = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= deadline:
                    break

                lo, hi = bounds[i]
                best_i_fx = fx
                best_i_val = x[i]

                si = steps[i]
                if si < min_step[i]:
                    si = min_step[i]
                    steps[i] = si

                # Try +/- with several step multipliers
                for m in mults:
                    d = si * m
                    for sgn in (-1.0, 1.0):
                        xi = x[i] + sgn * d
                        if xi < lo or xi > hi:
                            continue
                        xt = list(x)
                        xt[i] = xi
                        fxt = evaluate(xt)
                        if fxt < best_i_fx:
                            best_i_fx = fxt
                            best_i_val = xi

                if best_i_fx < fx:
                    x[i] = best_i_val
                    fx = best_i_fx
                    improved_any = True
                    steps[i] = min(spans[i] if spans[i] > 0 else steps[i] * grow, steps[i] * grow)

            if not improved_any:
                # global shrink if no coordinate helped
                for i in range(dim):
                    steps[i] = max(min_step[i], steps[i] * shrink)
                # stop if steps are extremely small across the board
                tiny = True
                for i in range(dim):
                    if steps[i] > min_step[i] * 10.0:
                        tiny = False
                        break
                if tiny:
                    break

        return x, fx

    # ---------- initialization ----------
    # Evaluate a small LHS batch quickly to get decent starting elites
    # batch size scaled by dimension but capped for time safety
    n0 = max(8, min(40, 4 * dim))
    elites = []
    best = float("inf")
    best_x = None

    for x in lhs_samples(n0):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites=max(5, min(25, 2 * dim)))

    # fallback if everything failed
    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)
        elites = [(best, best_x)]

    # Base step sizes
    base_steps = [0.3 * s if s > 0 else 1.0 for s in spans]

    # ---------- main loop ----------
    # Cycle between: refine elites, perturb elites, and occasional global LHS refresh
    phase = 0
    while time.time() < deadline:
        phase += 1

        # 1) Local refinement from a randomly chosen elite (bias towards best)
        # Choose index with quadratic bias to smaller indices (better points)
        m = len(elites)
        if m == 0:
            elites = [(best, best_x)]
            m = 1

        r = random.random()
        idx = int((r * r) * m)  # more likely near 0
        fx0, x0 = elites[idx]

        # Start steps: smaller for better points, bigger for worse
        rank_scale = 0.15 + 0.85 * (idx / max(1, m - 1))
        step0 = [max(1e-15, base_steps[i] * (0.25 + 1.5 * rank_scale)) for i in range(dim)]

        x1, f1 = local_refine(x0, fx0, step0, iters_limit=3 + dim)

        if f1 < best:
            best, best_x = f1, x1
        elites = elite_insert(elites, x1, f1, max_elites=max(5, min(25, 2 * dim)))

        if time.time() >= deadline:
            break

        # 2) Perturb around current best / a top elite (acts like a restart near good regions)
        topk = min(len(elites), max(3, dim // 2))
        pick = random.randrange(topk)
        fxp, xp = elites[pick]

        # sigma proportional to span, shrinking as phase grows
        # (keeps exploration early, exploitation later)
        decay = 1.0 / (1.0 + 0.03 * phase)
        sigmas = []
        for i in range(dim):
            s = spans[i] if spans[i] > 0 else 1.0
            # between ~2% and ~20% of span depending on rank and decay
            frac = (0.02 + 0.18 * (pick / max(1, topk - 1))) * decay
            sigmas.append(max(1e-15, frac * s))
        x2 = rand_gauss_around(xp, sigmas)
        f2 = evaluate(x2)

        if f2 < best:
            best, best_x = f2, x2
        elites = elite_insert(elites, x2, f2, max_elites=max(5, min(25, 2 * dim)))

        if time.time() >= deadline:
            break

        # 3) Occasionally inject new global points (fresh coverage)
        if phase % 7 == 0:
            nG = max(6, min(25, 2 * dim))
            for xg in lhs_samples(nG):
                if time.time() >= deadline:
                    break
                fg = evaluate(xg)
                if fg < best:
                    best, best_x = fg, xg
                elites = elite_insert(elites, xg, fg, max_elites=max(5, min(25, 2 * dim)))

        # 4) Rare full random restart (helps on deceptive landscapes)
        if phase % 17 == 0 and time.time() < deadline:
            xr = rand_uniform()
            fr = evaluate(xr)
            if fr < best:
                best, best_x = fr, xr
            elites = elite_insert(elites, xr, fr, max_elites=max(5, min(25, 2 * dim)))

    return best
