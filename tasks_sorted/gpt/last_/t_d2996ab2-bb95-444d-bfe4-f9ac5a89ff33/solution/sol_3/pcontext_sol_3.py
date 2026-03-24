import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Key improvements vs previous version:
      - Adds a true local optimizer: bounded Nelder–Mead (simplex) with
        reflection/expansion/contraction/shrink (very strong for continuous boxes).
      - Keeps global exploration + restarts: quasi-Latin sampling + heavy-tail jumps.
      - Uses a small cache to avoid wasting evaluations on duplicates/near-duplicates.
      - Robust time slicing: always returns best found within max_time.

    Returns:
      best (float): best objective value found.
    """
    # ---------------- utilities ----------------
    start = time.time()
    deadline = start + max(0.0, float(max_time))
    eps = 1e-12

    def time_left():
        return time.time() < deadline

    def clip(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(hi[i] - lo[i], 0.0) for i in range(dim)]

    def project(x):
        return [clip(float(x[i]), lo[i], hi[i]) for i in range(dim)]

    def center():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # Cheap "Latin-ish": sample each dimension from a random bin among m
    def stratified(m):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
                continue
            b = random.randrange(m)
            a = lo[i] + (b / m) * span[i]
            c = lo[i] + ((b + 1) / m) * span[i]
            x[i] = random.uniform(a, c)
        return x

    # --- evaluation with small cache (quantized) ---
    # Quantization size: ~1e-6 of span (or absolute fallback).
    qstep = []
    for i in range(dim):
        s = span[i]
        if s <= eps:
            qstep.append(0.0)
        else:
            qstep.append(max(1e-12, 1e-6 * s))

    cache = {}  # key -> best known f for that quantized point (store float)
    CACHE_MAX = 20000

    def key_of(x):
        k = []
        for i in range(dim):
            if qstep[i] == 0.0:
                k.append(0)
            else:
                k.append(int(round((x[i] - lo[i]) / qstep[i])))
        return tuple(k)

    def safe_eval(x):
        x = project(x)
        k = key_of(x)
        if k in cache:
            return cache[k], x

        try:
            y = func(x)
            if y is None:
                y = float("inf")
            y = float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")

        # bounded cache growth: random eviction when too big
        if len(cache) >= CACHE_MAX:
            # delete a few random keys cheaply
            for _ in range(20):
                if not cache:
                    break
                rk = next(iter(cache))
                del cache[rk]
        cache[k] = y
        return y, x

    # ---------------- global state ----------------
    best = float("inf")
    x_best = center()

    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    # Keep an elite pool (f, x)
    elites = [(best, x_best)]
    ELITE_MAX = max(12, 4 * dim)

    def push_elite(fx, x):
        nonlocal best, x_best, elites
        if fx < best:
            best, x_best = fx, x[:]
        elites.append((fx, x[:]))
        # prune
        if len(elites) > 3 * ELITE_MAX:
            elites.sort(key=lambda t: t[0])
            elites = elites[:ELITE_MAX]

    # ---------------- Nelder–Mead (bounded) ----------------
    def nelder_mead(x0, f0, max_evals, init_scale):
        """
        Run a bounded Nelder–Mead starting from x0.
        Returns (best_f, best_x, evals_used).
        """
        if max_evals <= 0:
            return f0, x0, 0

        # Build simplex: x0 + scale * e_i (projected)
        simplex = [(f0, x0[:])]
        evals = 0

        for i in range(dim):
            if span[i] <= eps:
                continue
            xi = x0[:]
            step = init_scale * span[i]
            if step <= 0.0:
                continue
            # push inside bounds (toward feasible)
            if xi[i] + step <= hi[i]:
                xi[i] = xi[i] + step
            elif xi[i] - step >= lo[i]:
                xi[i] = xi[i] - step
            else:
                # tiny fallback
                xi[i] = clip(xi[i] + 0.01 * span[i], lo[i], hi[i])
            fi, xi = safe_eval(xi)
            evals += 1
            simplex.append((fi, xi))
            if evals >= max_evals or not time_left():
                break

        # If simplex too small (e.g., many fixed dims), just return
        if len(simplex) < 2:
            simplex.sort(key=lambda t: t[0])
            return simplex[0][0], simplex[0][1], evals

        # NM coefficients
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho   = 0.5   # contraction
        sigma = 0.5   # shrink

        # Main loop
        while evals < max_evals and time_left():
            simplex.sort(key=lambda t: t[0])
            f_best, x_best_loc = simplex[0]
            f_worst, x_worst = simplex[-1]
            f_second = simplex[-2][0]

            # centroid of all but worst
            centroid = [0.0] * dim
            m = len(simplex) - 1
            for _, x in simplex[:-1]:
                for j in range(dim):
                    centroid[j] += x[j]
            for j in range(dim):
                centroid[j] /= max(1, m)

            # reflection
            xr = [centroid[j] + alpha * (centroid[j] - x_worst[j]) for j in range(dim)]
            fr, xr = safe_eval(xr)
            evals += 1

            if fr < f_best:
                # expansion
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(dim)]
                fe, xe = safe_eval(xe)
                evals += 1
                if fe < fr:
                    simplex[-1] = (fe, xe)
                else:
                    simplex[-1] = (fr, xr)
            elif fr < f_second:
                simplex[-1] = (fr, xr)
            else:
                # contraction
                if fr < f_worst:
                    # outside contraction
                    xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[j] - rho * (centroid[j] - x_worst[j]) for j in range(dim)]
                fc, xc = safe_eval(xc)
                evals += 1

                if fc < f_worst:
                    simplex[-1] = (fc, xc)
                else:
                    # shrink
                    x0s = simplex[0][1]
                    new_simplex = [simplex[0]]
                    for k in range(1, len(simplex)):
                        xs = [x0s[j] + sigma * (simplex[k][1][j] - x0s[j]) for j in range(dim)]
                        fs, xs = safe_eval(xs)
                        evals += 1
                        new_simplex.append((fs, xs))
                        if evals >= max_evals or not time_left():
                            break
                    simplex = new_simplex

            # early exit if simplex is tiny in all dims
            if len(simplex) >= 2:
                # measure max coordinate spread
                max_spread = 0.0
                for j in range(dim):
                    mn = min(p[1][j] for p in simplex)
                    mx = max(p[1][j] for p in simplex)
                    max_spread = max(max_spread, mx - mn)
                if max_spread <= 1e-10 * (max(span) if max(span) > 0 else 1.0):
                    break

        simplex.sort(key=lambda t: t[0])
        return simplex[0][0], simplex[0][1], evals

    # ---------------- main search ----------------
    # Initial global sampling
    init_n = max(30, 15 * dim)
    m_bins = max(4, int(math.sqrt(init_n)) + 1)

    for k in range(init_n):
        if not time_left():
            return best
        x = stratified(m_bins) if (k % 2 == 0) else rand_uniform()
        fx, x = safe_eval(x)
        push_elite(fx, x)

    # Main time loop: alternate (global proposals) + (local NM from elites)
    no_improve_rounds = 0
    round_id = 0

    while time_left():
        round_id += 1

        elites.sort(key=lambda t: t[0])
        elites = elites[:min(len(elites), ELITE_MAX)]

        improved_this_round = False

        # ---- Global proposals (mixture) ----
        # Heavy-tail around best + random restart candidates
        G = max(10, 5 * dim)
        for _ in range(G):
            if not time_left():
                return best

            r = random.random()
            if r < 0.35:
                x = rand_uniform()
            elif r < 0.60:
                x = stratified(max(4, 2 * dim))
            else:
                # heavy-tail jump around a random elite (bias toward better)
                idx = int((random.random() ** 2.0) * max(1, len(elites) - 1))
                base = elites[idx][1]
                x = base[:]
                for j in range(dim):
                    if span[j] <= eps:
                        x[j] = lo[j]
                        continue
                    # Cauchy-like step
                    u = random.random()
                    step = 0.15 * span[j] * math.tan(math.pi * (u - 0.5))
                    # small gaussian blend
                    step += random.gauss(0.0, 0.05 * span[j])
                    x[j] = clip(x[j] + step, lo[j], hi[j])

            fx, x = safe_eval(x)
            prev_best = best
            push_elite(fx, x)
            if best < prev_best:
                improved_this_round = True

        # ---- Local refinement: bounded Nelder–Mead on top elites ----
        # Allocate a slice of evaluations; keep small so we can restart often.
        # As dim grows, NM needs more, but time is limited: do a few short runs.
        nm_starts = 1 if dim > 25 else 2
        nm_starts = min(nm_starts, len(elites))
        # Initial simplex scale: start moderate, shrink as rounds progress
        init_scale = max(0.02, 0.18 / (1.0 + 0.08 * round_id))

        # evaluation budget per run (not time; still time-checked inside)
        per_run = max(20 + 6 * dim, 60)

        for s in range(nm_starts):
            if not time_left():
                return best
            x0 = elites[s][1]
            f0 = elites[s][0]

            f_loc, x_loc, _ = nelder_mead(x0, f0, per_run, init_scale)
            prev_best = best
            push_elite(f_loc, x_loc)
            if best < prev_best:
                improved_this_round = True

        # ---- Restart logic ----
        if improved_this_round:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= 4:
            # Inject a stronger diversification burst
            burst = max(12, 6 * dim)
            for _ in range(burst):
                if not time_left():
                    return best
                x = rand_uniform() if random.random() < 0.7 else stratified(max(6, 3 * dim))
                fx, x = safe_eval(x)
                push_elite(fx, x)
            no_improve_rounds = 0

    return best
