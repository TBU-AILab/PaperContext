import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Key changes vs the provided algorithms:
    - Uses a compact surrogate-assisted trust-region scheme (RBF over a small archive)
      to propose candidates that are likely good (sample-efficient).
    - Keeps strong global exploration via LHS-like seeding + periodic random injections.
    - Uses reflection bounds handling (robust, avoids border-sticking).
    - Adds a cheap local pattern search around the incumbent best.

    Returns:
        best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]
    active = [i for i in range(dim) if not fixed[i]]
    adim = len(active)

    # --- helpers ---
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def reflect_to_bounds(x):
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
                continue
            a, b = lo[i], hi[i]
            w = b - a
            if w <= 0.0:
                x[i] = a
                continue
            y = x[i] - a
            y = y % (2.0 * w)
            if y > w:
                y = 2.0 * w - y
            x[i] = a + y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                x[i] = lo[i] + random.random() * span[i]
        return x

    def dist2_active(x, y):
        s = 0.0
        for k in active:
            d = x[k] - y[k]
            s += d * d
        return s

    def median(lst):
        n = len(lst)
        if n == 0:
            return 1.0
        a = sorted(lst)
        mid = n // 2
        if n & 1:
            return a[mid]
        return 0.5 * (a[mid - 1] + a[mid])

    # Degenerate case
    if adim == 0:
        x = [lo[i] for i in range(dim)]
        return safe_eval(x)

    # --- archive of evaluated points ---
    X = []
    Y = []

    best = float("inf")
    best_x = rand_uniform()

    def push(x, y):
        nonlocal best, best_x
        X.append(x[:])
        Y.append(y)
        if y < best:
            best = y
            best_x = x[:]

    # --- LHS-like seeding (cheap) + opposites ---
    seed_n = min(30 + 5 * adim, 160)
    strata = []
    for i in range(dim):
        if fixed[i]:
            strata.append([lo[i]] * seed_n)
        else:
            perm = list(range(seed_n))
            random.shuffle(perm)
            vals = []
            for j in range(seed_n):
                u = (perm[j] + random.random()) / seed_n
                vals.append(lo[i] + u * span[i])
            strata.append(vals)

    for j in range(seed_n):
        if time.time() >= deadline:
            return best
        x = [strata[i][j] for i in range(dim)]
        if (j & 1) == 1:
            # opposite to diversify
            for i in range(dim):
                if not fixed[i]:
                    x[i] = lo[i] + (hi[i] - x[i])
        reflect_to_bounds(x)
        y = safe_eval(x)
        push(x, y)

    # --- trust-region state ---
    # radius relative to average active span
    avg_span = sum(span[i] for i in active) / max(1, adim)
    radius = 0.35 * (avg_span if avg_span > 0 else 1.0)
    r_min = 1e-14 * (avg_span if avg_span > 0 else 1.0)
    r_max = 1.5 * (avg_span if avg_span > 0 else 1.0)

    # local search step (per coordinate)
    step = [0.08 * s for s in span]
    step_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # --- RBF surrogate (local, weighted) ---
    # We use a local RBF predictor:
    #   pred(x) = sum_i w_i * y_i / sum_i w_i,  w_i = exp(-d^2/(2h^2))
    # h is median distance to neighbors to adapt scaling.
    def rbf_predict(x, neigh_idx, h2):
        num = 0.0
        den = 0.0
        inv2h2 = 1.0 / max(1e-30, 2.0 * h2)
        for idx in neigh_idx:
            d2 = dist2_active(x, X[idx])
            w = math.exp(-d2 * inv2h2)
            num += w * Y[idx]
            den += w
        if den <= 1e-300:
            return float("inf")
        return num / den

    def propose_from_surrogate(center, local_idx, h2, n_cand):
        # generate candidates in a ball around center; pick lowest predicted
        best_pred = float("inf")
        best_cand = None

        for _ in range(n_cand):
            x = center[:]
            # sample random direction in active space, with radius * U^(1/d)
            # use gaussian direction
            g = [random.gauss(0.0, 1.0) for _ in range(adim)]
            ng = math.sqrt(sum(v*v for v in g))
            if ng <= 0.0:
                continue
            scale = radius * (random.random() ** (1.0 / max(1, adim)))
            for t, i in enumerate(active):
                x[i] += scale * (g[t] / ng)

            # occasional coordinate perturbations (anisotropy robustness)
            if random.random() < 0.35:
                i = random.choice(active)
                x[i] += random.gauss(0.0, radius * 0.35)

            reflect_to_bounds(x)

            # encourage exploration a bit: subtract small term based on distance to archive
            # (simple "repulsion" so we don't resample same spot)
            pred = rbf_predict(x, local_idx, h2)
            # distance to nearest known point
            d2min = float("inf")
            for idx in local_idx:
                d2 = dist2_active(x, X[idx])
                if d2 < d2min:
                    d2min = d2
            pred_explore = pred - 0.01 * math.sqrt(max(0.0, d2min))  # tiny bias
            if pred_explore < best_pred:
                best_pred = pred_explore
                best_cand = x

        return best_cand

    # --- main loop ---
    it = 0
    stall = 0
    last_best = best

    # cap archive for speed
    MAX_ARC = max(80, min(420, 40 * adim + 120))

    while time.time() < deadline:
        it += 1

        # keep archive bounded by retaining best + some diverse points
        if len(X) > MAX_ARC:
            # keep top K best plus random remainder for diversity
            order = sorted(range(len(Y)), key=lambda i: Y[i])
            keep_best = min(len(order), max(30, MAX_ARC // 2))
            keep = set(order[:keep_best])
            # add random others
            while len(keep) < MAX_ARC:
                keep.add(random.randrange(len(Y)))
            X = [X[i] for i in keep]
            Y = [Y[i] for i in keep]

        # build local neighborhood around current best
        n = len(X)
        # choose k nearest to best_x in active dims
        k = min(n, max(10, 3 * adim + 8))
        dlist = [(dist2_active(best_x, X[i]), i) for i in range(n)]
        dlist.sort(key=lambda t: t[0])
        local_idx = [idx for _, idx in dlist[:k]]

        # kernel scale h^2 from median neighbor distance
        dvals = [d for d, _ in dlist[1:k]]  # exclude itself
        h2 = median(dvals)
        if h2 <= 1e-30:
            # if all collapsed, tie to radius
            h2 = max(1e-18, radius * radius)

        # decide how to sample this iteration
        # More surrogate steps when stalled; otherwise mix exploration.
        if stall > max(10, 2 * adim):
            n_cand = 26 + 4 * adim
            use_global = 0.10
        else:
            n_cand = 14 + 2 * adim
            use_global = 0.18

        x_try = None
        if random.random() < use_global:
            # global exploration
            x_try = rand_uniform()
        else:
            x_try = propose_from_surrogate(best_x, local_idx, h2, n_cand)
            if x_try is None:
                x_try = rand_uniform()

        y_try = safe_eval(x_try)
        push(x_try, y_try)

        improved = (best < last_best - 1e-15)
        if improved:
            last_best = best
            stall = 0
            # if improvement, expand trust region slightly
            radius = min(r_max, radius * 1.15)
        else:
            stall += 1
            # if no improvement, shrink slowly
            radius = max(r_min, radius * 0.985)

        # cheap local pattern search occasionally (uses real evals)
        if stall > max(8, 2 * adim) and time.time() + 0.002 < deadline:
            trials = min(10 + dim, 26)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.75:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    sd = max(step_min[d], step[d])
                    x[d] += sd if random.random() < 0.5 else -sd
                else:
                    # random direction in active dims
                    g = [random.gauss(0.0, 1.0) for _ in range(adim)]
                    ng = math.sqrt(sum(v*v for v in g))
                    if ng <= 0.0:
                        continue
                    sc = radius * (0.25 + 0.75 * random.random())
                    for t, i in enumerate(active):
                        x[i] += sc * (g[t] / ng)
                reflect_to_bounds(x)
                y = safe_eval(x)
                push(x, y)
                if y <= best + 1e-15:
                    stall = 0
                    radius = min(r_max, max(radius, 0.9 * radius + 0.1 * (0.45 * avg_span if avg_span > 0 else 1.0)))
                    break
                else:
                    for d in range(dim):
                        if not fixed[d] and step[d] > step_min[d]:
                            step[d] *= 0.997

        # occasional restart-ish diversification if very stalled
        if stall > max(30, 6 * adim) and time.time() + 0.01 < deadline:
            stall = 0
            radius = min(r_max, max(radius, 0.6 * (avg_span if avg_span > 0 else 1.0)))
            # inject a few global points
            inject = min(10 + 2 * adim, 30)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = rand_uniform()
                y = safe_eval(x)
                push(x, y)

    return best
