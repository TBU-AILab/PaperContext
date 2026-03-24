import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer: hybrid of
    - low-discrepancy-like exploration (stratified + occasional global samples)
    - adaptive local search (coordinate + full-dimensional Gaussian steps)
    - pattern search refinement around best
    - robust restarts with shrinking neighborhood

    Self-contained: uses only Python stdlib.
    Returns best (minimum) objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- helpers ----------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # "stratified" point: tries to cover space better than pure random
    # using a simple shuffled grid index per dimension (no numpy/halton needed).
    # k grows over time (more bins) to densify exploration.
    def stratified_point(k, perm_tables, ctr):
        x = [0.0] * dim
        for i in range(dim):
            # pick bin index from a permutation table, then jitter inside the bin
            p = perm_tables[i]
            idx = p[ctr % k]
            u = (idx + random.random()) / k
            x[i] = lo[i] + u * spans[i]
        return x

    # safe evaluation (in case func can error on some inputs)
    def eval_f(x):
        try:
            v = func(x)
            # guard NaN/inf
            if v is None or isinstance(v, complex):
                return float("inf")
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return float(v)
        except Exception:
            return float("inf")

    # ---------- initial sampling (quick global scan) ----------
    # Spend a small fraction of time to seed a decent starting point.
    best = float("inf")
    best_x = None

    # dynamic sampling budget: small but enough for higher dims
    seed_budget = max(20, 10 * dim)
    for _ in range(seed_budget):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        return float("inf")

    # ---------- hybrid local search state ----------
    x = list(best_x)
    fx = best

    # step sizes: start moderate, then adapt
    # base step is a fraction of span; if span=0, use 1.0
    base_step = [0.2 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    step = list(base_step)
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    max_step = [0.5 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # pattern-search radius around best
    radius = 0.25
    min_radius = 1e-9

    # restart control
    no_improve = 0
    patience = max(200, 60 * dim)

    # acceptance temperature for occasional uphill moves (very mild SA)
    T = 0.1
    Tmin = 1e-14
    alpha = 0.97

    # stratified exploration tables
    k = max(4, int(round(dim ** 0.5)) + 3)  # bins per dimension
    perm_tables = []
    for _ in range(dim):
        p = list(range(k))
        random.shuffle(p)
        perm_tables.append(p)
    strat_ctr = 0

    # ---------- main loop ----------
    it = 0
    accepted = 0
    tried = 0

    while time.time() < deadline:
        it += 1

        # Occasionally increase stratification resolution as time progresses
        # (rebuild permutation tables when k changes).
        if it % (200 + 20 * dim) == 0:
            elapsed = time.time() - t0
            frac = elapsed / max(1e-9, float(max_time))
            # slowly densify: k from ~4 up to ~20
            new_k = min(20, max(4, int(4 + 16 * frac)))
            if new_k != k:
                k = new_k
                perm_tables = []
                for _ in range(dim):
                    p = list(range(k))
                    random.shuffle(p)
                    perm_tables.append(p)
                strat_ctr = 0

        # Mix of moves:
        # 1) local coordinate move (cheap, good for ill-conditioned problems)
        # 2) local full-dimensional Gaussian
        # 3) pattern search around current best
        # 4) global/stratified sample (escape + exploration)
        r = random.random()

        candidate = None

        if r < 0.42:
            # Coordinate step: change 1-3 coordinates
            candidate = list(x)
            m = 1 if dim == 1 else (2 if dim < 8 else 3)
            for _ in range(m):
                j = random.randrange(dim)
                # symmetric step with occasional long jump
                s = step[j] * (1.0 if random.random() < 0.9 else 4.0)
                candidate[j] += random.gauss(0.0, s)
            clip_inplace(candidate)

        elif r < 0.74:
            # Full-dimensional Gaussian step
            candidate = list(x)
            long_jump = (random.random() < 0.08)
            for j in range(dim):
                s = step[j] * (1.0 if not long_jump else 3.5)
                candidate[j] += random.gauss(0.0, s)
            clip_inplace(candidate)

        elif r < 0.92:
            # Pattern search around best_x (exploit)
            candidate = list(best_x)
            # shrink radius over time, but keep some minimum
            # use a random direction in {-1,0,+1} per coordinate
            for j in range(dim):
                if random.random() < 0.35:
                    direction = random.choice((-1.0, 1.0))
                    candidate[j] += direction * radius * (spans[j] if spans[j] > 0 else 1.0)
            clip_inplace(candidate)

        else:
            # Global sample: mostly stratified, sometimes pure random
            if random.random() < 0.75:
                candidate = stratified_point(k, perm_tables, strat_ctr)
                strat_ctr += 1
            else:
                candidate = rand_point()

        fn = eval_f(candidate)
        tried += 1

        # Accept logic: greedy + mild SA for uphill moves
        d = fn - fx
        if d <= 0:
            accept = True
        else:
            if T <= Tmin:
                accept = False
            else:
                # cap to avoid overflow for huge d/T
                z = -d / T
                if z < -60.0:
                    accept = False
                else:
                    accept = (random.random() < math.exp(z))

        if accept:
            x, fx = candidate, fn
            accepted += 1

        # Update best
        if fn < best:
            best = fn
            best_x = list(candidate)
            no_improve = 0

            # When we improve, slightly expand steps for faster progress (up to max)
            for j in range(dim):
                step[j] = min(max_step[j], step[j] * 1.08)
            radius = min(0.5, radius * 1.05)
        else:
            no_improve += 1

        # Adapt steps + temperature periodically
        if it % max(30, 8 * dim) == 0:
            acc_rate = accepted / tried if tried else 0.0

            # Keep acceptance moderate; adjust step sizes
            if acc_rate < 0.18:
                for j in range(dim):
                    step[j] = max(min_step[j], step[j] * 0.72)
                radius = max(min_radius, radius * 0.85)
            elif acc_rate > 0.45:
                for j in range(dim):
                    step[j] = min(max_step[j], step[j] * 1.25)
                radius = min(0.75, radius * 1.10)

            # Cool temperature slowly; reheat a bit if stuck late
            T = max(Tmin, T * alpha)

            elapsed = time.time() - t0
            if elapsed > 0.65 * float(max_time) and no_improve > patience // 2:
                T = min(0.25, T * 2.0)

            accepted = 0
            tried = 0

        # Restart if stuck
        if no_improve >= patience:
            # restart near best with shrinking neighborhood + occasional full random
            if random.random() < 0.8:
                x = list(best_x)
                for j in range(dim):
                    s = 0.35 * step[j]
                    x[j] += random.gauss(0.0, s)
                clip_inplace(x)
            else:
                x = rand_point()

            fx = eval_f(x)
            no_improve = 0

            # Re-tune: reduce steps a bit to focus, but keep exploratory ability
            for j in range(dim):
                step[j] = max(min_step[j], min(max_step[j], step[j] * 0.85))
            radius = max(min_radius, radius * 0.9)
            T = max(T, 0.08)

    return best
