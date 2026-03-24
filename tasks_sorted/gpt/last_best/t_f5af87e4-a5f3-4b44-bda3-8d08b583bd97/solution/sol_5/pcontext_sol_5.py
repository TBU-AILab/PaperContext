import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved stdlib-only time-bounded minimizer.

    Key upgrades vs previous:
      - Uses a strong (and simple) CMA-ES core (diagonal-covariance variant for speed/robustness)
      - Restarts with increasing population (IPOP-style) to escape local minima
      - Bound handling via reflection (better than clamping for search dynamics)
      - Occasional local coordinate search polishing of the best
      - Evaluation budget auto-managed by time; no external libs

    Returns:
      best (float): best objective value found within max_time
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    if dim <= 0:
        # Degenerate case: evaluate empty vector if allowed
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def is_finite(x):
        return not (math.isinf(x) or math.isnan(x))

    def reflect_into_bounds(x):
        # Reflect each coordinate into [lo, hi] (handles steps beyond bounds smoothly).
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            # reflection with period 2*(hi-lo)
            r = hi - lo
            p = 2.0 * r
            # map to [0, p)
            y = (v - lo) % p
            if y < 0.0:
                y += p
            if y <= r:
                x[i] = lo + y
            else:
                x[i] = hi - (y - r)

    def eval_f(x):
        # func may throw or return non-finite; handle robustly
        try:
            fx = float(func(x))
        except Exception:
            return float("inf")
        if not is_finite(fx):
            return float("inf")
        return fx

    # fast approx N(0,1): sum of uniforms - 6 (12 uniforms)
    def randn():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def center_point():
        return [0.5 * (lows[i] + highs[i]) for i in range(dim)]

    # ---------- best tracking ----------
    best = float("inf")
    best_x = None

    def consider(x):
        nonlocal best, best_x
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = list(x)
        return fx

    # ---------- cheap local polish ----------
    def local_polish(x0, f0, budget=30):
        nonlocal best, best_x
        x = list(x0)
        fx = float(f0)
        # start with moderate step, shrink
        step = 0.08
        evals = 0
        while evals < budget and time.perf_counter() < deadline:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.perf_counter() >= deadline or evals >= budget:
                    break
                if spans[i] == 0.0:
                    continue
                delta = step * safe_spans[i]

                xp = list(x); xp[i] += delta
                reflect_into_bounds(xp)
                fp = eval_f(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    if fp < best:
                        best, best_x = fp, list(x)
                    continue

                xm = list(x); xm[i] -= delta
                reflect_into_bounds(xm)
                fm = eval_f(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    if fm < best:
                        best, best_x = fm, list(x)

            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
        return x, fx

    # ---------- initial probes ----------
    if time.perf_counter() < deadline:
        consider(center_point())
    if time.perf_counter() < deadline:
        # a few random probes to seed best
        for _ in range(min(12, 2 * dim + 4)):
            if time.perf_counter() >= deadline:
                break
            consider(rand_point())

    if best_x is None:
        return float("inf")

    # ---------- diagonal CMA-ES with restarts ----------
    # We implement a diagonal-covariance CMA-ES-like scheme:
    #   - sample: x_k = m + sigma * D * z_k, z_k ~ N(0,I)
    #   - update m as weighted mean of best mu
    #   - update diagonal variance (D^2) via weighted z^2 statistics
    # This is not full CMA-ES but is very effective and much simpler.

    # weights (recomputed per lambda)
    def make_weights(lam):
        mu = lam // 2
        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        if s <= 0.0:
            w = [1.0 / mu] * mu
        else:
            w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return mu, w, mueff

    # run restarts until time runs out
    restart = 0
    # base sigma relative to box size
    base_sigma = 0.25  # relative; adapted per restart
    # when any dimension span is zero, it's fine (handled by reflection + D scaling)
    while time.perf_counter() < deadline:
        restart += 1

        # IPOP-ish population increase
        lam = int(4 + 3 * math.log(dim + 1.0))
        lam = max(8, lam)
        lam = int(lam * (2 ** (restart - 1)))
        lam = min(lam, 60 + 10 * dim)  # cap for speed

        mu, weights, mueff = make_weights(lam)

        # learning rates (diagonal version; conservative defaults)
        # Adaptation strength increases slightly with dim/mueff
        c_sigma = min(0.6, (mueff + 2.0) / (dim + mueff + 5.0))
        c_var = min(0.4, 2.0 / (dim + 2.0))  # variance learning

        # initialize mean near current best, with some randomization on restarts
        if restart == 1:
            m = list(best_x)
        else:
            m = list(best_x)
            # jitter mean to escape local basin
            for i in range(dim):
                if spans[i] == 0.0:
                    m[i] = lows[i]
                else:
                    m[i] += (randn() * 0.15) * safe_spans[i]
            reflect_into_bounds(m)

        # diagonal scaling D (std dev per coordinate, relative to spans)
        D = [1.0 for _ in range(dim)]

        # sigma
        sigma = base_sigma * (0.9 ** (restart - 1))
        # ensure sigma not too small
        sigma = max(1e-6, sigma)

        # evolution path-ish scalar for step-size control (diagonal simplification)
        p_sigma = 0.0

        # best in this restart to detect stagnation
        local_best = float("inf")
        local_best_time = time.perf_counter()
        stall_time = max(0.10, 0.18 * float(max_time))

        # CMA loop
        while time.perf_counter() < deadline:
            # sample population
            pop = []
            fits = []

            for _ in range(lam):
                if time.perf_counter() >= deadline:
                    break

                z = [randn() for _ in range(dim)]
                x = [0.0] * dim
                for i in range(dim):
                    if spans[i] == 0.0:
                        x[i] = lows[i]
                    else:
                        x[i] = m[i] + (sigma * D[i]) * z[i]
                reflect_into_bounds(x)

                fx = eval_f(x)
                pop.append((fx, x, z))
                fits.append(fx)

                if fx < best:
                    best = fx
                    best_x = list(x)

                if fx < local_best:
                    local_best = fx
                    local_best_time = time.perf_counter()

            if not pop:
                break

            # sort by fitness
            pop.sort(key=lambda t: t[0])

            # recombination to update mean
            m_old = m
            m = [0.0] * dim
            for i in range(mu):
                _, xi, _ = pop[i]
                wi = weights[i]
                for d in range(dim):
                    m[d] += wi * xi[d]

            # step-size control (scalar), based on normalized mean step
            # compute y = (m - m_old) / (sigma * D)
            # then use its norm to adjust sigma
            norm2 = 0.0
            for d in range(dim):
                if spans[d] == 0.0:
                    continue
                denom = sigma * (D[d] if D[d] > 1e-12 else 1e-12)
                yd = (m[d] - m_old[d]) / denom
                norm2 += yd * yd
            norm = math.sqrt(norm2)

            # update p_sigma as a smoothed norm (scalar)
            p_sigma = (1.0 - c_sigma) * p_sigma + c_sigma * norm

            # target norm ~ sqrt(dim) (heuristic)
            target = math.sqrt(max(1.0, float(dim)))
            if target > 0.0:
                # if norm > target => increase sigma, else decrease
                # use mild exponential update
                sigma *= math.exp(0.25 * (p_sigma / target - 1.0))
                sigma = max(1e-10, min(sigma, 2.0))  # keep sane

            # update diagonal variances D using weighted z^2 of selected individuals
            # Approx: D^2 <- (1-c_var) D^2 + c_var * sum(w_i * z_i^2)
            # We need z from selected best; but reflection distorts z a bit. Still works well.
            new_D2 = [0.0] * dim
            for i in range(mu):
                _, _, zi = pop[i]
                wi = weights[i]
                for d in range(dim):
                    new_D2[d] += wi * (zi[d] * zi[d])

            for d in range(dim):
                # keep D positive and bounded
                d2 = D[d] * D[d]
                d2 = (1.0 - c_var) * d2 + c_var * max(1e-12, new_D2[d])
                D[d] = math.sqrt(max(1e-12, min(1e6, d2)))

            # occasional polish of global best
            if best_x is not None and (deadline - time.perf_counter()) < 0.25 * float(max_time):
                # near end polish more
                bx, bf = local_polish(best_x, best, budget=24)
                # ensure best consistent
                if bf < best:
                    best, best_x = bf, list(bx)
            elif best_x is not None and random.random() < 0.10:
                bx, bf = local_polish(best_x, best, budget=10)
                if bf < best:
                    best, best_x = bf, list(bx)

            # restart criteria: long stall or sigma too small
            now = time.perf_counter()
            if (now - local_best_time) > stall_time:
                break
            if sigma < 1e-10:
                break

        # If very little time left, don't restart again
        if (deadline - time.perf_counter()) < 0.02 * float(max_time):
            break

    return float(best)
