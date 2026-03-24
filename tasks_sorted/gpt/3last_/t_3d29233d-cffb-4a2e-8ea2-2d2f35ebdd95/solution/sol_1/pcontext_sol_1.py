import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization:
      - LHS-like initialization
      - (mu, lambda) Evolution Strategy with *diagonal* self-adapting step sizes
      - Rank-based recombination of top mu offspring
      - Per-dimension success adaptation (keeps useful coordinates moving)
      - Occasional short coordinate/pattern local search around the incumbent
      - Stagnation-triggered multi-start with adaptive exploration radius

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ----------------- helpers -----------------
    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def eval_f(x):
        return float(func(x))  # list is acceptable per template example

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def lhs_init(n):
        # Simple stratified (LHS-style) sampling per dimension.
        strata = []
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            strata.append(perm)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                a = (strata[d][i] + random.random()) / n
                lo, hi = bounds[d]
                x[d] = lo + a * (hi - lo)
            pts.append(x)
        return pts

    def mutate(x, sigmas):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if hi <= lo:
                y[i] = lo
            else:
                y[i] = clip(y[i] + random.gauss(0.0, sigmas[i]), lo, hi)
        return y

    def pattern_local_search(x0, f0, steps, max_sweeps=2):
        # Lightweight coordinate + pattern move search.
        x = x0[:]
        fx = f0
        for _ in range(max_sweeps):
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            # Coordinate probes
            for i in order:
                si = steps[i]
                if si <= 0.0:
                    continue
                lo, hi = bounds[i]
                xi = x[i]

                # +step
                xp = x[:]
                xp[i] = clip(xi + si, lo, hi)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                # -step
                xm = x[:]
                xm[i] = clip(xi - si, lo, hi)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            # Pattern move (useful when improvements align)
            if improved:
                # small push along last move direction: x + (x - x0)
                d = [x[i] - x0[i] for i in range(dim)]
                xt = [clip(x[i] + d[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft

            if not improved:
                for i in range(dim):
                    steps[i] *= 0.5

            x0 = x[:]  # update base for pattern
        return x, fx

    # ----------------- time mgmt -----------------
    start = time.time()
    deadline = start + float(max_time)

    # Degenerate bounds / widths
    widths = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    # Base sigma scale
    base_sigma = [w * 0.20 if w > 0 else 0.0 for w in widths]

    # ES parameters
    lam = max(12, 6 + int(4 * math.log(dim + 1)))          # offspring
    mu = max(3, lam // 4)                                   # parents to recombine
    # rank weights (positive, normalized)
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]

    # Adaptation knobs
    sigma_min_frac = 1e-8
    sigma_max_frac = 0.60
    # per-dimension success counters (for coordinate-wise adaptation)
    succ = [0.0] * dim
    succ_decay = 0.90
    target = 0.20  # desired success rate
    adapt_strength = 0.25

    # Initialization budget
    n0 = max(2 * lam, 24)

    best = float("inf")
    best_x = None

    # --- Initial sampling (LHS + random) ---
    init_pts = lhs_init(n0)
    for _ in range(max(4, dim // 2)):
        init_pts.append(rand_uniform_vec())

    for x in init_pts:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    # Start ES from the best initial point
    x = best_x[:] if best_x is not None else rand_uniform_vec()
    fx = best
    sigmas = base_sigma[:]

    # Stagnation controls
    it = 0
    no_global = 0
    no_parent = 0
    refine_every = 18
    restart_after = 40  # stagnation in iterations

    while time.time() < deadline:
        it += 1

        # ---- generate offspring ----
        offspring = []
        improved_any = False

        for _ in range(lam):
            if time.time() >= deadline:
                return best
            y = mutate(x, sigmas)
            fy = eval_f(y)
            offspring.append((fy, y))

            if fy < fx:
                improved_any = True
                # update per-dimension success based on what moved
                for i in range(dim):
                    if y[i] != x[i]:
                        succ[i] = succ_decay * succ[i] + (1.0 - succ_decay) * 1.0
            else:
                # mild decay when not improved
                for i in range(dim):
                    succ[i] = succ_decay * succ[i]

        offspring.sort(key=lambda t: t[0])

        # ---- (mu, lambda) recombination ----
        top = offspring[:mu]
        x_new = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += w[k] * top[k][1][i]
            x_new[i] = clip(s, bounds[i][0], bounds[i][1])
        fx_new = eval_f(x_new)

        # Select new parent as the better between recombined and best offspring
        best_off_f, best_off_x = offspring[0]
        if best_off_f <= fx_new:
            cand_x, cand_f = best_off_x, best_off_f
        else:
            cand_x, cand_f = x_new, fx_new

        if cand_f < fx:
            x, fx = cand_x[:], cand_f
            no_parent = 0
        else:
            no_parent += 1
            # still move to the best offspring to avoid being stuck on recombination artifacts
            x, fx = best_off_x[:], best_off_f

        # Update global best
        if fx < best:
            best, best_x = fx, x[:]
            no_global = 0
        else:
            no_global += 1

        # ---- step-size adaptation (coordinate-wise) ----
        # If succ[i] > target => increase sigma[i], else decrease.
        for i in range(dim):
            w_i = widths[i]
            if w_i <= 0.0:
                sigmas[i] = 0.0
                continue

            # succ[i] is an EWMA in [0,1], drive towards target
            delta = succ[i] - target
            # exponential update keeps positivity and is scale-free
            sigmas[i] *= math.exp(adapt_strength * delta)

            # clamp
            sig_min = w_i * sigma_min_frac
            sig_max = w_i * sigma_max_frac
            if sigmas[i] < sig_min:
                sigmas[i] = sig_min
            elif sigmas[i] > sig_max:
                sigmas[i] = sig_max

        # If the whole batch didn't improve parent, shrink a bit globally
        if not improved_any:
            for i in range(dim):
                sigmas[i] *= 0.85

        # ---- occasional local refinement around global best ----
        if (it % refine_every) == 0 and best_x is not None and time.time() < deadline:
            steps = [max(sigmas[i], widths[i] * 0.03) for i in range(dim)]
            rx, rf = pattern_local_search(best_x, best, steps, max_sweeps=2)
            if rf < best:
                best, best_x = rf, rx[:]
                x, fx = best_x[:], best
                no_global = 0
                no_parent = 0

        # ---- restart on stagnation ----
        if (no_global >= restart_after) and time.time() < deadline and best_x is not None:
            no_global = 0
            no_parent = 0

            # Mix best with random; exploration radius grows with stagnation
            r = rand_uniform_vec()
            alpha = 0.70  # weight on best
            x = [clip(alpha * best_x[i] + (1.0 - alpha) * r[i], bounds[i][0], bounds[i][1])
                 for i in range(dim)]
            fx = eval_f(x)

            # reset sigmas moderately (but keep some learned anisotropy)
            for i in range(dim):
                sigmas[i] = max(sigmas[i], base_sigma[i])
                # also avoid too tiny after long exploitation
                if widths[i] > 0:
                    sigmas[i] = max(sigmas[i], widths[i] * 1e-4)

            if fx < best:
                best, best_x = fx, x[:]

    return best
