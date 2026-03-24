import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization
    - (1+1)-ES style adaptive Gaussian steps (log-normal step-size)
    - Occasional coordinate search refinements
    - Random restarts to escape stagnation

    Returns:
        best (float): best (minimum) fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers (no external libs) ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        y = list(x)
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_func(x):
        # func expects "array-like"; we pass a plain list
        return float(func(x))

    # --- initial best ---
    best = float("inf")
    best_x = None

    # Create a quick stratified batch (cheap LHS-like)
    # This usually beats pure random for early performance.
    init_n = max(8, 4 * dim)
    strata = list(range(init_n))
    per_dim_perm = []
    for _ in range(dim):
        p = strata[:]
        random.shuffle(p)
        per_dim_perm.append(p)

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample uniformly within stratum
            u = (per_dim_perm[i][j] + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        fx = eval_func(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        best_x = rand_uniform_vec()
        best = eval_func(best_x)

    # --- main optimization loop ---
    # Start step size as a fraction of search range.
    base_sigma = [0.2 * s if s > 0 else 1.0 for s in spans]
    sigma_scale = 1.0

    # Adaptation parameters
    tau = 1.0 / math.sqrt(max(1.0, dim))       # global step-size learning rate
    tau_i = 1.0 / math.sqrt(max(1.0, 2.0*dim)) # per-dimension perturbation scale
    stall_limit = 50 + 10 * dim

    stall = 0
    it = 0

    # Track a "current" point for local search; restart sometimes
    cur_x = list(best_x)
    cur_f = best

    while time.time() < deadline:
        it += 1

        # Occasional random restart if stalled
        if stall >= stall_limit:
            # partial restart: mix best with random to keep some structure
            r = rand_uniform_vec()
            alpha = 0.3 + 0.4 * random.random()  # 0.3..0.7
            cur_x = clamp([alpha * best_x[i] + (1.0 - alpha) * r[i] for i in range(dim)])
            cur_f = eval_func(cur_x)
            stall = 0
            # reset step size modestly
            sigma_scale = 1.0

        # Log-normal step-size update (global), plus some per-dim noise
        # Similar spirit to evolution strategies but lightweight.
        global_noise = random.gauss(0.0, 1.0)
        sigma_scale *= math.exp(tau * global_noise)
        # keep within sane bounds
        if sigma_scale < 1e-6:
            sigma_scale = 1e-6
        elif sigma_scale > 1e3:
            sigma_scale = 1e3

        # Propose a candidate by Gaussian mutation
        cand = []
        for i in range(dim):
            per_noise = random.gauss(0.0, 1.0)
            step = base_sigma[i] * sigma_scale * math.exp(tau_i * per_noise) * random.gauss(0.0, 1.0)
            cand.append(cur_x[i] + step)

        cand = clamp(cand)
        f_cand = eval_func(cand)

        if f_cand < cur_f:
            cur_x, cur_f = cand, f_cand
            stall = 0
            # success: slightly expand
            sigma_scale *= 1.05
        else:
            stall += 1
            # failure: slightly contract
            sigma_scale *= 0.97

        # Update global best
        if cur_f < best:
            best = cur_f
            best_x = list(cur_x)

        # Occasionally do a quick coordinate search around the current best
        # (helps on separable / piecewise landscapes).
        if (it % (20 + dim)) == 0 and time.time() < deadline:
            x = list(best_x)
            fx = best
            # initial coordinate step: small fraction of span, scaled down by sigma_scale
            coord_step = [max(1e-12, 0.05 * spans[i] / (1.0 + 0.5 * sigma_scale)) for i in range(dim)]
            # one pass of greedy coordinate tweaks
            for i in range(dim):
                if time.time() >= deadline:
                    break
                for direction in (-1.0, 1.0):
                    y = list(x)
                    y[i] = y[i] + direction * coord_step[i]
                    y = clamp(y)
                    fy = eval_func(y)
                    if fy < fx:
                        x, fx = y, fy
            if fx < best:
                best, best_x = fx, x
                cur_x, cur_f = list(best_x), best
                stall = 0

    return best
