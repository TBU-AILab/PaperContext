import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization (per-dimension stratification)
    - (mu, lambda) evolution strategy with diagonal step sizes
    - Occasional coordinate/pattern local search around the best
    No external libraries required.
    Returns: best fitness found (float)
    """

    # ---- helpers ----
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_uniform(lo, hi):
        return lo + (hi - lo) * random.random()

    def randn():  # Box-Muller
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-12)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(vec):
        # Ensure Python list passed to func (works for funcs expecting "array-like")
        try:
            val = func(vec)
        except Exception:
            # If func is fragile, return very poor fitness on error
            return float("inf")
        if val is None:
            return float("inf")
        try:
            v = float(val)
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def random_point():
        return [rand_uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def reflect_into_bounds(x, lo, hi):
        # Reflection handles large steps better than clamping for ES
        if lo == hi:
            return lo
        # reflect until inside
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        # numerical safety
        return clamp(x, lo, hi)

    # ---- time bookkeeping ----
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---- handle degenerate cases ----
    if dim <= 0:
        # Evaluate at empty vector if possible
        return safe_eval([])

    # Precompute ranges and initial sigmas
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    # If a dimension is fixed, treat sigma as 0
    sigma0 = [0.15 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    # ---- stratified initial sampling ("LHS-like") ----
    # We create n_init samples where each dimension uses a random permutation of strata.
    # This gives broad coverage quickly.
    n_init = max(10, 6 * dim)
    strata = []
    for i in range(dim):
        perm = list(range(n_init))
        random.shuffle(perm)
        strata.append(perm)

    best_x = None
    best = float("inf")

    # Evaluate initial points
    for k in range(n_init):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            if span[i] <= 0:
                x.append(lo[i])
            else:
                a = strata[i][k]
                # sample within stratum
                u = (a + random.random()) / n_init
                x.append(lo[i] + u * span[i])
        f = safe_eval(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # ---- Evolution Strategy parameters ----
    mu = max(3, 2 * dim)          # parents
    lam = max(10, 6 * dim)        # offspring
    # Initialize parent population around best_x with noise
    parents = []
    parent_sig = []

    for _ in range(mu):
        x = []
        sig = sigma0[:]
        for i in range(dim):
            if span[i] <= 0:
                x.append(lo[i])
                sig[i] = 0.0
            else:
                # small random perturbation around best
                xi = best_x[i] + 0.05 * span[i] * randn()
                x.append(reflect_into_bounds(xi, lo[i], hi[i]))
        fx = safe_eval(x)
        parents.append((fx, x))
        parent_sig.append(sig)
        if fx < best:
            best = fx
            best_x = x

    # Sort parents by fitness
    parents_sorted = sorted(range(mu), key=lambda idx: parents[idx][0])
    parents = [parents[i] for i in parents_sorted]
    parent_sig = [parent_sig[i] for i in parents_sorted]

    # ---- Local search (coordinate/pattern) around best ----
    def local_search(x0, f0, base_step):
        x = x0[:]
        f = f0
        step = base_step[:]
        # A small, time-aware pattern search
        iters = 0
        while iters < 6 and time.time() < deadline:
            improved = False
            for i in range(dim):
                if time.time() >= deadline:
                    break
                if span[i] <= 0 or step[i] <= 0:
                    continue
                # try +step and -step
                for sgn in (1.0, -1.0):
                    if time.time() >= deadline:
                        break
                    cand = x[:]
                    cand[i] = reflect_into_bounds(cand[i] + sgn * step[i], lo[i], hi[i])
                    fc = safe_eval(cand)
                    if fc < f:
                        x, f = cand, fc
                        improved = True
                        break
            if not improved:
                # reduce step
                for i in range(dim):
                    step[i] *= 0.5
                iters += 1
            else:
                # mild expansion on success
                for i in range(dim):
                    step[i] *= 1.1
                iters += 1
        return x, f

    # ---- main loop ----
    # Self-adaptation: log-normal step-size update per dimension
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(dim))
    tau0 = 1.0 / math.sqrt(2.0 * dim)

    gen = 0
    while time.time() < deadline:
        gen += 1

        # Occasionally run local search around current best
        if gen % 8 == 0 and time.time() < deadline:
            # base steps proportional to spans, but not too small
            base_step = [max(1e-12, 0.05 * span[i]) if span[i] > 0 else 0.0 for i in range(dim)]
            x_ls, f_ls = local_search(best_x, best, base_step)
            if f_ls < best:
                best, best_x = f_ls, x_ls

        # Produce offspring
        offspring = []
        offspring_sig = []

        for _ in range(lam):
            if time.time() >= deadline:
                break
            # select a parent (tournament of 2)
            a = random.randrange(mu)
            b = random.randrange(mu)
            p_idx = a if parents[a][0] <= parents[b][0] else b
            p_fit, p_x = parents[p_idx]
            sig = parent_sig[p_idx][:]

            # mutate sigmas
            global_noise = tau0 * randn()
            for i in range(dim):
                if sig[i] > 0.0:
                    sig[i] *= math.exp(global_noise + tau * randn())
                    # keep sigma within sensible bounds
                    sig[i] = clamp(sig[i], 1e-15 * max(1.0, span[i]), 0.5 * max(1e-12, span[i]))

            # mutate solution
            child = []
            for i in range(dim):
                if span[i] <= 0:
                    child.append(lo[i])
                else:
                    xi = p_x[i] + sig[i] * randn()
                    child.append(reflect_into_bounds(xi, lo[i], hi[i]))

            f = safe_eval(child)
            offspring.append((f, child))
            offspring_sig.append(sig)

            if f < best:
                best = f
                best_x = child

        if not offspring:
            break

        # (mu, lambda): select best mu from offspring only
        idx_sorted = sorted(range(len(offspring)), key=lambda i: offspring[i][0])
        idx_sorted = idx_sorted[:mu]
        parents = [offspring[i] for i in idx_sorted]
        parent_sig = [offspring_sig[i] for i in idx_sorted]

        # If we are stagnating, inject diversity
        if gen % 15 == 0 and time.time() < deadline:
            # replace worst parent with a random sample
            rx = random_point()
            rf = safe_eval(rx)
            if rf < best:
                best, best_x = rf, rx
            # overwrite last parent (worst in sorted-by-selection list is last)
            parents[-1] = (rf, rx)
            parent_sig[-1] = sigma0[:]

    return best
