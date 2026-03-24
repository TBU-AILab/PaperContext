import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like stratified initialization
      - (1+λ) evolution strategy with decreasing Gaussian step-size
      - Occasional coordinate-wise local refinement
      - Soft restarts if stagnating

    Inputs:
      func: callable(list_or_array)-> float
      dim: int
      bounds: list of (low, high) for each dimension
      max_time: seconds (int/float)

    Returns:
      best (float): best objective value found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- helpers ----------
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; we pass a list to avoid external deps
        return float(func(x))

    # Normal sampler (Box-Muller), cached
    _bm_has = False
    _bm_val = 0.0
    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(max(1e-12, u1)))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _bm_has = True
        _bm_val = z1
        return z0

    # ---------- initialization: stratified per-dimension ----------
    # Create K initial points; per dim use shuffled strata in [0,1)
    K = max(8, 4 * dim)
    strata = [list(range(K)) for _ in range(dim)]
    for d in range(dim):
        random.shuffle(strata[d])

    best_x = None
    best = float("inf")

    for k in range(K):
        if time.time() >= deadline:
            return best
        x = []
        for d in range(dim):
            # sample within stratum
            u = (strata[d][k] + random.random()) / K
            x.append(lows[d] + u * spans[d])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]

    if best_x is None:
        # fallback
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------- main loop (1+λ ES + occasional local search) ----------
    # Initial step-size relative to bounds
    avg_span = sum(spans) / max(1, dim)
    sigma0 = 0.2 * avg_span
    sigma_min = 1e-12 * max(1.0, avg_span)

    # λ candidates per iteration
    lam = max(8, 6 * dim)

    # Stagnation / restart controls
    no_improve = 0
    best_iter = 0
    max_no_improve = 40 + 10 * dim

    # Coordinate local search parameters
    local_period = max(10, 3 * dim)   # every N iterations
    coord_trials = max(2, dim // 2)   # how many coords to try each local phase

    it = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best

        # time-adaptive cooling for sigma
        frac = (now - t0) / max(1e-12, (deadline - t0))
        sigma = max(sigma_min, sigma0 * (0.9 ** (5.0 * frac)) * (0.97 ** it))

        # Generate λ offspring around current best (elitist)
        parent = best_x
        cand_best_x = None
        cand_best_f = float("inf")

        for _ in range(lam):
            if time.time() >= deadline:
                return best
            child = parent[:]
            # mutate all dims with Gaussian noise scaled to span
            for d in range(dim):
                step = randn() * sigma
                child[d] = child[d] + step
            clip(child)
            fchild = evaluate(child)
            if fchild < cand_best_f:
                cand_best_f = fchild
                cand_best_x = child

        # Selection
        if cand_best_f < best:
            best = cand_best_f
            best_x = cand_best_x[:]
            no_improve = 0
            best_iter = it
        else:
            no_improve += 1

        # Occasional coordinate-wise refinement (cheap local search)
        if (it % local_period) == 0 and time.time() < deadline:
            x = best_x[:]
            fx = best

            # choose coordinates to probe
            coords = list(range(dim))
            random.shuffle(coords)
            coords = coords[:min(coord_trials, dim)]

            for d in coords:
                if time.time() >= deadline:
                    return best

                # try +/- delta on coordinate d
                delta = 0.05 * spans[d] * (0.7 ** (1 + it // local_period))
                if delta <= 0:
                    continue

                improved = False
                for sign in (-1.0, 1.0):
                    y = x[:]
                    y[d] = y[d] + sign * delta
                    clip(y)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True

                if improved:
                    best_x = x[:]
                    best = fx
                    no_improve = 0
                    best_iter = it

        # Soft restart if stagnating: re-seed around best and random points
        if no_improve >= max_no_improve:
            if time.time() >= deadline:
                return best
            # Mix: keep best, but jump to a new area sometimes
            if random.random() < 0.5:
                # random restart
                best_x = rand_uniform()
                best = evaluate(best_x)
            else:
                # broaden sigma temporarily around current best
                sigma0 = min(avg_span, sigma0 * 1.5)
            no_improve = 0

        it += 1
