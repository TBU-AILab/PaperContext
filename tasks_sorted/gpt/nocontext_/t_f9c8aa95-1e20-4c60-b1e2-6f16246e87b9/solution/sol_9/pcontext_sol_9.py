import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      1) Latin-hypercube-like stratified initialization
      2) (mu, lambda) Evolution Strategy with self-adaptive step-size
      3) Occasional coordinate/local refinements
    No external libraries required.

    func: callable(list[float]) -> float
    dim: int
    bounds: list[(low, high)] length == dim
    max_time: seconds (int/float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --------- helpers ----------
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Avoid zero spans
    spans = [s if s != 0.0 else 1.0 for s in spans]

    def clamp_vec(x):
        y = x[:]
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Box-Muller normal generator (no numpy)
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        # protect log(0)
        u1 = max(u1, 1e-15)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare = z1
        _has_spare = True
        return z0

    def evaluate(x):
        # func expects an array-like; list is fine per prompt example
        return float(func(x))

    # --------- initialization: stratified sampling per dimension ----------
    # sample count scales with dim but remains modest for time-bounded runs
    n_init = max(8, min(60, 10 + 4 * dim))
    strata = list(range(n_init))

    # Build per-dimension shuffled strata indices
    per_dim_bins = []
    for j in range(dim):
        bins = strata[:]
        random.shuffle(bins)
        per_dim_bins.append(bins)

    best_x = None
    best = float("inf")

    # Evaluate init points
    for k in range(n_init):
        if time.time() >= deadline:
            return best
        x = []
        for j in range(dim):
            b = per_dim_bins[j][k]
            # pick uniformly within stratum
            u = (b + random.random()) / n_init
            x.append(lows[j] + u * spans[j])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    # If func is extremely slow, we might only have init
    if best_x is None:
        return best

    # --------- Evolution Strategy parameters ----------
    # Population sizes chosen for general robustness without heavy overhead
    mu = max(2, min(12, 2 + dim // 2))          # parents kept
    lam = max(8, min(40, 6 + 3 * dim))          # offspring per generation

    # self-adaptive step size (sigma) relative to search space
    # start moderately wide to explore, then adapt
    sigma = 0.25  # fraction of span (applied per-dim)
    sigma_min = 1e-8
    sigma_max = 2.0

    # Parent pool starts as small set around best + random points
    parents = []
    parents_fx = []

    # seed parent set
    # include best and a few randoms
    seed_count = mu
    for i in range(seed_count):
        if i == 0:
            x = best_x[:]
        else:
            x = rand_uniform_vec()
        fx = evaluate(x) if i != 0 else best
        parents.append(x)
        parents_fx.append(fx)
        if fx < best:
            best = fx
            best_x = x

    # --------- local refinement: coordinate pattern search around best ----------
    def local_refine(x0, f0, base_step_frac):
        """
        Simple coordinate search with decreasing step.
        Returns (x_best, f_best).
        """
        x = x0[:]
        fx = f0
        step = base_step_frac
        # limited iterations to keep time bounded
        for _ in range(12):
            if time.time() >= deadline:
                break
            improved = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                delta = step * spans[j]
                if delta <= 0:
                    continue

                # try +delta
                xp = x[:]
                xp[j] = min(highs[j], xp[j] + delta)
                fp = evaluate(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                # try -delta
                xm = x[:]
                xm[j] = max(lows[j], xm[j] - delta)
                fm = evaluate(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                step *= 0.5
                if step * max(spans) < 1e-12:
                    break
        return x, fx

    # --------- main loop ----------
    # Use 1/5th success rule style update for sigma
    successes = 0
    trials = 0
    last_refine_time = t0

    while time.time() < deadline:
        # Generate offspring from parents by mutation and intermediate recombination
        offspring = []
        offspring_fx = []

        for _ in range(lam):
            if time.time() >= deadline:
                break

            # pick two parents for recombination
            a = random.randrange(len(parents))
            b = random.randrange(len(parents))
            pa = parents[a]
            pb = parents[b]

            # intermediate recombination
            child = []
            for j in range(dim):
                w = random.random()
                val = w * pa[j] + (1.0 - w) * pb[j]
                child.append(val)

            # gaussian mutation scaled per dimension
            # jitter sigma slightly (log-normal) for self-adaptation
            # keep it simple & stable
            sigma *= math.exp(0.15 * randn())
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            for j in range(dim):
                child[j] += (sigma * spans[j]) * randn()

            child = clamp_vec(child)
            fchild = evaluate(child)

            offspring.append(child)
            offspring_fx.append(fchild)

            # track global best and ES success stats
            trials += 1
            if fchild < best:
                best = fchild
                best_x = child
                successes += 1

        if not offspring:
            break

        # Select next parents: best mu among offspring + current parents (elitism)
        # This improves stability on noisy/hard landscapes.
        combined = parents + offspring
        combined_fx = parents_fx + offspring_fx

        # partial sort by fx (simple O(n log n) is fine for small n)
        idx = list(range(len(combined)))
        idx.sort(key=lambda i: combined_fx[i])

        parents = [combined[i] for i in idx[:mu]]
        parents_fx = [combined_fx[i] for i in idx[:mu]]

        # 1/5th success rule adaptation every "generation"
        if trials >= lam:
            rate = successes / float(trials) if trials > 0 else 0.0
            # if success rate > 0.2 increase sigma, else decrease
            if rate > 0.2:
                sigma *= 1.15
            else:
                sigma *= 0.85
            sigma = max(sigma_min, min(sigma_max, sigma))
            successes = 0
            trials = 0

        # occasional local refinement around current best (time permitting)
        now = time.time()
        if now - last_refine_time > 0.25 and now < deadline:
            # refine with step depending on sigma (smaller when already fine)
            step_frac = max(1e-4, min(0.2, 0.5 * sigma))
            xr, fr = local_refine(best_x, best, step_frac)
            if fr < best:
                best, best_x = fr, xr
                # inject refined solution into parent pool
                parents[0] = best_x
                parents_fx[0] = best
            last_refine_time = now

    # return fitness of the best found solution
    return best
