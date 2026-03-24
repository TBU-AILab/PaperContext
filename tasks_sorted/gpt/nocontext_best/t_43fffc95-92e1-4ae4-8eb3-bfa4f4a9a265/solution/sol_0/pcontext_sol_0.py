import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      1) low-discrepancy-ish sampling (Latin hypercube per batch),
      2) best-so-far archive,
      3) adaptive local search around elites (Gaussian + Cauchy steps),
      4) occasional global re-sampling to avoid stagnation.

    Requirements: no external libs.
    func(params) -> float, where params is a list/sequence of length dim.
    bounds: list of (low, high) for each dimension.
    max_time: seconds.
    Returns: best (float) fitness found.
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Avoid degenerate spans
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 0.0

    def clip(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_uniform_point():
        return [lows[i] + spans[i] * random.random() for i in range(dim)]

    def latin_hypercube_batch(n):
        # For each dim: stratified n bins, shuffled
        # Returns list of n points (each list[dim])
        perms = []
        for i in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)

        pts = []
        inv_n = 1.0 / n
        for k in range(n):
            p = []
            for i in range(dim):
                # sample within bin
                u = (perms[i][k] + random.random()) * inv_n
                p.append(lows[i] + spans[i] * u)
            pts.append(p)
        return pts

    def safe_eval(x):
        # Robust evaluation: handle exceptions / NaNs
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # --- algorithm parameters (lightweight, time-adaptive) ---
    # population/archive sizes
    archive_size = max(8, min(40, 4 * dim))
    elite_count = max(2, min(10, archive_size // 2))

    # local search scale: start relatively broad, adapt down with stagnation
    base_sigma = 0.15  # as fraction of span
    min_sigma = 1e-6

    # batch size for global sampling
    batch = max(16, min(128, 16 + 8 * dim))

    # initialize archive with LHS
    archive = []  # list of (fitness, point)
    best = float("inf")
    best_x = None

    # initial global sampling (at least one batch, but respect max_time)
    while time.time() < deadline and len(archive) < archive_size:
        for x in latin_hypercube_batch(min(batch, archive_size - len(archive))):
            if time.time() >= deadline:
                break
            f = safe_eval(x)
            archive.append((f, x))
            if f < best:
                best, best_x = f, x

    if not archive:
        return float("inf")

    archive.sort(key=lambda t: t[0])
    archive = archive[:archive_size]

    # --- main loop ---
    stagnation = 0
    iters = 0

    while time.time() < deadline:
        iters += 1

        # Recompute elites
        archive.sort(key=lambda t: t[0])
        archive = archive[:archive_size]
        elites = archive[:elite_count]

        # Adapt sigma: decrease with stagnation, slightly increase if improving
        # sigma fraction relative to span
        sigma_frac = base_sigma * (0.5 ** (stagnation / 15.0))
        if sigma_frac < min_sigma:
            sigma_frac = min_sigma

        # Choose a parent (biased toward best)
        # rank-based selection: p(rank=k) ~ 1/(k+1)
        r = random.random()
        denom = sum(1.0 / (k + 1) for k in range(len(elites)))
        acc = 0.0
        parent = elites[0][1]
        for k in range(len(elites)):
            acc += (1.0 / (k + 1)) / denom
            if r <= acc:
                parent = elites[k][1]
                break

        # Decide between local step and global exploration
        # More global if stagnating
        p_global = 0.05 + min(0.45, stagnation / 100.0)
        do_global = (random.random() < p_global)

        if do_global:
            x = rand_uniform_point()
        else:
            # Local perturbation: mixture of Gaussian and Cauchy-like steps
            x = parent[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    x[i] = lows[i]
                    continue

                s = sigma_frac * spans[i]
                # mixture: 70% Gaussian, 30% heavy-tail
                if random.random() < 0.7:
                    step = random.gauss(0.0, s)
                else:
                    # Cauchy via tan(pi*(u-0.5)), scale s
                    u = random.random()
                    step = s * math.tan(math.pi * (u - 0.5))

                # occasional coordinate-wise "reset" to add diversity
                if random.random() < 0.02:
                    x[i] = lows[i] + spans[i] * random.random()
                else:
                    x[i] = clip(x[i] + step, i)

        f = safe_eval(x)

        # Update best/archives
        improved = False
        if f < best:
            best = f
            best_x = x
            improved = True

        archive.append((f, x))
        # Keep archive limited; prune worst occasionally
        if len(archive) > archive_size * 2:
            archive.sort(key=lambda t: t[0])
            archive = archive[:archive_size]

        if improved:
            stagnation = max(0, stagnation - 3)
        else:
            stagnation += 1

        # If very stagnant, inject a small LHS batch (global restart-ish)
        if stagnation > 60 and time.time() < deadline:
            n = min(batch, archive_size)
            for gx in latin_hypercube_batch(n):
                if time.time() >= deadline:
                    break
                gf = safe_eval(gx)
                archive.append((gf, gx))
                if gf < best:
                    best = gf
                    best_x = gx
                    stagnation = 0
            # prune
            archive.sort(key=lambda t: t[0])
            archive = archive[:archive_size]

    # return fitness of the best found solution
    return best
