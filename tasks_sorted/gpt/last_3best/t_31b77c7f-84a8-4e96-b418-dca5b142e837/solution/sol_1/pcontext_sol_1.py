import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs):
      - Quasi-random (Halton) + stratified seeding for better initial coverage
      - Small elite pool instead of single parent (helps multi-modality)
      - Adaptive diagonal Gaussian ES around elites with per-dimension sigmas
      - Coordinate/pattern refinement on improvements
      - Heavy-tail occasional jumps + controlled restarts
    Returns: best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------------------- helpers ----------------------------

    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def eval_f(x):
        return float(func(x))

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]
    avgw = sum(widths) / max(1, dim)

    def clamp_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ---- Halton sequence (simple, deterministic, no libs) ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    primes = first_primes(max(1, dim))

    def halton_point(index):
        # index >= 1
        x = []
        for i in range(dim):
            u = van_der_corput(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ----------------------- initialization/seeding -----------------------

    best = float("inf")
    best_x = None

    # Evaluate a mixture of:
    #  - center (often decent for bounded problems)
    #  - random points
    #  - Halton points
    #  - stratified points
    candidates = []

    # center
    candidates.append(centers[:])

    # seed counts
    seed_n = max(16, 8 * dim)
    halton_n = seed_n
    rand_n = seed_n // 2
    strata_n = seed_n

    # random
    for _ in range(rand_n):
        candidates.append(rand_uniform_vec())

    # halton
    # start at a random offset to avoid always same early points
    halton_offset = random.randint(1, 1000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(halton_offset + k))

    # stratified (LHS-like)
    strata = []
    for i in range(dim):
        idx = list(range(strata_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(strata_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            a = strata[i][k]
            u = (a + random.random()) / strata_n
            x.append(lo + u * (hi - lo))
        candidates.append(x)

    # Evaluate seeds, maintain small elite set
    elite_size = max(4, min(16, 2 * dim))
    elites = []  # list of (fitness, x)

    def elite_add(fx, x):
        nonlocal best, best_x, elites
        if fx < best:
            best = fx
            best_x = x[:]
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_size:
            elites.pop()

    for x in candidates:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        elite_add(fx, x)

    if best_x is None:
        x = rand_uniform_vec()
        best = eval_f(x)
        best_x = x[:]
        elites = [(best, best_x[:])]

    # ------------------------- search parameters -------------------------

    # per-dimension sigma (diagonal)
    base_sigma = [0.25 * (w if w > 0 else 1.0) for w in widths]
    min_sigma = [1e-12 * ((w if w > 0 else 1.0) + 1.0) for w in widths]
    max_sigma = [0.75 * (w if w > 0 else avgw if avgw > 0 else 1.0) for w in widths]
    sigma = base_sigma[:]

    # offspring per iteration
    lam = max(12, 6 * dim)

    # coordinate refinement step
    coord_step = [0.10 * (w if w > 0 else 1.0) for w in widths]
    min_coord_step = [1e-12 * ((w if w > 0 else 1.0) + 1.0) for w in widths]

    # stagnation/restart control
    it = 0
    stagnation = 0

    # ------------------------------ main loop ------------------------------

    while time.time() < deadline:
        it += 1

        # Pick an elite parent with bias to the best but not always the best
        # rank-based selection
        r = random.random()
        # geometric-ish: smaller rank more likely
        rank = int((r ** 2) * len(elites))
        if rank >= len(elites):
            rank = len(elites) - 1
        parent_f, parent = elites[rank]
        parent = parent[:]  # copy

        # Produce offspring; track best child
        best_child = None
        best_child_f = float("inf")

        # Update sampling scale: sometimes heavy-tail jump
        heavy_tail = (random.random() < 0.15)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            child = parent[:]

            if heavy_tail:
                # Cauchy-like: gauss / |gauss|
                for i in range(dim):
                    g1 = random.gauss(0.0, 1.0)
                    g2 = random.gauss(0.0, 1.0)
                    step = (g1 / (abs(g2) + 1e-12)) * (0.15 * (widths[i] if widths[i] > 0 else 1.0))
                    child[i] = child[i] + step
            else:
                for i in range(dim):
                    child[i] = child[i] + random.gauss(0.0, sigma[i])

            child = clamp_vec(child)
            f = eval_f(child)

            if f < best_child_f:
                best_child_f = f
                best_child = child

        improved_global = False
        if best_child is not None:
            elite_add(best_child_f, best_child)
            if best_child_f <= best:
                improved_global = True

        # Step-size adaptation:
        # - if we improved global best: expand a bit (encourage progress)
        # - else: shrink slowly; occasional re-expand to avoid premature collapse
        if improved_global:
            stagnation = 0
            for i in range(dim):
                sigma[i] = min(max_sigma[i], max(min_sigma[i], sigma[i] * 1.12))
        else:
            stagnation += 1
            for i in range(dim):
                sigma[i] = max(min_sigma[i], sigma[i] * 0.96)
            if stagnation % 25 == 0:
                # mild re-expansion to escape tight local basin
                for i in range(dim):
                    sigma[i] = min(max_sigma[i], sigma[i] * 1.5)

        # Local coordinate/pattern refinement when we get an improvement or periodically
        if improved_global or (it % 10 == 0 and stagnation < 60):
            x0 = best_x[:]
            f0 = best
            # one pass, greedy
            for i in range(dim):
                if time.time() >= deadline:
                    return best
                step_i = coord_step[i]
                if step_i < min_coord_step[i]:
                    continue

                # try + and -
                xp = x0[:]
                xm = x0[:]
                xp[i] = clamp(xp[i] + step_i, bounds[i][0], bounds[i][1])
                xm[i] = clamp(xm[i] - step_i, bounds[i][0], bounds[i][1])

                fp = eval_f(xp)
                fm = eval_f(xm)

                if fp < f0 or fm < f0:
                    if fp <= fm:
                        x0, f0 = xp, fp
                    else:
                        x0, f0 = xm, fm
                    elite_add(f0, x0)
                    # if helpful, increase step a bit
                    coord_step[i] = min(coord_step[i] * 1.25, widths[i] if widths[i] > 0 else coord_step[i])
                    # also allow sigma to grow in that dim
                    sigma[i] = min(max_sigma[i], sigma[i] * 1.10)
                else:
                    # shrink step if not helpful
                    coord_step[i] = max(min_coord_step[i], coord_step[i] * 0.5)

        # Controlled restart if stuck
        if stagnation >= 120:
            stagnation = 0
            # keep best elite, refresh others
            keep = elites[:max(1, len(elites) // 3)]
            elites = keep[:]
            # reset sigmas
            sigma = base_sigma[:]
            coord_step = [0.10 * (w if w > 0 else 1.0) for w in widths]

            # add a few fresh samples (Halton + random)
            for k in range(1, 1 + max(8, 2 * dim)):
                if time.time() >= deadline:
                    return best
                if random.random() < 0.5:
                    x = rand_uniform_vec()
                else:
                    halton_offset += 1
                    x = halton_point(halton_offset + k)
                fx = eval_f(x)
                elite_add(fx, x)

    return best
