import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improvements over the provided ES-style code:
      - Better global exploration early: scrambled Halton + LHS seeding + opposition points
      - DE/rand/1/bin "global" steps from an elite pool (good for rugged/multimodal)
      - Adaptive local search around current best using diagonal step sizes
      - Lightweight quadratic 1D line-search per coordinate (3-point parabola fit)
      - Stagnation-triggered partial restarts that keep the best

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ------------------------ helpers ------------------------

    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def clamp_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    avgw = sum(widths) / max(1, dim)
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposition(x):
        # "Opposition-based" point mirrored around center of bounds
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    # ---- Halton (scrambled a bit via random digit permutation) ----
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

    primes = first_primes(max(1, dim))

    # digit permutations for each base (very light "scramble")
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def van_der_corput_scrambled(n, base):
        v = 0.0
        denom = 1.0
        perm = digit_perm[base]
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += perm[rem] / denom
        return v

    def halton_point(index):
        x = []
        for i in range(dim):
            u = van_der_corput_scrambled(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ------------------------ state ------------------------

    best = float("inf")
    best_x = None

    elite_size = max(10, min(40, 5 * dim))
    elites = []  # (f, x)

    def elite_add(fx, x):
        nonlocal best, best_x, elites
        if fx < best:
            best = fx
            best_x = x[:]
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_size:
            elites.pop()

    # ------------------------ initialization (stronger seeding) ------------------------

    # Evaluate: center, some random, LHS-like, scrambled Halton, plus opposition points
    seed_base = max(24, 12 * dim)
    lhs_n = seed_base
    halton_n = seed_base
    rand_n = max(8, seed_base // 3)

    # center and its opposition (same as center, but keep pattern consistent)
    init_points = [centers[:]]

    # random
    for _ in range(rand_n):
        init_points.append(rand_uniform_vec())

    # LHS-like stratified
    strata = []
    for i in range(dim):
        idx = list(range(lhs_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(lhs_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            a = strata[i][k]
            u = (a + random.random()) / lhs_n
            x.append(lo + u * (hi - lo))
        init_points.append(x)

    # scrambled halton with random offset
    offset = random.randint(1, 5000)
    for k in range(1, halton_n + 1):
        init_points.append(halton_point(offset + k))

    # add opposition points for a subset (cheap, often helps)
    # (avoid doubling everything in very tight time budgets)
    opp_subset = init_points[: max(8, len(init_points) // 4)]
    for x in opp_subset:
        init_points.append(opposition(x))

    for x in init_points:
        if time.time() >= deadline:
            return best
        x = clamp_vec(x)
        fx = eval_f(x)
        elite_add(fx, x)

    if best_x is None:
        x = rand_uniform_vec()
        best = eval_f(x)
        best_x = x[:]
        elites = [(best, best_x[:])]

    # ------------------------ parameters ------------------------

    # Local step sizes per dimension (adaptive)
    step = [0.15 * (w if w > 0 else 1.0) for w in widths]
    step_min = [1e-12 * ((w if w > 0 else 1.0) + 1.0) for w in widths]
    step_max = [0.75 * (w if w > 0 else (avgw if avgw > 0 else 1.0)) for w in widths]

    # DE parameters
    F = 0.7
    CR = 0.9

    stagnation = 0
    last_best = best

    # ------------------------ local refinement tools ------------------------

    def quadratic_refine_1d(x, i, delta):
        """
        1D refinement along coordinate i using 3 points: x-d, x, x+d.
        Fits a parabola and tries the vertex if it lies inside [-d, d].
        Returns (best_fx, best_x).
        """
        if delta <= 0:
            return None

        lo, hi = bounds[i]
        x0 = x[:]
        f0 = eval_f(x0)

        xm = x[:]
        xp = x[:]
        xm[i] = clamp(xm[i] - delta, lo, hi)
        xp[i] = clamp(xp[i] + delta, lo, hi)

        fm = eval_f(xm)
        fp = eval_f(xp)

        # If any endpoint improves, keep the best
        best_local_f = f0
        best_local_x = x0
        if fm < best_local_f:
            best_local_f, best_local_x = fm, xm
        if fp < best_local_f:
            best_local_f, best_local_x = fp, xp

        # Parabolic fit around x0 in terms of t in {-1,0,1} scaled by delta
        # y(-1)=fm, y(0)=f0, y(1)=fp; vertex t* = (fm - fp) / (2*(fm - 2f0 + fp))
        denom = (fm - 2.0 * f0 + fp)
        if denom != 0.0:
            t_star = 0.5 * (fm - fp) / denom  # in "steps" of delta
            if -1.0 <= t_star <= 1.0:
                xv = x[:]
                xv[i] = clamp(xv[i] + t_star * delta, lo, hi)
                fv = eval_f(xv)
                if fv < best_local_f:
                    best_local_f, best_local_x = fv, xv

        return best_local_f, best_local_x

    # ------------------------ main loop ------------------------

    while time.time() < deadline:
        # Track stagnation
        if best < last_best:
            last_best = best
            stagnation = 0
        else:
            stagnation += 1

        # Ensure we have enough points for DE; otherwise inject randoms
        while len(elites) < 4 and time.time() < deadline:
            x = rand_uniform_vec()
            fx = eval_f(x)
            elite_add(fx, x)

        # -------- Global step: DE/rand/1/bin from elites --------
        # Pick 3 distinct indices
        nE = len(elites)
        idxs = list(range(nE))
        random.shuffle(idxs)
        a = elites[idxs[0]][1]
        b = elites[idxs[1]][1]
        c = elites[idxs[2]][1]

        # Bias target to best_x sometimes (current-to-best flavor)
        if random.random() < 0.45:
            target = best_x[:]
        else:
            target = elites[idxs[3 % nE]][1][:]

        # Create mutant
        mutant = [a[i] + F * (b[i] - c[i]) for i in range(dim)]
        mutant = clamp_vec(mutant)

        # Binomial crossover
        trial = target[:]
        jrand = random.randrange(dim) if dim > 0 else 0
        for j in range(dim):
            if j == jrand or random.random() < CR:
                trial[j] = mutant[j]

        trial = clamp_vec(trial)

        # Occasionally apply a heavy-tailed kick (Cauchy-like) to a few dims
        if random.random() < 0.12:
            k = max(1, dim // 6)
            for _ in range(k):
                j = random.randrange(dim)
                g1 = random.gauss(0.0, 1.0)
                g2 = random.gauss(0.0, 1.0)
                scale = 0.10 * (widths[j] if widths[j] > 0 else 1.0)
                trial[j] = clamp(trial[j] + (g1 / (abs(g2) + 1e-12)) * scale, bounds[j][0], bounds[j][1])

        if time.time() >= deadline:
            return best
        f_trial = eval_f(trial)
        elite_add(f_trial, trial)

        # -------- Local step around current best (adaptive diagonal) --------
        # A few quick Gaussian-like local proposals (implemented with gauss)
        local_tries = max(6, 2 * dim)
        for _ in range(local_tries):
            if time.time() >= deadline:
                return best
            x = best_x[:]
            # perturb only a subset of coordinates for efficiency in high-dim
            k = max(1, dim // 3)
            for __ in range(k):
                j = random.randrange(dim)
                x[j] = clamp(x[j] + random.gauss(0.0, step[j]), bounds[j][0], bounds[j][1])
            fx = eval_f(x)
            elite_add(fx, x)

        # -------- Coordinate quadratic refinement (cheap but powerful) --------
        # Do a small number of coordinates per iteration to stay time-bounded.
        if time.time() < deadline and (stagnation < 80 or random.random() < 0.35):
            xref = best_x[:]
            # sample coordinates in random order, but only a few
            coord_budget = max(1, min(dim, 6))
            coords = list(range(dim))
            random.shuffle(coords)
            for t in range(coord_budget):
                if time.time() >= deadline:
                    return best
                i = coords[t]
                d = step[i]
                if d < step_min[i]:
                    continue
                res = quadratic_refine_1d(xref, i, d)
                if res is None:
                    continue
                f_new, x_new = res
                if f_new < best:
                    elite_add(f_new, x_new)
                    xref = x_new[:]
                    step[i] = min(step_max[i], step[i] * 1.20)  # expand if successful
                else:
                    step[i] = max(step_min[i], step[i] * 0.70)  # shrink if not

        # -------- Adapt DE parameters very lightly --------
        # If improving recently, slightly increase exploration; otherwise tighten.
        if stagnation == 0:
            F = min(0.95, max(0.4, F * 1.03))
            CR = min(0.98, max(0.5, CR * 1.01))
        elif stagnation % 25 == 0:
            F = min(0.95, max(0.35, F * 0.92))
            CR = min(0.98, max(0.3, CR * 0.95))

        # -------- Restart logic --------
        # If stuck, inject diversity but keep the best few.
        if stagnation >= 140:
            stagnation = 0
            keep_k = max(3, elite_size // 4)
            elites = elites[:keep_k]

            # reset steps moderately
            step = [0.20 * (w if w > 0 else 1.0) for w in widths]

            # add fresh points
            inject = max(12, 4 * dim)
            offset = random.randint(1, 5000)
            for k in range(inject):
                if time.time() >= deadline:
                    return best
                if random.random() < 0.5:
                    x = rand_uniform_vec()
                else:
                    x = halton_point(offset + k + 1)
                if random.random() < 0.5:
                    # sometimes mix with best to create "directed random" samples
                    alpha = random.random()
                    x = [alpha * best_x[i] + (1.0 - alpha) * x[i] for i in range(dim)]
                x = clamp_vec(x)
                fx = eval_f(x)
                elite_add(fx, x)

    return best
