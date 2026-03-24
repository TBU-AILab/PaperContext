import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded optimizer (no external libs):
      - Low-discrepancy (Halton) init + opposition points
      - (mu, lambda)-style Evolution Strategy with:
          * elitism (keep best few)
          * mixed Gaussian + Cauchy mutations
          * per-dimension step-sizes with 1/5 success rule adaptation
      - Lightweight local refinement around the incumbent best
      - Automatic restarts with shrinking "trust region" around best

    Returns:
      best (float): best (minimum) function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0 else 1.0 for s in span]

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # fast-ish Halton sequence (bases are first primes)
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):  # k >= 1
        u = [halton_value(k, primes[i]) for i in range(dim)]
        return [lo[i] + u[i] * span_safe[i] for i in range(dim)]

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- ES parameters (chosen to be robust across dims) ----------
    # population sizes depend mildly on dim, but kept small for time-bounded use
    mu = max(4, min(20, 4 + dim // 4))      # parents kept
    lam = max(12, min(60, 8 + 3 * mu))      # offspring per generation
    elite = max(1, mu // 3)                 # best parents always kept

    # initial step sizes (per dimension)
    sigma = [0.20 * span_safe[i] for i in range(dim)]
    sigma_min = [1e-12 * span_safe[i] for i in range(dim)]
    sigma_max = [0.60 * span_safe[i] for i in range(dim)]

    # success rule parameters
    adapt_every = 15  # generations
    target_success = 0.20  # close to 1/5th success
    adapt_rate = 0.35

    # restart control
    last_restart = time.time()
    restart_period = max_time / 5.0 if max_time > 0 else 0.1
    trust = 1.0  # shrinks when restarting around best

    # ---------- initialization ----------
    best = float("inf")
    best_x = None

    pop = []
    # Use a mix of Halton points, their opposites, and a few uniform random points.
    init_n = max(lam, min(4 * lam, 40 + 4 * dim))
    k = 1
    while len(pop) < init_n and time.time() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, x

        if len(pop) < init_n and time.time() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, xo

    # If extremely tight time, return best found so far
    if time.time() >= deadline:
        return best

    pop.sort(key=lambda t: t[0])
    pop = pop[:mu]

    # incumbent for mutation center: weighted toward best
    def pick_parent():
        # rank-based selection: higher prob for better ranks
        # (no numpy; do simple geometric distribution)
        # p(rank=r) ~ (1-q)*q^r
        q = 0.75
        r = 0
        while r < len(pop) - 1 and random.random() < q:
            r += 1
        return pop[r][1], pop[r][0]

    # mutation operators
    def mutate(parent):
        # mixture: mostly gaussian, sometimes cauchy for heavy tails
        child = parent[:]
        # update subset size: coordinate-wise often helps in higher dim
        if dim <= 8:
            n_mut = dim
        else:
            # mutate about 25% of coordinates (at least 1)
            n_mut = max(1, dim // 4)

        # choose coords
        if n_mut == dim:
            idxs = range(dim)
        else:
            idxs = random.sample(range(dim), n_mut)

        heavy = (random.random() < 0.18)  # occasional big jumps
        for i in idxs:
            if heavy:
                # Cauchy-like step: tan(pi*(u-0.5))
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))
                step = 0.60 * sigma[i] * c
            else:
                step = random.gauss(0.0, sigma[i])

            # small bias to stay inside bounds if near edges
            v = child[i] + step
            if v < lo[i] or v > hi[i]:
                # reflect once, then clip
                v = lo[i] + (lo[i] - v) if v < lo[i] else hi[i] - (v - hi[i])
            child[i] = clip(v, lo[i], hi[i])
        return child

    # local refinement around current best (cheap coordinate search)
    def local_refine(x0, f0):
        x = x0[:]
        fx = f0
        # a few quick passes; keep it lightweight
        for _ in range(2):
            improved = False
            for i in range(dim):
                if time.time() >= deadline:
                    return x, fx
                # try +/- step on coordinate i
                step = 0.10 * sigma[i]
                if step <= 0:
                    continue
                for sgn in (-1.0, 1.0):
                    y = x[:]
                    y[i] = clip(y[i] + sgn * step, lo[i], hi[i])
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        if fx < best:
                            # best is outer var; safe update handled by caller too
                            pass
            if not improved:
                break
        return x, fx

    # ---------- main loop ----------
    gen = 0
    success_count = 0
    trial_count = 0

    while True:
        now = time.time()
        if now >= deadline:
            return best

        # Restart logic: periodically diversify; if we have a good best, sample around it.
        if (now - last_restart) >= restart_period:
            last_restart = now
            # shrink trust region gradually, but not below a floor
            trust = max(0.20, trust * 0.85)

            # re-seed parents: some around best, some global
            new_pop = []
            if best_x is not None:
                for _ in range(mu):
                    x = []
                    for i in range(dim):
                        sd = trust * 0.35 * span_safe[i]
                        v = random.gauss(best_x[i], sd)
                        x.append(clip(v, lo[i], hi[i]))
                    fx = evaluate(x)
                    new_pop.append((fx, x))
                    if fx < best:
                        best, best_x = fx, x
            # add a few global points
            for _ in range(max(2, mu // 2)):
                x = rand_uniform_point()
                fx = evaluate(x)
                new_pop.append((fx, x))
                if fx < best:
                    best, best_x = fx, x

            new_pop.sort(key=lambda t: t[0])
            pop = new_pop[:mu]

            # slightly reset sigmas upward to escape potential stagnation
            for i in range(dim):
                sigma[i] = min(sigma_max[i], max(sigma[i], 0.10 * span_safe[i] * trust))
            continue

        # generate offspring
        offspring = []
        best_parent_f = pop[0][0]

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            parent_x, parent_f = pick_parent()
            child = mutate(parent_x)
            fchild = evaluate(child)

            offspring.append((fchild, child))
            trial_count += 1

            # count success vs the best parent (simple, robust)
            if fchild < best_parent_f:
                success_count += 1

            if fchild < best:
                best = fchild
                best_x = child

        # selection: elitism from parents + best offspring
        offspring.sort(key=lambda t: t[0])
        pop.sort(key=lambda t: t[0])

        next_pop = pop[:elite] + offspring[:max(0, mu - elite)]
        next_pop.sort(key=lambda t: t[0])
        pop = next_pop[:mu]

        # occasional local refinement on the global best (only if time allows)
        if best_x is not None and (gen % 6 == 0):
            if time.time() < deadline:
                xr, fr = local_refine(best_x, best)
                if fr < best:
                    best, best_x = fr, xr
                    # inject refined best into population
                    pop.append((fr, xr))
                    pop.sort(key=lambda t: t[0])
                    pop = pop[:mu]

        gen += 1

        # adapt step sizes using success rate
        if gen % adapt_every == 0 and trial_count > 0:
            rate = success_count / float(trial_count)
            # log-space-like scaling, but implemented multiplicatively
            if rate > target_success:
                factor = 1.0 + adapt_rate
            else:
                factor = 1.0 - adapt_rate * 0.75

            # avoid too aggressive changes; scale mildly with dim
            dim_damp = 1.0 / (1.0 + 0.03 * max(0, dim - 10))
            factor = 1.0 + (factor - 1.0) * dim_damp

            for i in range(dim):
                sigma[i] = max(sigma_min[i], min(sigma_max[i], sigma[i] * factor))

            success_count = 0
            trial_count = 0
