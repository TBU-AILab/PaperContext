import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid algorithm:
      1) Quasi-random (Halton) + random initialization (good coverage)
      2) Maintain a small elite set (best K points)
      3) Generate new candidates by:
           - DE-style "current-to-best/1" recombination from elites
           - Cauchy/Gaussian local perturbations around best
           - Occasional global random / quasi-random injections
      4) Lightweight coordinate pattern search refinement on the best-so-far
      5) Success-based adaptation of mutation scale

    Returns:
        best fitness (float)
    """

    t0 = time.time()
    t_end = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    # Handle degenerate bounds
    for i in range(dim):
        if span[i] < 0:
            lo[i], hi[i] = hi[i], lo[i]
            span[i] = -span[i]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def eval_f(x):
        return float(func(list(x)))

    # ---------------- Halton sequence (quasi-random) ----------------
    def first_primes(n):
        primes = []
        c = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(c))
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    bases = first_primes(max(1, dim))

    def halton_value(index, base):
        # index >= 1
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        # k >= 1
        x = [0.0] * dim
        for j in range(dim):
            u = halton_value(k, bases[j])
            x[j] = lo[j] + u * span[j]
        return x

    def random_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---------------- small helpers ----------------
    def cauchy():
        # tan(pi*(u-0.5)) gives standard Cauchy
        u = random.random()
        # avoid extreme infinities
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------------- initialization ----------------
    best_x = None
    best = float("inf")

    # Keep an elite list of (fitness, x)
    elite_size = max(6, min(30, 4 + 2 * dim))
    elites = []

    def consider(x, fx):
        nonlocal best, best_x, elites
        if fx < best:
            best = fx
            best_x = list(x)
        elites.append((fx, list(x)))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_size:
            elites = elites[:elite_size]

    # Budget some time for initialization
    # Use a mix of Halton + random
    init_n = max(20, min(120, 10 * dim))
    hal_n = int(init_n * 0.7)
    rnd_n = init_n - hal_n

    k_hal = 1
    for _ in range(hal_n):
        if time.time() >= t_end:
            return best
        x = halton_point(k_hal)
        k_hal += 1
        fx = eval_f(x)
        consider(x, fx)

    for _ in range(rnd_n):
        if time.time() >= t_end:
            return best
        x = random_point()
        fx = eval_f(x)
        consider(x, fx)

    if best_x is None:
        return best

    # ---------------- main loop parameters ----------------
    # Mutation scale in normalized space; adapted by success
    sigma = 0.25  # start moderately exploratory
    sigma_min, sigma_max = 1e-4, 0.75

    # Coordinate search steps (absolute per dimension)
    step = [max(1e-12, 0.2 * span[i]) for i in range(dim)]
    min_step = [max(1e-12, 1e-10 * span[i] if span[i] > 0 else 1e-12) for i in range(dim)]

    # Attempt counters for adaptation
    attempts = 0
    successes = 0

    # ---------------- candidate generators ----------------
    def normalized(x):
        z = [0.0] * dim
        for i in range(dim):
            if span[i] > 0:
                z[i] = (x[i] - lo[i]) / span[i]
            else:
                z[i] = 0.0
        return z

    def denormalized(z):
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] + z[i] * span[i]
        return x

    def gen_de_candidate():
        # DE-like move using elites:
        # v = x + F*(best - x) + F2*(a - b)
        # then binomial crossover with x
        if len(elites) < 4:
            return None

        # pick "current" from elites biased to better ones
        # triangular bias: pick index ~ floor(U^2 * m)
        m = len(elites)
        idx = int((random.random() ** 2) * m)
        x = elites[idx][1]
        best_e = elites[0][1]

        # pick a, b distinct
        ia = random.randrange(m)
        ib = random.randrange(m)
        while ib == ia:
            ib = random.randrange(m)
        a = elites[ia][1]
        b = elites[ib][1]

        # scales
        F = 0.4 + 0.5 * random.random()  # [0.4, 0.9]
        F2 = 0.2 + 0.6 * random.random() # [0.2, 0.8]
        cr = 0.7

        z = normalized(x)
        zb = normalized(best_e)
        za = normalized(a)
        zb2 = normalized(b)

        v = [0.0] * dim
        for i in range(dim):
            v[i] = z[i] + F * (zb[i] - z[i]) + F2 * (za[i] - zb2[i])

        # crossover
        jrand = random.randrange(dim)
        u = z[:]  # start from current
        for i in range(dim):
            if random.random() < cr or i == jrand:
                u[i] = v[i]

        # small extra gaussian jitter (normalized)
        for i in range(dim):
            # Box-Muller
            if random.random() < 0.3:
                u1 = max(1e-12, random.random())
                u2 = random.random()
                g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                u[i] += 0.15 * sigma * g

        # clamp normalized
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0

        return denormalized(u)

    def gen_local_candidate():
        # Local around best using mixture of gaussian + cauchy in normalized space
        z = normalized(best_x)
        u = z[:]
        for i in range(dim):
            # choose distribution
            if random.random() < 0.7:
                # gaussian
                u1 = max(1e-12, random.random())
                u2 = random.random()
                g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                u[i] += sigma * 0.35 * g
            else:
                # cauchy (heavy tails)
                u[i] += sigma * 0.15 * cauchy()

        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return denormalized(u)

    def pattern_refine_once():
        # One quick coordinate pattern pass around best_x
        nonlocal best, best_x
        improved = False
        order = list(range(dim))
        random.shuffle(order)
        x0 = best_x
        for j in order:
            if time.time() >= t_end:
                return improved
            s = step[j]
            if s <= min_step[j] or span[j] == 0.0:
                continue

            # evaluate +/- s
            for direction in (+1.0, -1.0):
                if time.time() >= t_end:
                    return improved
                x = list(x0)
                x[j] = x[j] + direction * s
                if x[j] < lo[j]:
                    x[j] = lo[j]
                elif x[j] > hi[j]:
                    x[j] = hi[j]
                if x[j] == x0[j]:
                    continue
                fx = eval_f(x)
                consider(x, fx)
                if fx < best:
                    best = fx
                    best_x = list(x)
                    x0 = best_x
                    improved = True
        # update step sizes
        if improved:
            for j in range(dim):
                step[j] = min(span[j] if span[j] > 0 else step[j], step[j] * 1.15)
        else:
            for j in range(dim):
                step[j] *= 0.6
        return improved

    # ---------------- main optimization loop ----------------
    hal_counter = k_hal
    while time.time() < t_end:
        # Choose a generation mode
        r = random.random()
        if r < 0.55:
            x = gen_de_candidate()
            if x is None:
                x = gen_local_candidate()
        elif r < 0.85:
            x = gen_local_candidate()
        elif r < 0.95:
            # quasi-random injection
            x = halton_point(hal_counter)
            hal_counter += 1
        else:
            # pure random restart
            x = random_point()

        if time.time() >= t_end:
            break

        fx = eval_f(x)
        prev_best = best
        consider(x, fx)

        attempts += 1
        if fx < prev_best:
            successes += 1

        # success-based adaptation every so often
        if attempts >= 30:
            rate = successes / float(attempts)
            # If improving often, shrink sigma for exploitation; else increase to explore
            if rate > 0.2:
                sigma = max(sigma_min, sigma * 0.85)
            elif rate < 0.05:
                sigma = min(sigma_max, sigma * 1.25)
            # reset counters
            attempts = 0
            successes = 0

        # Occasionally refine with a quick coordinate search
        if random.random() < 0.25:
            pattern_refine_once()

        # If coordinate steps collapsed, re-expand a bit around current best
        if all(step[j] <= min_step[j] for j in range(dim)):
            step = [max(min_step[i], 0.1 * span[i]) for i in range(dim)]
            sigma = min(sigma_max, max(sigma, 0.15))

    return best
