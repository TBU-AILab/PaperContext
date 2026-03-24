import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer.

    Key upgrades vs your current best:
      - Better global exploration: Sobol-like (LDS) via scrambled Halton + opposition points
      - Maintains a small elite population (not just single incumbent)
      - Uses DE/rand-to-best style recombination for global moves (very effective on many black-box tasks)
      - Uses a lightweight local optimizer around the best (adaptive coordinate search + ES steps)
      - Time-aware scheduling: exploration early, exploitation late
      - Robust evaluation (handles exceptions/NaN)

    Returns:
      best (float): best objective value found within max_time
    """

    # ---------------- helpers ----------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Box-Muller gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # primes for Halton
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2: return False
        if k % 2 == 0: return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def opposite_point(x):
        # opposition-based learning: reflect across center of bounds
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    # ---------------- setup ----------------
    start = time.time()
    deadline = start + max(0.0, float(max_time))
    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # Elite population size (small to keep eval budget for actual improvements)
    pop_size = max(8, min(24, 4 * dim))
    elite_size = max(3, min(10, pop_size // 2))

    # Global sampling budget
    # keep it moderate; DE/local phases usually do the heavy lifting
    init_budget = max(pop_size, 10 * dim)

    # store population: list of (f, x)
    pop = []
    best = float("inf")
    best_x = None

    # ---------------- phase 1: scrambled Halton + random + opposition ----------------
    shift = [random.random() for _ in range(dim)]
    k = 1
    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1

        fx = eval_f(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, list(x)

        # also evaluate opposite with some probability (cheap but helpful)
        if time.time() >= deadline:
            break
        if random.random() < 0.60:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, list(xo)

        i += 1

    if not pop:
        x = rand_vec()
        return eval_f(x)

    # keep only best pop_size
    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_size]
    best, best_x = pop[0][0], list(pop[0][1])

    # ---------------- phase 2: DE-style evolution with elitism ----------------
    # DE parameters (jittered over time)
    F_base = 0.55
    CR_base = 0.85

    # local search step scales (for late exploitation)
    sigma = [0.20 * s for s in span]
    min_sigma = [1e-14 * s for s in span]
    max_sigma = [0.70 * s for s in span]

    # For step adaptation (1/5 rule-ish)
    succ = 0
    trials = 0
    window = 25

    # stagnation handling
    last_best_t = time.time()
    stagnation = max(0.25, 0.12 * max_time)

    def pick3(exclude_index):
        # pick 3 distinct indices not equal to exclude_index
        idxs = list(range(len(pop)))
        # swap-remove exclude for speed
        if 0 <= exclude_index < len(idxs):
            idxs[exclude_index], idxs[-1] = idxs[-1], idxs[exclude_index]
            idxs.pop()
        a = random.choice(idxs); idxs.remove(a)
        b = random.choice(idxs); idxs.remove(b)
        c = random.choice(idxs)
        return a, b, c

    def local_refine(x, fx, intensity):
        """
        Small budget local refinement:
          - adaptive coordinate probing on a subset of dims
          - followed by a few mirrored gaussian steps
        intensity in [0,1]: higher near the end of time.
        """
        nonlocal best, best_x, last_best_t, succ, trials, sigma

        # coordinate subset size grows with intensity
        m = 1 if dim == 1 else min(dim, 2 + int(4 * intensity))
        idxs = random.sample(range(dim), m)

        # coordinate pattern tries (plus/minus)
        for j in idxs:
            lo, hi = bounds[j]
            step = (0.35 + 0.45 * intensity) * sigma[j]
            if step <= 0:
                continue

            # try + then -
            xp = list(x); xp[j] = clamp(xp[j] + step, lo, hi)
            fp = eval_f(xp); trials += 1
            if fp < fx:
                x, fx = xp, fp
                succ += 1
                if fp < best:
                    best, best_x = fp, list(x)
                    last_best_t = time.time()
                continue

            xm = list(x); xm[j] = clamp(xm[j] - step, lo, hi)
            fm = eval_f(xm); trials += 1
            if fm < fx:
                x, fx = xm, fm
                succ += 1
                if fm < best:
                    best, best_x = fm, list(x)
                    last_best_t = time.time()

            if time.time() >= deadline:
                return x, fx

        # a few mirrored gaussian steps (cheap exploitation)
        reps = 1 + int(2 * intensity)
        for _ in range(reps):
            if time.time() >= deadline:
                break
            z = [gauss() for _ in range(dim)]
            x1 = [0.0] * dim
            x2 = [0.0] * dim
            for j in range(dim):
                lo, hi = bounds[j]
                step = z[j] * sigma[j]
                x1[j] = clamp(x[j] + step, lo, hi)
                x2[j] = clamp(x[j] - step, lo, hi)
            f1 = eval_f(x1); trials += 1
            if time.time() >= deadline:
                break
            f2 = eval_f(x2); trials += 1

            if f1 <= f2 and f1 < fx:
                x, fx = x1, f1
                succ += 1
            elif f2 < f1 and f2 < fx:
                x, fx = x2, f2
                succ += 1

            if fx < best:
                best, best_x = fx, list(x)
                last_best_t = time.time()

        return x, fx

    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0
        if deadline > start:
            tfrac = min(1.0, max(0.0, (now - start) / (deadline - start)))

        # exploration early, exploitation late
        intensity = tfrac

        # restart/diversify if stagnating
        if (now - last_best_t) > stagnation:
            # inject a few new samples (half near-best, half global)
            new = []
            inject = max(2, pop_size // 4)
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if random.random() < 0.65 and best_x is not None:
                    x = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        x.append(clamp(best_x[j] + gauss() * (0.25 * span[j]), lo, hi))
                else:
                    x = rand_vec()
                fx = eval_f(x)
                new.append((fx, x))
                if fx < best:
                    best, best_x = fx, list(x)
                    last_best_t = time.time()
            pop.extend(new)
            pop.sort(key=lambda t: t[0])
            pop = pop[:pop_size]
            # reset local step a bit
            sigma = [0.25 * s for s in span]
            succ = 0
            trials = 0

        # choose target index
        i = random.randrange(len(pop))
        fi, xi = pop[i]

        # parameters jitter
        F = clamp(F_base + 0.20 * (random.random() - 0.5), 0.25, 0.95)
        CR = clamp(CR_base + 0.20 * (random.random() - 0.5), 0.20, 0.98)

        # pick donors
        a, b, c = pick3(i)
        xa = pop[a][1]
        xb = pop[b][1]
        xc = pop[c][1]

        # rand-to-best/1: v = xa + F*(best-xa) + F*(xb-xc)
        v = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            vj = xa[j] + F * (best_x[j] - xa[j]) + F * (xb[j] - xc[j])
            # occasional mild gaussian to de-align (more early than late)
            if random.random() < (0.12 * (1.0 - intensity)):
                vj += gauss() * (0.02 * span[j])
            v[j] = clamp(vj, lo, hi)

        # binomial crossover
        u = [0.0] * dim
        jrand = random.randrange(dim)
        for j in range(dim):
            if random.random() < CR or j == jrand:
                u[j] = v[j]
            else:
                u[j] = xi[j]

        fu = eval_f(u)

        # selection
        if fu <= fi:
            pop[i] = (fu, u)
            if fu < best:
                best, best_x = fu, list(u)
                last_best_t = time.time()

            # late-stage local refinement when we get an improvement
            if intensity > 0.55 and time.time() < deadline:
                u2, fu2 = local_refine(u, fu, intensity)
                pop[i] = (fu2, u2)
                if fu2 < best:
                    best, best_x = fu2, list(u2)
                    last_best_t = time.time()

        # keep population sorted occasionally
        if random.random() < 0.15:
            pop.sort(key=lambda t: t[0])
            pop = pop[:pop_size]
            best, best_x = pop[0][0], list(pop[0][1])

        # adapt sigma using success ratio from local refines / accepted steps proxy
        if trials >= window:
            rate = succ / float(trials) if trials > 0 else 0.0
            if rate > 0.22:
                factor = 1.30
            elif rate < 0.14:
                factor = 0.75
            else:
                factor = 1.0
            if factor != 1.0:
                for j in range(dim):
                    sj = sigma[j] * factor
                    if sj < min_sigma[j]: sj = min_sigma[j]
                    if sj > max_sigma[j]: sj = max_sigma[j]
                    sigma[j] = sj
            succ = 0
            trials = 0

        # small chance of pure random candidate injection (prevents premature convergence)
        if random.random() < (0.02 * (1.0 - intensity) + 0.003):
            x = rand_vec()
            fx = eval_f(x)
            # replace worst if better
            pop.sort(key=lambda t: t[0])
            if fx < pop[-1][0]:
                pop[-1] = (fx, x)
                if fx < best:
                    best, best_x = fx, list(x)
                    last_best_t = time.time()

    return best
