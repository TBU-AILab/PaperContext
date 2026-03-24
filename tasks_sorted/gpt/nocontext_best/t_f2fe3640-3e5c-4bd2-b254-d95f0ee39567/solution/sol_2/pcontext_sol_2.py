import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization, self-contained (stdlib only).
    Hybrid optimizer (robust on many black-box problems):
      - Quasi-random init (Halton) + random
      - Multi-start local search with:
          * SPSA-style gradient sign steps (2 eval/step) for fast descent in high-d
          * (1+λ)-ES trust-region sampling around incumbent
          * Occasional coordinate/pattern refinement
      - Stagnation detection + diversified restarts
      - Fitness caching to avoid duplicate evaluations
    Returns:
      best (float): best (minimum) fitness found within max_time seconds
    """

    t0 = time.time()
    deadline = t0 + max_time

    # -------------------- utilities --------------------
    def now():
        return time.time()

    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def project(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    rng = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    rng = [r if r > 0 else 1.0 for r in rng]

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # --- caching (important when steps get small) ---
    cache = {}
    # quantization relative to range: avoids exploding cache yet catches duplicates
    qlevel = 1e-12

    def key_of(x):
        return tuple(int((x[i] - bounds[i][0]) / (rng[i] * qlevel + 1e-300)) for i in range(dim))

    def eval_f(x):
        k = key_of(x)
        fx = cache.get(k)
        if fx is None:
            fx = float(func(x))
            cache[k] = fx
        return fx

    # -------------------- Halton sequence --------------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
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

    def van_der_corput(idx, base):
        vdc = 0.0
        denom = 1.0
        while idx > 0:
            idx, rem = divmod(idx, base)
            denom *= base
            vdc += rem / denom
        return vdc

    bases = first_primes(dim)
    halton_k = 1

    def halton_point(k):
        x = []
        for i in range(dim):
            u = van_der_corput(k, bases[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None

    # More coverage early, but still bounded
    init_n = max(60, 20 * dim)
    for _ in range(init_n):
        if now() >= deadline:
            return best
        if random.random() < 0.85:
            x = halton_point(halton_k)
            halton_k += 1
        else:
            x = rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        return best

    # -------------------- local search components --------------------
    # Global exploration / local exploitation scale (fraction of range)
    sigma = 0.25
    sigma_min = 1e-14
    sigma_max = 0.80

    # ES offspring count
    lam = max(10, 5 * dim)

    # Coordinate step (absolute units)
    coord = [0.08 * rng[i] for i in range(dim)]
    coord_min = [1e-14 * rng[i] for i in range(dim)]

    # SPSA step parameters (normalized)
    # ak controls step size; ck controls perturb size for gradient estimate
    ak = 0.12
    ck = 0.08
    spsa_iter = 0

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    def es_candidate(center):
        # Mix Gaussian (local) and occasional heavy-tail (escape)
        use_c = (random.random() < 0.08)
        x = []
        for i in range(dim):
            if use_c:
                step = 0.18 * sigma * rng[i] * cauchy()
            else:
                step = random.gauss(0.0, sigma * rng[i])
            x.append(center[i] + step)
        return project(x)

    def try_coordinate_pass(x0, f0):
        x = x0[:]
        f = f0
        improved = False
        # randomize order to avoid bias
        idxs = list(range(dim))
        random.shuffle(idxs)
        for i in idxs:
            if now() >= deadline:
                break
            step = coord[i]
            if step <= coord_min[i]:
                continue

            lo, hi = bounds[i]
            # + step
            xp = x[:]
            xp[i] = clip(xp[i] + step, lo, hi)
            fp = eval_f(xp)
            if fp < f:
                x, f = xp, fp
                improved = True
                continue

            if now() >= deadline:
                break

            # - step
            xm = x[:]
            xm[i] = clip(xm[i] - step, lo, hi)
            fm = eval_f(xm)
            if fm < f:
                x, f = xm, fm
                improved = True

        return x, f, improved

    def spsa_step(x0, f0):
        """
        One SPSA step:
          - sample random +/-1 perturb vector d
          - evaluate f(x+ck*d) and f(x-ck*d)
          - approximate gradient and take a step
        Costs 2 evaluations, good in higher dimensions.
        """
        nonlocal spsa_iter, ak, ck

        # time guard for 2 evals
        if now() >= deadline:
            return x0, f0, False

        spsa_iter += 1
        # mild decay
        a = ak / (1.0 + 0.02 * spsa_iter)
        c = ck / (1.0 + 0.01 * spsa_iter)

        d = [1.0 if random.getrandbits(1) else -1.0 for _ in range(dim)]
        x_plus = [x0[i] + c * rng[i] * d[i] for i in range(dim)]
        x_minus = [x0[i] - c * rng[i] * d[i] for i in range(dim)]
        x_plus = project(x_plus)
        x_minus = project(x_minus)

        f_plus = eval_f(x_plus)
        if now() >= deadline:
            return x0, f0, False
        f_minus = eval_f(x_minus)

        # gradient estimate and update
        g = []
        denom = 2.0 * c
        for i in range(dim):
            # scale by range to make update roughly invariant
            gi = (f_plus - f_minus) / (denom * (d[i] * rng[i] + 1e-300))
            g.append(gi)

        # normalize step (prevents blow-ups)
        gnorm = math.sqrt(sum(gi * gi for gi in g)) + 1e-300
        # step in normalized space then map to variable units via rng
        x1 = [x0[i] - (a / gnorm) * g[i] * (rng[i] ** 0.5) for i in range(dim)]
        x1 = project(x1)
        f1 = eval_f(x1)
        return x1, f1, (f1 < f0)

    # -------------------- main loop (multi-start) --------------------
    no_improve = 0
    stagnate = 70 + 20 * dim
    gen = 0

    while True:
        if now() >= deadline:
            return best

        gen += 1

        # 1) A few SPSA steps (cheap, often strong early)
        if gen % 2 == 1:
            steps = 2 if dim >= 10 else 3
            for _ in range(steps):
                if now() >= deadline:
                    return best
                x1, f1, imp = spsa_step(best_x, best)
                if f1 < best:
                    best, best_x = f1, x1
                    no_improve = 0
                    # exploit a bit more after success
                    sigma = max(sigma_min, sigma * 0.90)
                    for i in range(dim):
                        coord[i] = min(0.25 * rng[i], coord[i] * 1.05)
                else:
                    no_improve += 1

        # 2) (1+λ)-ES sample around incumbent + occasional global points
        best_gen_f = float("inf")
        best_gen_x = None
        successes = 0

        for _ in range(lam):
            if now() >= deadline:
                return best

            if random.random() < 0.10:
                # global injection: Halton-biased
                if random.random() < 0.75:
                    cand = halton_point(halton_k)
                    halton_k += 1
                else:
                    cand = rand_point()
            else:
                cand = es_candidate(best_x)

            fc = eval_f(cand)

            if fc < best_gen_f:
                best_gen_f = fc
                best_gen_x = cand
            if fc < best:
                successes += 1

        # Success-based sigma adaptation (target ~0.2)
        sr = successes / float(lam)
        if sr > 0.22:
            sigma = min(sigma_max, sigma * 1.18)
        else:
            sigma = max(sigma_min, sigma * 0.86)

        if best_gen_f < best:
            best, best_x = best_gen_f, best_gen_x
            no_improve = 0
        else:
            no_improve += 1

        # 3) Periodic coordinate refinement (very robust late-stage)
        if gen % 5 == 0 and now() < deadline:
            x2, f2, imp = try_coordinate_pass(best_x, best)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0
                sigma = max(sigma_min, sigma * 0.85)
            else:
                # shrink coordinate steps gently when not useful
                for i in range(dim):
                    coord[i] = max(coord_min[i], coord[i] * 0.93)

        # 4) Stagnation: restart from a good global sample + reset scales
        if no_improve >= stagnate:
            no_improve = 0

            # sample a batch, pick one of the top-k for diversity
            k = max(4, dim // 2)
            trials = max(20, 6 * dim)

            pool = []
            for _ in range(trials):
                if now() >= deadline:
                    return best
                if random.random() < 0.80:
                    x = halton_point(halton_k)
                    halton_k += 1
                else:
                    x = rand_point()
                fx = eval_f(x)
                pool.append((fx, x))
                if fx < best:
                    best, best_x = fx, x

            pool.sort(key=lambda t: t[0])
            pick = pool[random.randrange(min(k, len(pool)))]
            best_x = pick[1]

            # reset scales for exploration again
            sigma = min(0.40, max(0.10, sigma * 1.35))
            coord = [0.10 * rng[i] for i in range(dim)]
            # also refresh SPSA scales (helps on new basin)
            ak = 0.12
            ck = 0.08
            spsa_iter = 0
