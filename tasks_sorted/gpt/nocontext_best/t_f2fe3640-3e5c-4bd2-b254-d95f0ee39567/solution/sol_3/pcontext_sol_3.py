import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (stdlib-only).

    Key upgrades vs. the provided hybrid:
      - Better quasi-random coverage early (Halton + Latin-ish stratification)
      - Maintain an elite set + "center of mass" (CMA-ES-ish without numpy)
      - Adaptive diagonal scaling per-dimension from successful steps
      - More reliable local refinement: pattern search with step halving
      - Robust restarts using elites (mix: best, random elite, and global samples)
      - Cheap memoization with coarser quantization (avoids huge dict + misses)

    Returns:
      best (float): best (minimum) fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps = 1e-300

    # -------------------- helpers --------------------
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

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if span[i] <= 0:
            span[i] = 1.0

    # cache with quantization in normalized space
    # (coarser than 1e-12: avoids enormous keys and improves hit rate)
    cache = {}
    q = 1e-9

    def key_of(x):
        # normalized [0,1], quantized
        return tuple(int(((x[i] - lo[i]) / span[i]) / q) for i in range(dim))

    def eval_f(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        fx = float(func(x))
        cache[k] = fx
        return fx

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # -------------------- Halton sequence --------------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            r = int(math.isqrt(x))
            ok = True
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def vdc(k, base):
        out = 0.0
        denom = 1.0
        while k:
            k, r = divmod(k, base)
            denom *= base
            out += r / denom
        return out

    bases = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = []
        for i in range(dim):
            u = vdc(k, bases[i])
            x.append(lo[i] + u * span[i])
        return x

    # -------------------- initialization: mixed Halton + stratified jitter --------------------
    best = float("inf")
    best_x = None

    # elite archive (fx, x)
    elite_max = max(12, 2 * dim)
    elites = []

    def push_elite(fx, x):
        nonlocal elites
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_max:
            elites = elites[:elite_max]

    # stratified per-dimension jittered sampling (cheap, improves early coverage)
    def strat_point(k, m):
        # for each dim choose stratum based on (k + perm(i)) % m
        x = []
        for i in range(dim):
            s = (k + (i * 17 + 3)) % m
            u = (s + random.random()) / m
            x.append(lo[i] + u * span[i])
        return x

    init_n = max(80, 30 * dim)
    m = int(math.sqrt(init_n)) + 5

    for t in range(init_n):
        if now() >= deadline:
            return best

        r = random.random()
        if r < 0.55:
            x = halton_point(hal_k); hal_k += 1
        elif r < 0.90:
            x = strat_point(t, m)
        else:
            x = rand_point()

        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        push_elite(fx, x)

    if best_x is None:
        return best

    # -------------------- search state --------------------
    # diagonal sampling scales (fraction of span)
    sig = [0.20 for _ in range(dim)]
    sig_min = 1e-14
    sig_max = 0.90

    # pattern/coordinate step sizes
    step = [0.10 * span[i] for i in range(dim)]
    step_min = [1e-14 * span[i] for i in range(dim)]

    # ES parameters
    lam = max(12, 6 * dim)
    heavy_tail_p = 0.07

    def cauchy():
        # standard Cauchy
        return math.tan(math.pi * (random.random() - 0.5))

    def sample_around(center):
        x = []
        use_ht = (random.random() < heavy_tail_p)
        for i in range(dim):
            if use_ht:
                z = 0.12 * sig[i] * cauchy()
            else:
                z = random.gauss(0.0, sig[i])
            x.append(center[i] + z * span[i])
        return project(x)

    def weighted_center_of_elites():
        # weight better elites more (rank-based)
        if not elites:
            return best_x[:]
        # compute rank weights ~ 1/(rank+1)
        wsum = 0.0
        c = [0.0] * dim
        for rnk, (fx, x) in enumerate(elites):
            w = 1.0 / (1.0 + rnk)
            wsum += w
            for i in range(dim):
                c[i] += w * x[i]
        inv = 1.0 / (wsum + eps)
        for i in range(dim):
            c[i] *= inv
        return project(c)

    def pattern_refine(x0, f0):
        # coordinate pattern search with step halving until no progress
        x = x0[:]
        f = f0
        improved_any = False

        # one sweep tries +/- step on randomized dims
        idxs = list(range(dim))
        random.shuffle(idxs)
        for i in idxs:
            if now() >= deadline:
                break
            si = step[i]
            if si <= step_min[i]:
                continue

            # + direction
            xp = x[:]
            xp[i] = clip(xp[i] + si, lo[i], hi[i])
            fp = eval_f(xp)
            if fp < f:
                x, f = xp, fp
                improved_any = True
                continue

            if now() >= deadline:
                break

            # - direction
            xm = x[:]
            xm[i] = clip(xm[i] - si, lo[i], hi[i])
            fm = eval_f(xm)
            if fm < f:
                x, f = xm, fm
                improved_any = True

        return x, f, improved_any

    # success tracking for per-dim sigma adaptation
    succ = [0.0] * dim
    tries = [0.0] * dim

    def adapt_sig():
        # push sig up/down based on recent success ratio
        for i in range(dim):
            if tries[i] <= 0:
                continue
            r = succ[i] / (tries[i] + eps)
            # target around 0.2
            if r > 0.25:
                sig[i] = min(sig_max, sig[i] * 1.15)
            elif r < 0.12:
                sig[i] = max(sig_min, sig[i] * 0.85)
        # decay counters
        for i in range(dim):
            succ[i] *= 0.6
            tries[i] *= 0.6

    # -------------------- main loop with restarts --------------------
    no_imp = 0
    stagnate = 80 + 15 * dim
    gen = 0

    while True:
        if now() >= deadline:
            return best

        gen += 1

        # choose a center: mostly best, sometimes elite COM, sometimes random elite
        r = random.random()
        if r < 0.62:
            center = best_x
        elif r < 0.86:
            center = weighted_center_of_elites()
        else:
            center = elites[random.randrange(len(elites))][1] if elites else best_x

        # offspring sampling
        best_gen_f = float("inf")
        best_gen_x = None
        improved = False

        for _ in range(lam):
            if now() >= deadline:
                return best

            # occasional global injection
            if random.random() < 0.10:
                if random.random() < 0.70:
                    cand = halton_point(hal_k); hal_k += 1
                else:
                    cand = rand_point()
            else:
                cand = sample_around(center)

            fc = eval_f(cand)
            push_elite(fc, cand)

            if fc < best_gen_f:
                best_gen_f = fc
                best_gen_x = cand

            if fc < best:
                improved = True

                # per-dim success: credit dims that moved notably from center
                for i in range(dim):
                    tries[i] += 1.0
                    if abs(cand[i] - center[i]) > 0.25 * sig[i] * span[i]:
                        succ[i] += 1.0
            else:
                # still count tries for adaptation stability
                for i in range(dim):
                    tries[i] += 0.25

        adapt_sig()

        if best_gen_f < best:
            best, best_x = best_gen_f, best_gen_x
            no_imp = 0
        else:
            no_imp += 1

        # periodic deterministic-ish refinement
        if gen % 4 == 0 and now() < deadline:
            x2, f2, imp2 = pattern_refine(best_x, best)
            if f2 < best:
                best, best_x = f2, x2
                no_imp = 0
                # tighten after local improvement
                for i in range(dim):
                    sig[i] = max(sig_min, sig[i] * 0.90)
                    step[i] = max(step_min[i], step[i] * 0.85)
            else:
                # slow step decay
                for i in range(dim):
                    step[i] = max(step_min[i], step[i] * 0.92)

        # stagnation handling: restart center and re-expand scales a bit
        if no_imp >= stagnate:
            no_imp = 0

            # pick among top-k elites for diversity
            k = min(max(6, dim), len(elites))
            if k > 0:
                _, rx = elites[random.randrange(k)]
                best_x = rx[:]

            # re-expand scales (but not too much)
            for i in range(dim):
                sig[i] = min(sig_max, max(0.12, sig[i] * 1.35))
                step[i] = min(0.25 * span[i], max(step[i], 0.08 * span[i]))

            # inject a small fresh global batch to refresh elites
            batch = max(12, 3 * dim)
            for _ in range(batch):
                if now() >= deadline:
                    return best
                if random.random() < 0.75:
                    x = halton_point(hal_k); hal_k += 1
                else:
                    x = rand_point()
                fx = eval_f(x)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x
