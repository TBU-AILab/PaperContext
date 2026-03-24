import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Better time-bounded minimizer (no external libs):
    Multi-start, adaptive hybrid:
      - Scrambled Halton seeding for space-filling coverage
      - (1+λ)-ES local search with diagonal step-sizes (per-dimension sigmas)
      - Success-rule adaptation (global + per-dim)
      - Periodic finite-difference gradient step (SPSA-like but deterministic FD)
      - Elite archive + heavy-tailed restarts (Cauchy) + occasional global samples

    Returns:
        best (float): best fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    invsp = [1.0 / s if s > 0 else 0.0 for s in spans]

    # ---------- helpers ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # normal approx: sum of uniforms
    def gauss01():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    # heavy tail
    def cauchy(scale=1.0):
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 30.0: v = 30.0
        if v < -30.0: v = -30.0
        return scale * v

    # ---------- scrambled Halton ----------
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
              193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
              269, 271, 277, 281, 283, 293]

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(n ** 0.5)
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True

    def ensure_primes(k):
        nonlocal PRIMES
        if len(PRIMES) >= k:
            return
        p = PRIMES[-1] + 2
        while len(PRIMES) < k:
            if is_prime(p):
                PRIMES.append(p)
            p += 2

    ensure_primes(dim)
    digit_perms = []
    for i in range(dim):
        base = PRIMES[i]
        perm = list(range(base))
        random.shuffle(perm)
        digit_perms.append(perm)

    def halton_scrambled(idx, base, perm):
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_scrambled(k, PRIMES[i], digit_perms[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- init / seeding ----------
    best = float("inf")
    best_x = None

    # Seeding size: more than before but capped; usually pays off.
    seed_n = max(40, 18 * dim)
    seed_n = min(seed_n, 420)

    extra_rand = min(max(10, 3 * dim), 80)

    k = 1
    for _ in range(seed_n):
        if time.time() >= deadline:
            return best
        x = halton_point(k); k += 1
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    for _ in range(extra_rand):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        return best

    # ---------- elite archive ----------
    elite_cap = 10
    elite = [(best, best_x[:])]

    def push_elite(fv, xv):
        nonlocal elite
        elite.append((fv, xv[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            elite = elite[:elite_cap]

    # ---------- ES state ----------
    x = best_x[:]
    fx = best

    # per-dim sigmas (diagonal covariance)
    sigma = [0.18 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    sig_min = [1e-15 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    sig_max = [0.65 * spans[i] for i in range(dim)]

    # global step multiplier
    gscale = 1.0
    gscale_min, gscale_max = 1e-6, 10.0

    # adaptation tracking
    succ_ema = 0.0
    ema_a = 0.08
    target = 0.22  # slightly > 1/5

    # controls
    p_global = 0.035
    p_restart = 0.040
    p_grad = 0.09
    lam_base = 6
    lam_cap = 28

    # stagnation
    it = 0
    last_improve_it = 0

    # precomputed coordinate order buffer to reduce allocations
    coord = list(range(dim))

    # ---------- gradient-ish step (finite differences along random +/-1 vector) ----------
    def grad_step(center_x, center_f):
        # Choose an epsilon relative to current sigma/spans
        # one-sided-ish SPSA would need 2 evals; FD along random sign vector also 2 evals.
        eps = 0.0
        for i in range(dim):
            if spans[i] > 0:
                eps = max(eps, 0.15 * sigma[i] * gscale)
        if eps <= 0.0:
            return None, None

        # random Rademacher direction
        d = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

        x1 = center_x[:]
        x2 = center_x[:]
        for i in range(dim):
            if spans[i] > 0:
                step = eps * d[i]
                x1[i] += step
                x2[i] -= step
        clip_inplace(x1)
        clip_inplace(x2)

        f1 = eval_f(x1)
        if time.time() >= deadline:
            return None, None
        f2 = eval_f(x2)

        # directional derivative estimate
        gdir = (f1 - f2) / (2.0 * eps + 1e-300)

        # take step opposite direction (normalize by dim to avoid too large moves)
        eta = 0.55 * eps  # step length
        xn = center_x[:]
        for i in range(dim):
            if spans[i] > 0:
                xn[i] += -eta * gdir * d[i] / max(1.0, math.sqrt(dim))
        clip_inplace(xn)
        fn = eval_f(xn)
        return xn, fn

    # ---------- main loop ----------
    while time.time() < deadline:
        it += 1

        # occasional global exploration
        if random.random() < p_global:
            xg = halton_point(k); k += 1
            fg = eval_f(xg)
            if fg < best:
                best, best_x = fg, xg[:]
                push_elite(fg, xg)
                last_improve_it = it
            # sometimes jump current state
            if fg < fx or random.random() < 0.08:
                x, fx = xg, fg
            continue

        # restart: heavy-tail around an elite (mostly the best)
        if random.random() < p_restart:
            base = elite[0][1] if random.random() < 0.75 else random.choice(elite)[1]
            xr = base[:]
            for i in range(dim):
                if spans[i] > 0:
                    xr[i] += 0.12 * spans[i] * cauchy(1.0)
            clip_inplace(xr)
            fr = eval_f(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                push_elite(fr, xr)
                last_improve_it = it
            x, fx = xr, fr
            # re-widen after restart
            gscale = min(gscale_max, gscale * 1.15 + 0.05)
            for i in range(dim):
                sigma[i] = min(sig_max[i], max(sigma[i], 0.10 * spans[i] if spans[i] > 0 else 0.0))
            continue

        center = best_x if random.random() < 0.82 else x
        center_f = best if center is best_x else fx

        # periodic gradient-ish refinement (costs 3 evals; only do sometimes)
        if random.random() < p_grad and time.time() < deadline:
            xn, fn = grad_step(center, center_f)
            if xn is not None:
                if fn < best:
                    best, best_x = fn, xn[:]
                    push_elite(fn, xn)
                    last_improve_it = it
                if fn < fx or random.random() < 0.03:
                    x, fx = xn, fn

        # (1+λ)-ES sample-and-select
        lam = lam_base + dim // 4
        if lam > lam_cap:
            lam = lam_cap

        best_cand_f = float("inf")
        best_cand_x = None
        best_step = None

        # mix two proposal types: gaussian + cross-over from elites
        use_mix = (len(elite) >= 3 and random.random() < 0.18)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            xc = center[:]
            step_vec = [0.0] * dim

            if use_mix and random.random() < 0.35:
                # elite crossover + small noise (often helps on rugged landscapes)
                a = random.choice(elite)[1]
                b = random.choice(elite)[1]
                for i in range(dim):
                    if spans[i] > 0:
                        w = random.random()
                        val = w * a[i] + (1.0 - w) * b[i]
                        # add small anisotropic noise
                        n = 0.35 * sigma[i] * gscale * gauss01()
                        s = (val + n) - xc[i]
                        xc[i] += s
                        step_vec[i] = s
            else:
                # diagonal gaussian
                for i in range(dim):
                    if spans[i] > 0:
                        s = sigma[i] * gscale * gauss01()
                        xc[i] += s
                        step_vec[i] = s

                # occasional coordinate kick (improves conditioning)
                if dim > 1 and random.random() < 0.30:
                    j = random.randrange(dim)
                    if spans[j] > 0:
                        s = 1.8 * sigma[j] * gscale * gauss01()
                        xc[j] += s
                        step_vec[j] += s

            clip_inplace(xc)
            fc = eval_f(xc)
            if fc < best_cand_f:
                best_cand_f = fc
                best_cand_x = xc
                best_step = step_vec

        improved = False
        if best_cand_x is not None:
            # greedy accept for current; occasional non-greedy for mobility
            if best_cand_f <= fx:
                x, fx = best_cand_x, best_cand_f
                improved = True
            elif random.random() < 0.05:
                x, fx = best_cand_x, best_cand_f

            if best_cand_f < best:
                best, best_x = best_cand_f, best_cand_x[:]
                push_elite(best, best_x)
                improved = True
                last_improve_it = it

        # adapt success EMA
        succ_ema = (1.0 - ema_a) * succ_ema + ema_a * (1.0 if improved else 0.0)

        # global scale update (1/5-like)
        if succ_ema > target:
            gscale = min(gscale_max, gscale * 1.05 + 1e-12)
            mult = 1.03
        else:
            gscale = max(gscale_min, gscale * 0.93)
            mult = 0.97

        # per-dimension sigma update based on utilized step magnitudes
        if best_step is not None:
            for i in range(dim):
                if spans[i] <= 0:
                    continue
                mag = abs(best_step[i]) * invsp[i]  # normalized
                if mag > 0.03:
                    sigma[i] *= mult * (1.0 + 0.40 * min(1.0, mag))
                else:
                    sigma[i] *= (0.998 if improved else 0.990)

                if sigma[i] < sig_min[i]:
                    sigma[i] = sig_min[i]
                elif sigma[i] > sig_max[i]:
                    sigma[i] = sig_max[i]

        # if stuck for a while, force a structured coordinate probe around best
        if (it - last_improve_it) > (18 + 3 * dim) and random.random() < 0.25:
            random.shuffle(coord)
            xb = best_x[:]
            fb = best
            trials = min(3 * dim, 60)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                j = coord[_ % dim]
                if spans[j] <= 0:
                    continue
                for sgn in (1.0, -1.0):
                    xc = xb[:]
                    s = sgn * sigma[j] * max(0.6, gscale)
                    xc[j] += s
                    clip_inplace(xc)
                    fc = eval_f(xc)
                    if fc < fb:
                        xb, fb = xc, fc
            if fb < best:
                best, best_x = fb, xb[:]
                push_elite(best, best_x)
                x, fx = xb, fb
                last_improve_it = it
                gscale = min(gscale_max, gscale * 1.12 + 0.02)

    return best
