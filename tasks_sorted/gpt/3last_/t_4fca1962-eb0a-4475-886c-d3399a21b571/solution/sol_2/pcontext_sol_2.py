import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libraries):
    Hybrid optimizer:
      - Sobol-like coverage via scrambled Halton seeding (fast, good space-filling)
      - CMA-ES-inspired diagonal adaptation (per-dimension sigmas updated from successes)
      - Trust-region local refinement around best (coordinate + random orthogonal-ish moves)
      - Heavy-tailed restarts around best + occasional global samples

    Returns:
        best (float): best (minimum) fitness found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s > 0 else 0.0 for s in spans]

    # ---------------- helpers ----------------
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

    # ---- Halton with per-dimension random digit scramble (cheap "scrambled" LDS) ----
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
               127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191]

    def _is_prime(n):
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

    def _ensure_primes(k):
        nonlocal _PRIMES
        if len(_PRIMES) >= k:
            return
        p = _PRIMES[-1] + 2
        while len(_PRIMES) < k:
            if _is_prime(p):
                _PRIMES.append(p)
            p += 2

    # digit permutation per base (simple scramble): for each dim, a random permutation of digits [0..base-1]
    digit_perms = []
    def _init_scramble():
        _ensure_primes(dim)
        for i in range(dim):
            base = _PRIMES[i]
            perm = list(range(base))
            random.shuffle(perm)
            digit_perms.append(perm)

    def halton_scrambled(index, base, perm):
        # index >= 1
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            digit = i % base
            r += f * perm[digit]
            i //= base
        # map to [0,1)
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            base = _PRIMES[i]
            u = halton_scrambled(k, base, digit_perms[i])
            x[i] = lows[i] + u * spans[i]
        return x

    def approx_gauss():
        # 12-uniform trick: ~N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_clip(scale=1.0):
        # heavy-tailed: tan(pi*(u-0.5)), clipped
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        if v > 25.0: v = 25.0
        if v < -25.0: v = -25.0
        return scale * v

    # ---------------- initialization ----------------
    _init_scramble()

    best = float("inf")
    best_x = None

    # Seeding budget: modest but better than pure random
    seed_n = max(24, 12 * dim)
    seed_n = min(seed_n, 240)

    # A few purely random points too (guards against LDS pathology)
    extra_rand = min(max(8, 2 * dim), 60)

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

    # ---------------- main search state ----------------
    x = best_x[:]
    fx = best

    # Diagonal "CMA-like" sigmas (per-dimension step stddev)
    # start relatively broad, then adapt from success directions
    sigma = [0.22 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    sigma_min = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    sigma_max = [0.60 * spans[i] for i in range(dim)]

    # Trust region radius around best (normalized)
    trust = 0.35  # in [0,1] roughly (as fraction of span)
    trust_min = 1e-9
    trust_max = 0.70

    # Success tracking to adapt sigma
    succ_ema = 0.0
    alpha = 0.10  # EMA rate

    # Occasional global / restart controls
    p_global = 0.03
    p_restart = 0.03

    # Keep a small archive of good points for diversified restarts
    elite = [(best, best_x[:])]
    elite_cap = 6

    def consider_elite(fv, xv):
        nonlocal elite
        # insert sorted unique-ish by fitness
        elite.append((fv, xv[:]))
        elite.sort(key=lambda t: t[0])
        # prune
        if len(elite) > elite_cap:
            elite = elite[:elite_cap]

    consider_elite(best, best_x)

    # ---------------- main loop ----------------
    while time.time() < deadline:
        # Occasionally sample globally (keeps exploration alive)
        if random.random() < p_global:
            if time.time() >= deadline:
                break
            xr = halton_point(k); k += 1
            fr = eval_f(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                consider_elite(fr, xr)
            # small chance to move the chain
            if fr < fx or random.random() < 0.05:
                x, fx = xr, fr
            continue

        # Restart: heavy-tailed around a random elite (usually the best)
        if random.random() < p_restart:
            base = elite[0][1] if random.random() < 0.75 else random.choice(elite)[1]
            xr = base[:]
            for i in range(dim):
                if spans[i] > 0:
                    xr[i] += 0.12 * spans[i] * cauchy_clip(1.0)
            clip_inplace(xr)
            fr = eval_f(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                consider_elite(fr, xr)
            x, fx = xr, fr
            # re-widen slightly after restart
            trust = min(trust_max, trust * 1.15 + 0.03)
            for i in range(dim):
                if spans[i] > 0:
                    sigma[i] = min(sigma_max[i], max(sigma[i], 0.10 * spans[i]))
            continue

        # Choose a "center": mostly best, sometimes current (helps local walking)
        center = best_x if random.random() < 0.80 else x

        # Generate a small batch and take the best (cheap selection pressure)
        # Batch size scales mildly with dim, but capped for time.
        lam = min(16, max(6, 2 + dim // 3))

        best_cand_f = float("inf")
        best_cand_x = None
        best_cand_step = None

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # Mix move types: diagonal gaussian, coordinate, and "trust-box" uniform
            r = random.random()
            xc = center[:]
            step_vec = [0.0] * dim

            if r < 0.60:
                # diagonal gaussian
                for i in range(dim):
                    if spans[i] > 0:
                        s = sigma[i] * approx_gauss()
                        xc[i] += s
                        step_vec[i] = s
            elif r < 0.85:
                # coordinate move (good for separable-ish problems)
                j = random.randrange(dim)
                if spans[j] > 0:
                    s = sigma[j] * (1.5 * approx_gauss())
                    xc[j] += s
                    step_vec[j] = s
            else:
                # uniform within trust region box around center
                for i in range(dim):
                    if spans[i] > 0:
                        s = (2.0 * random.random() - 1.0) * (trust * spans[i])
                        xc[i] += s
                        step_vec[i] = s

            clip_inplace(xc)
            fc = eval_f(xc)

            if fc < best_cand_f:
                best_cand_f = fc
                best_cand_x = xc
                best_cand_step = step_vec

        # Accept best candidate if it improves current, else sometimes still move a bit
        improved = False
        if best_cand_x is not None:
            if best_cand_f <= fx:
                x, fx = best_cand_x, best_cand_f
                improved = True
            else:
                # small probability to move (keeps exploration and adapts sigma)
                if random.random() < 0.08:
                    x, fx = best_cand_x, best_cand_f

            if best_cand_f < best:
                best, best_x = best_cand_f, best_cand_x[:]
                consider_elite(best, best_x)
                improved = True

        # --- Adaptation ---
        # Update success rate EMA (treat any improvement of current or best as success)
        succ_ema = (1.0 - alpha) * succ_ema + alpha * (1.0 if improved else 0.0)

        # 1/5th success rule flavor: target ~0.2 success
        # If too successful -> increase steps; if not -> decrease.
        target = 0.20
        if succ_ema > target:
            trust = min(trust_max, trust * 1.06 + 1e-6)
            mult = 1.04
        else:
            trust = max(trust_min, trust * 0.92)
            mult = 0.96

        # Per-dimension sigma adaptation from last selected step direction:
        if best_cand_step is not None:
            for i in range(dim):
                if spans[i] <= 0:
                    continue
                # If dimension moved meaningfully, slightly boost; otherwise decay a bit.
                # Also bias by normalized magnitude to avoid shrinking useful dims.
                mag = abs(best_cand_step[i]) * inv_spans[i]
                if mag > 0.02:
                    sigma[i] *= (mult * (1.0 + 0.6 * min(1.0, mag)))
                else:
                    sigma[i] *= (0.995 if improved else 0.985)

                if sigma[i] < sigma_min[i]:
                    sigma[i] = sigma_min[i]
                elif sigma[i] > sigma_max[i]:
                    sigma[i] = sigma_max[i]

        # If stuck badly, do a directed local refinement around best (cheap pattern)
        if not improved and random.random() < 0.12:
            xb = best_x[:]
            fb = best
            # try a few coordinate nudges with current sigma (trust-region-ish)
            trials = min(2 * dim, 40)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                j = random.randrange(dim)
                if spans[j] <= 0:
                    continue
                xc = xb[:]
                s = sigma[j] * (1.0 if random.random() < 0.5 else -1.0)
                xc[j] += s
                clip_inplace(xc)
                fc = eval_f(xc)
                if fc < fb:
                    xb, fb = xc, fc
            if fb < best:
                best, best_x = fb, xb[:]
                consider_elite(best, best_x)
                x, fx = xb, fb
                trust = min(trust_max, trust * 1.10 + 0.01)

    return best
