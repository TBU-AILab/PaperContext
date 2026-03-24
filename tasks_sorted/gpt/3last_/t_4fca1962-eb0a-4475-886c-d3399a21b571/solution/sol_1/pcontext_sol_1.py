import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs):
    Hybrid of:
      1) Low-discrepancy sampling (Halton) for strong initial coverage
      2) Adaptive local search (coordinate pattern + diagonal moves)
      3) Simulated-annealing-style occasional acceptance to escape basins
      4) Restarts biased toward best-so-far with heavy-tailed steps (Cauchy-like)

    Returns:
        best (float): best (minimum) fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

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

    # First primes for Halton bases (extend if needed)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

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
        # Ensure we have at least k primes
        nonlocal _PRIMES
        if len(_PRIMES) >= k:
            return
        p = _PRIMES[-1] + 2
        while len(_PRIMES) < k:
            if _is_prime(p):
                _PRIMES.append(p)
            p += 2

    def halton_value(index, base):
        # index >= 1 recommended
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        # k starts at 1
        _ensure_primes(dim)
        x = [0.0] * dim
        for i in range(dim):
            u = halton_value(k, _PRIMES[i])
            x[i] = lows[i] + u * spans[i]
        return x

    def approx_cauchy():
        # Heavy-tailed step: tan(pi*(u-0.5)), clipped to avoid infinities
        u = random.random()
        v = math.tan(math.pi * (u - 0.5))
        # clip extreme outliers for numerical stability
        if v > 20.0:
            v = 20.0
        elif v < -20.0:
            v = -20.0
        return v

    # --------------- initialization (Halton + a few random) ---------------
    best = float("inf")
    best_x = None

    # Spend a small, bounded portion of time on global coverage
    # (more robust than LHS-ish for many dims).
    seed_n = max(16, 10 * dim)
    seed_n = min(seed_n, 160)

    # Mix in some pure random points too (helps if Halton aligns poorly)
    extra_rand = max(4, dim)
    extra_rand = min(extra_rand, 40)

    k = 1
    for _ in range(seed_n):
        if time.time() >= deadline:
            return best
        x = halton_point(k)
        k += 1
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

    # --------------- main loop: adaptive local + annealing + restarts ---------------
    x = best_x[:]
    fx = best

    # Initial step as fraction of span, with sane fallback if span==0
    step = [0.25 * s if s > 0 else 0.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]
    max_step = [0.5 * s for s in spans]

    # Annealing temperature based on |best| scale; adapt online
    # (kept small; this is mainly for occasional uphill escapes)
    T = 1.0
    if math.isfinite(best) and abs(best) > 1e-12:
        T = 0.05 * abs(best)
    T_min = 1e-12
    cool = 0.995

    # Control
    restart_prob = 0.02
    diag_prob = 0.35  # attempt diagonal move sometimes
    shrink = 0.7
    expand = 1.12

    # Stagnation tracking
    no_improve_iters = 0
    last_best = best

    while time.time() < deadline:
        # occasional restart (biased around best with heavy tails)
        if random.random() < restart_prob:
            if best_x is not None and random.random() < 0.75:
                xr = best_x[:]
                for i in range(dim):
                    if spans[i] > 0:
                        # heavy-tailed jump proportional to span
                        xr[i] += 0.15 * spans[i] * approx_cauchy()
                clip_inplace(xr)
            else:
                xr = rand_point()

            fr = eval_f(xr)
            # accept if better, or sometimes anyway to diversify
            if fr < fx or random.random() < 0.10:
                x, fx = xr, fr
            if fr < best:
                best, best_x = fr, xr[:]
            step = [0.25 * s if s > 0 else 0.0 for s in spans]
            T = max(T * 0.8, T_min)
            continue

        improved = False

        # --- local proposals: coordinate moves + occasional diagonal move ---
        # Choose a random order to reduce bias
        order = list(range(dim))
        random.shuffle(order)

        # Coordinate pattern search
        for i in order:
            if time.time() >= deadline:
                return best
            if spans[i] == 0 or step[i] <= min_step[i]:
                continue

            base = x[:]
            si = step[i]

            # try + and -
            best_cand = None
            best_cand_f = fx
            for d in (1.0, -1.0):
                xc = base[:]
                xc[i] += d * si
                clip_inplace(xc)
                fc = eval_f(xc)
                if fc < best_cand_f:
                    best_cand_f = fc
                    best_cand = xc

            # SA-style acceptance (small chance to accept uphill)
            if best_cand is None:
                # try one stochastic perturbation in this coordinate
                xc = base[:]
                xc[i] += (2.0 * random.random() - 1.0) * si
                clip_inplace(xc)
                fc = eval_f(xc)
                delta = fc - fx
                if delta <= 0.0 or (T > T_min and random.random() < math.exp(-delta / max(T, 1e-300))):
                    x, fx = xc, fc
                    improved = improved or (fc < fx + 0.0)
                    if fc < best:
                        best, best_x = fc, xc[:]
                        improved = True
            else:
                x, fx = best_cand, best_cand_f
                improved = True
                if fx < best:
                    best, best_x = fx, x[:]

        # Occasional diagonal move (helps on rotated valleys)
        if time.time() >= deadline:
            return best
        if random.random() < diag_prob:
            xc = x[:]
            for i in range(dim):
                if spans[i] > 0 and step[i] > min_step[i]:
                    # random +/- for each dim
                    direction = -1.0 if random.random() < 0.5 else 1.0
                    xc[i] += direction * step[i]
            clip_inplace(xc)
            fc = eval_f(xc)
            delta = fc - fx
            if delta <= 0.0 or (T > T_min and random.random() < math.exp(-delta / max(T, 1e-300))):
                x, fx = xc, fc
                if fc < best:
                    best, best_x = fc, xc[:]
                    improved = True

        # Step adaptation
        if improved:
            for i in range(dim):
                if spans[i] > 0:
                    step[i] = min(step[i] * expand, max_step[i])
            no_improve_iters = 0
        else:
            for i in range(dim):
                if spans[i] > 0:
                    step[i] *= shrink
            no_improve_iters += 1

        # Cooling
        T = max(T * cool, T_min)

        # If we are stuck with tiny steps, do a strong local restart around best
        tiny = True
        for i in range(dim):
            if spans[i] > 0 and step[i] > 50.0 * min_step[i]:
                tiny = False
                break

        if tiny or no_improve_iters > (25 + 5 * dim):
            if best_x is not None:
                x = best_x[:]
                # moderate heavy-tailed shake
                for i in range(dim):
                    if spans[i] > 0:
                        x[i] += 0.05 * spans[i] * approx_cauchy()
                clip_inplace(x)
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x[:]
            else:
                x = rand_point()
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x[:]
            step = [0.25 * s if s > 0 else 0.0 for s in spans]
            no_improve_iters = 0

        # track best trend (optional minor adjustment)
        if best < last_best:
            last_best = best
            # when finding improvements, slightly reheat to allow bigger escapes later
            T = max(T, 0.02 * abs(best) + 1e-12)

    return best
