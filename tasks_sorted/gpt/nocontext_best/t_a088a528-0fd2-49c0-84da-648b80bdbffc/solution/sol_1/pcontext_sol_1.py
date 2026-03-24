import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Main ideas (compact, no external libs):
      - LHS-style initialization to get a good starting point quickly
      - (1+λ)-ES style sampling around an evolving mean with Cauchy/Gaussian mix
      - Lightweight diagonal covariance adaptation (per-dimension scales)
      - Success-based global step-size control (1/5-ish rule)
      - Elite archive + periodic restarts (best / random / blended)

    Returns:
      best fitness value found (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive.")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim.")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i, s in enumerate(spans):
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")

    # --- helpers ---
    def clip(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    # Box-Muller normal
    have_spare = False
    spare = 0.0
    def randn():
        nonlocal have_spare, spare
        if have_spare:
            have_spare = False
            return spare
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        spare = z1
        have_spare = True
        return z0

    # Cauchy via tan(pi*(u-0.5)) (heavy-tail)
    def randc():
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    def evaluate(x):
        return float(func(x))

    # --- LHS-like initialization ---
    best = float("inf")
    best_x = None

    init_n = max(12, min(80, 12 * dim))
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            if spans[i] == 0:
                x.append(lows[i])
            else:
                a = perms[i][j]
                u = (a + random.random()) / init_n
                x.append(lows[i] + u * spans[i])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # --- Evolution Strategy core ---
    # Mean starts at current best
    m = best_x[:]

    # Diagonal "sigma per dimension" (relative scales), plus a global multiplier
    sig_diag = [0.25 * s if s > 0 else 0.0 for s in spans]
    # Ensure non-zero exploration when spans are tiny but non-zero
    for i in range(dim):
        if spans[i] > 0 and sig_diag[i] <= 0:
            sig_diag[i] = spans[i] * 0.25

    # Global step multiplier
    sigma_g = 1.0

    # Bounds on sigma_g to avoid collapse/explosion
    sigma_g_min = 1e-6
    sigma_g_max = 50.0

    # Elite archive for restarts
    archive = [(best, best_x[:])]
    arch_max = 10

    # Controls
    lam = max(10, min(50, 4 * dim))       # offspring per generation
    mu = max(2, lam // 4)                 # number of elites to recombine
    stall = 0
    stall_limit = 25 + 6 * dim
    restart_every = 80 + 15 * dim

    # Success-rate adaptation
    succ = 0
    trials = 0
    adapt_window = 20

    # For diagonal adaptation: keep an EMA of successful step magnitudes
    ema = [sd if sd > 0 else 0.0 for sd in sig_diag]
    ema_beta = 0.15

    gen = 0
    while time.time() < deadline:
        gen += 1

        # Periodic or stall-triggered restart
        if (gen % restart_every == 0) or (stall >= stall_limit):
            stall = 0
            # Choose restart mode: best-based, archive-based, or global random
            r = random.random()
            if r < 0.45 and archive:
                _, base = random.choice(archive)
                m = base[:]
                sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g * 1.8))
            elif r < 0.80:
                m = best_x[:]
                sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g * 1.5))
            else:
                m = rand_uniform()
                sigma_g = 1.0

        # Generate offspring
        offspring = []
        improved_this_gen = False

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # Mix Gaussian with occasional Cauchy jumps for escaping local minima
            use_cauchy = (random.random() < 0.12)

            x = m[:]
            for i in range(dim):
                if spans[i] == 0:
                    x[i] = lows[i]
                    continue
                step = sig_diag[i] * sigma_g
                if step <= 0:
                    continue
                z = randc() if use_cauchy else randn()
                x[i] = x[i] + step * z

            x = clip(x)
            fx = evaluate(x)
            offspring.append((fx, x))

            trials += 1
            if fx < best:
                best = fx
                best_x = x[:]
                improved_this_gen = True
                succ += 1

                archive.append((best, best_x[:]))
                archive.sort(key=lambda t: t[0])
                if len(archive) > arch_max:
                    archive = archive[:arch_max]

        offspring.sort(key=lambda t: t[0])

        # Recombine top-mu to update mean (weighted)
        # Weights: log scheme
        weights = []
        for k in range(mu):
            weights.append(max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)))
        wsum = sum(weights) or 1.0
        weights = [w / wsum for w in weights]

        new_m = [0.0] * dim
        for k in range(mu):
            _, xk = offspring[k]
            wk = weights[k]
            for i in range(dim):
                new_m[i] += wk * xk[i]
        m = clip(new_m)

        # Diagonal adaptation using best offspring displacement (stabilizes sig per dimension)
        # Use the best candidate as "direction" proxy
        fx0, x0 = offspring[0]
        # Update EMA of absolute displacement (normalized)
        for i in range(dim):
            if spans[i] == 0:
                continue
            d = abs(x0[i] - m[i])
            # Convert to a target sigma scale; keep within reasonable span fraction
            target = max(1e-15, min(0.5 * spans[i], d * 1.5 + 1e-12 * spans[i]))
            ema[i] = (1.0 - ema_beta) * ema[i] + ema_beta * target
            # Blend into sig_diag slowly; keep floors
            floor = 1e-12 * spans[i] + 1e-15
            sig_diag[i] = max(floor, min(0.5 * spans[i], 0.85 * sig_diag[i] + 0.15 * ema[i]))

        # Global step-size adaptation via success rate
        if trials >= adapt_window:
            rate = succ / float(trials)
            # if too few successes -> shrink, too many -> grow
            if rate < 0.18:
                sigma_g *= 0.82
            elif rate > 0.28:
                sigma_g *= 1.18
            sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g))
            succ = 0
            trials = 0

        stall = 0 if improved_this_gen else (stall + 1)

    return best
