import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained, no external libs).

    Main upgrades vs your best (Halton + coordinate/random TR):
      - Uses a *small CMA-ES-like* stochastic search (diagonal covariance) as the main engine
        (very strong on continuous black-box problems).
      - Keeps Halton as a cheap warm-start / occasional diversification.
      - Uses ask/tell with mirrored sampling and rank-based weighting.
      - Strict time guarding and safe objective evaluation.
      - Works well across dimensions without needing gradients.

    Returns:
        best (float): best objective value found within time limit.
    """

    # ------------------------- utilities ------------------------- #
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def sanitize_bounds(b):
        out = []
        for lo, hi in b:
            lo = float(lo); hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    def safe_eval(x):
        # Clamp and evaluate robustly
        xx = [clamp(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    # ------------------------- Halton (optional warm start) ------------------------- #
    def first_primes(k):
        primes = []
        n = 2
        while len(primes) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        return primes

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        return [vdc(index, b) for b in bases]

    def to_bounds(u):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi == lo:
                x.append(lo)
            else:
                x.append(lo + u[i] * (hi - lo))
        return x

    def random_point():
        return [random.uniform(bnds[i][0], bnds[i][1]) for i in range(dim)]

    # ------------------------- math helpers ------------------------- #
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(v):
        return math.sqrt(dot(v, v))

    def randn():
        # Gaussian using Box-Muller
        # Avoid log(0)
        u1 = random.random()
        u2 = random.random()
        u1 = 1e-12 if u1 < 1e-12 else u1
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ------------------------- setup ------------------------- #
    if dim <= 0:
        return float("inf")
    bnds = sanitize_bounds(bounds)
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    start = time.time()
    deadline = start + float(max_time)

    # Domain scale
    ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
    mean_range = 0.0
    active = 0
    for r in ranges:
        if r > 0:
            mean_range += r
            active += 1
    mean_range = (mean_range / active) if active else 1.0

    # Best so far
    best = float("inf")
    best_x = None

    # ------------------------- warm start (Halton + random) ------------------------- #
    # Very small cost; helps choose a decent initial mean.
    bases = first_primes(dim)
    idx = 1
    warm_cap = 8 + 4 * dim  # modest
    warm = []
    while idx <= warm_cap and time.time() < deadline:
        if idx <= warm_cap // 2:
            x0 = to_bounds(halton_point(idx, bases))
        else:
            x0 = random_point()
        f0, x0 = safe_eval(x0)
        warm.append((f0, x0))
        if f0 < best:
            best, best_x = f0, x0
        idx += 1

    # Initial mean
    if best_x is None:
        m = random_point()
        best, best_x = safe_eval(m)
    else:
        m = best_x[:]

    # ------------------------- diagonal CMA-ES-like optimizer ------------------------- #
    # Parameters (lightweight defaults)
    n = dim
    lam = max(8, 4 + int(3 * math.log(n + 1.0)) * 2)  # population size
    if lam % 2 == 1:
        lam += 1
    mu = lam // 2

    # Log weights (rank-based, normalized)
    w = [0.0] * mu
    for i in range(mu):
        w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
    w_sum = sum(w) if mu > 0 else 1.0
    w = [wi / w_sum for wi in w]
    mu_eff = 1.0 / sum(wi * wi for wi in w) if mu > 0 else 1.0

    # Step size and diag "cov" (actually std factors)
    # Start moderate relative to domain; shrink if clamping is frequent.
    sigma = 0.25 * mean_range
    if sigma <= 0:
        sigma = 1.0

    # Diagonal scaling factors (like sqrt of covariance diagonal)
    diag = [1.0] * n

    # Evolution path and adaptation constants (simplified)
    # For diagonal-only, keep it simple and stable.
    c_sigma = min(0.3, (mu_eff + 2.0) / (n + mu_eff + 5.0))
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1e-9)) - 1.0) + c_sigma
    # "cov" adaptation rate
    c_c = min(0.4, (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n))
    c1 = 0.10 / ((n + 1.3) ** 2 + mu_eff)  # small/stable
    c_mu = min(0.25, 0.15 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    if c_mu < 0.0:
        c_mu = 0.0

    p_sigma = [0.0] * n
    p_c = [0.0] * n

    # Expected length of N(0,I)
    # Approx: E||N(0,I)|| ~ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
    if n > 1:
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    else:
        chi_n = 1.0

    # Restart / stagnation controls
    no_improve_gens = 0
    last_best = best
    evals_since_restart = 0

    # ------------------------- main loop ------------------------- #
    while time.time() < deadline:
        # If sigma collapsed too much, restart around best with larger sigma
        if sigma < 1e-15 * mean_range:
            sigma = 0.20 * mean_range
            diag = [1.0] * n
            p_sigma = [0.0] * n
            p_c = [0.0] * n
            if best_x is not None:
                m = best_x[:]
            else:
                m = random_point()
            evals_since_restart = 0

        # Ask: generate lam candidates (mirrored sampling)
        pop = []
        # Track clamping amount to reduce sigma if we slam into bounds a lot
        clamp_hits = 0

        half = lam // 2
        z_store = []  # store z for covariance update (selected only later)
        x_store = []
        f_store = []

        for k in range(half):
            if time.time() >= deadline:
                break
            z = [randn() for _ in range(n)]
            # Mirror
            for sign in (1.0, -1.0):
                if time.time() >= deadline:
                    break
                y = [diag[i] * (sign * z[i]) for i in range(n)]
                x = [m[i] + sigma * y[i] for i in range(n)]
                # clamp and count hits
                for i in range(n):
                    lo, hi = bnds[i]
                    xi = x[i]
                    if xi < lo:
                        x[i] = lo
                        clamp_hits += 1
                    elif xi > hi:
                        x[i] = hi
                        clamp_hits += 1
                f, x = safe_eval(x)
                pop.append((f, x, y))  # store y for updates (approx)
                if f < best:
                    best, best_x = f, x
                evals_since_restart += 1

        if not pop:
            break

        pop.sort(key=lambda t: t[0])

        # Tell: update mean using top mu
        old_m = m[:]
        m = [0.0] * n
        for i in range(mu):
            fi, xi, yi = pop[i]
            wi = w[i]
            for j in range(n):
                m[j] += wi * xi[j]

        # Update evolution path for sigma (use normalized step in diag space)
        # y_mean in "y" coordinates (approx)
        y_w = [0.0] * n
        for i in range(mu):
            wi = w[i]
            yi = pop[i][2]
            for j in range(n):
                y_w[j] += wi * yi[j]

        # p_sigma
        for j in range(n):
            p_sigma[j] = (1.0 - c_sigma) * p_sigma[j] + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * y_w[j]

        # Sigma adaptation
        p_norm = norm(p_sigma)
        sigma *= math.exp((c_sigma / d_sigma) * (p_norm / (chi_n + 1e-12) - 1.0))

        # p_c (for covariance)
        # Using h_sigma-like gate simplified: if p_sigma not too large
        h_sigma = 1.0 if (p_norm / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (evals_since_restart / lam + 1.0))) ) < (1.6 + 0.3 * n) else 0.0
        for j in range(n):
            p_c[j] = (1.0 - c_c) * p_c[j] + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w[j]

        # Diagonal covariance update (diag factors)
        # diag^2 <- (1-c1-cmu)*diag^2 + c1*p_c^2 + cmu*sum(w_i * y_i^2)
        # operate directly on variances then sqrt
        for j in range(n):
            var = diag[j] * diag[j]
            var *= (1.0 - c1 - c_mu)
            var += c1 * (p_c[j] * p_c[j])
            if c_mu > 0.0:
                s = 0.0
                for i in range(mu):
                    yi = pop[i][2]
                    s += w[i] * (yi[j] * yi[j])
                var += c_mu * s
            if var < 1e-30:
                var = 1e-30
            diag[j] = math.sqrt(var)

        # If too much clamping, reduce sigma modestly (helps bounded problems)
        if clamp_hits > (0.20 * lam * n):
            sigma *= 0.7

        # Stagnation handling / occasional diversification via Halton/random restart
        if best < last_best - 1e-15:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= 10:
            no_improve_gens = 0
            # restart mean: best with jitter, or new LDS point
            if time.time() < deadline:
                if best_x is not None and random.random() < 0.7:
                    m = best_x[:]
                    for j in range(n):
                        r = ranges[j]
                        if r > 0:
                            m[j] = clamp(m[j] + random.uniform(-0.08, 0.08) * r, bnds[j][0], bnds[j][1])
                else:
                    m = to_bounds(halton_point(idx, bases))
                    idx += 1
                # reset strategy parameters moderately
                sigma = 0.22 * mean_range
                diag = [1.0] * n
                p_sigma = [0.0] * n
                p_c = [0.0] * n
                evals_since_restart = 0

    return best
