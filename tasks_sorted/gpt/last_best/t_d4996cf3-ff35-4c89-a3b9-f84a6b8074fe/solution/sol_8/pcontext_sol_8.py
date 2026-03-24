import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libraries).

    Key upgrades vs previous code:
      - Uses a compact CMA-ES-like optimizer (separable/diagonal covariance) as the main engine.
        This is typically stronger than DE on many continuous problems under tight budgets.
      - Interleaves occasional trust-region coordinate pattern search around the incumbent best.
      - Uses quasi-random (Halton) + opposition initialization for better early coverage.
      - Time-aware: adapts population size and local-search budget based on estimated eval cost.
      - Robust bound handling (fold+reflect).

    Returns: best fitness (float) found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    var_idx = [i for i in range(dim) if spans[i] > 0.0]

    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # ---------- utilities ----------
    def now():
        return time.time()

    def reflect_inplace(x):
        # fold into [lo, lo+2w), then reflect into [lo,hi]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            w = hi - lo
            xi = x[i]
            if xi < lo or xi > hi:
                xi = lo + (xi - lo) % (2.0 * w)
                if xi > hi:
                    xi = hi - (xi - hi)
                x[i] = xi
        return x

    eval_count = 0
    eval_time = 0.0

    def eval_f(x):
        nonlocal eval_count, eval_time
        t = time.time()
        fx = float(func(x))
        eval_time += (time.time() - t)
        eval_count += 1
        return fx

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] > 0.0:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Halton init for coverage
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    # ---------- local pattern search (cheap intensification) ----------
    def local_pattern_search(x_best, f_best, budget, init_frac):
        if budget <= 0 or now() >= deadline:
            return f_best, x_best

        x = x_best[:]
        fx = f_best
        used = 0

        # coordinate steps
        steps = [0.0] * dim
        for j in var_idx:
            steps[j] = max(1e-14, init_frac * spans[j])

        # prioritize larger spans first
        dims = var_idx[:]
        dims.sort(key=lambda j: spans[j], reverse=True)

        no_improve_rounds = 0
        while used < budget and now() < deadline:
            improved = False
            best_cand = None
            best_fc = fx

            for j in dims:
                if used >= budget or now() >= deadline:
                    break
                s = steps[j]
                if s <= 0.0:
                    continue

                xj = x[j]
                # try +/- s
                for sgn in (1.0, -1.0):
                    y = x[:]
                    y[j] = xj + sgn * s
                    reflect_inplace(y)
                    fy = eval_f(y)
                    used += 1
                    if fy < best_fc:
                        best_fc = fy
                        best_cand = y
                    if used >= budget or now() >= deadline:
                        break

            if best_cand is not None and best_fc < fx:
                x, fx = best_cand, best_fc
                improved = True
                for j in var_idx:
                    steps[j] *= 1.15
                no_improve_rounds = 0
            else:
                for j in var_idx:
                    steps[j] *= 0.55
                no_improve_rounds += 1
                if no_improve_rounds >= 2:
                    break

            # stop if all steps tiny
            tiny = True
            for j in var_idx:
                if steps[j] > 1e-12 * spans[j]:
                    tiny = False
                    break
            if tiny:
                break

        return fx, x

    # ---------- probe for time awareness ----------
    x_probe = rand_point()
    reflect_inplace(x_probe)
    f_probe = eval_f(x_probe)
    x_best = x_probe[:]
    f_best = f_probe

    def evals_per_sec():
        spent = max(1e-9, eval_time)
        return eval_count / spent

    eps = evals_per_sec()

    # ---------- initialization pool ----------
    # size tuned to time & dim
    init_n = 8 + 3 * len(var_idx)
    if max_time <= 1.0:
        init_n = min(init_n, 20)
    if eps < 200:
        init_n = max(10, init_n // 2)

    mid = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    for i in range(init_n):
        if now() >= deadline:
            return f_best
        r = i % 5
        if r == 0:
            x = halton_point()
        elif r == 1:
            x = rand_point()
        elif r == 2:
            x = mid[:]
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.25 * spans[j])
        elif r == 3:
            x = halton_point()
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.18 * spans[j])
        else:
            x = rand_point()
            xo = opposite_point(x)
            reflect_inplace(x)
            reflect_inplace(xo)
            fx = eval_f(x)
            if now() >= deadline:
                return min(f_best, fx)
            fo = eval_f(xo)
            if fo < fx:
                x, fx = xo, fo
            if fx < f_best:
                f_best, x_best = fx, x[:]
            continue

        reflect_inplace(x)
        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # ---------- Separable CMA-ES-like loop ----------
    # We sample in normalized coordinates y in R^n, map to x = m + sigma * D * z
    # where D is per-dimension scale. We adapt m, sigma, and diagonal D.
    n = len(var_idx)

    # initial mean from best found so far
    m = [x_best[j] for j in var_idx]

    # diagonal scales (std per dimension in x-space)
    D = [0.3 * spans[j] for j in var_idx]
    for k in range(n):
        D[k] = max(D[k], 1e-12)

    # global step-size multiplier
    sigma = 0.6  # fairly exploratory, will adapt
    sigma_min = 1e-12
    sigma_max = 2.0

    # population size: CMA-ES-ish, with time scaling
    lam = 4 + int(3 * math.log(n + 1.0))
    # if evaluations are cheap, increase lambda
    if eps > 1500:
        lam = min(60, lam + 10 + 2 * n)
    elif eps > 600:
        lam = min(50, lam + 6 + n)
    # if expensive, keep small
    if eps < 200:
        lam = max(6, min(lam, 10 + n))
    lam = int(clamp(lam, 6, 80))

    mu = lam // 2

    # recombination weights (log)
    weights = [0.0] * mu
    for i in range(mu):
        weights[i] = math.log(mu + 0.5) - math.log(i + 1.0)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # learning rates (diagonal)
    c_sigma = (mueff + 2.0) / (n + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))

    # evolution paths (in normalized coords)
    p_sigma = [0.0] * n
    p_c = [0.0] * n

    # expected norm of N(0, I)
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if n > 0 else 1.0

    gen = 0
    last_improve_gen = 0

    def sample_candidate():
        # returns (x_full, z_vec, x_sub) where:
        #   z ~ N(0,I) in n dims
        #   x_sub are values for var_idx
        z = [random.gauss(0.0, 1.0) for _ in range(n)]
        x_sub = [m[i] + sigma * D[i] * z[i] for i in range(n)]
        x_full = [lows[i] for i in range(dim)]
        for i in range(dim):
            if spans[i] > 0.0:
                x_full[i] = x_best[i]  # will overwrite var dims below
            else:
                x_full[i] = lows[i]
        for k, j in enumerate(var_idx):
            x_full[j] = x_sub[k]
        reflect_inplace(x_full)
        # ensure x_sub matches reflected (for proper adaptation, re-read)
        for k, j in enumerate(var_idx):
            x_sub[k] = x_full[j]
        return x_full, z, x_sub

    # Main loop
    while now() < deadline:
        gen += 1

        # periodic local search (more near the end or after stagnation)
        tleft = max(0.0, deadline - now())
        frac_left = tleft / max(1e-9, float(max_time))
        if (gen % 10 == 0) or (gen - last_improve_gen >= 12) or (frac_left < 0.30 and gen % 4 == 0):
            eps = evals_per_sec()
            base = 6 * n + 10
            if eps < 200:
                base = 3 * n + 8
            if eps < 80:
                base = 2 * n + 6
            fb2, xb2 = local_pattern_search(x_best, f_best, budget=base, init_frac=0.10 if frac_left > 0.4 else 0.06)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                # pull mean slightly toward improved best
                for k, j in enumerate(var_idx):
                    m[k] = 0.7 * m[k] + 0.3 * x_best[j]
                last_improve_gen = gen

        # sample and evaluate lambda candidates
        pop = []
        for _ in range(lam):
            if now() >= deadline:
                return f_best
            x_full, z, x_sub = sample_candidate()
            fx = eval_f(x_full)
            pop.append((fx, z, x_sub, x_full))
            if fx < f_best:
                f_best = fx
                x_best = x_full[:]
                last_improve_gen = gen

        pop.sort(key=lambda t: t[0])
        best_fx = pop[0][0]

        # recombination: new mean in x-space
        old_m = m[:]
        m = [0.0] * n
        for i in range(mu):
            w = weights[i]
            x_sub = pop[i][2]
            for k in range(n):
                m[k] += w * x_sub[k]

        # compute y = (m - old_m) / (sigma * D) in normalized coords (approx)
        y = [0.0] * n
        for k in range(n):
            denom = max(1e-30, sigma * D[k])
            y[k] = (m[k] - old_m[k]) / denom

        # update p_sigma
        cs = c_sigma
        for k in range(n):
            p_sigma[k] = (1.0 - cs) * p_sigma[k] + math.sqrt(cs * (2.0 - cs) * mueff) * y[k]

        # step-size adaptation
        norm_ps = math.sqrt(sum(v * v for v in p_sigma))
        sigma *= math.exp((cs / d_sigma) * (norm_ps / chi_n - 1.0))
        sigma = clamp(sigma, sigma_min, sigma_max)

        # update p_c
        cc = c_c
        hsig = 1.0 if norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
        for k in range(n):
            p_c[k] = (1.0 - cc) * p_c[k] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y[k]

        # covariance (diagonal) update: D^2 acts like diag(C)
        # Let Ck = Dk^2. Update with rank-1 and rank-mu using z vectors.
        # We approximate using selected individuals' z.
        # C <- (1-c1-cmu)*C + c1*(pc^2) + cmu*sum(wi*(sigma*D*zi)^2)/(sigma^2) -> effectively D^2 * zi^2
        # Work directly on Ck = Dk^2
        C = [D[k] * D[k] for k in range(n)]

        # rank-mu term
        rank_mu = [0.0] * n
        for i in range(mu):
            w = weights[i]
            z = pop[i][1]
            for k in range(n):
                rank_mu[k] += w * (z[k] * z[k])

        # apply update
        for k in range(n):
            Ck = C[k]
            Ck = (1.0 - c1 - c_mu) * Ck + c1 * (p_c[k] * p_c[k]) + c_mu * Ck * rank_mu[k]
            # numerical safety
            if Ck < 1e-30:
                Ck = 1e-30
            C[k] = Ck
            D[k] = math.sqrt(Ck)

        # restart / diversify if stuck (increase sigma, jitter mean)
        if gen - last_improve_gen >= 25:
            last_improve_gen = gen  # avoid repeated triggers
            sigma = min(sigma_max, sigma * 1.6)
            for k, j in enumerate(var_idx):
                m[k] = clamp(m[k] + random.gauss(0.0, 0.10 * spans[j]), lows[j], highs[j])
            # also expand some D
            for k, j in enumerate(var_idx):
                D[k] = max(D[k], 0.08 * spans[j])

        # also keep mean within bounds (important if reflection caused clipping)
        for k, j in enumerate(var_idx):
            m[k] = clamp(m[k], lows[j], highs[j])

        # mild exploitation if generation's best beats global best by a lot: pull mean a bit
        if best_fx < f_best + 1e-15:
            # already reflected in f_best; keep stable
            pass

    return f_best
