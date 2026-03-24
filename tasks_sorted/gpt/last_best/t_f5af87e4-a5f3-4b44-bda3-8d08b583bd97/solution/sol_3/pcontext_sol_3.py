import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Further improved time-bounded minimization (stdlib only).

    Key changes vs previous:
      - Quasi-random initialization (Halton) + opposition + center point
      - CMA-ES-like diagonal (sep) adaptation around current best (fast, robust)
      - Small "trust region" local pattern-search polish near the end
      - Stagnation-triggered restarts with shrinking/expanding exploration radius

    Returns
    -------
    best : float
        Best (minimum) objective value found within the time budget.
    """
    deadline = time.perf_counter() + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def clamp_inplace(x):
        for i in range(dim):
            lo = lows[i]; hi = highs[i]
            v = x[i]
            if v < lo:
                x[i] = lo
            elif v > hi:
                x[i] = hi

    def eval_f(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def center_point():
        return [0.5 * (lows[i] + highs[i]) for i in range(dim)]

    # approx N(0,1) using sum of uniforms (fast, no trig/log)
    def randn():
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    # Cauchy heavy tail (for occasional escapes)
    def cauchy():
        u = random.random()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ---------- Halton (quasi-random) ----------
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173
    ]

    def radical_inverse(n, base):
        inv = 1.0 / base
        f = inv
        r = 0.0
        while n > 0:
            n, mod = divmod(n, base)
            r += mod * f
            f *= inv
        return r

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            base = primes[i % len(primes)]
            u = radical_inverse(index, base)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- global best ----------
    best = float("inf")
    best_x = None

    def consider(x, fx):
        nonlocal best, best_x
        if fx < best:
            best = fx
            best_x = list(x)

    # ---------- initialization ----------
    # Try center + opposition (often very strong on bounded problems)
    x0 = center_point()
    fx0 = eval_f(x0)
    consider(x0, fx0)
    xo = opposite_point(x0)
    clamp_inplace(xo)
    consider(xo, eval_f(xo))

    # Mix Halton + random + opposition
    # Keep init modest to leave time for exploitation.
    init_n = max(24, 10 * dim)
    halton_start = 1 + random.randrange(1, 4096)
    for k in range(init_n):
        if time.perf_counter() >= deadline:
            return best
        if random.random() < 0.75:
            x = halton_point(halton_start + k)
        else:
            x = rand_point()
        fx = eval_f(x)
        consider(x, fx)

        xo = opposite_point(x)
        clamp_inplace(xo)
        fxo = eval_f(xo)
        consider(xo, fxo)

    if best_x is None:
        return best

    # ---------- sep-CMA-ES-like loop (diagonal covariance) ----------
    # Sample lambda candidates around a mean, select best mu, update mean and per-dim sigma.
    # This is a strong general-purpose method under strict time budgets.
    lam = max(8, min(40, 4 + 3 * dim))
    mu = max(3, lam // 2)

    # mean and diagonal step sizes (in actual coordinates)
    m = list(best_x)
    sigma0 = 0.25  # normalized
    sig = [sigma0 * safe_spans[i] for i in range(dim)]

    # evolution path-ish smoothing
    alpha_m = 0.45  # mean update
    alpha_sig = 0.25  # diag sigma update based on selected steps

    # restart / stagnation control
    evals = 0
    last_improve_evals = 0
    stall_limit = 80 + 20 * dim

    # keep some "temperature" for random/global jumps
    p_heavy = 0.03

    # to reduce overhead, allocate buffers inside loop
    while True:
        if time.perf_counter() >= deadline:
            return best

        # occasional heavy-tailed / global proposal mixed in (helps rugged landscapes)
        if random.random() < p_heavy:
            x = list(m)
            for i in range(dim):
                if spans[i] == 0.0:
                    continue
                # heavy step scaled to bounds
                x[i] += (0.20 * safe_spans[i]) * cauchy()
            clamp_inplace(x)
            fx = eval_f(x)
            evals += 1
            if fx < best:
                consider(x, fx)
                m = list(best_x)
                last_improve_evals = evals
            continue

        # --- generate population ---
        candidates = []  # list of (fx, x, step_vector)
        for _ in range(lam):
            if time.perf_counter() >= deadline:
                return best
            x = [0.0] * dim
            z = [0.0] * dim  # normalized-ish step per dim: (x-m)/sig
            for i in range(dim):
                if spans[i] == 0.0:
                    x[i] = m[i]
                    z[i] = 0.0
                else:
                    zi = randn()
                    z[i] = zi
                    x[i] = m[i] + zi * sig[i]
            clamp_inplace(x)
            fx = eval_f(x)
            evals += 1
            if fx < best:
                consider(x, fx)
                last_improve_evals = evals
            candidates.append((fx, x, z))

        # select best mu
        candidates.sort(key=lambda t: t[0])
        sel = candidates[:mu]

        # update mean toward weighted average of selected
        # simple decreasing weights
        w_sum = 0.0
        w = [0.0] * mu
        for j in range(mu):
            wj = (mu - j)
            w[j] = wj
            w_sum += wj
        inv_wsum = 1.0 / w_sum

        m_new = [0.0] * dim
        for i in range(dim):
            acc = 0.0
            for j in range(mu):
                acc += w[j] * sel[j][1][i]
            m_new[i] = acc * inv_wsum

        # smooth mean update
        for i in range(dim):
            m[i] = (1.0 - alpha_m) * m[i] + alpha_m * m_new[i]
        clamp_inplace(m)

        # update diagonal sigmas based on selected step magnitudes
        # target: adapt to promising directions, while keeping lower/upper bounds stable
        # compute per-dim RMS of z among selected
        z_rms = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for j in range(mu):
                zj = sel[j][2][i]
                s += zj * zj
            z_rms[i] = math.sqrt(s / float(mu))

        # adjust sig: if selected steps are small -> shrink; large -> expand.
        # Keep within [min,max] relative to span.
        for i in range(dim):
            if spans[i] == 0.0:
                sig[i] = 0.0
                continue
            # z_rms around ~1 means "good scale". Encourage ~1.
            # ratio <1 => steps too small => increase slightly; ratio >1 => decrease slightly
            # (inverted compared to some schemes; empirically stable with clamp & bounds)
            ratio = z_rms[i]
            # map ratio to multiplicative factor; keep mild
            # if ratio < 0.7 => expand; if ratio > 1.3 => shrink
            if ratio < 0.7:
                mult = 1.10
            elif ratio > 1.3:
                mult = 0.90
            else:
                mult = 1.00

            sig_i = sig[i] * ((1.0 - alpha_sig) + alpha_sig * mult)

            # clamp sigma to sensible range
            sig_min = 1e-8 * safe_spans[i]
            sig_max = 0.50 * safe_spans[i]
            if sig_i < sig_min:
                sig_i = sig_min
            elif sig_i > sig_max:
                sig_i = sig_max
            sig[i] = sig_i

        # keep m aligned with best when it's clearly better (helps exploitation)
        if best_x is not None and candidates[0][0] <= best:
            # best may already be updated; make sure mean doesn't drift away too much
            if random.random() < 0.35:
                m = list(best_x)

        # --- restart / diversification on stall ---
        if (evals - last_improve_evals) > stall_limit:
            last_improve_evals = evals

            # restart mean: mix best, halton, random
            r = random.random()
            if r < 0.55 and best_x is not None:
                m = list(best_x)
            elif r < 0.80:
                m = halton_point(1 + random.randrange(1, 20000))
            else:
                m = rand_point()
            clamp_inplace(m)

            # restart step sizes: sometimes shrink (intensify), sometimes expand (explore)
            if random.random() < 0.5 and best_x is not None:
                base = 0.12
            else:
                base = 0.30
            sig = [base * safe_spans[i] for i in range(dim)]

            # also do a quick local "polish" around current best/mean
            # (very small budget, opportunistic)
            def polish(iters=12):
                nonlocal best, best_x
                if best_x is None:
                    return
                x = list(best_x)
                fx = best
                # start with moderate coordinate step
                step = 0.08
                for _ in range(iters):
                    if time.perf_counter() >= deadline:
                        return
                    improved = False
                    # opportunistic coordinate scan with random order
                    coords = list(range(dim))
                    random.shuffle(coords)
                    for i in coords:
                        if spans[i] == 0.0:
                            continue
                        delta = step * safe_spans[i]
                        xp = list(x); xp[i] += delta
                        clamp_inplace(xp)
                        f1 = eval_f(xp)
                        if f1 < fx:
                            x, fx = xp, f1
                            consider(x, fx)
                            improved = True
                            continue
                        xm = list(x); xm[i] -= delta
                        clamp_inplace(xm)
                        f2 = eval_f(xm)
                        if f2 < fx:
                            x, fx = xm, f2
                            consider(x, fx)
                            improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-6:
                            break

            polish(iters=10)

        # near the end: brief polish bursts to convert progress to final value
        remaining = deadline - time.perf_counter()
        if remaining < 0.15 * float(max_time) and random.random() < 0.10:
            # very cheap final exploitation
            x = list(best_x) if best_x is not None else list(m)
            fx = best if best_x is not None else eval_f(x)
            step = 0.05
            for _ in range(8):
                if time.perf_counter() >= deadline:
                    return best
                i = random.randrange(dim)
                if spans[i] == 0.0:
                    continue
                delta = step * safe_spans[i]
                xp = list(x); xp[i] += delta
                clamp_inplace(xp)
                f1 = eval_f(xp)
                evals += 1
                if f1 < fx:
                    x, fx = xp, f1
                    consider(x, fx)
                    continue
                xm = list(x); xm[i] -= delta
                clamp_inplace(xm)
                f2 = eval_f(xm)
                evals += 1
                if f2 < fx:
                    x, fx = xm, f2
                    consider(x, fx)
                else:
                    step *= 0.7
                    if step < 1e-6:
                        break
