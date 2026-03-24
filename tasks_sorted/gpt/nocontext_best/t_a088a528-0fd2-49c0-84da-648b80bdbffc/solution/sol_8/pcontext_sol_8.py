import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Improvements vs provided algorithm:
      - Uses CMA-ES style evolution path + diagonal covariance (fast, no matrices)
      - Proper mirrored (reflection) box handling
      - Two-phase loop: global search (ES) + deterministic local pattern search
      - Asynchronous "best tracking" and periodic restarts with adaptive radius
      - Low overhead: keeps only mu elites via insertion, avoids full sorts

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i, s in enumerate(spans):
        if s < 0:
            raise ValueError("Each bound must satisfy low <= high")

    # ---------- RNG helpers ----------
    # Box-Muller normal with cache
    _have_spare = False
    _spare = 0.0
    def randn():
        nonlocal _have_spare, _spare
        if _have_spare:
            _have_spare = False
            return _spare
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        _spare = r * math.sin(th)
        _have_spare = True
        return z0

    def randc():
        # Cauchy for occasional long jumps
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling ----------
    def reflect_1d(x, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (x - lo) % (2.0 * w)
        if y < 0.0:
            y += 2.0 * w
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def reflect_vec(x):
        y = x[:]  # copy
        for i in range(dim):
            if spans[i] <= 0.0:
                y[i] = lows[i]
            else:
                xi = y[i]
                if xi < lows[i] or xi > highs[i]:
                    y[i] = reflect_1d(xi, lows[i], highs[i])
        return y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            si = spans[i]
            x[i] = lows[i] + (random.random() * si if si > 0.0 else 0.0)
        return x

    def evaluate(x):
        return float(func(x))

    # ---------- keep top-mu without sorting all ----------
    def push_elite(elites, item, mu):
        # elites sorted ascending by fitness, max len mu
        f, x = item
        n = len(elites)
        if n == 0:
            elites.append(item)
            return
        if n == mu and f >= elites[-1][0]:
            return
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if f < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > mu:
            elites.pop()

    # ---------- initialization: LHS-ish + opposition + center probes ----------
    best = float("inf")
    best_x = None

    def try_point(x):
        nonlocal best, best_x
        if time.time() >= deadline:
            return
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]

    center = [lows[i] + 0.5 * spans[i] for i in range(dim)]
    try_point(center)
    for i in range(min(dim, 10)):
        if time.time() >= deadline:
            return best
        x1 = center[:]
        x2 = center[:]
        x1[i] = lows[i] + 0.25 * spans[i]
        x2[i] = lows[i] + 0.75 * spans[i]
        try_point(x1)
        try_point(x2)

    init_n = max(32, min(220, 14 * dim))
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            si = spans[i]
            if si <= 0.0:
                x[i] = lows[i]
            else:
                u = (perms[i][j] + random.random()) / init_n
                x[i] = lows[i] + u * si
        try_point(x)
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        try_point(reflect_vec(xo))

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------- ES parameters (diag-CMA flavor) ----------
    # population sizes
    lam = max(18, min(96, 6 * dim + 12))
    mu = max(3, lam // 3)

    # log weights
    w = [max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)) for k in range(mu)]
    wsum = sum(w) or 1.0
    w = [wk / wsum for wk in w]
    mueff = 1.0 / sum(wk * wk for wk in w)

    # learning rates (diag)
    c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

    # state
    m = best_x[:]
    sigma = 0.25  # global scalar (relative; combined with sig_d)
    sig_d = [(0.35 * spans[i] if spans[i] > 0.0 else 0.0) for i in range(dim)]  # per-dim scales

    # evolution paths
    p_sigma = [0.0] * dim
    p_c = [0.0] * dim

    # expected norm of N(0,I)
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    def clamp_scales():
        nonlocal sigma
        sigma = max(1e-15, min(50.0, sigma))
        for i in range(dim):
            if spans[i] <= 0.0:
                sig_d[i] = 0.0
            else:
                sig_d[i] = max(1e-15 * spans[i], min(0.9 * spans[i], sig_d[i]))

    clamp_scales()

    # ---------- local search (pattern/coordinate) ----------
    step = [0.10 * spans[i] if spans[i] > 0.0 else 0.0 for i in range(dim)]
    step_floor = [1e-12 * (spans[i] if spans[i] > 0.0 else 1.0) for i in range(dim)]

    # restart / stagnation
    stall = 0
    last_improve = time.time()
    next_restart = t0 + 0.30 * max_time

    def restart():
        nonlocal m, sigma, sig_d, step, stall, p_sigma, p_c
        stall = 0
        p_sigma = [0.0] * dim
        p_c = [0.0] * dim

        r = random.random()
        if r < 0.55:
            m = best_x[:]
        elif r < 0.85:
            m = rand_uniform()
        else:
            u = rand_uniform()
            a = 0.75
            m = reflect_vec([a * best_x[i] + (1.0 - a) * u[i] for i in range(dim)])

        # re-expand
        sigma = min(12.0, sigma * 2.0 + 0.05)
        for i in range(dim):
            if spans[i] > 0.0:
                sig_d[i] = max(sig_d[i], 0.30 * spans[i])
                step[i] = max(step[i], 0.12 * spans[i])
        clamp_scales()

    # ---------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        now = time.time()
        if now >= next_restart or stall > (45 + 8 * dim):
            restart()
            next_restart = now + 0.30 * max_time

        # ===== sample population =====
        elites = []
        pop_steps = []  # store y = (x-m)/(sigma*sig_d) for covariance update of elites only
        gen_best = float("inf")
        gen_best_x = None

        heavy = (random.random() < 0.10)
        old_m = m[:]

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # sample y ~ N(0, I) (or with occasional heavy tail)
            y = [0.0] * dim
            if heavy:
                for i in range(dim):
                    y[i] = randc() if random.random() < 0.18 else randn()
            else:
                for i in range(dim):
                    y[i] = randn()

            x = [0.0] * dim
            for i in range(dim):
                if spans[i] <= 0.0:
                    x[i] = lows[i]
                else:
                    x[i] = old_m[i] + (sigma * sig_d[i]) * y[i]

            x = reflect_vec(x)
            fx = evaluate(x)

            if fx < gen_best:
                gen_best = fx
                gen_best_x = x[:]

            push_elite(elites, (fx, x, y), mu)

            if fx < best:
                best = fx
                best_x = x[:]
                last_improve = time.time()
                stall = 0

        # ===== recombination =====
        m = [0.0] * dim
        y_w = [0.0] * dim
        for k in range(len(elites)):
            fk, xk, yk = elites[k]
            wk = w[k]
            for i in range(dim):
                m[i] += wk * xk[i]
                y_w[i] += wk * yk[i]
        m = reflect_vec(m)

        # ===== step-size control (CSA) =====
        # p_sigma <- (1-c)*p_sigma + sqrt(c(2-c)mueff)*y_w
        coeff_ps = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
        for i in range(dim):
            p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + coeff_ps * y_w[i]

        norm_ps = math.sqrt(sum(v * v for v in p_sigma))
        sigma *= math.exp((c_sigma / d_sigma) * (norm_ps / (chi_n + 1e-30) - 1.0))
        clamp_scales()

        # ===== covariance / diagonal scales update =====
        # Heaviside for conjugate evolution path
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * lam)) < (1.4 + 2.0 / (dim + 1.0)) * chi_n) else 0.0

        coeff_pc = math.sqrt(c_c * (2.0 - c_c) * mueff)
        for i in range(dim):
            p_c[i] = (1.0 - c_c) * p_c[i] + hsig * coeff_pc * y_w[i]

        # update diagonal "variances" in sig_d via multiplicative update on squared scales
        # interpret sig_d as std per-dim; update std^2 then take sqrt
        for i in range(dim):
            if spans[i] <= 0.0:
                continue
            # rank-one + rank-mu on diagonal
            rank_one = p_c[i] * p_c[i]
            rank_mu = 0.0
            for k in range(len(elites)):
                _, _, yk = elites[k]
                rank_mu += w[k] * (yk[i] * yk[i])

            # exponential moving update of variance proxy
            # v <- (1-c1-cmu)*v + c1*rankone + cmu*rankmu ; but we store std directly in sig_d
            v = (sig_d[i] / (spans[i] + 1e-300)) ** 2  # normalized variance proxy
            v = (1.0 - c1 - c_mu) * v + c1 * rank_one + c_mu * rank_mu
            v = max(1e-30, min(1e6, v))
            sig_d[i] = math.sqrt(v) * spans[i]

        clamp_scales()

        # ===== local improvement around current best (pattern search) =====
        if time.time() >= deadline:
            return best

        improved = False
        x = best_x[:]

        # deterministic-ish coordinate pattern
        order = list(range(dim))
        random.shuffle(order)
        probes = min(dim, 18)

        for t in range(probes):
            if time.time() >= deadline:
                return best
            i = order[t]
            if spans[i] <= 0.0 or step[i] <= step_floor[i]:
                continue
            base = x[i]
            for sgn in (1.0, -1.0):
                cand = x[:]
                cand[i] = base + sgn * step[i]
                cand = reflect_vec(cand)
                fc = evaluate(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    x = cand
                    improved = True
                    break

        # occasional 2D correlated probe
        if dim >= 2 and time.time() < deadline and random.random() < 0.45:
            i = random.randrange(dim)
            j = random.randrange(dim - 1)
            if j >= i:
                j += 1
            if spans[i] > 0.0 and spans[j] > 0.0:
                si = step[i]
                sj = step[j]
                if si > step_floor[i] and sj > step_floor[j]:
                    ang = (random.random() * 2.0 - 1.0) * math.pi
                    dx = math.cos(ang) * si
                    dy = math.sin(ang) * sj
                    cand = x[:]
                    cand[i] += dx
                    cand[j] += dy
                    cand = reflect_vec(cand)
                    fc = evaluate(cand)
                    if fc < best:
                        best = fc
                        best_x = cand[:]
                        improved = True

        if improved:
            stall = 0
            # pull mean toward best modestly
            a = 0.55
            m = reflect_vec([a * best_x[i] + (1.0 - a) * m[i] for i in range(dim)])
            # enlarge local steps slightly; also tighten ES around improved region
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = min(0.35 * spans[i], step[i] * 1.10)
            sigma = max(1e-15, sigma * 0.92)
        else:
            stall += 1
            # shrink local steps and (if stagnating) increase exploration
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = max(step_floor[i], step[i] * 0.72)
            if time.time() - last_improve > max(0.10 * max_time, 0.35):
                sigma = min(20.0, sigma * 1.12)
                clamp_scales()
