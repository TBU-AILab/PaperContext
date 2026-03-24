import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer.

    Key improvements over the provided code:
      - Uses a robust "ask/tell" core: (mu,lambda)-ES with diagonal covariance + 1/5th success rule
      - Adds *cheap* surrogate-free local improvement via adaptive coordinate search + occasional 2D rotates
      - Better handling of box constraints: reflect (instead of clamp) to reduce boundary sticking
      - Multi-start schedule driven by stagnation + time, with re-seeding around best and random
      - Lower overhead: avoids sorting full populations when possible, uses partial selection

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
    for s in spans:
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
        # Cauchy for rare escapes
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling ----------
    def reflect_1d(x, lo, hi):
        # Reflect into [lo,hi] without loops for most cases; loop only if hugely out.
        if lo == hi:
            return lo
        w = hi - lo
        # bring to [0, 2w) then reflect
        y = (x - lo) % (2.0 * w)
        if y < 0:
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
                yi = y[i]
                if yi < lows[i] or yi > highs[i]:
                    y[i] = reflect_1d(yi, lows[i], highs[i])
        return y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            si = spans[i]
            x[i] = lows[i] + (random.random() * si if si > 0.0 else 0.0)
        return x

    def evaluate(x):
        return float(func(x))

    # ---------- init (stratified + opposition + center) ----------
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

    # center + a couple axis probes
    center = [lows[i] + 0.5 * spans[i] for i in range(dim)]
    try_point(center)
    if dim > 0:
        for i in range(min(dim, 8)):
            if time.time() >= deadline:
                return best
            x = center[:]
            x[i] = lows[i] + 0.25 * spans[i]
            try_point(x)
            x = center[:]
            x[i] = lows[i] + 0.75 * spans[i]
            try_point(x)

    # stratified LHS-ish
    init_n = max(24, min(160, 12 * dim))
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

    # ---------- ES state ----------
    # population size
    lam = max(16, min(80, 5 * dim + 10))
    mu = max(3, lam // 4)

    # recombination weights (log)
    w = [max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)) for k in range(mu)]
    wsum = sum(w) or 1.0
    w = [wk / wsum for wk in w]

    # mean at best
    m = best_x[:]

    # diag step scales (per-dim), and global sigma
    sig_d = [0.0] * dim
    for i in range(dim):
        sig_d[i] = (0.35 * spans[i] if spans[i] > 0.0 else 0.0)
    sigma = 1.0

    # keep scales reasonable
    def clamp_sig():
        nonlocal sigma
        sigma = max(1e-12, min(50.0, sigma))
        for i in range(dim):
            if spans[i] <= 0.0:
                sig_d[i] = 0.0
            else:
                sig_d[i] = max(1e-15 * spans[i], min(0.75 * spans[i], sig_d[i]))

    clamp_sig()

    # success-rule for sigma (1/5th), measured on *best-of-generation*
    succ = 0
    gen_count = 0

    # diagonal variance adaptation using elite steps (simpler + fast)
    # learning rate
    eta = 0.20 / math.sqrt(dim + 1.0)

    # ---------- local search state ----------
    step = [0.12 * spans[i] if spans[i] > 0.0 else 0.0 for i in range(dim)]
    step_floor = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    shrink = 0.70

    # stagnation/restart
    stall = 0
    last_improve = time.time()
    next_restart = t0 + 0.35 * max_time

    def restart():
        nonlocal m, sigma, sig_d, step, stall
        stall = 0
        r = random.random()
        if r < 0.55:
            m = best_x[:]
        elif r < 0.85:
            m = rand_uniform()
        else:
            u = rand_uniform()
            a = 0.7
            m = [a * best_x[i] + (1.0 - a) * u[i] for i in range(dim)]
            m = reflect_vec(m)

        # re-expand exploration moderately
        sigma = min(10.0, sigma * 1.8 + 0.2)
        for i in range(dim):
            if spans[i] > 0:
                sig_d[i] = max(sig_d[i], 0.25 * spans[i])
                step[i] = max(step[i], 0.10 * spans[i])
        clamp_sig()

    # small helper: keep top-mu without sorting all (simple insertion; mu is small)
    def push_elite(elites, item):
        # elites: list of (f,x) sorted ascending, max len mu
        f, x = item
        n = len(elites)
        if n == 0:
            elites.append(item)
            return
        if n == mu and f >= elites[-1][0]:
            return
        # insertion
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

    # ---------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        now = time.time()
        if now >= next_restart or stall > (35 + 7 * dim):
            restart()
            next_restart = now + 0.35 * max_time

        # ===== ES generation (ask) =====
        elites = []  # top mu
        gen_best = float("inf")
        gen_best_x = None

        heavy = (random.random() < 0.08)  # rare heavy-tail burst

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            x = m[:]
            if heavy:
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                        continue
                    # mostly Gaussian, sometimes Cauchy on a few dims
                    z = randc() if random.random() < 0.20 else randn()
                    x[i] += sigma * sig_d[i] * z
            else:
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                    else:
                        x[i] += sigma * sig_d[i] * randn()

            x = reflect_vec(x)
            fx = evaluate(x)

            if fx < gen_best:
                gen_best = fx
                gen_best_x = x[:]

            push_elite(elites, (fx, x))

            if fx < best:
                best = fx
                best_x = x[:]
                last_improve = time.time()
                stall = 0

        # ===== ES update (tell) =====
        old_m = m[:]
        m = [0.0] * dim
        for k in range(len(elites)):
            _, xk = elites[k]
            wk = w[k]
            for i in range(dim):
                m[i] += wk * xk[i]
        m = reflect_vec(m)

        # 1/5th success rule for sigma based on generation best improving the global best recently
        gen_count += 1
        # define "success" as generation best improves previous mean point evaluation proxy:
        # evaluate old_m once per gen (cheap) and compare.
        if time.time() < deadline:
            f_oldm = evaluate(reflect_vec(old_m))
            if gen_best < f_oldm:
                succ += 1

        if gen_count >= 10:
            rate = succ / float(gen_count)
            # target ~0.2
            if rate > 0.22:
                sigma *= 1.25
            elif rate < 0.18:
                sigma *= 0.80
            succ = 0
            gen_count = 0
            clamp_sig()

        # diagonal scale adaptation from elite steps (multiplicative log-normal update)
        # use weighted mean squared normalized steps
        for i in range(dim):
            if spans[i] <= 0.0:
                continue
            s2 = 0.0
            denom = (sigma * sig_d[i] + 1e-30)
            for k in range(len(elites)):
                _, xk = elites[k]
                y = (xk[i] - old_m[i]) / denom
                s2 += w[k] * (y * y)
            # want s2 ~ 1; update log-scale
            # clamp s2 to avoid extreme jumps
            s2 = max(0.2, min(5.0, s2))
            sig_d[i] *= math.exp(0.5 * eta * (math.log(s2)))
        clamp_sig()

        # ===== Local search around best (coordinate + occasional 2D rotate) =====
        if time.time() >= deadline:
            return best

        improved = False
        x = best_x[:]

        # coordinate order
        order = list(range(dim))
        random.shuffle(order)

        probes = min(dim, 14)
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
                    best, best_x = fc, cand[:]
                    x = cand
                    improved = True
                    break

        # occasional 2D rotated probe (helps on correlated valleys)
        if dim >= 2 and time.time() < deadline and random.random() < 0.35:
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
                        best, best_x = fc, cand[:]
                        improved = True

        if improved:
            stall = 0
            # gently pull mean to best
            a = 0.45
            m = reflect_vec([a * best_x[i] + (1.0 - a) * m[i] for i in range(dim)])
            # expand local steps slightly
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = min(0.30 * spans[i], step[i] * 1.07)
        else:
            stall += 1
            for i in range(dim):
                if spans[i] > 0.0 and step[i] > step_floor[i]:
                    step[i] = max(step_floor[i], step[i] * shrink)

            # if no improvements for a bit, encourage exploration
            if time.time() - last_improve > max(0.12 * max_time, 0.4):
                sigma = min(20.0, sigma * 1.10)
                clamp_sig()
