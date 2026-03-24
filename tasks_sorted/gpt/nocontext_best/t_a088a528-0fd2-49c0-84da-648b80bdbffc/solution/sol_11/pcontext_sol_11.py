import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Main upgrades vs the provided algorithm:
      - Uses a proper (mu,lambda)-ES with mirrored sampling (antithetic) to reduce noise/variance.
      - Success-based global step control with a sliding window (more stable than coarse batches).
      - Diagonal scale adaptation from elite steps + mild damping.
      - Stronger, cheaper local search: randomized pattern search + occasional pairwise (2D) steps.
      - Smarter restarts: triggered by stagnation AND shrinking steps; mixes random + best-centered.
      - Still fully self-contained (stdlib only), box constraints handled by reflection.

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

    # ---------------- RNG: fast normal + cauchy ----------------
    _have = False
    _spare = 0.0

    def randn():
        nonlocal _have, _spare
        if _have:
            _have = False
            return _spare
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        _spare = r * math.sin(th)
        _have = True
        return z0

    def randc():
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------------- bounds: reflection ----------------
    def reflect_1d(x, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (x - lo) % (2.0 * w)
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

    # ---------------- init: center + LHS-ish + opposition ----------------
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

    # quick axis probes
    for i in range(min(dim, 10)):
        if time.time() >= deadline:
            return best
        if spans[i] <= 0.0:
            continue
        x1 = center[:]
        x2 = center[:]
        x1[i] = lows[i] + 0.25 * spans[i]
        x2[i] = lows[i] + 0.75 * spans[i]
        try_point(x1)
        try_point(x2)

    init_n = max(30, min(220, 14 * dim))
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
        # opposition point (then reflect)
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        try_point(reflect_vec(xo))

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------------- ES parameters ----------------
    # population; keep moderate to reduce overhead
    lam = max(18, min(90, 6 * dim + 12))
    if lam % 2 == 1:
        lam += 1  # for mirrored sampling pairs
    mu = max(4, lam // 4)

    # log weights
    w = [max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)) for k in range(mu)]
    wsum = sum(w) or 1.0
    w = [wk / wsum for wk in w]

    m = best_x[:]

    # per-dim step scales and global sigma
    sig_d = [0.0] * dim
    for i in range(dim):
        sig_d[i] = (0.30 * spans[i] if spans[i] > 0.0 else 0.0)
    sigma = 1.0

    # local search step sizes
    step = [0.10 * spans[i] if spans[i] > 0.0 else 0.0 for i in range(dim)]
    step_floor = [max(1e-12, 1e-14 * (spans[i] if spans[i] > 0 else 1.0)) for i in range(dim)]

    def clamp_scales():
        nonlocal sigma
        sigma = max(1e-14, min(50.0, sigma))
        for i in range(dim):
            if spans[i] <= 0.0:
                sig_d[i] = 0.0
                step[i] = 0.0
            else:
                sig_d[i] = max(1e-15 * spans[i], min(0.80 * spans[i], sig_d[i]))
                step[i] = max(step_floor[i], min(0.35 * spans[i], step[i]))

    clamp_scales()

    # keep elites without sorting whole population
    def push_elite(elites, item, cap):
        f, x = item
        n = len(elites)
        if n == 0:
            elites.append(item)
            return
        if n == cap and f >= elites[-1][0]:
            return
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if f < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > cap:
            elites.pop()

    # success control: sliding window on "best-of-gen improves old mean"
    succ_win = [0] * 18
    succ_idx = 0
    succ_sum = 0

    # diagonal adaptation learning rate
    eta = 0.18 / math.sqrt(dim + 1.0)

    # restart/stagnation
    stall = 0
    last_improve = time.time()

    def restart():
        nonlocal m, sigma, sig_d, step, stall, last_improve
        stall = 0
        # three modes: best-centered, random, blend
        r = random.random()
        if r < 0.55:
            m = best_x[:]
        elif r < 0.85:
            m = rand_uniform()
        else:
            u = rand_uniform()
            a = 0.75
            m = reflect_vec([a * best_x[i] + (1.0 - a) * u[i] for i in range(dim)])

        # expand exploration; also re-inflate local steps
        sigma = min(20.0, sigma * 1.9 + 0.25)
        for i in range(dim):
            if spans[i] > 0.0:
                sig_d[i] = max(sig_d[i], 0.22 * spans[i])
                step[i] = max(step[i], 0.09 * spans[i])
        clamp_scales()
        last_improve = time.time()

    # ---------------- main loop ----------------
    while True:
        if time.time() >= deadline:
            return best

        # restart triggers: long stall, or steps got tiny without progress
        tiny_steps = True
        for i in range(dim):
            if spans[i] > 0.0 and step[i] > 8.0 * step_floor[i]:
                tiny_steps = False
                break

        if stall > (28 + 6 * dim) or (tiny_steps and (time.time() - last_improve) > max(0.10 * max_time, 0.4)):
            restart()

        # ----- ES generation (mirrored sampling) -----
        elites = []
        gen_best = float("inf")
        gen_best_x = None

        old_m = m[:]
        # evaluate old mean once (proxy for success)
        f_oldm = evaluate(reflect_vec(old_m)) if time.time() < deadline else float("inf")

        heavy = (random.random() < 0.06)  # rare heavy-tail mode

        # sample in pairs: z and -z (antithetic)
        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [0.0] * dim
            if heavy:
                # mostly gaussian, some cauchy components
                for i in range(dim):
                    if spans[i] <= 0.0:
                        z[i] = 0.0
                    else:
                        z[i] = (randc() if random.random() < 0.15 else randn())
            else:
                for i in range(dim):
                    z[i] = randn() if spans[i] > 0.0 else 0.0

            for sign in (1.0, -1.0):
                x = old_m[:]
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                    else:
                        x[i] += sign * (sigma * sig_d[i] * z[i])
                x = reflect_vec(x)
                fx = evaluate(x)

                if fx < gen_best:
                    gen_best, gen_best_x = fx, x[:]
                push_elite(elites, (fx, x), mu)

                if fx < best:
                    best, best_x = fx, x[:]
                    stall = 0
                    last_improve = time.time()

        # ----- ES recombination -----
        m = [0.0] * dim
        for k in range(len(elites)):
            _, xk = elites[k]
            wk = w[k]
            for i in range(dim):
                m[i] += wk * xk[i]
        m = reflect_vec(m)

        # ----- success-based sigma control (sliding window) -----
        success = 1 if gen_best < f_oldm else 0
        succ_sum -= succ_win[succ_idx]
        succ_win[succ_idx] = success
        succ_sum += success
        succ_idx = (succ_idx + 1) % len(succ_win)

        rate = succ_sum / float(len(succ_win))
        # target ~0.2; gentle, frequent updates
        if rate > 0.24:
            sigma *= 1.12
        elif rate < 0.16:
            sigma *= 0.88

        # ----- diagonal scale adaptation from elite steps -----
        # keep it stable with damping and clamping
        for i in range(dim):
            if spans[i] <= 0.0:
                continue
            denom = (sigma * sig_d[i] + 1e-30)
            s2 = 0.0
            for k in range(len(elites)):
                _, xk = elites[k]
                y = (xk[i] - old_m[i]) / denom
                s2 += w[k] * (y * y)
            s2 = max(0.25, min(4.0, s2))
            sig_d[i] *= math.exp(0.5 * eta * math.log(s2))

        clamp_scales()

        # ----- local search: randomized pattern search around best -----
        x = best_x[:]
        improved = False

        order = list(range(dim))
        random.shuffle(order)

        # coordinate probes (few per generation)
        probes = min(dim, 18)
        for t in range(probes):
            if time.time() >= deadline:
                return best
            i = order[t]
            if spans[i] <= 0.0 or step[i] <= step_floor[i]:
                continue

            base = x[i]
            # try +/- step and a smaller fallback
            for s in (1.0, -1.0):
                cand = x[:]
                cand[i] = base + s * step[i]
                cand = reflect_vec(cand)
                fc = evaluate(cand)
                if fc < best:
                    best, best_x = fc, cand[:]
                    x = cand
                    improved = True
                    break
            if improved:
                continue
            # smaller step attempt (cheap second chance)
            for s in (1.0, -1.0):
                cand = x[:]
                cand[i] = base + s * (0.35 * step[i])
                cand = reflect_vec(cand)
                fc = evaluate(cand)
                if fc < best:
                    best, best_x = fc, cand[:]
                    x = cand
                    improved = True
                    break

        # pairwise (2D) step sometimes helps on rotated valleys
        if dim >= 2 and time.time() < deadline and random.random() < 0.40:
            i = random.randrange(dim)
            j = random.randrange(dim - 1)
            if j >= i:
                j += 1
            if spans[i] > 0.0 and spans[j] > 0.0:
                si = step[i]
                sj = step[j]
                if si > step_floor[i] and sj > step_floor[j]:
                    # try a couple correlated directions
                    for _ in range(2):
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
                            break

        if improved:
            stall = 0
            last_improve = time.time()
            # pull mean toward best (exploitation) but not too aggressively
            a = 0.55
            m = reflect_vec([a * best_x[i] + (1.0 - a) * m[i] for i in range(dim)])
            # slightly increase local steps
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = min(0.30 * spans[i], step[i] * 1.08)
        else:
            stall += 1
            # shrink local steps; if stalled for long, increase global exploration a bit
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = max(step_floor[i], step[i] * 0.72)
            if (time.time() - last_improve) > max(0.12 * max_time, 0.5):
                sigma = min(25.0, sigma * 1.06)
                clamp_scales()
