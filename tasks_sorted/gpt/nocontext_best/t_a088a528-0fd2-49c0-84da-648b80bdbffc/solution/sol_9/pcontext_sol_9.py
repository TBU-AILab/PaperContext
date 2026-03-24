import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Main upgrades vs the provided algorithm:
      1) Better global search: (mu, lambda)-ES with diagonal step-sizes + *mirrored sampling*
         to reduce noise and increase sample efficiency.
      2) More reliable step-size control: 1/5-success rule computed against *previous generation best*
         (cheaper + more meaningful than re-evaluating old mean).
      3) More aggressive, still cheap local improvement: short pattern-search around current best
         with adaptive steps + a few random subspace probes.
      4) Stronger restarts: schedule based on stall + time slices; mixes random, best-centered,
         and "wide" restarts.
      5) Lower overhead: keeps only top-mu via insertion; avoids heavy allocations.

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

    # ---------- RNG: Box-Muller normal with cache ----------
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
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- bounds handling: reflection ----------
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

    # ---------- keep top-mu without full sort ----------
    def push_elite(elites, item, mu):
        f, _ = item
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

    # ---------- initialization: center + LHS-ish + opposition ----------
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
        if spans[i] <= 0.0:
            continue
        x1 = center[:]
        x2 = center[:]
        x1[i] = lows[i] + 0.25 * spans[i]
        x2[i] = lows[i] + 0.75 * spans[i]
        try_point(x1)
        try_point(x2)

    init_n = max(28, min(220, 14 * dim))
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

    # ---------- ES parameters ----------
    lam = max(18, min(96, 6 * dim + 12))
    # mirrored sampling -> use even lambda
    if lam % 2 == 1:
        lam += 1
    mu = max(3, lam // 4)

    # log weights
    w = [max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)) for k in range(mu)]
    wsum = sum(w) or 1.0
    w = [wk / wsum for wk in w]

    m = best_x[:]

    # per-dimension step scales and global sigma
    sig_d = [0.0] * dim
    for i in range(dim):
        sig_d[i] = (0.30 * spans[i] if spans[i] > 0.0 else 0.0)
    sigma = 1.0

    def clamp_scales():
        nonlocal sigma
        sigma = max(1e-12, min(50.0, sigma))
        for i in range(dim):
            if spans[i] <= 0.0:
                sig_d[i] = 0.0
            else:
                sig_d[i] = max(1e-15 * spans[i], min(0.85 * spans[i], sig_d[i]))

    clamp_scales()

    # diagonal adaptation rate
    eta = 0.25 / math.sqrt(dim + 2.0)

    # success rule state (compare generation-best to prev generation-best)
    succ = 0
    gen_win = 12
    gen_count = 0
    prev_gen_best = float("inf")

    # ---------- local search (pattern search-ish) ----------
    step = [0.10 * spans[i] if spans[i] > 0.0 else 0.0 for i in range(dim)]
    step_floor = [1e-12 * (spans[i] if spans[i] > 0.0 else 1.0) for i in range(dim)]
    shrink = 0.72

    # ---------- restart logic ----------
    stall = 0
    last_improve_t = time.time()
    slice_len = max(0.2, 0.30 * max_time)
    next_restart = t0 + slice_len

    def restart(wide=False):
        nonlocal m, sigma, sig_d, step, stall
        stall = 0
        r = random.random()
        if wide:
            m = rand_uniform()
        else:
            if r < 0.60:
                m = best_x[:]
            elif r < 0.85:
                m = rand_uniform()
            else:
                u = rand_uniform()
                a = 0.75
                m = [a * best_x[i] + (1.0 - a) * u[i] for i in range(dim)]
                m = reflect_vec(m)

        # expand exploration
        sigma = min(20.0, sigma * (2.0 if wide else 1.6) + 0.15)
        for i in range(dim):
            if spans[i] > 0.0:
                sig_d[i] = max(sig_d[i], (0.35 if wide else 0.22) * spans[i])
                step[i] = max(step[i], (0.12 if wide else 0.09) * spans[i])
        clamp_scales()

    # ---------- main loop ----------
    while True:
        if time.time() >= deadline:
            return best

        now = time.time()
        if now >= next_restart or stall > (28 + 6 * dim):
            wide = (now - last_improve_t) > max(0.18 * max_time, 0.6)
            restart(wide=wide)
            next_restart = now + slice_len

        # ===== ES generation (mirrored sampling) =====
        elites = []
        gen_best = float("inf")
        gen_best_x = None

        heavy = (random.random() < 0.06)

        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                return best

            z = [0.0] * dim
            if heavy:
                for i in range(dim):
                    z[i] = (randc() if random.random() < 0.18 else randn())
            else:
                for i in range(dim):
                    z[i] = randn()

            # sample pair: m + step, m - step (mirroring)
            for sign in (1.0, -1.0):
                x = m[:]
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                    else:
                        x[i] += sign * sigma * sig_d[i] * z[i]
                x = reflect_vec(x)
                fx = evaluate(x)

                if fx < gen_best:
                    gen_best = fx
                    gen_best_x = x[:]

                push_elite(elites, (fx, x), mu)

                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_improve_t = time.time()
                    stall = 0

        # ===== ES recombination/update =====
        old_m = m[:]
        m = [0.0] * dim
        for k in range(len(elites)):
            _, xk = elites[k]
            wk = w[k]
            for i in range(dim):
                m[i] += wk * xk[i]
        m = reflect_vec(m)

        # 1/5 success: compare to previous generation best (no extra evals)
        gen_count += 1
        if gen_best < prev_gen_best:
            succ += 1
        prev_gen_best = gen_best

        if gen_count >= gen_win:
            rate = succ / float(gen_count)
            if rate > 0.22:
                sigma *= 1.28
            elif rate < 0.18:
                sigma *= 0.78
            succ = 0
            gen_count = 0
            clamp_scales()

        # diagonal scale adaptation from elites
        for i in range(dim):
            if spans[i] <= 0.0:
                continue
            denom = sigma * sig_d[i] + 1e-30
            s2 = 0.0
            for k in range(len(elites)):
                _, xk = elites[k]
                y = (xk[i] - old_m[i]) / denom
                s2 += w[k] * (y * y)
            s2 = max(0.15, min(6.0, s2))
            sig_d[i] *= math.exp(0.5 * eta * math.log(s2))
        clamp_scales()

        # ===== local improvement around best =====
        if time.time() >= deadline:
            return best

        improved = False
        x = best_x[:]

        # coordinate probes (a few)
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
            cand1 = x[:]
            cand2 = x[:]
            cand1[i] = base + step[i]
            cand2[i] = base - step[i]
            cand1 = reflect_vec(cand1)
            cand2 = reflect_vec(cand2)

            f1 = evaluate(cand1)
            if f1 < best:
                best, best_x = f1, cand1[:]
                x = cand1
                improved = True
                break

            if time.time() >= deadline:
                return best
            f2 = evaluate(cand2)
            if f2 < best:
                best, best_x = f2, cand2[:]
                x = cand2
                improved = True
                break

        # small random subspace probes (helps in rotated landscapes)
        if not improved and dim >= 2 and time.time() < deadline:
            tries = 2
            for _ in range(tries):
                if time.time() >= deadline:
                    return best
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1
                if spans[i] <= 0.0 or spans[j] <= 0.0:
                    continue
                si = max(step_floor[i], step[i])
                sj = max(step_floor[j], step[j])
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
            # pull mean towards best and slightly enlarge local step
            a = 0.55
            m = reflect_vec([a * best_x[i] + (1.0 - a) * m[i] for i in range(dim)])
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = min(0.32 * spans[i], step[i] * 1.10)
        else:
            stall += 1
            for i in range(dim):
                if spans[i] > 0.0 and step[i] > step_floor[i]:
                    step[i] = max(step_floor[i], step[i] * shrink)

            # if stuck for a while, boost exploration modestly
            if (time.time() - last_improve_t) > max(0.14 * max_time, 0.5):
                sigma = min(25.0, sigma * 1.12)
                clamp_scales()
