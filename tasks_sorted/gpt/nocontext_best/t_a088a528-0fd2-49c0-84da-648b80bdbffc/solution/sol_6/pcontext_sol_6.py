import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained).

    Improvements vs previous:
      - Much stronger initialization: opposition + stratified (LHS-ish) + center probes
      - Two complementary local searches running under one budget:
          (A) Diagonal CMA-ES-lite (rank-1+rank-mu style variance update, evolution path)
          (B) Coordinate/pattern search with shrinking step (very good on separable / boxy landscapes)
      - Trust-region style step control + periodic heavy-tail escapes
      - Budget-aware: avoids overhead, uses monotone best tracking, frequent early exits

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

    # ---------- helpers ----------
    def clip(x):
        # inlined-ish for speed
        y = x[:]  # copy
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            si = spans[i]
            x[i] = lows[i] + (random.random() * si if si > 0.0 else 0.0)
        return x

    # Box-Muller normal (cached)
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
        spare = r * math.sin(th)
        have_spare = True
        return z0

    # Cauchy (heavy tail)
    def randc():
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    def evaluate(x):
        return float(func(x))

    # ---------- initialization (stronger than before) ----------
    best = float("inf")
    best_x = None

    # center point
    center = [lows[i] + 0.5 * spans[i] for i in range(dim)]
    if time.time() < deadline:
        f = evaluate(center)
        best, best_x = f, center[:]

    # opposition point (helps on some landscapes)
    opp = [highs[i] - (center[i] - lows[i]) for i in range(dim)]
    if time.time() < deadline:
        f = evaluate(opp)
        if f < best:
            best, best_x = f, opp[:]

    # LHS-ish stratified samples + their opposites
    init_n = max(20, min(120, 10 * dim))
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
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition of this stratified point (cheap diversity)
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        xo = clip(xo)
        fx = evaluate(xo)
        if fx < best:
            best, best_x = fx, xo[:]

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------- Optimizer A: diagonal CMA-ES-lite ----------
    # Mean
    m = best_x[:]

    # diag stds (start relatively broad)
    sig = [0.0] * dim
    for i in range(dim):
        si = spans[i]
        sig[i] = 0.30 * si if si > 0 else 0.0

    # global scale
    sigma_g = 1.0
    sigma_g_min = 1e-9
    sigma_g_max = 50.0

    # evolution path for diagonal adaptation
    ps = [0.0] * dim

    # population settings
    lam = max(12, min(64, 4 * dim + 8))
    mu = max(2, lam // 3)

    # log weights
    w = [0.0] * mu
    for k in range(mu):
        w[k] = max(0.0, math.log(mu + 0.5) - math.log(k + 1.0))
    wsum = sum(w) or 1.0
    w = [wk / wsum for wk in w]

    # learning rates (diag-only, stable defaults)
    c_sigma = 0.30 / math.sqrt(dim + 1.0)
    c_c = 0.20 / math.sqrt(dim + 1.0)      # path learning
    c1 = 0.12 / (dim + 2.0)                # rank-1
    cmu = 0.25 / (dim + 2.0)               # rank-mu
    # keep reasonable
    c1 = min(0.15, max(0.02, c1))
    cmu = min(0.35, max(0.05, cmu))

    # ---------- Optimizer B: coordinate/pattern search ----------
    # (good for quick monotone improvements under box constraints)
    step = [0.0] * dim
    for i in range(dim):
        si = spans[i]
        step[i] = 0.15 * si if si > 0 else 0.0
    step_floor = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    shrink = 0.65

    # scheduling / restarts
    stall = 0
    last_improve_time = time.time()
    restart_period = 0.45 * max_time  # soft restart
    next_restart = t0 + restart_period

    # success adaptation counters
    succ = 0
    trials = 0
    adapt_window = max(12, lam)

    def do_restart():
        nonlocal m, sigma_g, ps, sig, step, stall
        stall = 0
        ps = [0.0] * dim
        # choose base: best, random, or blend
        r = random.random()
        if r < 0.55:
            m = best_x[:]
        elif r < 0.80:
            m = rand_uniform()
        else:
            u = rand_uniform()
            # blend towards best
            a = 0.65
            m = [a * best_x[i] + (1.0 - a) * u[i] for i in range(dim)]
            m = clip(m)

        # re-expand exploration a bit
        sigma_g = min(sigma_g_max, max(1.0, sigma_g * 1.6))
        for i in range(dim):
            if spans[i] > 0:
                sig[i] = max(sig[i], 0.22 * spans[i])
                step[i] = max(step[i], 0.12 * spans[i])

    # ---------- main loop ----------
    while True:
        now = time.time()
        if now >= deadline:
            return best

        # periodic restart or if stalled too long
        if now >= next_restart or stall > (40 + 8 * dim):
            do_restart()
            next_restart = now + restart_period

        # ===== A) ES generation around m =====
        offspring = []
        # occasionally allow heavy-tail escape
        heavy = (random.random() < 0.10)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            x = m[:]  # start at mean
            if heavy:
                # occasional big jump on a few dims
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                        continue
                    if random.random() < 0.25:
                        x[i] += (sig[i] * sigma_g) * randc()
                    else:
                        x[i] += (sig[i] * sigma_g) * randn()
            else:
                for i in range(dim):
                    if spans[i] <= 0.0:
                        x[i] = lows[i]
                    else:
                        x[i] += (sig[i] * sigma_g) * randn()

            x = clip(x)
            fx = evaluate(x)
            offspring.append((fx, x))

            trials += 1
            if fx < best:
                best = fx
                best_x = x[:]
                succ += 1
                stall = 0
                last_improve_time = time.time()

        offspring.sort(key=lambda t: t[0])

        # update mean with top mu
        old_m = m[:]
        m = [0.0] * dim
        for k in range(mu):
            _, xk = offspring[k]
            wk = w[k]
            for i in range(dim):
                m[i] += wk * xk[i]
        m = clip(m)

        # update evolution path ps (approx using normalized displacement)
        # ps <- (1-c_c)*ps + sqrt(c_c*(2-c_c))* (m-old_m)/(sigma_g*sig)
        cc_fac = math.sqrt(max(1e-12, c_c * (2.0 - c_c)))
        for i in range(dim):
            if spans[i] <= 0.0 or sig[i] <= 0.0:
                ps[i] = 0.0
            else:
                ps[i] = (1.0 - c_c) * ps[i] + cc_fac * ((m[i] - old_m[i]) / (sigma_g * sig[i] + 1e-30))

        # diagonal variance update (rank-1 + rank-mu on normalized steps)
        for i in range(dim):
            if spans[i] <= 0.0:
                sig[i] = 0.0
                continue

            # rank-1 target
            r1 = ps[i] * ps[i]

            # rank-mu target: weighted mean of squared steps of elites
            rmu = 0.0
            denom = (sigma_g * sig[i] + 1e-30)
            for k in range(mu):
                _, xk = offspring[k]
                y = (xk[i] - old_m[i]) / denom
                rmu += w[k] * (y * y)

            # Update "variance" implicitly via sig scaling
            # sig^2 *= (1 - c1 - cmu) + c1*r1 + cmu*rmu
            cur2 = 1.0
            upd = (1.0 - c1 - cmu) * cur2 + c1 * r1 + cmu * rmu
            upd = max(1e-6, min(1e3, upd))  # stability clamp
            sig[i] *= math.sqrt(upd)

            # clamp to fraction of span
            sig[i] = min(0.60 * spans[i], max(1e-15 * spans[i] + 1e-30, sig[i]))

        # global step-size adaptation from success-rate
        if trials >= adapt_window:
            rate = succ / float(trials)
            if rate < 0.18:
                sigma_g *= 0.82
            elif rate > 0.30:
                sigma_g *= 1.20
            sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g))
            succ = 0
            trials = 0

        # ===== B) Coordinate/pattern search around best_x =====
        # run a small number of cheap probes each loop; helps refine rapidly
        if time.time() >= deadline:
            return best

        improved = False
        x = best_x[:]  # local search anchor at current best
        # random dimension order
        order = list(range(dim))
        random.shuffle(order)

        # limit probes so we don't blow the budget
        probes = min(dim, 12)
        for idx in range(probes):
            if time.time() >= deadline:
                return best
            i = order[idx]
            if spans[i] <= 0.0 or step[i] <= step_floor[i]:
                continue

            # try + and - step
            xi = x[i]
            for sgn in (1.0, -1.0):
                cand = x[:]
                cand[i] = xi + sgn * step[i]
                cand = clip(cand)
                fc = evaluate(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    x = cand
                    improved = True
                    break  # move on to next dimension
            # if not improved for that dim, keep x unchanged

        if improved:
            stall = 0
            # pull ES mean towards new best gently
            a = 0.35
            m = clip([a * best_x[i] + (1.0 - a) * m[i] for i in range(dim)])
            # slightly enlarge coordinate steps when making progress
            for i in range(dim):
                if spans[i] > 0.0:
                    step[i] = min(0.25 * spans[i], step[i] * 1.05)
        else:
            stall += 1
            # shrink coordinate steps on stall
            for i in range(dim):
                if spans[i] > 0.0 and step[i] > step_floor[i]:
                    step[i] = max(step_floor[i], step[i] * shrink)

            # if stuck for a while, force a more explorative ES scale bump
            if time.time() - last_improve_time > max(0.15 * max_time, 0.5):
                sigma_g = min(sigma_g_max, sigma_g * 1.12)
