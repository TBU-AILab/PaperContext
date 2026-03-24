import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Upgrade over your current best (#5 ES+SA+pattern+restarts):
      - Uses a stronger global engine: Differential Evolution "current-to-best/1" with jDE-style
        self-adaptation of F and CR (more reliable than plain ES on many black-boxes).
      - Keeps your good ideas: diversified init (LHS-like + Halton + random + corners),
        stagnation restarts biased toward best, and cheap coordinate pattern search.
      - Adds a very cheap "quadratic/line" 1D refine along coordinate directions near the endgame
        (3-point parabolic step if possible; otherwise back to small pattern steps).
      - Uses normalized-space duplicate rejection (but not too strict to avoid hurting DE).

    Returns:
        best (float): best (minimum) objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    eps_time = 1e-4

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0
    inv_spans = [1.0 / spans[i] for i in range(dim)]
    avg_span = sum(spans) / float(dim)

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def eval_x(x):
        return float(func(x))

    def to_unit(x):
        return [(x[i] - lows[i]) * inv_spans[i] for i in range(dim)]

    def from_unit(u):
        return [clamp(lows[i] + u[i] * spans[i], i) for i in range(dim)]

    # ---------------- RNG helpers ----------------
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # ---------------- Halton + LHS-like init ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(dim)

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        n = index
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_u(k):
        return [van_der_corput(k, primes[i]) for i in range(dim)]

    def stratified_u(n):
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append(vals)
        pts = []
        for j in range(n):
            pts.append([strata[i][j] for i in range(dim)])
        return pts

    # ---------------- near-duplicate rejection in unit space ----------------
    recent_u = []
    recent_cap = 220
    close_thr2 = (3e-7) ** 2  # slightly stricter than DE-islands version but still permissive

    def dist2_u(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def accept_u(u):
        for r in recent_u:
            if dist2_u(u, r) < close_thr2:
                return False
        recent_u.append(u[:])
        if len(recent_u) > recent_cap:
            del recent_u[0:len(recent_u) - recent_cap]
        return True

    # ---------------- cheap local refinements ----------------
    def pattern_search(x0, f0, base_scale):
        x = x0[:]
        fx = f0
        step = [base_scale * spans[i] for i in range(dim)]
        for _round in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                if si <= 0.0:
                    continue
                xi = x[i]

                cand = x[:]
                cand[i] = clamp(xi + si, i)
                u = to_unit(cand)
                if accept_u(u):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        continue

                cand = x[:]
                cand[i] = clamp(xi - si, i)
                u = to_unit(cand)
                if accept_u(u):
                    fc = eval_x(cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    def parabolic_1d(x, fx, i, h):
        """
        Try a 3-point parabolic step along coordinate i:
          evaluate f(x-h), f(x), f(x+h) -> propose parabola minimum if stable.
        Returns (fbest, xbest). Uses up to 2 new evaluations if endpoints accepted.
        """
        if time.time() >= deadline - eps_time:
            return fx, x

        xi = x[i]
        h = abs(h)
        if h <= 0.0:
            return fx, x

        # Build two candidates
        x1 = x[:]
        x2 = x[:]
        x1[i] = clamp(xi - h, i)
        x2[i] = clamp(xi + h, i)

        f1 = None
        f2 = None

        u1 = to_unit(x1)
        if accept_u(u1):
            f1 = eval_x(x1)
        else:
            f1 = float("inf")

        if time.time() >= deadline - eps_time:
            # even if f1 computed, we return best among x and x1
            if f1 < fx:
                return f1, x1
            return fx, x

        u2 = to_unit(x2)
        if accept_u(u2):
            f2 = eval_x(x2)
        else:
            f2 = float("inf")

        # best among sampled
        bestf = fx
        bestx = x
        if f1 < bestf:
            bestf, bestx = f1, x1
        if f2 < bestf:
            bestf, bestx = f2, x2

        # Fit parabola through (-h,f1),(0,fx),(+h,f2) in local coordinate t
        # t* = h*(f1 - f2) / (2*(f1 - 2*fx + f2))
        denom = (f1 - 2.0 * fx + f2)
        if not math.isfinite(denom) or abs(denom) < 1e-18:
            return bestf, bestx

        t_star = 0.5 * h * (f1 - f2) / denom
        # keep within [-2h,2h] to avoid wild extrapolation
        if t_star < -2.0 * h:
            t_star = -2.0 * h
        elif t_star > 2.0 * h:
            t_star = 2.0 * h

        x3 = x[:]
        x3[i] = clamp(xi + t_star, i)
        u3 = to_unit(x3)
        if not accept_u(u3):
            return bestf, bestx

        if time.time() >= deadline - eps_time:
            return bestf, bestx

        f3 = eval_x(x3)
        if f3 < bestf:
            return f3, x3
        return bestf, bestx

    def endgame_refine(best_x, best_f):
        # small budgeted refinement pass
        x = best_x[:]
        fx = best_f
        # coarse pattern then coordinate parabolic attempts
        fx, x = pattern_search(x, fx, base_scale=0.05)
        if time.time() >= deadline - eps_time:
            return fx, x
        # coordinate parabolic steps with diminishing h
        base_h = 0.08 * avg_span
        for rep in range(2):
            if time.time() >= deadline - eps_time:
                break
            order = list(range(dim))
            random.shuffle(order)
            h = base_h * (0.6 ** rep)
            for i in order:
                if time.time() >= deadline - eps_time:
                    break
                # scale by span
                hi = h * (spans[i] / max(1e-12, avg_span))
                f2, x2 = parabolic_1d(x, fx, i, hi)
                if f2 < fx:
                    fx, x = f2, x2
        return fx, x

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # Build candidate unit points
    init_u = []
    init_u.extend(stratified_u(max(18, min(90, 8 * dim))))
    for k in range(1, max(20, min(220, 18 * dim)) + 1):
        init_u.append(halton_u(k))
    for _ in range(max(18, min(120, 9 * dim))):
        init_u.append([random.random() for _ in range(dim)])
    for _ in range(min(2 * dim, 26)):
        init_u.append([1.0 if random.random() < 0.5 else 0.0 for _ in range(dim)])

    random.shuffle(init_u)

    # DE population size
    NP = max(22, min(90, 12 + 4 * dim))
    pop = []
    tries = 0
    idx = 0
    while len(pop) < NP and time.time() < deadline - eps_time and (idx < len(init_u) or tries < 5 * NP):
        tries += 1
        if idx < len(init_u):
            u = init_u[idx]; idx += 1
        else:
            u = [random.random() for _ in range(dim)]
        if not accept_u(u):
            continue
        x = from_unit(u)
        f = eval_x(x)
        ind = {"u": u[:], "x": x[:], "f": f,
               "F": 0.4 + 0.5 * random.random(),
               "CR": 0.2 + 0.7 * random.random()}
        pop.append(ind)
        if f < best:
            best, best_x = f, x[:]

    if best_x is None:
        # fallback
        x = [lows[i] + random.random() * spans[i] for i in range(dim)]
        return eval_x(x)

    # ---------------- main loop (DE + restarts + local) ----------------
    stall = 0
    last_best = best

    next_local_time = t0 + 0.18
    next_restart_time = t0 + 0.60  # time-based restart check

    while time.time() < deadline - eps_time:
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))

        # Sort indices by fitness occasionally (cheap with small NP)
        # Also provides p-best set.
        idxs = list(range(len(pop)))
        idxs.sort(key=lambda k: pop[k]["f"])
        pbest_cnt = max(2, int(0.2 * len(pop)))

        # evolve
        for ii in range(len(pop)):
            if time.time() >= deadline - eps_time:
                break

            target = pop[ii]

            # jDE adaptation
            if random.random() < 0.10:
                target["F"] = 0.1 + 0.9 * random.random()
            if random.random() < 0.10:
                target["CR"] = random.random()

            F = target["F"]
            CR = target["CR"]

            # choose pbest
            pbest = pop[idxs[random.randrange(pbest_cnt)]]
            # choose r1,r2 distinct and not ii
            pool = list(range(len(pop)))
            pool.remove(ii)
            r1, r2 = random.sample(pool, 2)
            x1 = pop[r1]["u"]
            x2 = pop[r2]["u"]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = target["u"][d] + F * (pbest["u"][d] - target["u"][d]) + F * (x1[d] - x2[d])
                # bounce-back to [0,1]
                if v[d] < 0.0:
                    v[d] = -v[d] * 0.5
                if v[d] > 1.0:
                    v[d] = 1.0 - (v[d] - 1.0) * 0.5
                if v[d] < 0.0:
                    v[d] = 0.0
                elif v[d] > 1.0:
                    v[d] = 1.0

            # crossover
            u_trial = target["u"][:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u_trial[d] = v[d]

            # occasional heavy-tail kick in unit space (early more exploration)
            if random.random() < (0.09 if frac < 0.45 else 0.04):
                for d in range(dim):
                    u_trial[d] = min(1.0, max(0.0, u_trial[d] + (0.05 if frac < 0.5 else 0.03) * cauchy()))

            if not accept_u(u_trial):
                # small jitter then retry accept once
                for d in range(dim):
                    u_trial[d] = min(1.0, max(0.0, u_trial[d] + 0.004 * (random.random() - 0.5)))
                if not accept_u(u_trial):
                    continue

            x_trial = from_unit(u_trial)
            f_trial = eval_x(x_trial)

            if f_trial <= target["f"]:
                pop[ii] = {"u": u_trial[:], "x": x_trial[:], "f": f_trial, "F": target["F"], "CR": target["CR"]}
                if f_trial < best:
                    best, best_x = f_trial, x_trial[:]
                    stall = 0

        # stall tracking
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        # periodic local improve
        if now >= next_local_time and time.time() < deadline - eps_time:
            if frac < 0.7:
                f2, x2 = pattern_search(best_x, best, base_scale=0.06)
            else:
                f2, x2 = endgame_refine(best_x, best)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best
                stall = 0
            next_local_time = now + (0.22 if frac < 0.6 else 0.13)

        # time-based / stagnation-based restart: re-inject diversity but keep best
        if (stall >= max(25, 4 * dim) or now >= next_restart_time) and time.time() < deadline - eps_time:
            stall = 0
            next_restart_time = now + (0.55 if frac < 0.6 else 0.35)

            # Replace a fraction of worst with biased mixes around best + random/halton
            idxs = list(range(len(pop)))
            idxs.sort(key=lambda k: pop[k]["f"])
            worst_cnt = max(2, len(pop) // 4)
            worst = idxs[-worst_cnt:]

            for wi in worst:
                if time.time() >= deadline - eps_time:
                    break
                # mix best with a diversified sample
                if random.random() < 0.5:
                    u_div = halton_u(random.randint(1, max(60, 14 * dim)))
                else:
                    u_div = [random.random() for _ in range(dim)]
                u_best = to_unit(best_x)
                a = 0.35 + 0.60 * random.random()  # bias toward best
                u_new = [min(1.0, max(0.0, a * u_best[d] + (1.0 - a) * u_div[d])) for d in range(dim)]

                # jitter
                for d in range(dim):
                    u_new[d] = min(1.0, max(0.0, u_new[d] + 0.01 * (random.random() - 0.5)))

                if not accept_u(u_new):
                    continue
                x_new = from_unit(u_new)
                f_new = eval_x(x_new)
                pop[wi] = {"u": u_new[:], "x": x_new[:], "f": f_new,
                           "F": 0.3 + 0.7 * random.random(), "CR": random.random()}
                if f_new < best:
                    best, best_x = f_new, x_new[:]
                    last_best = best

    # final endgame refinement if any time remains
    if time.time() < deadline - eps_time:
        f2, x2 = endgame_refine(best_x, best)
        if f2 < best:
            best = f2

    return best
