import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Upgrade over your current best (#6 ES+SA+pattern+restarts):
      - Stronger global search: Differential Evolution "current-to-best/1" with jDE self-adaptation (F, CR).
      - Keeps diversified init: stratified (LHS-like) + Halton + random + corners.
      - Adds a small *archive* (JADE-style) to improve mutation diversity and reduce stagnation.
      - Adds *resource-aware* local refinement:
          * fast coordinate pattern steps
          * small parabolic 1D steps along coordinates in the endgame
      - Uses normalized-space near-duplicate rejection but permissive enough not to break DE.

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
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
    spans = [0.0] * dim
    for i in range(dim):
        lo, hi = lows[i], highs[i]
        if hi < lo:
            lo, hi = hi, lo
            lows[i], highs[i] = lo, hi
        s = hi - lo
        if s == 0.0:
            s = 1.0
        spans[i] = s
    inv_spans = [1.0 / spans[i] for i in range(dim)]
    avg_span = sum(spans) / float(dim)

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def to_unit(x):
        return [(x[i] - lows[i]) * inv_spans[i] for i in range(dim)]

    def from_unit(u):
        return [clamp(lows[i] + u[i] * spans[i], i) for i in range(dim)]

    def eval_x(x):
        return float(func(x))

    # ---------------- RNG helpers ----------------
    def randn():
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        if u < 1e-12:
            u = 1e-12
        elif u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------------- Halton + stratified init ----------------
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

    # ---------------- near-duplicate rejection (unit space) ----------------
    recent_u = []
    recent_cap = 260
    close_thr2 = (6e-7) ** 2  # permissive: avoid exact/tiny duplicates without hurting DE

    def dist2_u(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def accept_u(u):
        # reject only if extremely close to recent points
        for r in recent_u:
            if dist2_u(u, r) < close_thr2:
                return False
        recent_u.append(u[:])
        if len(recent_u) > recent_cap:
            del recent_u[0:len(recent_u) - recent_cap]
        return True

    # ---------------- local refinements ----------------
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
        if time.time() >= deadline - eps_time:
            return fx, x
        h = abs(h)
        if h <= 0.0:
            return fx, x

        xi = x[i]
        x1 = x[:]
        x2 = x[:]
        x1[i] = clamp(xi - h, i)
        x2[i] = clamp(xi + h, i)

        u1 = to_unit(x1)
        f1 = eval_x(x1) if accept_u(u1) else float("inf")
        if time.time() >= deadline - eps_time:
            if f1 < fx:
                return f1, x1
            return fx, x

        u2 = to_unit(x2)
        f2 = eval_x(x2) if accept_u(u2) else float("inf")

        bestf, bestx = fx, x
        if f1 < bestf:
            bestf, bestx = f1, x1
        if f2 < bestf:
            bestf, bestx = f2, x2

        denom = (f1 - 2.0 * fx + f2)
        if (not math.isfinite(denom)) or abs(denom) < 1e-18:
            return bestf, bestx

        t_star = 0.5 * h * (f1 - f2) / denom
        if t_star < -2.0 * h:
            t_star = -2.0 * h
        elif t_star > 2.0 * h:
            t_star = 2.0 * h

        x3 = x[:]
        x3[i] = clamp(xi + t_star, i)
        u3 = to_unit(x3)
        if not accept_u(u3) or time.time() >= deadline - eps_time:
            return bestf, bestx
        f3 = eval_x(x3)
        if f3 < bestf:
            return f3, x3
        return bestf, bestx

    def endgame_refine(best_x, best_f):
        x, fx = best_x[:], best_f
        fx, x = pattern_search(x, fx, base_scale=0.05)
        if time.time() >= deadline - eps_time:
            return fx, x
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
                hi = h * (spans[i] / max(1e-12, avg_span))
                f2, x2 = parabolic_1d(x, fx, i, hi)
                if f2 < fx:
                    fx, x = f2, x2
        return fx, x

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    init_u = []
    init_u.extend(stratified_u(max(18, min(90, 8 * dim))))
    for k in range(1, max(20, min(240, 18 * dim)) + 1):
        init_u.append(halton_u(k))
    for _ in range(max(18, min(140, 9 * dim))):
        init_u.append([random.random() for _ in range(dim)])
    for _ in range(min(2 * dim, 28)):
        init_u.append([1.0 if random.random() < 0.5 else 0.0 for _ in range(dim)])
    random.shuffle(init_u)

    NP = max(26, min(96, 14 + 4 * dim))
    pop = []
    idx = 0
    tries = 0

    # JADE-style external archive (stores unit vectors)
    archive = []
    arch_max = max(10, min(140, 2 * NP))

    def arch_add(u):
        archive.append(u[:])
        if len(archive) > arch_max:
            # random removal
            del archive[random.randrange(len(archive))]

    while len(pop) < NP and time.time() < deadline - eps_time and (idx < len(init_u) or tries < 6 * NP):
        tries += 1
        if idx < len(init_u):
            u = init_u[idx]
            idx += 1
        else:
            u = [random.random() for _ in range(dim)]
        if not accept_u(u):
            continue
        x = from_unit(u)
        f = eval_x(x)
        ind = {
            "u": u[:],
            "x": x[:],
            "f": f,
            "F": 0.35 + 0.55 * random.random(),
            "CR": 0.15 + 0.75 * random.random()
        }
        pop.append(ind)
        if f < best:
            best, best_x = f, x[:]

    if best_x is None:
        x = [lows[i] + random.random() * spans[i] for i in range(dim)]
        return eval_x(x)

    stall = 0
    last_best = best
    next_local_time = t0 + 0.18
    next_restart_time = t0 + 0.65

    # ---------------- main loop ----------------
    while time.time() < deadline - eps_time:
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))

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

            pbest = pop[idxs[random.randrange(pbest_cnt)]]

            # select r1 != ii
            pool = list(range(len(pop)))
            pool.remove(ii)
            r1 = random.choice(pool)

            # select r2 from pop or archive (JADE-style)
            use_arch = (archive and random.random() < 0.35)
            if use_arch:
                u2 = random.choice(archive)
            else:
                # ensure not same as r1
                pool2 = pool[:]
                if r1 in pool2:
                    pool2.remove(r1)
                r2 = random.choice(pool2) if pool2 else r1
                u2 = pop[r2]["u"]

            u1 = pop[r1]["u"]

            # current-to-pbest/1 with archive difference
            v = [0.0] * dim
            tu = target["u"]
            pu = pbest["u"]
            for d in range(dim):
                vd = tu[d] + F * (pu[d] - tu[d]) + F * (u1[d] - u2[d])

                # bounce-back to [0,1]
                if vd < 0.0:
                    vd = -vd * 0.5
                if vd > 1.0:
                    vd = 1.0 - (vd - 1.0) * 0.5
                if vd < 0.0:
                    vd = 0.0
                elif vd > 1.0:
                    vd = 1.0
                v[d] = vd

            # crossover
            u_trial = tu[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u_trial[d] = v[d]

            # exploration kick (early more)
            if random.random() < (0.10 if frac < 0.40 else 0.04):
                kick = 0.05 if frac < 0.5 else 0.03
                for d in range(dim):
                    u_trial[d] = min(1.0, max(0.0, u_trial[d] + kick * cauchy()))

            if not accept_u(u_trial):
                # tiny jitter then retry
                for d in range(dim):
                    u_trial[d] = min(1.0, max(0.0, u_trial[d] + 0.004 * (random.random() - 0.5)))
                if not accept_u(u_trial):
                    continue

            x_trial = from_unit(u_trial)
            f_trial = eval_x(x_trial)

            if f_trial <= target["f"]:
                # successful -> archive the replaced target
                arch_add(target["u"])
                pop[ii] = {
                    "u": u_trial[:], "x": x_trial[:], "f": f_trial,
                    "F": target["F"], "CR": target["CR"]
                }
                if f_trial < best:
                    best, best_x = f_trial, x_trial[:]
                    stall = 0

        # stall tracking
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        # periodic local improvement
        if now >= next_local_time and time.time() < deadline - eps_time:
            if frac < 0.70:
                f2, x2 = pattern_search(best_x, best, base_scale=0.06)
            else:
                f2, x2 = endgame_refine(best_x, best)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best
                stall = 0
            next_local_time = now + (0.22 if frac < 0.6 else 0.12)

        # restart / diversity injection
        if (stall >= max(26, 4 * dim) or now >= next_restart_time) and time.time() < deadline - eps_time:
            stall = 0
            next_restart_time = now + (0.58 if frac < 0.6 else 0.36)

            idxs = list(range(len(pop)))
            idxs.sort(key=lambda k: pop[k]["f"])
            worst_cnt = max(2, len(pop) // 4)
            worst = idxs[-worst_cnt:]
            u_best = to_unit(best_x)

            for wi in worst:
                if time.time() >= deadline - eps_time:
                    break
                if random.random() < 0.5:
                    u_div = halton_u(random.randint(1, max(70, 16 * dim)))
                else:
                    u_div = [random.random() for _ in range(dim)]
                a = 0.35 + 0.60 * random.random()
                u_new = [min(1.0, max(0.0, a * u_best[d] + (1.0 - a) * u_div[d])) for d in range(dim)]
                for d in range(dim):
                    u_new[d] = min(1.0, max(0.0, u_new[d] + 0.012 * (random.random() - 0.5)))

                if not accept_u(u_new):
                    continue
                x_new = from_unit(u_new)
                f_new = eval_x(x_new)

                arch_add(pop[wi]["u"])
                pop[wi] = {
                    "u": u_new[:], "x": x_new[:], "f": f_new,
                    "F": 0.25 + 0.75 * random.random(),
                    "CR": random.random()
                }
                if f_new < best:
                    best, best_x = f_new, x_new[:]
                    last_best = best

    # final endgame refinement if time remains
    if time.time() < deadline - eps_time:
        f2, x2 = endgame_refine(best_x, best)
        if f2 < best:
            best = f2

    return best
