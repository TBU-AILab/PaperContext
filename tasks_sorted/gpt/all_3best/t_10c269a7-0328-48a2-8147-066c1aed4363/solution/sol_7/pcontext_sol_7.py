import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Improvements over your current best (ES+SA+pattern+restarts):
      - Switches to a more reliable backbone: Adaptive Differential Evolution (jDE/SaDE-like)
        + opposition-ish candidates + diversity preservation.
      - Adds a *budgeted* local refinement stage: bounded Powell-style direction set search
        (derivative-free, strong on smooth-ish problems) triggered periodically and at endgame.
      - Uses normalized-space distance filtering to avoid redundant evaluations.
      - Uses multiple sub-populations (small islands) with occasional migration to reduce stagnation.

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

    def to_unit(x):
        return [(x[i] - lows[i]) * inv_spans[i] for i in range(dim)]

    def from_unit(u):
        return [clamp(lows[i] + u[i] * spans[i], i) for i in range(dim)]

    def eval_x(x):
        return float(func(x))

    # --- RNG helpers ---
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # --- Halton + stratified init (cheap diversity) ---
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

    # --- near-duplicate rejection in unit space ---
    recent_u = []
    recent_cap = 180
    # allow some closeness; too strict hurts DE
    close_thr2 = (1e-6) ** 2

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

    # --- bounded line minimization along direction (1D) ---
    def line_search_1d(x, fx, d, max_evals):
        # Golden-section search in t within [-tmax, tmax] such that x+t*d remains in bounds.
        # Determine feasible t interval.
        t_lo = -1e100
        t_hi = 1e100
        for i in range(dim):
            di = d[i]
            if abs(di) < 1e-18:
                continue
            if di > 0:
                lo = (lows[i] - x[i]) / di
                hi = (highs[i] - x[i]) / di
            else:
                lo = (highs[i] - x[i]) / di
                hi = (lows[i] - x[i]) / di
            if lo > t_lo:
                t_lo = lo
            if hi < t_hi:
                t_hi = hi
        if not (t_lo < t_hi):
            return fx, x

        # also cap by a reasonable radius to avoid wasting time
        rad = 0.6  # in normalized units
        # convert to absolute-ish by projecting to spans; approximate with avg_span
        cap = max(1e-12, rad * avg_span / (max(1e-12, math.sqrt(sum((d[i] * inv_spans[i])**2 for i in range(dim))))))
        t_lo = max(t_lo, -cap)
        t_hi = min(t_hi, cap)
        if not (t_lo < t_hi):
            return fx, x

        phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = t_lo, t_hi
        c = b - phi * (b - a)
        e = a + phi * (b - a)

        def f_at(t):
            xt = [clamp(x[i] + t * d[i], i) for i in range(dim)]
            u = to_unit(xt)
            if not accept_u(u):
                return None, None, None
            return eval_x(xt), xt, u

        evals = 0

        # ensure endpoints are not absurdly bad by sampling a couple points
        fc, xc, uc = f_at(c)
        if fc is None:
            fc, xc, uc = fx, x, to_unit(x)
        evals += 1

        if time.time() >= deadline - eps_time or evals >= max_evals:
            return fc if fc < fx else fx, xc if fc < fx else x

        fe, xe, ue = f_at(e)
        if fe is None:
            fe, xe, ue = fx, x, to_unit(x)
        evals += 1

        best_local_f = fx
        best_local_x = x
        if fc < best_local_f:
            best_local_f, best_local_x = fc, xc
        if fe < best_local_f:
            best_local_f, best_local_x = fe, xe

        while evals < max_evals and time.time() < deadline - eps_time and (b - a) > 1e-10:
            if fc < fe:
                b = e
                e, fe, xe = c, fc, xc
                c = b - phi * (b - a)
                fc, xc, _ = f_at(c)
                if fc is None:
                    fc, xc = fe, xe
                evals += 1
                if fc < best_local_f:
                    best_local_f, best_local_x = fc, xc
            else:
                a = c
                c, fc, xc = e, fe, xe
                e = a + phi * (b - a)
                fe, xe, _ = f_at(e)
                if fe is None:
                    fe, xe = fc, xc
                evals += 1
                if fe < best_local_f:
                    best_local_f, best_local_x = fe, xe

        return best_local_f, best_local_x

    # --- Powell-style local refinement (very budgeted) ---
    def powell_refine(x0, f0, budget_evals):
        x = x0[:]
        fx = f0
        # start with coordinate directions scaled by spans
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = spans[i]
            dirs.append(d)

        # small random directions help in rotated valleys
        k_extra = min(2, max(0, dim // 10))
        for _ in range(k_extra):
            d = [randn() * spans[i] for i in range(dim)]
            dirs.append(d)

        remaining = max(0, int(budget_evals))
        if remaining <= 0:
            return fx, x

        for _outer in range(2):
            if time.time() >= deadline - eps_time or remaining <= 0:
                break
            x_start = x[:]
            f_start = fx

            for j in range(len(dirs)):
                if time.time() >= deadline - eps_time or remaining <= 0:
                    break
                # each line search gets a tiny slice
                ls_budget = max(2, min(10, remaining // max(1, len(dirs) - j)))
                f_new, x_new = line_search_1d(x, fx, dirs[j], ls_budget)
                used = ls_budget
                remaining -= used
                if f_new < fx:
                    x, fx = x_new[:], f_new

            # update direction: net move
            dnet = [x[i] - x_start[i] for i in range(dim)]
            dnet_norm = math.sqrt(sum((dnet[i] * inv_spans[i]) ** 2 for i in range(dim)))
            if fx < f_start and dnet_norm > 1e-12:
                # replace worst direction with net direction
                dirs.pop(0)
                dirs.append(dnet)

        return fx, x

    # ---------------- Initialization: build islands ----------------
    best = float("inf")
    best_x = None

    # population sizing: try to keep evaluations per loop stable across dim
    base_pop = max(18, min(70, 10 + 3 * dim))
    islands = 3 if dim >= 8 else 2
    island_pop = max(8, base_pop // islands)

    # build initial unit points
    init_u = []
    # stratified
    init_u.extend(stratified_u(max(8, min(40, 6 * dim))))
    # halton
    for k in range(1, max(12, min(120, 12 * dim)) + 1):
        init_u.append(halton_u(k))
    # random + corners
    for _ in range(max(8, min(60, 6 * dim))):
        init_u.append([random.random() for _ in range(dim)])
    for _ in range(min(2 * dim, 24)):
        init_u.append([1.0 if random.random() < 0.5 else 0.0 for _ in range(dim)])

    # Evaluate some and create initial populations per island
    random.shuffle(init_u)
    pops = [[] for _ in range(islands)]   # each entry: list of dicts {u,x,f,F,CR}
    idx = 0

    def new_individual(u):
        x = from_unit(u)
        if not accept_u(u):
            return None
        f = eval_x(x)
        ind = {
            "u": u[:],
            "x": x[:],
            "f": f,
            "F": 0.5 + 0.3 * random.random(),     # initial DE scale
            "CR": 0.6 + 0.3 * random.random()     # initial crossover
        }
        return ind

    # fill populations
    while time.time() < deadline - eps_time and any(len(p) < island_pop for p in pops) and idx < len(init_u):
        for isl in range(islands):
            if len(pops[isl]) >= island_pop:
                continue
            if idx >= len(init_u) or time.time() >= deadline - eps_time:
                break
            u = init_u[idx]; idx += 1
            ind = new_individual(u)
            if ind is None:
                continue
            pops[isl].append(ind)
            if ind["f"] < best:
                best, best_x = ind["f"], ind["x"][:]

    # if still empty (pathological), just do one random eval
    if best_x is None:
        x = [lows[i] + random.random() * spans[i] for i in range(dim)]
        return eval_x(x)

    # ensure each island has at least 4 individuals for DE mutation; pad if needed
    for isl in range(islands):
        while len(pops[isl]) < 4 and time.time() < deadline - eps_time:
            u = [random.random() for _ in range(dim)]
            ind = new_individual(u)
            if ind is None:
                continue
            pops[isl].append(ind)
            if ind["f"] < best:
                best, best_x = ind["f"], ind["x"][:]

    # ---------------- Main loop: jDE + occasional Powell refinement ----------------
    stall = 0
    last_best = best

    next_local_time = t0 + 0.22
    next_migrate_time = t0 + 0.35

    while time.time() < deadline - eps_time:
        now = time.time()
        frac = (now - t0) / max(1e-12, float(max_time))

        # migrate best individuals across islands sometimes
        if now >= next_migrate_time:
            # pick global top few and inject
            all_inds = []
            for isl in range(islands):
                all_inds.extend((isl, i, pops[isl][i]["f"]) for i in range(len(pops[isl])))
            all_inds.sort(key=lambda t: t[2])
            top = all_inds[:max(1, islands)]
            for rank, (src_isl, src_i, _) in enumerate(top):
                donor = pops[src_isl][src_i]
                dst_isl = (src_isl + 1 + rank) % islands
                # replace worst in destination
                worst_i = max(range(len(pops[dst_isl])), key=lambda k: pops[dst_isl][k]["f"])
                pops[dst_isl][worst_i] = {
                    "u": donor["u"][:], "x": donor["x"][:], "f": donor["f"],
                    "F": 0.5 + 0.3 * random.random(), "CR": 0.6 + 0.3 * random.random()
                }
            next_migrate_time = now + (0.28 if frac < 0.6 else 0.20)

        # DE evolution per island
        for isl in range(islands):
            pop = pops[isl]
            n = len(pop)
            if n < 4:
                continue

            # iterate over members
            for i in range(n):
                if time.time() >= deadline - eps_time:
                    break

                xi = pop[i]

                # jDE adaptation (self-adapt F/CR)
                if random.random() < 0.1:
                    xi["F"] = 0.1 + 0.9 * random.random()
                if random.random() < 0.1:
                    xi["CR"] = random.random()

                F = xi["F"]
                CR = xi["CR"]

                # choose r1,r2,r3 != i
                idxs = list(range(n))
                idxs.remove(i)
                r1, r2, r3 = random.sample(idxs, 3)

                a = pop[r1]["u"]
                b = pop[r2]["u"]
                c = pop[r3]["u"]

                # mutation in unit space
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = a[d] + F * (b[d] - c[d])

                    # bounce-back / wrap to keep in [0,1]
                    if v[d] < 0.0:
                        v[d] = -v[d] * 0.5
                    if v[d] > 1.0:
                        v[d] = 1.0 - (v[d] - 1.0) * 0.5
                    if v[d] < 0.0:
                        v[d] = 0.0
                    elif v[d] > 1.0:
                        v[d] = 1.0

                # binomial crossover
                u_trial = xi["u"][:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        u_trial[d] = v[d]

                # occasional opposition-like candidate (cheap diversity)
                if random.random() < 0.07:
                    u_opp = [1.0 - u_trial[d] for d in range(dim)]
                    # slight jitter toward center
                    for d in range(dim):
                        u_opp[d] = 0.5 * u_opp[d] + 0.5 * (0.5 + 0.15 * (random.random() - 0.5))
                        if u_opp[d] < 0.0:
                            u_opp[d] = 0.0
                        elif u_opp[d] > 1.0:
                            u_opp[d] = 1.0
                    # pick better predicted by a quick evaluation? just evaluate the one farther from current
                    if dist2_u(u_opp, xi["u"]) > dist2_u(u_trial, xi["u"]):
                        u_trial = u_opp

                # if too close to recent, random-jitter it a bit
                if not accept_u(u_trial):
                    for d in range(dim):
                        u_trial[d] = min(1.0, max(0.0, u_trial[d] + 0.01 * (random.random() - 0.5)))
                    if not accept_u(u_trial):
                        continue

                x_trial = from_unit(u_trial)
                f_trial = eval_x(x_trial)

                if f_trial <= xi["f"]:
                    pop[i] = {
                        "u": u_trial[:],
                        "x": x_trial[:],
                        "f": f_trial,
                        "F": xi["F"],
                        "CR": xi["CR"],
                    }
                    if f_trial < best:
                        best = f_trial
                        best_x = x_trial[:]
                        stall = 0

        # stall / progress tracking
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        # periodic local refinement (Powell) with tiny budget
        if now >= next_local_time and time.time() < deadline - eps_time:
            # endgame: spend more budget on local search
            budget = 14 if frac < 0.65 else 24
            f2, x2 = powell_refine(best_x, best, budget_evals=budget)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_best = best
                stall = 0
            next_local_time = now + (0.22 if frac < 0.6 else 0.14)

        # stagnation rescue: inject a few heavy-tail samples around best
        if stall >= max(25, 4 * dim) and time.time() < deadline - eps_time:
            stall = 0
            for _ in range(3):
                if time.time() >= deadline - eps_time:
                    break
                u = to_unit(best_x)
                # heavy tail in unit space
                for d in range(dim):
                    u[d] = min(1.0, max(0.0, u[d] + 0.08 * cauchy()))
                if not accept_u(u):
                    continue
                x = from_unit(u)
                fx = eval_x(x)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
            # also re-randomize a worst member in each island
            for isl in range(islands):
                if time.time() >= deadline - eps_time:
                    break
                pop = pops[isl]
                if not pop:
                    continue
                worst_i = max(range(len(pop)), key=lambda k: pop[k]["f"])
                u = [random.random() for _ in range(dim)]
                if accept_u(u):
                    x = from_unit(u)
                    fx = eval_x(x)
                    pop[worst_i] = {"u": u[:], "x": x[:], "f": fx, "F": 0.5, "CR": 0.9}
                    if fx < best:
                        best, best_x = fx, x[:]
                        last_best = best

    # final quick local refinement if time remains
    if time.time() < deadline - eps_time:
        f2, x2 = powell_refine(best_x, best, budget_evals=18)
        if f2 < best:
            best = f2

    return best
