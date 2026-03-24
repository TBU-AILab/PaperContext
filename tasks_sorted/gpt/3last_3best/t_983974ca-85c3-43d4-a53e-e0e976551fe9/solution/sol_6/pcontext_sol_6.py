import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements over your current best (JADE + coord-polish):
      - Adds *success-adapted strategy ensemble* (jDE-style): current-to-pbest/1 + rand/1 + best/1
      - Adds *jDE parameter self-adaptation per individual* (Fi, CRi) + success reinforcement
      - Uses *reflection repair* (less boundary sticking)
      - Uses *Top-k partial sorting* (avoid full sort when only p-best set is needed)
      - Stronger, budgeted *best-only local search*: adaptive coordinate + occasional 2D block
      - Stagnation handling: targeted worst replacement using best-jitter + random/opposition immigrants
      - Initialization: LHS-like + corners + opposition + small best-jitter once a best exists

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- edge cases ----------
    if dim <= 0:
        try:
            v = func([])
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]

    # fix inverted / degenerate bounds
    spans = [0.0] * dim
    for i in range(dim):
        lo, hi = lows[i], highs[i]
        if lo > hi:
            lo, hi = hi, lo
            lows[i], highs[i] = lo, hi
        s = hi - lo
        if s <= 0.0:
            s = 1.0
        spans[i] = s

    # ---------- helpers ----------
    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def repair_reflect(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            xi = x[i]
            if xi < lo:
                xi = lo + (lo - xi)
                if xi > hi:
                    xi = lo + random.random() * (hi - lo) if hi > lo else lo
            elif xi > hi:
                xi = hi - (xi - hi)
                if xi < lo:
                    xi = lo + random.random() * (hi - lo) if hi > lo else lo
            x[i] = xi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return repair_reflect(xo)

    def corner_vec(jitter=0.02):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        if jitter > 0.0:
            for i in range(dim):
                x[i] += random.gauss(0.0, jitter * spans[i])
        return repair_reflect(x)

    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        invn = 1.0 / float(n)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][k] + random.random()) * invn
                x[d] = lows[d] + u * (highs[d] - lows[d])
            pts.append(x)
        return pts

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def randn_clip2():
        while True:
            z = random.gauss(0.0, 1.0)
            if -2.0 <= z <= 2.0:
                return z

    # fast pick from [0..n-1] not in a small forbidden set
    def pick_excluding(n, forbid):
        j = random.randrange(n)
        while j in forbid:
            j = random.randrange(n)
        return j

    # partial selection: get indices of k best without full sort
    def topk_indices(fits, k):
        # maintain a small list of (fit, idx), insertion O(k) which is cheap for small k
        best_list = []
        for idx, f in enumerate(fits):
            if len(best_list) < k:
                best_list.append((f, idx))
                if len(best_list) == k:
                    best_list.sort(key=lambda t: t[0])
            else:
                if f < best_list[-1][0]:
                    # insert in sorted position
                    j = k - 1
                    while j > 0 and f < best_list[j - 1][0]:
                        j -= 1
                    best_list.insert(j, (f, idx))
                    best_list.pop()
        return [idx for _, idx in best_list]

    # ---------- parameters ----------
    pop_size = int(16 + 6 * math.log(dim + 1.0))
    pop_size = max(24, min(90, pop_size))

    archive_max = pop_size

    # jDE self-adaptation rates
    tau1 = 0.1  # F update prob
    tau2 = 0.1  # CR update prob

    # initial means (also used for generating when self-adapt doesn't trigger)
    mu_F = 0.6
    mu_CR = 0.9

    # strategy probabilities (adapt by success)
    # 0: current-to-pbest/1, 1: rand/1, 2: best/1
    strat_p = [0.55, 0.30, 0.15]

    # local search
    min_step = 1e-15
    polish_every = 9
    polish_coords = min(dim, max(8, int(0.40 * dim)))
    polish_step = [max(min_step, 0.012 * spans[i]) for i in range(dim)]

    # stagnation control
    last_improve_t = time.time()
    stagnate_time = max(0.24 * max_time, 0.7)

    # ---------- initialization ----------
    init_until = min(deadline, t0 + 0.18 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1.0))))
    candidates.extend(lhs_points(n_lhs))

    for _ in range(max(2, pop_size // 6)):
        candidates.append(corner_vec(0.02))

    # random + opposition
    while len(candidates) < pop_size:
        x = rand_vec()
        candidates.append(x)
        if len(candidates) < pop_size:
            candidates.append(opposite(x))

    pop = []
    fits = []
    F_i = []
    CR_i = []

    best = float("inf")
    best_x = None

    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(10, pop_size // 2):
            break
        x = repair_reflect(list(x))
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        # per-individual params
        F_i.append(min(1.0, max(0.05, mu_F + 0.1 * cauchy())))
        CR_i.append(min(1.0, max(0.0, mu_CR + 0.1 * randn_clip2())))
        if f < best:
            best = f
            best_x = list(x)
            last_improve_t = time.time()
        if len(pop) >= pop_size:
            break

    while len(pop) < pop_size and time.time() < deadline:
        x = repair_reflect(rand_vec())
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        F_i.append(min(1.0, max(0.05, mu_F + 0.1 * cauchy())))
        CR_i.append(min(1.0, max(0.0, mu_CR + 0.1 * randn_clip2())))
        if f < best:
            best = f
            best_x = list(x)
            last_improve_t = time.time()

    if best_x is None:
        best_x = repair_reflect(rand_vec())
        best = safe_eval(best_x)
        last_improve_t = time.time()

    archive = []

    # ---------- local refinement ----------
    def polish_best(x, fx):
        nonlocal best, best_x, last_improve_t, polish_step

        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:polish_coords]

        improved_any = False

        # 1D coordinate moves
        for i in idxs:
            if time.time() >= deadline:
                break
            si = polish_step[i]
            if si < min_step:
                si = min_step

            base = x[i]

            x[i] = base + si
            repair_reflect(x)
            f1 = safe_eval(x)

            x[i] = base - si
            repair_reflect(x)
            f2 = safe_eval(x)

            x[i] = base

            if f1 < fx or f2 < fx:
                improved_any = True
                if f1 <= f2:
                    x[i] = base + si
                    repair_reflect(x)
                    fx = f1
                else:
                    x[i] = base - si
                    repair_reflect(x)
                    fx = f2

                polish_step[i] = min(0.35 * spans[i], polish_step[i] * 1.30)
                if fx < best:
                    best = fx
                    best_x = list(x)
                    last_improve_t = time.time()
            else:
                polish_step[i] = max(min_step, polish_step[i] * 0.75)

        # small 2D block tries
        if time.time() < deadline and dim >= 2:
            for _ in range(1 if dim >= 20 else 2):
                if time.time() >= deadline:
                    break
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1

                si = max(min_step, 0.6 * polish_step[i])
                sj = max(min_step, 0.6 * polish_step[j])
                bi, bj = x[i], x[j]

                best_loc = fx
                best_ij = (bi, bj)
                for di, dj in ((si, sj), (si, -sj), (-si, sj), (-si, -sj)):
                    x[i] = bi + di
                    x[j] = bj + dj
                    repair_reflect(x)
                    fv = safe_eval(x)
                    if fv < best_loc:
                        best_loc = fv
                        best_ij = (x[i], x[j])

                x[i], x[j] = bi, bj
                if best_loc < fx:
                    x[i], x[j] = best_ij
                    fx = best_loc
                    improved_any = True
                    if fx < best:
                        best = fx
                        best_x = list(x)
                        last_improve_t = time.time()

        if not improved_any:
            for i in idxs:
                polish_step[i] = max(min_step, polish_step[i] * 0.92)

        return x, fx

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # time-adaptive p-best rate
        tfrac = (time.time() - t0) / max(1e-12, max_time)
        p_best_rate = 0.30 - 0.22 * min(1.0, tfrac)
        if p_best_rate < 0.06:
            p_best_rate = 0.06

        # occasional polish
        if (gen % polish_every) == 0 and time.time() < deadline:
            bx, bf = polish_best(list(best_x), best)
            if bf < best:
                best, best_x = bf, list(bx)

        # stagnation: replace some worst
        if time.time() - last_improve_t > stagnate_time:
            # get worst indices via full sort (rare)
            idx_sorted = list(range(pop_size))
            idx_sorted.sort(key=lambda i: fits[i])
            worst_k = max(3, pop_size // 6)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted[-1 - t]
                if random.random() < 0.65:
                    x = list(best_x)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.16 * spans[d])
                    repair_reflect(x)
                else:
                    x = corner_vec(0.08) if random.random() < 0.5 else rand_vec()
                    repair_reflect(x)
                f = safe_eval(x)
                pop[wi] = x
                fits[wi] = f
                # reset individual params a bit
                F_i[wi] = min(1.0, max(0.05, 0.7 * F_i[wi] + 0.3 * (mu_F + 0.1 * cauchy())))
                CR_i[wi] = min(1.0, max(0.0, 0.7 * CR_i[wi] + 0.3 * (mu_CR + 0.1 * randn_clip2())))
                if f < best:
                    best = f
                    best_x = list(x)
                    last_improve_t = time.time()
            last_improve_t = time.time()

        # build p-best pool without full sort
        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))
        pbest_pool = topk_indices(fits, p_num)

        # compute current best index cheaply
        best_idx = 0
        best_fit = fits[0]
        for i in range(1, pop_size):
            if fits[i] < best_fit:
                best_fit = fits[i]
                best_idx = i
        xbest = pop[best_idx]

        # success trackers for adaptation
        strat_succ = [1e-12, 1e-12, 1e-12]
        strat_try = [1e-12, 1e-12, 1e-12]
        succ_F = []
        succ_CR = []
        succ_w = []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # jDE self-adaptation
            Fi = F_i[i]
            CRi = CR_i[i]
            if random.random() < tau1:
                Fi = 0.1 + 0.9 * random.random()
            else:
                # tiny drift toward global mean
                Fi = 0.85 * Fi + 0.15 * max(0.05, min(0.95, mu_F + 0.08 * cauchy()))
            if random.random() < tau2:
                CRi = random.random()
            else:
                CRi = 0.85 * CRi + 0.15 * max(0.0, min(1.0, mu_CR + 0.08 * randn_clip2()))
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0
            if CRi < 0.0:
                CRi = 0.0
            if CRi > 1.0:
                CRi = 1.0

            # choose strategy
            r = random.random()
            if r < strat_p[0]:
                strat = 0
            elif r < strat_p[0] + strat_p[1]:
                strat = 1
            else:
                strat = 2
            strat_try[strat] += 1.0

            # mutation
            if strat == 0:
                pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
                xpbest = pop[pbest_idx]

                r1 = pick_excluding(pop_size, {i})
                use_arch = (archive and random.random() < 0.55)
                if use_arch and random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding(pop_size, {i, r1})
                    x2 = pop[r2]
                x1 = pop[r1]

                v = [xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (x1[d] - x2[d]) for d in range(dim)]

            elif strat == 1:
                r1 = pick_excluding(pop_size, {i})
                r2 = pick_excluding(pop_size, {i, r1})
                r3 = pick_excluding(pop_size, {i, r1, r2})
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                v = [x1[d] + Fi * (x2[d] - x3[d]) for d in range(dim)]

            else:
                r1 = pick_excluding(pop_size, {i})
                r2 = pick_excluding(pop_size, {i, r1})
                x1, x2 = pop[r1], pop[r2]
                v = [xbest[d] + Fi * (x1[d] - x2[d]) for d in range(dim)]

            repair_reflect(v)

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CRi) else xi[d]

            fu = safe_eval(u)

            if fu <= fi:
                # archive parent
                archive.append(list(xi))
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                # keep adapted params if success
                F_i[i] = Fi
                CR_i[i] = CRi

                strat_succ[strat] += 1.0
                succ_F.append(Fi)
                succ_CR.append(CRi)
                imp = fi - fu
                if imp <= 0.0:
                    imp = 1e-12
                succ_w.append(imp)

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve_t = time.time()
            else:
                # mild reversion on failure (prevents bad parameters persisting)
                F_i[i] = 0.90 * F_i[i] + 0.10 * mu_F
                CR_i[i] = 0.90 * CR_i[i] + 0.10 * mu_CR

        # adapt global mu_F/mu_CR using successful trials (JADE-style)
        if succ_F:
            mu_CR = 0.9 * mu_CR + 0.1 * (sum(succ_CR) / float(len(succ_CR)))
            num = 0.0
            den = 0.0
            for Fv, w in zip(succ_F, succ_w):
                num += w * Fv * Fv
                den += w * Fv
            if den > 0.0:
                mu_F = 0.9 * mu_F + 0.1 * (num / den)
            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.98, max(0.05, mu_CR))

        # adapt strategy probabilities toward success rates
        rates = [strat_succ[k] / strat_try[k] for k in range(3)]
        s = rates[0] + rates[1] + rates[2]
        if s > 0.0:
            target = [rates[k] / s for k in range(3)]
            floors = [0.28, 0.12, 0.06]
            newp = [max(floors[k], 0.75 * strat_p[k] + 0.25 * target[k]) for k in range(3)]
            ss = newp[0] + newp[1] + newp[2]
            strat_p = [newp[0] / ss, newp[1] / ss, newp[2] / ss]

        # periodic best-jitter injection (cheap exploitation)
        if gen % 7 == 0 and time.time() < deadline:
            # find a couple of worse-half indices without sorting fully
            # sample candidates and replace if better
            inject = 2 if dim <= 40 else 1
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                j = random.randrange(pop_size)
                # try to bias toward worse: a few picks
                for __ in range(3):
                    k = random.randrange(pop_size)
                    if fits[k] > fits[j]:
                        j = k
                x = list(best_x)
                scale = 0.10 - 0.07 * min(1.0, tfrac)
                if scale < 0.015:
                    scale = 0.015
                for d in range(dim):
                    x[d] += random.gauss(0.0, scale * spans[d])
                repair_reflect(x)
                f = safe_eval(x)
                if f < fits[j]:
                    pop[j] = x
                    fits[j] = f
                    if f < best:
                        best = f
                        best_x = list(x)
                        last_improve_t = time.time()

    return best
