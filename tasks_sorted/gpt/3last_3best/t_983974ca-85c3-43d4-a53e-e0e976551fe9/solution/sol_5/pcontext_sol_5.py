import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improved over your current best (JADE+polish) by adding:
      - Multi-strategy DE ensemble per individual (current-to-pbest, rand/1, best/1)
      - Success-based adaptation of strategy probabilities
      - More robust bound handling (reflect with randomized fallback)
      - Stronger, *budgeted* local refinement on best: adaptive coordinate + occasional 2D block steps
      - Time-aware population sizing + periodic partial restarts/immigrants
      - Initialization: LHS-like + corners + opposition + a few Gaussian jitters around best-so-far

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

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

    # spans + handle inverted/degenerate bounds
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

    # ------------ helpers ------------
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
        # reflect into bounds; if still out due to huge jump -> random reinsert
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

    def corner_vec(jitter_scale=0.02):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        if jitter_scale > 0.0:
            for i in range(dim):
                x[i] += random.gauss(0.0, jitter_scale * spans[i])
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

    def opposite(x):
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return repair_reflect(xo)

    def randn_clip2():
        # N(0,1) truncated to [-2,2]
        while True:
            z = random.gauss(0.0, 1.0)
            if -2.0 <= z <= 2.0:
                return z

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ------------ parameters (time-aware) ------------
    # population size: moderate, not too big (Python overhead)
    pop_size = int(16 + 6 * math.log(dim + 1.0))
    pop_size = max(22, min(86, pop_size))

    archive_max = pop_size

    # JADE-like adaptation
    c_adapt = 0.1
    mu_F = 0.6
    mu_CR = 0.9

    # strategy probabilities (adapted by success)
    # 0: current-to-pbest/1 (JADE)
    # 1: rand/1
    # 2: best/1 (more exploitative)
    strat_p = [0.55, 0.30, 0.15]

    # local search budget controls
    polish_every = 9
    polish_coords = min(dim, max(8, int(0.40 * dim)))
    min_step = 1e-15

    # stagnation control
    last_improve_t = time.time()
    stagnate_time = max(0.24 * max_time, 0.7)

    # per-coordinate steps for polish (adaptive)
    polish_step = [max(min_step, 0.012 * spans[i]) for i in range(dim)]

    # ------------ initialization ------------
    init_until = min(deadline, t0 + 0.18 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1.0))))
    candidates.extend(lhs_points(n_lhs))

    # corners
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
    best = float("inf")
    best_x = None

    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(10, pop_size // 2):
            break
        x = repair_reflect(list(x))
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
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
        if f < best:
            best = f
            best_x = list(x)
            last_improve_t = time.time()

    if best_x is None:
        best_x = repair_reflect(rand_vec())
        best = safe_eval(best_x)
        last_improve_t = time.time()

    # archive for DE
    archive = []

    # ------------ local refinement ------------
    def polish_best(x, fx):
        nonlocal best, best_x, last_improve_t, polish_step
        # coordinate subset
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

        # occasional 2D block tweak (cheap second-order-ish)
        if time.time() < deadline and dim >= 2:
            # try a few random pairs
            kpair = 2 if dim < 20 else 1
            for _ in range(kpair):
                if time.time() >= deadline:
                    break
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1
                si = max(min_step, 0.6 * polish_step[i])
                sj = max(min_step, 0.6 * polish_step[j])
                bi, bj = x[i], x[j]

                # four diagonal tries
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
            # gentle global shrink of tried steps
            for i in idxs:
                polish_step[i] = max(min_step, polish_step[i] * 0.92)

        return x, fx

    # ------------ main loop ------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # time-adaptive p-best rate (explore early -> exploit late)
        tfrac = (time.time() - t0) / max(1e-12, max_time)
        p_best_rate = 0.28 - 0.20 * min(1.0, tfrac)
        if p_best_rate < 0.06:
            p_best_rate = 0.06

        # sort indices by fitness
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda i: fits[i])

        # occasional local polish
        if (gen % polish_every) == 0 and time.time() < deadline:
            bx = list(best_x)
            bf = best
            bx, bf = polish_best(bx, bf)
            if bf < best:
                best = bf
                best_x = list(bx)

        # stagnation handling: inject immigrants + jittered best
        if time.time() - last_improve_t > stagnate_time:
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
                if f < best:
                    best = f
                    best_x = list(x)
                    last_improve_t = time.time()
            last_improve_t = time.time()

        # success trackers for parameter and strategy adaptation
        SF, SCR, dW = [], [], []
        strat_succ = [1e-12, 1e-12, 1e-12]  # avoid zero-prob
        strat_try = [1e-12, 1e-12, 1e-12]

        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))

        # precompute best vector for best/1 strategy
        xbest = pop[idx_sorted[0]]

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # sample CR
            CR = mu_CR + 0.10 * randn_clip2()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # sample F (cauchy)
            F = mu_F + 0.10 * cauchy()
            tries = 0
            while F <= 0.0 and tries < 8:
                F = mu_F + 0.10 * cauchy()
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose strategy
            r = random.random()
            if r < strat_p[0]:
                strat = 0
            elif r < strat_p[0] + strat_p[1]:
                strat = 1
            else:
                strat = 2
            strat_try[strat] += 1.0

            # pick distinct indices
            def pick_not(exclude):
                j = random.randrange(pop_size)
                while j in exclude:
                    j = random.randrange(pop_size)
                return j

            # mutation
            if strat == 0:
                # JADE current-to-pbest/1 with archive
                pbest_idx = idx_sorted[random.randrange(p_num)]
                xpbest = pop[pbest_idx]

                r1 = pick_not({i})
                x1 = pop[r1]

                use_arch = (archive and random.random() < 0.55)
                if use_arch and random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_not({i, r1})
                    x2 = pop[r2]

                v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d]) for d in range(dim)]

            elif strat == 1:
                # rand/1
                r1 = pick_not({i})
                r2 = pick_not({i, r1})
                r3 = pick_not({i, r1, r2})
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                v = [x1[d] + F * (x2[d] - x3[d]) for d in range(dim)]

            else:
                # best/1
                r1 = pick_not({i})
                r2 = pick_not({i, r1})
                x1, x2 = pop[r1], pop[r2]
                v = [xbest[d] + F * (x1[d] - x2[d]) for d in range(dim)]

            repair_reflect(v)

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CR) else xi[d]

            fu = safe_eval(u)

            if fu <= fi:
                # archive parent for diversity (only for strategies that benefit; ok to always do)
                archive.append(list(xi))
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                strat_succ[strat] += 1.0

                SF.append(F)
                SCR.append(CR)
                imp = fi - fu
                if imp <= 0.0:
                    imp = 1e-12
                dW.append(imp)

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve_t = time.time()

        # adapt mu_F / mu_CR (JADE)
        if SF:
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))
            num = 0.0
            den = 0.0
            for Fv, w in zip(SF, dW):
                num += w * Fv * Fv
                den += w * Fv
            if den > 0.0:
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * (num / den)

            # clamp
            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.98, max(0.05, mu_CR))

        # adapt strategy probabilities toward observed success rates (soft update)
        # success rate per strategy
        rates = [strat_succ[k] / strat_try[k] for k in range(3)]
        s = rates[0] + rates[1] + rates[2]
        if s > 0:
            target = [rates[k] / s for k in range(3)]
            # keep exploration alive with floors
            floors = [0.30, 0.12, 0.06]
            newp = [max(floors[k], 0.75 * strat_p[k] + 0.25 * target[k]) for k in range(3)]
            # renormalize
            ss = newp[0] + newp[1] + newp[2]
            strat_p = [newp[0] / ss, newp[1] / ss, newp[2] / ss]

        # periodically add a couple of "best-jitter" individuals to exploit basin
        if gen % 7 == 0 and time.time() < deadline:
            inject = 1 if dim > 40 else 2
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                # replace a random worse-half member
                j = idx_sorted[random.randrange(pop_size // 2, pop_size)]
                x = list(best_x)
                # jitter scale shrinks with time
                scale = (0.10 - 0.07 * min(1.0, tfrac))
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
