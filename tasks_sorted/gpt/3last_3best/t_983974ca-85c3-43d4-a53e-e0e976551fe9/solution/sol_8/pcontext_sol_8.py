import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement focus vs your current best (JADE + coord-polish):
      - Keep JADE (current-to-pbest/1 + archive) as the robust global engine
      - Add a *real local optimizer* at the end: bounded SPSA (gradient approx) + backtracking
        (works very well when near a basin and evaluations are noisy-ish or non-smooth)
      - Add *multi-start elite restarts*: maintain a small elite set; periodically spawn local runs
      - Make the diagonal sampler cheaper & more "trust-region": sample fewer, but update sigma smarter
      - Tight time-slicing: dedicate last ~25% of time to exploitation (local runs + polish)

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- edge cases ----------------
    if dim <= 0:
        try:
            v = float(func([]))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]

    # fix inverted/degenerate
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

    # ---------------- helpers ----------------
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

    def randn_clip2():
        while True:
            z = random.gauss(0.0, 1.0)
            if -2.0 <= z <= 2.0:
                return z

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def topk_indices(fits, k):
        best_list = []
        for idx, f in enumerate(fits):
            if len(best_list) < k:
                best_list.append((f, idx))
                if len(best_list) == k:
                    best_list.sort(key=lambda t: t[0])
            else:
                if f < best_list[-1][0]:
                    j = k - 1
                    while j > 0 and f < best_list[j - 1][0]:
                        j -= 1
                    best_list.insert(j, (f, idx))
                    best_list.pop()
        return [idx for _, idx in best_list]

    def pick_excluding(n, forbid):
        j = random.randrange(n)
        while j in forbid:
            j = random.randrange(n)
        return j

    # ---------------- parameters ----------------
    pop_size = int(14 + 6 * math.log(dim + 1.0))
    pop_size = max(24, min(96, pop_size))

    archive_max = pop_size

    # JADE adaptation
    c_adapt = 0.1
    mu_F = 0.6
    mu_CR = 0.9

    # local coordinate polish (cheap)
    min_step = 1e-15
    polish_every = 9
    polish_coords = min(dim, max(8, int(0.40 * dim)))
    polish_step = [max(min_step, 0.012 * spans[i]) for i in range(dim)]

    # diagonal trust-region sampler around best
    diag_sigma = [0.20 * spans[i] for i in range(dim)]
    diag_sigma_min = [max(min_step, 1e-7 * spans[i]) for i in range(dim)]
    diag_sigma_max = [0.65 * spans[i] for i in range(dim)]
    diag_every = 7
    diag_samples = max(4, min(14, 2 + dim // 5))

    # stagnation
    last_improve_t = time.time()
    stagnate_time = max(0.22 * max_time, 0.7)

    # elite pool for restarts/local runs
    elite_k = max(3, min(10, 2 + pop_size // 10))
    elites = []  # list of (fit, x)

    def elite_add(x, f):
        nonlocal elites
        # insert keep sorted
        if math.isinf(f) or math.isnan(f):
            return
        if not elites:
            elites = [(f, list(x))]
            return
        if len(elites) < elite_k or f < elites[-1][0]:
            # insert
            pos = len(elites)
            for i in range(len(elites)):
                if f < elites[i][0]:
                    pos = i
                    break
            elites.insert(pos, (f, list(x)))
            if len(elites) > elite_k:
                elites.pop()

    # ---------------- initialization ----------------
    init_until = min(deadline, t0 + 0.16 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1.0))))
    candidates.extend(lhs_points(n_lhs))
    for _ in range(max(2, pop_size // 7)):
        candidates.append(corner_vec(0.02))

    while len(candidates) < pop_size:
        x = rand_vec()
        candidates.append(x)
        if len(candidates) < pop_size:
            candidates.append(opposite(x))

    pop, fits = [], []
    best = float("inf")
    best_x = None

    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(10, pop_size // 2):
            break
        x = repair_reflect(list(x))
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        elite_add(x, f)
        if f < best:
            best, best_x = f, list(x)
            last_improve_t = time.time()
        if len(pop) >= pop_size:
            break

    while len(pop) < pop_size and time.time() < deadline:
        x = repair_reflect(rand_vec())
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        elite_add(x, f)
        if f < best:
            best, best_x = f, list(x)
            last_improve_t = time.time()

    if best_x is None:
        best_x = repair_reflect(rand_vec())
        best = safe_eval(best_x)
        elite_add(best_x, best)
        last_improve_t = time.time()

    archive = []

    # ---------------- coordinate polish ----------------
    def polish_best(x, fx):
        nonlocal best, best_x, last_improve_t, polish_step
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:polish_coords]

        improved_any = False
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

                polish_step[i] = min(0.35 * spans[i], polish_step[i] * 1.25)
                if fx < best:
                    best, best_x = fx, list(x)
                    last_improve_t = time.time()
            else:
                polish_step[i] = max(min_step, polish_step[i] * 0.78)

        if not improved_any:
            for i in idxs:
                polish_step[i] = max(min_step, polish_step[i] * 0.93)

        return x, fx

    # ---------------- diagonal sampler ----------------
    def diag_sampler():
        nonlocal best, best_x, last_improve_t, diag_sigma
        if time.time() >= deadline:
            return

        # update sigma from elites spread (cheap)
        if elites:
            # compute mean and abs-dev (more robust than variance)
            m = [0.0] * dim
            for _, ex in elites:
                for d in range(dim):
                    m[d] += ex[d]
            inv = 1.0 / float(len(elites))
            for d in range(dim):
                m[d] *= inv

            mad = [0.0] * dim
            for _, ex in elites:
                for d in range(dim):
                    mad[d] += abs(ex[d] - m[d])
            for d in range(dim):
                mad[d] *= inv
                target = 1.3 * mad[d] + 1e-12
                target = max(diag_sigma_min[d], min(diag_sigma_max[d], target))
                diag_sigma[d] = 0.80 * diag_sigma[d] + 0.20 * target

        # time shrink
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        shrink = 1.0 - 0.60 * min(1.0, tfrac)
        if shrink < 0.22:
            shrink = 0.22

        bx = best_x
        for _ in range(diag_samples):
            if time.time() >= deadline:
                return
            x = [bx[d] + random.gauss(0.0, shrink * diag_sigma[d]) for d in range(dim)]
            repair_reflect(x)
            f = safe_eval(x)
            elite_add(x, f)
            if f < best:
                best, best_x = f, list(x)
                last_improve_t = time.time()

            # inject by replacing likely-worse
            j = random.randrange(pop_size)
            for __ in range(3):
                k = random.randrange(pop_size)
                if fits[k] > fits[j]:
                    j = k
            if f < fits[j]:
                pop[j] = x
                fits[j] = f

    # ---------------- SPSA local optimizer (bounded) ----------------
    def spsa_local(x0, f0, time_budget):
        nonlocal best, best_x, last_improve_t
        endt = min(deadline, time.time() + max(0.0, time_budget))
        x = list(x0)
        fx = float(f0)

        # initial scale ~ 1% of span (smaller late)
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        base_step = (0.03 - 0.02 * min(1.0, tfrac))
        if base_step < 0.008:
            base_step = 0.008

        a0 = base_step
        c0 = 0.08 * base_step

        k = 0
        while time.time() < endt:
            k += 1
            # decreasing schedules
            ak = a0 / (k ** 0.35)
            ck = c0 / (k ** 0.10)

            # Rademacher perturbation
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

            x_plus = [x[d] + (ck * spans[d]) * delta[d] for d in range(dim)]
            x_minus = [x[d] - (ck * spans[d]) * delta[d] for d in range(dim)]
            repair_reflect(x_plus)
            repair_reflect(x_minus)

            f_plus = safe_eval(x_plus)
            if time.time() >= endt:
                break
            f_minus = safe_eval(x_minus)

            # gradient estimate
            g = [0.0] * dim
            denom = (2.0 * ck)
            if denom <= 0.0:
                denom = 1e-12
            diff = (f_plus - f_minus) / denom
            for d in range(dim):
                # scale to original coords
                g[d] = diff * delta[d] / max(1e-12, spans[d])

            # normalized step + backtracking
            gnorm = 0.0
            for d in range(dim):
                gnorm += g[d] * g[d]
            gnorm = math.sqrt(gnorm)
            if not (gnorm > 0.0) or math.isinf(gnorm) or math.isnan(gnorm):
                continue

            step = ak
            improved = False
            for _bt in range(5):
                if time.time() >= endt:
                    break
                xn = [x[d] - (step * spans[d]) * (g[d] / gnorm) for d in range(dim)]
                repair_reflect(xn)
                fn = safe_eval(xn)
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    elite_add(x, fx)
                    if fx < best:
                        best, best_x = fx, list(x)
                        last_improve_t = time.time()
                    break
                step *= 0.5

            if not improved:
                # if stuck, a tiny random nudge (keeps SPSA moving)
                if time.time() < endt:
                    xn = [x[d] + random.gauss(0.0, 0.01 * spans[d]) for d in range(dim)]
                    repair_reflect(xn)
                    fn = safe_eval(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        elite_add(x, fx)
                        if fx < best:
                            best, best_x = fx, list(x)
                            last_improve_t = time.time()

        return x, fx

    # ---------------- main loop (JADE) ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        now = time.time()
        tfrac = (now - t0) / max(1e-12, float(max_time))

        # allocate exploitation late
        if tfrac > 0.75:
            # run a few local searches from elites/best
            # keep it very time-bounded
            per = 0.06 * max_time / max(1, elite_k)
            # always try best
            spsa_local(best_x, best, 0.10 * max_time)
            for j in range(min(elite_k, len(elites))):
                if time.time() >= deadline:
                    return best
                spsa_local(elites[j][1], elites[j][0], per)
            # final polish pass
            polish_best(list(best_x), best)
            return best

        # time-adaptive pbest rate
        p_best_rate = 0.28 - 0.20 * min(1.0, tfrac)
        if p_best_rate < 0.08:
            p_best_rate = 0.08

        # occasional polish and diag sampler
        if gen % polish_every == 0:
            bx, bf = polish_best(list(best_x), best)
            if bf < best:
                best, best_x = bf, list(bx)

        if gen % diag_every == 0:
            diag_sampler()

        # stagnation handling
        if time.time() - last_improve_t > stagnate_time:
            idx_sorted = list(range(pop_size))
            idx_sorted.sort(key=lambda i: fits[i])
            worst_k = max(3, pop_size // 6)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted[-1 - t]
                r = random.random()
                if r < 0.60:
                    x = list(best_x)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.16 * spans[d])
                    repair_reflect(x)
                elif r < 0.78:
                    x = opposite(best_x)
                else:
                    x = corner_vec(0.09) if random.random() < 0.5 else rand_vec()
                    repair_reflect(x)
                f = safe_eval(x)
                pop[wi] = x
                fits[wi] = f
                elite_add(x, f)
                if f < best:
                    best, best_x = f, list(x)
                    last_improve_t = time.time()
            last_improve_t = time.time()

        # pbest pool
        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))
        pbest_pool = topk_indices(fits, p_num)

        SF, SCR, dW = [], [], []

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            CR = mu_CR + 0.10 * randn_clip2()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = mu_F + 0.10 * cauchy()
            tries = 0
            while F <= 0.0 and tries < 8:
                F = mu_F + 0.10 * cauchy()
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
            xpbest = pop[pbest_idx]

            r1 = pick_excluding(pop_size, {i})
            x1 = pop[r1]

            use_arch = (archive and random.random() < 0.5)
            if use_arch and random.random() < 0.5:
                x2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_excluding(pop_size, {i, r1})
                x2 = pop[r2]

            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d])
            repair_reflect(v)

            jrand = random.randrange(dim)
            u = [v[d] if (d == jrand or random.random() < CR) else xi[d] for d in range(dim)]
            fu = safe_eval(u)

            if fu <= fi:
                archive.append(list(xi))
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                SF.append(F)
                SCR.append(CR)
                imp = fi - fu
                if imp <= 0.0:
                    imp = 1e-12
                dW.append(imp)

                elite_add(u, fu)

                if fu < best:
                    best, best_x = fu, list(u)
                    last_improve_t = time.time()

        if SF:
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))
            num = 0.0
            den = 0.0
            for Fv, w in zip(SF, dW):
                num += w * Fv * Fv
                den += w * Fv
            if den > 0.0:
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * (num / den)

            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.98, max(0.05, mu_CR))

        # cheap best-jitter injection (kept, but slightly reduced)
        if gen % 9 == 0 and time.time() < deadline:
            inject = 1 if dim > 35 else 2
            scale = 0.09 - 0.06 * min(1.0, tfrac)
            if scale < 0.012:
                scale = 0.012
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                j = random.randrange(pop_size)
                for __ in range(3):
                    k = random.randrange(pop_size)
                    if fits[k] > fits[j]:
                        j = k
                x = list(best_x)
                for d in range(dim):
                    x[d] += random.gauss(0.0, scale * spans[d])
                repair_reflect(x)
                f = safe_eval(x)
                elite_add(x, f)
                if f < fits[j]:
                    pop[j] = x
                    fits[j] = f
                if f < best:
                    best, best_x = f, list(x)
                    last_improve_t = time.time()

    return best
