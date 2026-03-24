import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs your current best (JADE + coord-polish):
      1) Hybrid global search:
         - JADE current-to-pbest/1 + archive (main engine)
         - Periodic *CMA-ES-like diagonal* sampler around best (very effective basin exploitation)
      2) Better time/budget allocation:
         - Early: coverage (LHS/opposition/corners)
         - Mid: JADE exploration/exploitation
         - Late: stronger local improvement (diag sampler + polish)
      3) Stronger local search:
         - Adaptive coordinate pattern search with per-dimension step control
         - Occasional 2D pair moves
      4) Stagnation response:
         - Partial restart: replace worst with (best+jitter), opposition, and random immigrants
      5) Fast p-best selection:
         - Avoid full sort most generations (top-k maintenance)

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- edge cases ----------------
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
        # reflect into bounds; if still out => random reinsert
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
        # maintain sorted list of size k: O(n*k), OK for small k
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
    pop_size = max(22, min(90, pop_size))

    archive_max = pop_size

    # JADE adaptation
    c_adapt = 0.1
    mu_F = 0.6
    mu_CR = 0.9

    # polish / local search
    min_step = 1e-15
    polish_every = 9
    polish_coords = min(dim, max(8, int(0.40 * dim)))
    polish_step = [max(min_step, 0.012 * spans[i]) for i in range(dim)]

    # diagonal-CMA-like sampler parameters
    # (maintains per-dim std; updated from an elite set periodically)
    diag_sigma = [0.18 * spans[i] for i in range(dim)]
    diag_sigma_min = [max(min_step, 1e-6 * spans[i]) for i in range(dim)]
    diag_sigma_max = [0.60 * spans[i] for i in range(dim)]
    diag_every = 6  # generations
    diag_samples = max(6, min(24, 2 + dim // 3))  # samples per activation (kept moderate)

    # stagnation
    last_improve_t = time.time()
    stagnate_time = max(0.26 * max_time, 0.8)

    # ---------------- initialization ----------------
    init_until = min(deadline, t0 + 0.18 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1.0))))
    candidates.extend(lhs_points(n_lhs))
    for _ in range(max(2, pop_size // 6)):
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

    archive = []

    # ---------------- local refinement ----------------
    def polish_best(x, fx):
        nonlocal best, best_x, last_improve_t, polish_step
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:polish_coords]

        improved_any = False

        # 1D moves
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

        # occasional 2D diagonal try
        if time.time() < deadline and dim >= 2:
            tries = 2 if dim < 20 else 1
            for _ in range(tries):
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

    # ---------------- diagonal CMA-like sampling ----------------
    def diag_sampler_update_and_sample():
        """
        Update diag_sigma from current population elites, then sample around best_x.
        Uses only objective values (no gradients).
        """
        nonlocal best, best_x, last_improve_t, diag_sigma

        if time.time() >= deadline:
            return

        # elite set: small top fraction
        k = max(4, min(pop_size, 6 + pop_size // 6))
        elite_idx = topk_indices(fits, k)

        # compute per-dim mean/var over elites (diagonal covariance estimate)
        means = [0.0] * dim
        for idx in elite_idx:
            x = pop[idx]
            for d in range(dim):
                means[d] += x[d]
        invk = 1.0 / float(len(elite_idx))
        for d in range(dim):
            means[d] *= invk

        vars_ = [0.0] * dim
        for idx in elite_idx:
            x = pop[idx]
            for d in range(dim):
                t = x[d] - means[d]
                vars_[d] += t * t
        for d in range(dim):
            vars_[d] *= invk

        # update sigma with smoothing; ensure bounds
        for d in range(dim):
            target = math.sqrt(max(0.0, vars_[d])) + 1e-12
            # keep some exploration; avoid collapse too early
            target = max(diag_sigma_min[d], min(diag_sigma_max[d], 1.15 * target))
            diag_sigma[d] = 0.75 * diag_sigma[d] + 0.25 * target
            if diag_sigma[d] < diag_sigma_min[d]:
                diag_sigma[d] = diag_sigma_min[d]
            elif diag_sigma[d] > diag_sigma_max[d]:
                diag_sigma[d] = diag_sigma_max[d]

        # sample
        # shrink over time (more exploit late)
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        shrink = 1.0 - 0.55 * min(1.0, tfrac)
        if shrink < 0.25:
            shrink = 0.25

        bx = best_x
        for _ in range(diag_samples):
            if time.time() >= deadline:
                return
            x = [0.0] * dim
            for d in range(dim):
                x[d] = bx[d] + random.gauss(0.0, shrink * diag_sigma[d])
            repair_reflect(x)
            f = safe_eval(x)
            if f < best:
                best = f
                best_x = list(x)
                last_improve_t = time.time()

            # also: if good, inject into population by replacing a worse individual (cheap)
            # pick a likely-worse index by a few samples
            j = random.randrange(pop_size)
            for __ in range(3):
                kidx = random.randrange(pop_size)
                if fits[kidx] > fits[j]:
                    j = kidx
            if f < fits[j]:
                pop[j] = x
                fits[j] = f

    # ---------------- main loop (JADE) ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # time-adaptive pbest rate
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        p_best_rate = 0.26 - 0.18 * min(1.0, tfrac)  # ~0.26 -> ~0.08
        if p_best_rate < 0.08:
            p_best_rate = 0.08

        # occasional polish + diagonal sampler
        if gen % polish_every == 0 and time.time() < deadline:
            bx, bf = polish_best(list(best_x), best)
            if bf < best:
                best, best_x = bf, list(bx)

        if gen % diag_every == 0 and time.time() < deadline:
            diag_sampler_update_and_sample()

        # stagnation handling
        if time.time() - last_improve_t > stagnate_time:
            # replace worst set (need a sort rarely)
            idx_sorted = list(range(pop_size))
            idx_sorted.sort(key=lambda i: fits[i])
            worst_k = max(3, pop_size // 6)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted[-1 - t]
                r = random.random()
                if r < 0.55:
                    x = list(best_x)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.14 * spans[d])
                    repair_reflect(x)
                elif r < 0.75:
                    x = opposite(best_x)
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

        # pbest pool without full sort
        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))
        pbest_pool = topk_indices(fits, p_num)

        # success trackers for JADE parameter update
        SF, SCR, dW = [], [], []

        # per-individual loop
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # sample CR, F (JADE)
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

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d])
            repair_reflect(v)

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CR) else xi[d]

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

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve_t = time.time()

        # update mu_F, mu_CR (JADE)
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

        # light best-jitter injection (very cheap exploitation)
        if gen % 7 == 0 and time.time() < deadline:
            inject = 2 if dim <= 40 else 1
            scale = 0.10 - 0.07 * min(1.0, tfrac)
            if scale < 0.012:
                scale = 0.012
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                j = random.randrange(pop_size)
                for __ in range(3):
                    kidx = random.randrange(pop_size)
                    if fits[kidx] > fits[j]:
                        j = kidx
                x = list(best_x)
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
