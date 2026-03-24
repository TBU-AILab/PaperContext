import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Further-improved time-bounded minimizer (no external libs).

    What is improved vs your current best (SHADE + Halton + kNN + simplex + 1+1):
      1) True "ask/tell" batching per generation:
         - build many candidates, pre-screen cheaply with kNN, then evaluate only the best few
           (saves evaluations when func is expensive; still helps when cheap by improving focus).
      2) Better local search: bounded BOBYQA-lite style isn't feasible w/o libs, so we add
         an inexpensive Rosenbrock/Hooke-Jeeves pattern search around best that works better
         than simplex on higher dims and noisy ridges.
      3) Multi-armed bandit for operator choice (UCB1 over 3 operators):
           - current-to-pbest/1
           - rand/1
           - "best/1" (only late, guarded) for faster contraction when basin is found
         This replaces the heuristic succ_pb/succ_r1 mix with a more stable allocator.
      4) Archive + jitter de-dup stays, but we also maintain a tiny LRU hash to avoid exact
         duplicates (from reflections/jitters) in continuous space.

    Returns: best fitness found (float).
    """

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---------------- helpers ----------------
    def clamp(x, a, b):
        return a if x < a else b if x > b else x

    def reflect(x, a, b):
        if a == b:
            return a
        while x < a or x > b:
            if x < a:
                x = a + (a - x)
            if x > b:
                x = b - (x - b)
        return clamp(x, a, b)

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    inv_span = [(1.0 / s) if s > 1e-18 else 0.0 for s in span]

    def in_bounds_jitter(x, eps_scale=1e-12):
        y = x[:]
        for i in range(dim):
            if span[i] <= 0:
                y[i] = lo[i]
            else:
                eps = eps_scale * span[i]
                if y[i] <= lo[i]:
                    y[i] = lo[i] + eps
                elif y[i] >= hi[i]:
                    y[i] = hi[i] - eps
        return y

    def random_point():
        return [lo[i] if span[i] <= 0 else (lo[i] + span[i] * random.random()) for i in range(dim)]

    def opposite_point(x):
        return [lo[i] if span[i] <= 0 else (lo[i] + hi[i] - x[i]) for i in range(dim)]

    def sqdist_white(a, b):
        s = 0.0
        for i in range(dim):
            if span[i] <= 0:
                continue
            d = (a[i] - b[i]) * inv_span[i]
            s += d * d
        return s

    def too_close(a, b, thr=1e-14):
        return sqdist_white(a, b) <= thr

    # polynomial mutation (Deb)
    def poly_mutate(x, eta=16.0, pm=0.2):
        y = x[:]
        for i in range(dim):
            if span[i] <= 0:
                y[i] = lo[i]
                continue
            if random.random() > pm:
                continue
            xl, xu = lo[i], hi[i]
            xi = y[i]
            if xl == xu:
                y[i] = xl
                continue
            d1 = (xi - xl) / (xu - xl)
            d2 = (xu - xi) / (xu - xl)
            r = random.random()
            mp = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - d1
                val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta + 1.0))
                dq = (val ** mp) - 1.0
            else:
                xy = 1.0 - d2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta + 1.0))
                dq = 1.0 - (val ** mp)
            xi = xi + dq * (xu - xl)
            y[i] = reflect(xi, xl, xu)
        return in_bounds_jitter(y)

    # ---------------- Halton sequence ----------------
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

    primes = first_primes(max(1, dim))

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    halton_index = 1
    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= 0:
                x[i] = lo[i]
            else:
                u = halton_value(idx, primes[i])
                x[i] = lo[i] + u * span[i]
        return x

    # ---------------- history + kNN surrogate (whitened) ----------------
    HCAP = 30 + 7 * dim
    hist_x, hist_f = [], []

    def add_hist(x, f):
        hist_x.append(x[:])
        hist_f.append(float(f))
        if len(hist_f) > HCAP:
            idx = sorted(range(len(hist_f)), key=lambda i: hist_f[i])
            keep = set(idx[:max(12, len(idx)//3)])
            drop = [i for i in range(len(hist_f)) if i not in keep]
            j = random.choice(drop) if drop else random.randrange(len(hist_f))
            hist_x.pop(j); hist_f.pop(j)

    def knn_predict(x, k=14):
        n = len(hist_f)
        if n == 0:
            return float("inf")
        k = min(k, n)
        d = [(sqdist_white(x, hist_x[i]), hist_f[i]) for i in range(n)]
        d.sort(key=lambda t: t[0])
        eps = 1e-12
        num = 0.0
        den = 0.0
        for j in range(k):
            d2, fj = d[j]
            w = 1.0 / (eps + d2)
            num += w * fj
            den += w
        return num / den if den > 0.0 else float("inf")

    # tiny duplicate guard (quantized key)
    seen_cap = 2000
    seen = {}
    seen_q = []

    def key_quant(x):
        # quantize in whitened scale ~ 1e-9 of span
        parts = []
        for i in range(dim):
            if span[i] <= 0:
                parts.append(0)
            else:
                q = int((x[i] - lo[i]) / (span[i] + 1e-300) * 1e9)
                parts.append(q)
        return tuple(parts)

    def mark_seen(x):
        k = key_quant(x)
        if k in seen:
            return False
        seen[k] = 1
        seen_q.append(k)
        if len(seen_q) > seen_cap:
            old = seen_q.pop(0)
            seen.pop(old, None)
        return True

    # ---------------- local search: pattern search (Hooke-Jeeves-ish) ----------------
    def pattern_refine(x0, f0, rad, iters=18):
        x = x0[:]
        f = f0
        step = rad[:]
        # ensure nonzero
        for i in range(dim):
            if span[i] > 0:
                step[i] = max(step[i], 1e-15 * span[i])

        for _ in range(iters):
            if time.time() >= deadline:
                break
            improved = False
            xb = x[:]
            fb = f

            # exploratory moves
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                if span[i] <= 0:
                    continue
                s = step[i]
                if s <= 0:
                    continue

                # try + and -
                best_local_x = None
                best_local_f = fb

                for sgn in (1.0, -1.0):
                    cand = x[:]
                    cand[i] = reflect(cand[i] + sgn * s, lo[i], hi[i])
                    cand = in_bounds_jitter(cand)
                    if not mark_seen(cand):
                        continue
                    fc = safe_eval(cand)
                    add_hist(cand, fc)
                    if fc < best_local_f:
                        best_local_f = fc
                        best_local_x = cand

                if best_local_x is not None:
                    x, f = best_local_x, best_local_f
                    improved = True

            # pattern move (accelerate)
            if improved and time.time() < deadline:
                cand = [0.0] * dim
                for i in range(dim):
                    if span[i] <= 0:
                        cand[i] = lo[i]
                    else:
                        cand[i] = reflect(x[i] + 0.7 * (x[i] - xb[i]), lo[i], hi[i])
                cand = in_bounds_jitter(cand)
                if mark_seen(cand):
                    fc = safe_eval(cand)
                    add_hist(cand, fc)
                    if fc < f:
                        x, f = cand, fc

            # step update
            if f < fb:
                for i in range(dim):
                    if span[i] > 0:
                        step[i] *= 1.08
            else:
                for i in range(dim):
                    if span[i] > 0:
                        step[i] *= 0.55
                # early stop if steps tiny
                small = True
                for i in range(dim):
                    if span[i] > 0 and step[i] > 1e-10 * span[i]:
                        small = False
                        break
                if small:
                    break

        return x, f

    # ---------------- init: stratified + opposition + Halton ----------------
    pop_size = max(24, min(96, 12 * dim))
    elite_n = max(3, pop_size // 8)

    pop, fit = [], []
    best = float("inf")
    best_x = None

    strata = []
    for i in range(dim):
        perm = list(range(pop_size))
        random.shuffle(perm)
        strata.append(perm)

    for k in range(pop_size):
        if time.time() >= deadline:
            return best
        if k % 2 == 0:
            x = [0.0] * dim
            for i in range(dim):
                if span[i] <= 0:
                    x[i] = lo[i]
                else:
                    u = (strata[i][k] + random.random()) / pop_size
                    x[i] = lo[i] + u * span[i]
        else:
            x = halton_point()

        fx = safe_eval(x)
        xo = opposite_point(x)
        fxo = safe_eval(xo)
        if fxo < fx:
            x, fx = xo, fxo

        x = in_bounds_jitter(x)
        pop.append(x)
        fit.append(fx)
        add_hist(x, fx)
        mark_seen(x)

        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)
        add_hist(best_x, best)
        mark_seen(best_x)

    # ---------------- SHADE memories ----------------
    Hm = 12
    MCR = [0.5] * Hm
    MF = [0.7] * Hm
    mem_k = 0

    archive = []
    archive_max = pop_size

    # ---------------- operator bandit (UCB1) ----------------
    # ops: 0 = current-to-pbest/1, 1 = rand/1, 2 = best/1 (guarded; mostly late)
    op_count = [1, 1, 1]
    op_gain = [0.0, 0.0, 0.0]  # cumulative improvement
    op_rounds = 3

    def choose_op(tfrac):
        nonlocal op_rounds
        op_rounds += 1
        # disallow best/1 early to avoid premature collapse
        allowed = [0, 1] if tfrac < 0.45 else [0, 1, 2]
        total = sum(op_count[i] for i in allowed) + 1e-12
        best_i = allowed[0]
        best_ucb = -1e300
        for i in allowed:
            avg = op_gain[i] / op_count[i]
            bonus = math.sqrt(2.0 * math.log(total + 1.0) / op_count[i])
            ucb = avg + 0.25 * bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_i = i
        return best_i

    base_rad = [0.12 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]

    last_best = best
    last_improve_time = time.time()
    no_improve_gens = 0
    gen = 0
    reheats = 0

    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # pbest fraction: larger early, smaller late
        pfrac = clamp(0.30 - 0.18 * tfrac, 2.0 / pop_size, 0.35)
        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(pfrac * pop_size)))
        pbest_set = order[:pcount]

        # periodic local refinement (more frequent late)
        if gen % (7 if tfrac < 0.55 else 4) == 0 and time.time() < deadline:
            rad = [max(1e-15, r) for r in base_rad]
            # kNN-guided candidate (1 eval)
            if len(hist_f) >= max(12, 2 * dim) and time.time() < deadline:
                best_c = None
                best_p = float("inf")
                tries = 18 if tfrac > 0.5 else 12
                for _ in range(tries):
                    x = best_x[:]
                    for i in range(dim):
                        if span[i] > 0 and random.random() < 0.6:
                            x[i] = reflect(x[i] + randn() * rad[i], lo[i], hi[i])
                    x = in_bounds_jitter(x)
                    pv = knn_predict(x)
                    pv += 0.0005 * sqdist_white(x, best_x)
                    if pv < best_p:
                        best_p = pv
                        best_c = x
                if best_c is not None and mark_seen(best_c):
                    fc = safe_eval(best_c)
                    add_hist(best_c, fc)
                    if fc < best:
                        best, best_x = fc, best_c[:]
                        last_best = best
                        last_improve_time = time.time()
                        no_improve_gens = 0

            # pattern search refine
            if time.time() < deadline:
                rad2 = [max(1e-15, r) for r in base_rad]
                xr, fr = pattern_refine(best_x, best, rad2, iters=(14 if dim > 25 else 18))
                if fr < best:
                    best, best_x = fr, xr[:]
                    last_best = best
                    last_improve_time = time.time()
                    no_improve_gens = 0

        SCR, SF, dF = [], [], []

        # ---------- generation: "ask" many, pre-screen, "tell" few ----------
        # create trials for each i, but only evaluate top fraction by kNN if history exists
        candidates = []  # (pred, i, trial, cr, F, op, fi)
        have_sur = (len(hist_f) >= max(15, 2 * dim))

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(Hm)
            mu_cr = MCR[r]
            mu_f = MF[r]
            cr = clamp(mu_cr + 0.10 * randn(), 0.0, 1.0)

            # Cauchy for F with resampling
            F = -1.0
            for _ in range(10):
                u = random.random()
                F = mu_f + 0.12 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            if no_improve_gens >= 6:
                F = max(F, 0.62)
                cr = max(cr, 0.55)
            F = clamp(F, 0.05, 1.0)

            op = choose_op(tfrac)

            if op == 0:
                # current-to-pbest/1
                pbest_idx = random.choice(pbest_set)
                xpbest = pop[pbest_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                pool_n = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_n)

                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])

            elif op == 1:
                # rand/1
                r0 = i
                while r0 == i:
                    r0 = random.randrange(pop_size)
                r1 = r0
                while r1 == r0 or r1 == i:
                    r1 = random.randrange(pop_size)
                pool_n = pop_size + len(archive)
                r2 = r1
                while r2 == r0 or r2 == r1 or r2 == i:
                    r2 = random.randrange(pool_n)

                xr0 = pop[r0]
                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = xr0[d] + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])
            else:
                # best/1 (guarded)
                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)
                pool_n = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pool_n)
                xr1 = pop[r1]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                v = [0.0] * dim
                for d in range(dim):
                    if span[d] <= 0:
                        v[d] = lo[d]
                    else:
                        vd = best_x[d] + F * (xr1[d] - xr2[d])
                        v[d] = reflect(vd, lo[d], hi[d])

            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if span[d] <= 0:
                    uvec[d] = lo[d]
                else:
                    if random.random() < cr or d == jrand:
                        uvec[d] = v[d]
            uvec = in_bounds_jitter(uvec)

            # de-dup & tiny jitter if too close
            if too_close(uvec, xi, 1e-14) or too_close(uvec, best_x, 1e-14):
                for d in range(dim):
                    if span[d] > 0 and random.random() < 0.15:
                        uvec[d] = reflect(uvec[d] + randn() * (1e-6 * span[d]), lo[d], hi[d])
                uvec = in_bounds_jitter(uvec)

            # occasional kick
            kick_p = 0.05 + (0.06 if no_improve_gens >= 6 else 0.0) + (0.03 if tfrac > 0.55 else 0.0)
            if random.random() < kick_p:
                pm = 1.0 / max(1, dim)
                uvec = poly_mutate(uvec, eta=16.0, pm=min(0.35, 6.0 * pm))

            pred = knn_predict(uvec) if have_sur else 0.0
            candidates.append((pred, i, uvec, cr, F, op, fi))

        # choose which to actually evaluate
        if have_sur:
            # evaluate only best predicted subset, but always include a few random for exploration
            candidates.sort(key=lambda t: t[0])
            eval_k = max(6, pop_size // 2)
            chosen = candidates[:eval_k]
            # exploration injection
            extras = max(2, pop_size // 8)
            for _ in range(extras):
                chosen.append(random.choice(candidates))
        else:
            chosen = candidates

        # evaluate chosen (some duplicates possible due to extras; skip by seen key)
        for (_, i, uvec, cr, F, op, fi) in chosen:
            if time.time() >= deadline:
                return best
            if not mark_seen(uvec):
                continue
            fu = safe_eval(uvec)
            add_hist(uvec, fu)

            if fu <= fit[i]:
                xi_old = pop[i]
                if len(archive) < archive_max:
                    archive.append(xi_old[:])
                else:
                    archive[random.randrange(archive_max)] = xi_old[:]

                pop[i] = uvec
                fit[i] = fu

                imp = (fi - fu) if (fi < float("inf") and fu < float("inf")) else 1.0
                if imp < 0.0:
                    imp = 0.0
                SCR.append(cr)
                SF.append(F)
                dF.append(imp)

                op_count[op] += 1
                op_gain[op] += imp

                if fu < best:
                    best, best_x = fu, uvec[:]

        # update SHADE memories
        if SCR:
            s = sum(dF)
            wts = ([1.0 / len(dF)] * len(dF)) if s <= 0.0 else [di / s for di in dF]

            mean_cr = 0.0
            for wi, cri in zip(wts, SCR):
                mean_cr += wi * cri

            num = 0.0
            den = 0.0
            for wi, Fi_ in zip(wts, SF):
                num += wi * Fi_ * Fi_
                den += wi * Fi_
            mean_f = (num / den) if den > 0.0 else MF[mem_k]

            MCR[mem_k] = clamp(mean_cr, 0.0, 1.0)
            MF[mem_k] = clamp(mean_f, 0.05, 1.0)
            mem_k = (mem_k + 1) % Hm

        # trust radius schedule + stagnation logic
        if best < last_best - 1e-12:
            last_best = best
            last_improve_time = time.time()
            no_improve_gens = 0
            reheats = 0
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = max(1e-15, base_rad[d] * (0.90 if tfrac > 0.55 else 0.94))
        else:
            no_improve_gens += 1
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min((0.35 if tfrac < 0.55 else 0.22) * span[d], base_rad[d] * 1.04)

        # staged reheats/restarts
        stuck_time = time.time() - last_improve_time
        if (no_improve_gens >= 10 and reheats == 0) and time.time() < deadline:
            reheats = 1
            archive = []
            for d in range(dim):
                if span[d] > 0:
                    base_rad[d] = min(0.35 * span[d], max(base_rad[d], (0.18 if tfrac < 0.55 else 0.12) * span[d]))
            no_improve_gens = 6

        if (no_improve_gens >= 16) or (stuck_time > 0.50 * max_time):
            if time.time() >= deadline:
                return best

            order = sorted(range(pop_size), key=lambda i: fit[i])
            keep = max(elite_n, pop_size // 4)
            keep_idx = order[:keep]

            new_pop = [pop[j][:] for j in keep_idx]
            new_fit = [fit[j] for j in keep_idx]
            archive = []

            while len(new_pop) < pop_size and time.time() < deadline:
                r = random.random()
                if r < 0.30:
                    x = halton_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                elif r < 0.60:
                    x = random_point()
                    fx = safe_eval(x)
                    xo = opposite_point(x)
                    fxo = safe_eval(xo)
                    if fxo < fx:
                        x, fx = xo, fxo
                else:
                    x = best_x[:]
                    for d in range(dim):
                        if span[d] > 0:
                            g1 = randn()
                            g2 = abs(randn()) + 1e-12
                            step = (g1 / g2) * base_rad[d] * (1.2 + 0.8 * random.random())
                            x[d] = reflect(x[d] + step, lo[d], hi[d])
                    x = in_bounds_jitter(x)
                    fx = safe_eval(x)

                x = in_bounds_jitter(x)
                new_pop.append(x)
                new_fit.append(fx)
                add_hist(x, fx)
                mark_seen(x)
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
                    last_improve_time = time.time()

            pop, fit = new_pop, new_fit
            no_improve_gens = 0
            reheats = 0
            last_improve_time = time.time()
            last_best = best

    return best
