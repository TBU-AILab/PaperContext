import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved hybrid time-bounded minimizer (self-contained, no external libs).

    Goal: outperform the best you reported (8.014...) by tightening local exploitation
    without sacrificing global exploration.

    Key improvements vs your current best (#3 / Powell-like + SHADE either-or):
      1) Adds a *reliable* 1D bracket+golden line-search, but only when a direction is
         proven improving (cheap, very effective).
      2) Upgrades local search into a "Powell/pattern + opportunistic line-search"
         routine, run on multiple elites with a time-aware budget.
      3) Adds a small late-stage *Nelder–Mead* finisher (bounded), which often squeezes
         extra improvements from already-good solutions.
      4) Keeps the strong global engine: L-SHADE-ish current-to-pbest/1 + archive,
         with either-or mutation and subspace crossover, plus gentler stall restarts.
      5) Fixes a common budget bug in earlier codes: local search previously "charged"
         eval budgets approximately; here we strictly count actual evaluations.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------------- RNG helpers ----------------
    _bm_has = False
    _bm_val = 0.0

    def randn():
        nonlocal _bm_has, _bm_val
        if _bm_has:
            _bm_has = False
            return _bm_val
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _bm_val = z1
        _bm_has = True
        return z0

    def cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- helpers ----------------
    def bounce_repair(x):
        # reflection folding in [lo,hi], then clamp
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                if y > w:
                    y = 2.0 * w - y
                xi = a + y
            if xi < a:
                xi = a
            elif xi > b:
                xi = b
            x[i] = xi
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def eval_point(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def opposition_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------------- low discrepancy seeding (scrambled Halton) ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def scrambled_halton_points(n):
        bases = first_primes(max(1, dim))
        shifts = [random.random() for _ in range(dim)]
        pts = []
        for k in range(1, n + 1):
            x = []
            for d in range(dim):
                u = (halton_value(k, bases[d]) + shifts[d]) % 1.0
                x.append(lo[d] + u * span_safe[d])
            pts.append(x)
        return pts

    # ---------------- eval time estimate ----------------
    def estimate_eval_time():
        k = 3
        ts = []
        for _ in range(k):
            if time.time() >= deadline:
                break
            x = rand_point()
            t1 = time.time()
            _ = eval_point(x)
            t2 = time.time()
            ts.append(max(1e-6, t2 - t1))
        if not ts:
            return 1e-3
        ts.sort()
        return ts[len(ts) // 2]

    eval_dt = estimate_eval_time()

    # ---------------- 1D line search (bracket + golden) with strict eval counting ----------------
    def _step_along(x, d_unit, a):
        y = [x[i] + a * d_unit[i] for i in range(dim)]
        return bounce_repair(y)

    def line_search(x0, f0, d, budget, step0):
        """
        Minimizes f(x0 + a*d_unit) starting from a=0, with a bracket then golden search.
        Returns (x_best, f_best, evals_used). Uses bounce repair.
        Only meaningful if d is non-zero.
        """
        if budget < 4:
            return x0, f0, 0

        norm = 0.0
        for i in range(dim):
            norm += d[i] * d[i]
        if norm <= 1e-30:
            return x0, f0, 0
        invn = 1.0 / math.sqrt(norm)
        d_unit = [d[i] * invn for i in range(dim)]

        used = 0

        # try +step0 then -step0 to find improving side
        a1 = step0
        x1 = _step_along(x0[:], d_unit, a1)
        f1 = eval_point(x1)
        used += 1
        if f1 >= f0:
            if used >= budget:
                return x0, f0, used
            a1 = -step0
            x1 = _step_along(x0[:], d_unit, a1)
            f1 = eval_point(x1)
            used += 1
            if f1 >= f0:
                return x0, f0, used

        # expand bracket
        a_prev, f_prev = 0.0, f0
        a_curr, f_curr = a1, f1
        grow = 1.8
        aL = None
        aR = None

        while used < budget and time.time() < deadline:
            a_next = a_curr + grow * (a_curr - a_prev)
            x_next = _step_along(x0[:], d_unit, a_next)
            f_next = eval_point(x_next)
            used += 1
            if f_next >= f_curr:
                aL, aR = (a_prev, a_next) if a_prev < a_next else (a_next, a_prev)
                break
            a_prev, f_prev = a_curr, f_curr
            a_curr, f_curr = a_next, f_next

        if aL is None:
            # no bracket found within budget; return best seen among (0, a_curr)
            if f_curr < f0:
                return _step_along(x0[:], d_unit, a_curr), f_curr, used
            return x0, f0, used

        # golden section inside [aL, aR]
        if used + 2 > budget:
            # no room to do golden properly; return best among current
            bestx, bestf = x0, f0
            if f_curr < bestf:
                bestx, bestf = _step_along(x0[:], d_unit, a_curr), f_curr
            return bestx, bestf, used

        phi = 0.5 * (3.0 - math.sqrt(5.0))
        c = aR - phi * (aR - aL)
        d_ = aL + phi * (aR - aL)

        xc = _step_along(x0[:], d_unit, c)
        xd = _step_along(x0[:], d_unit, d_)
        fc = eval_point(xc)
        fd = eval_point(xd)
        used += 2

        bestx, bestf = (xc, fc) if fc < fd else (xd, fd)
        if f_curr < bestf:
            bestx, bestf = _step_along(x0[:], d_unit, a_curr), f_curr

        while used < budget and time.time() < deadline and abs(aR - aL) > 1e-12:
            if fc < fd:
                aR = d_
                d_ = c
                fd = fc
                c = aR - phi * (aR - aL)
                xc = _step_along(x0[:], d_unit, c)
                fc = eval_point(xc)
                used += 1
                if fc < bestf:
                    bestx, bestf = xc, fc
            else:
                aL = c
                c = d_
                fc = fd
                d_ = aL + phi * (aR - aL)
                xd = _step_along(x0[:], d_unit, d_)
                fd = eval_point(xd)
                used += 1
                if fd < bestf:
                    bestx, bestf = xd, fd

        return bestx, bestf, used

    # ---------------- local search: pattern + opportunistic line searches ----------------
    def pattern_search_with_lines(x0, f0, step_frac, budget):
        """
        Coordinate exploratory moves + pattern move.
        When a direction improves, optionally spend a small line-search budget along it.
        Returns (x_best, f_best, evals_used).
        """
        if x0 is None or budget <= 0:
            return x0, f0, 0

        x = x0[:]
        fx = f0

        step = [max(1e-16, step_frac) * span_safe[i] for i in range(dim)]
        min_step = [1e-12 * span_safe[i] for i in range(dim)]

        used = 0
        no_improve = 0

        # small per-sweep line search budget cap to avoid burning everything
        while used < budget and time.time() < deadline:
            improved = False
            x_base = x[:]
            f_base = fx

            idx = list(range(dim))
            random.shuffle(idx)

            # exploratory
            for i in idx:
                if used >= budget or time.time() >= deadline:
                    break
                if step[i] <= min_step[i]:
                    continue

                si = step[i]
                best_local_f = fx
                best_local_x = None
                best_dir = None

                for sgn in (1.0, -1.0):
                    if used >= budget or time.time() >= deadline:
                        break
                    y = x[:]
                    y[i] += sgn * si
                    bounce_repair(y)
                    fy = eval_point(y)
                    used += 1
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y
                        best_dir = [0.0] * dim
                        best_dir[i] = sgn * si

                if best_local_x is not None:
                    x, fx = best_local_x, best_local_f
                    improved = True

                    # opportunistic line-search along successful coordinate direction
                    if best_dir is not None and used < budget and random.random() < 0.35:
                        ls_budget = min(max(6, 2 * dim), budget - used)
                        x2, f2, u2 = line_search(x, fx, best_dir, budget=ls_budget, step0=0.7)
                        used += u2
                        if f2 < fx:
                            x, fx = x2, f2

            # pattern move
            if improved and used < budget and time.time() < deadline:
                d = [x[i] - x_base[i] for i in range(dim)]
                y = [x[i] + 1.5 * d[i] for i in range(dim)]
                bounce_repair(y)
                fy = eval_point(y)
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    # line-search along pattern direction
                    if used < budget and random.random() < 0.55:
                        ls_budget = min(max(8, 2 * dim), budget - used)
                        x3, f3, u3 = line_search(x, fx, d, budget=ls_budget, step0=0.6)
                        used += u3
                        if f3 < fx:
                            x, fx = x3, f3

            # adapt step
            if fx < f_base:
                for i in range(dim):
                    step[i] = min(0.6 * span_safe[i], step[i] * 1.18)
                no_improve = 0
            else:
                for i in range(dim):
                    step[i] *= 0.5
                no_improve += 1

            if no_improve >= 2:
                if all(step[i] <= min_step[i] for i in range(dim)):
                    break

        return x, fx, used

    # ---------------- Nelder–Mead finisher (bounded) ----------------
    def nelder_mead(x0, f0, budget, init_frac):
        if dim <= 0 or budget < (dim + 2):
            return x0, f0, 0

        simplex = [x0[:]]
        fvals = [f0]
        used = 0

        # build simplex (dim additional points)
        for i in range(dim):
            if used >= budget or time.time() >= deadline:
                break
            x = x0[:]
            x[i] += init_frac * span_safe[i] * (1.0 if random.random() < 0.5 else -1.0)
            bounce_repair(x)
            simplex.append(x)
            fvals.append(eval_point(x))
            used += 1

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while used < budget and time.time() < deadline and len(simplex) >= 2:
            order = list(range(len(simplex)))
            order.sort(key=lambda k: fvals[k])
            simplex = [simplex[k] for k in order]
            fvals = [fvals[k] for k in order]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]

            m = len(simplex) - 1
            centroid = [0.0] * dim
            invm = 1.0 / max(1, m)
            for j in range(m):
                xj = simplex[j]
                for i in range(dim):
                    centroid[i] += xj[i]
            for i in range(dim):
                centroid[i] *= invm

            xr = [centroid[i] + alpha * (centroid[i] - worstx[i]) for i in range(dim)]
            bounce_repair(xr)
            fr = eval_point(xr)
            used += 1
            if used >= budget:
                break

            if fr < bestf:
                xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(dim)]
                bounce_repair(xe)
                fe = eval_point(xe)
                used += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                if fr < worstf:
                    xc = [centroid[i] + rho * (xr[i] - centroid[i]) for i in range(dim)]
                else:
                    xc = [centroid[i] - rho * (centroid[i] - worstx[i]) for i in range(dim)]
                bounce_repair(xc)
                fc = eval_point(xc)
                used += 1
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink
                    for j in range(1, len(simplex)):
                        if used >= budget or time.time() >= deadline:
                            break
                        xs = [bestx[i] + sigma * (simplex[j][i] - bestx[i]) for i in range(dim)]
                        bounce_repair(xs)
                        fs = eval_point(xs)
                        simplex[j], fvals[j] = xs, fs
                        used += 1

        kbest = 0
        for k in range(1, len(fvals)):
            if fvals[k] < fvals[kbest]:
                kbest = k
        return simplex[kbest], fvals[kbest], used

    # ---------------- initialization ----------------
    remaining = max(0.0, deadline - time.time())
    eval_budget = max(30, int(0.82 * remaining / max(eval_dt, 1e-9)))

    NP0 = int(18 + 4.5 * dim)
    NP0 = max(20, min(90, NP0))
    if eval_budget < 250:
        NP0 = max(12, min(NP0, 28))
    elif eval_budget < 600:
        NP0 = max(16, min(NP0, 45))

    NPmin = max(8, min(24, 6 + 2 * dim))

    n_seed = min(max(NP0, 3 * NP0), max(60, min(260, eval_budget // 3)))
    n_halton = max(2, int(0.70 * n_seed))
    n_rand = n_seed - n_halton

    seeds = scrambled_halton_points(n_halton)
    for _ in range(n_rand):
        seeds.append(rand_point())

    seeds2 = []
    for x in seeds:
        seeds2.append(x)
        seeds2.append(opposition_point(x))

    boundary_k = max(6, min(40, 2 * dim + 8))
    for _ in range(boundary_k):
        x = []
        for d in range(dim):
            r = random.random()
            if r < 0.34:
                u = (random.random() ** 2) * 0.02
                x.append(lo[d] + u * span_safe[d])
            elif r < 0.68:
                u = (random.random() ** 2) * 0.02
                x.append(hi[d] - u * span_safe[d])
            else:
                x.append(lo[d] + random.random() * span_safe[d])
        seeds2.append(x)

    best = float("inf")
    best_x = None
    scored = []
    for x in seeds2:
        if time.time() >= deadline:
            return best
        bounce_repair(x)
        fx = eval_point(x)
        scored.append((fx, x[:]))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    scored = scored[:NP0]
    pop = [x for (fx, x) in scored]
    fit = [fx for (fx, x) in scored]

    elite_max = max(4, min(14, 3 + dim))
    elite = [(fit[i], pop[i][:]) for i in range(min(elite_max, len(pop)))]

    # ---------------- SHADE memory ----------------
    H = 10
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    archive = []
    archive_max = NP0

    pmin = 2.0 / max(2, NP0)
    pmax = 0.30

    last_improve = time.time()
    stall_seconds = max(0.20, 0.15 * max_time)

    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, max_time)))

        # linear population reduction
        target_NP = int(round(NP0 - (NP0 - NPmin) * frac))
        if target_NP < NPmin:
            target_NP = NPmin
        if len(pop) > target_NP:
            order_idx = sorted(range(len(pop)), key=lambda i: fit[i])
            keep = set(order_idx[:target_NP])
            pop = [pop[i] for i in range(len(pop)) if i in keep]
            fit = [fit[i] for i in range(len(fit)) if i in keep]
            archive_max = max(target_NP, 8)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

        NP = len(pop)
        if NP < 4:
            return best

        # refresh elite
        if gen % 5 == 0:
            order_idx = sorted(range(NP), key=lambda i: fit[i])
            elite = [(fit[order_idx[k]], pop[order_idx[k]][:]) for k in range(min(elite_max, NP))]

        stalled = (time.time() - last_improve) > stall_seconds

        # time-aware local intensification (multi-elite)
        if best_x is not None and (gen <= 2 or gen % 8 == 0 or stalled or frac > 0.70):
            remaining = max(0.0, deadline - time.time())
            evals_avail = int(0.92 * remaining / max(eval_dt, 1e-9))

            # more local late, but capped
            local_budget = int((0.06 + 0.24 * frac + (0.10 if stalled else 0.0)) * evals_avail)
            local_budget = max(0, min(local_budget, 34 * (dim + 1)))

            if local_budget > 0 and elite:
                k_use = min(len(elite), 1 + (1 if frac < 0.45 else 2))
                each = max(10, local_budget // (k_use + 1))
                step_frac = max(1e-6, 0.11 * (1.0 - 0.82 * frac))

                for k in range(k_use):
                    if time.time() >= deadline:
                        return best
                    fx0, x0 = elite[k]
                    x1, f1, _ = pattern_search_with_lines(x0, fx0, step_frac=step_frac, budget=each)
                    if f1 < fx0:
                        elite[k] = (f1, x1[:])
                    if f1 < best:
                        best, best_x = f1, x1[:]
                        last_improve = time.time()

                # late-stage Nelder–Mead finisher
                if time.time() < deadline and (frac > 0.60 or stalled):
                    nm_budget = max(0, local_budget // 2)
                    if nm_budget >= dim + 2:
                        init_frac = max(1e-6, 0.05 * (1.0 - 0.85 * frac))
                        x2, f2, _ = nelder_mead(best_x, best, budget=nm_budget, init_frac=init_frac)
                        if f2 < best:
                            best, best_x = f2, x2[:]
                            last_improve = time.time()

        # pbest ordering
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = pmin + (pmax - pmin) * random.random()
        pcount = max(2, int(math.ceil(p * NP)))

        S_CR, S_F, S_df = [], [], []

        # DE evolve (either-or)
        for i in range(NP):
            if time.time() >= deadline:
                return best

            r = random.randrange(H)
            CRi = MCR[r] + 0.1 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            Fi = cauchy(MF[r], 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 10:
                Fi = cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            x_i = pop[i]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)

            pool_size = NP + len(archive)
            if pool_size <= 2:
                r3 = random.randrange(NP)
            else:
                r3 = i
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = random.randrange(pool_size)

            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_r3 = archive[r3 - NP] if r3 >= NP else pop[r3]

            use_ctp = random.random() < (0.72 - 0.30 * frac)
            if use_ctp:
                pbest_idx = order[random.randrange(pcount)]
                x_pbest = pop[pbest_idx]
                v = [x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r3[d]) for d in range(dim)]
            else:
                v = [x_r1[d] + Fi * (x_r2[d] - x_r3[d]) for d in range(dim)]

            # mild late best/1 pulse
            if best_x is not None and frac > 0.80 and random.random() < 0.12:
                v = [best_x[d] + Fi * (x_r1[d] - x_r3[d]) for d in range(dim)]

            # crossover (subspace sometimes)
            u = x_i[:]
            if dim > 0:
                if dim >= 10 and random.random() < 0.28:
                    m = max(2, int(0.35 * dim))
                    idxs = random.sample(range(dim), m)
                    for d in idxs:
                        if random.random() < CRi:
                            u[d] = v[d]
                    u[idxs[0]] = v[idxs[0]]
                else:
                    jrand = random.randrange(dim)
                    for d in range(dim):
                        if d == jrand or random.random() < CRi:
                            u[d] = v[d]

            bounce_repair(u)
            fu = eval_point(u)

            if fu <= fit[i]:
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                df = fit[i] - fu
                if df > 0.0:
                    S_CR.append(CRi)
                    S_F.append(Fi)
                    S_df.append(df)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve = time.time()

        # update SHADE memories
        if S_df:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                wsum = 1.0

            cr_new = 0.0
            for k in range(len(S_df)):
                cr_new += (S_df[k] / wsum) * S_CR[k]

            num = 0.0
            den = 0.0
            for k in range(len(S_df)):
                wk = S_df[k] / wsum
                fk = S_F[k]
                num += wk * fk * fk
                den += wk * fk
            f_new = (num / den) if den > 1e-18 else MF[mem_idx]

            MCR[mem_idx] = cr_new
            MF[mem_idx] = f_new
            mem_idx = (mem_idx + 1) % H

        # stall handling: replace redundant/worst points by jittering around elites/best
        if (time.time() - last_improve) > stall_seconds and time.time() < deadline and best_x is not None:
            order_desc = sorted(range(NP), key=lambda i: fit[i], reverse=True)
            m = max(2, int(0.30 * NP))

            for idx in order_desc[:m]:
                if time.time() >= deadline:
                    return best

                center = best_x
                if elite and random.random() < 0.60:
                    center = elite[random.randrange(len(elite))][1]

                y = center[:]
                rad = (0.10 + 0.28 * abs(cauchy(0.0, 1.0))) * (1.0 - 0.55 * frac)
                if rad > 0.90:
                    rad = 0.90
                for d in range(dim):
                    y[d] += (random.random() * 2.0 - 1.0) * rad * span_safe[d]
                bounce_repair(y)

                fy = eval_point(y)
                pop[idx] = y
                fit[idx] = fy
                if fy < best:
                    best, best_x = fy, y[:]
                    last_improve = time.time()

            last_improve = time.time()

    return best
