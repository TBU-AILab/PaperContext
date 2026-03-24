import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ----------------- basic helpers -----------------
    def eval_f(x):
        return float(func(x))

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect_1d(v, lo, hi):
        # reflection keeps steps "alive" near boundaries
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    span = []
    for lo, hi in bounds:
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # -------------- Halton (scrambled) --------------
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_scrambled(index, base, perm):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * perm[i % base]
            i //= base
        return r

    primes = first_primes(dim)
    digit_perm = {}
    for b in set(primes):
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    k_hal = 1
    def halton_vec(k):
        x = []
        for i in range(dim):
            b = primes[i]
            u = halton_scrambled(k, b, digit_perm[b])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # -------------- elite archive (diverse) --------------
    archive = []  # list of (f, x)
    archive_cap = 48

    def norm_l1(a, b):
        d = 0.0
        for i in range(dim):
            d += abs(a[i] - b[i]) / span[i]
        return d / max(1, dim)

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        pruned = []
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 2e-4:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    # -------------- best-so-far init --------------
    best_x = rand_vec()
    best = eval_f(best_x)
    push_archive(best, best_x)

    # ============================================================
    # Strong local optimizer: Subspace Powell-like + coordinate + 2-point quad
    # (still derivative-free). Uses cached direction set, adapts steps.
    # ============================================================
    def line_search_reflect(x, f, d, step0, max_evals):
        """1D search along direction d with reflection, bracket + golden."""
        evals = 0
        # normalize direction
        norm = 0.0
        for j in range(dim):
            norm += d[j] * d[j]
        if norm <= 1e-30:
            return f, x, evals
        inv = 1.0 / math.sqrt(norm)
        dd = [dj * inv for dj in d]

        # initial probe points
        a = 0.0
        fa = f
        step = step0

        def point_at(alpha):
            y = x[:]
            for j in range(dim):
                lo, hi = bounds[j]
                y[j] = reflect_1d(y[j] + alpha * dd[j], lo, hi)
            return y

        # Try to find a downhill direction by testing +/- step
        y1 = point_at(step)
        f1 = eval_f(y1); evals += 1
        y2 = point_at(-step) if evals < max_evals and time.time() < deadline else None
        f2 = eval_f(y2) if y2 is not None else float('inf')
        if y2 is not None:
            evals += 1

        if f1 >= fa and f2 >= fa:
            return fa, x, evals  # no improvement

        # choose better direction
        if f2 < f1:
            step = -step
            b = step
            fb = f2
        else:
            b = step
            fb = f1

        # bracket by expanding
        c = b
        fc = fb
        grow = 1.8
        while evals < max_evals and time.time() < deadline:
            c = c * grow
            yc = point_at(c)
            fc = eval_f(yc); evals += 1
            if fc >= fb:
                break
            a, fa = b, fb
            b, fb = c, fc

        # Now we have roughly a < b < c with fb best, use golden section in [a,c]
        loA = min(a, c)
        hiA = max(a, c)
        phi = (math.sqrt(5.0) - 1.0) * 0.5  # 0.618...
        x1a = hiA - phi * (hiA - loA)
        x2a = loA + phi * (hiA - loA)

        yx1 = point_at(x1a)
        fx1 = eval_f(yx1); evals += 1
        if evals >= max_evals or time.time() >= deadline:
            # return best among evaluated
            best_local = fa
            best_alpha = 0.0
            cand = [(f, 0.0), (fb, b), (fx1, x1a)]
            if y2 is not None: cand.append((f2, -step0))
            cand.sort(key=lambda t: t[0])
            best_local, best_alpha = cand[0]
            return (best_local, point_at(best_alpha), evals)

        yx2 = point_at(x2a)
        fx2 = eval_f(yx2); evals += 1

        for _ in range(28):
            if evals >= max_evals or time.time() >= deadline:
                break
            if fx1 <= fx2:
                hiA = x2a
                x2a, fx2 = x1a, fx1
                x1a = hiA - phi * (hiA - loA)
                yx1 = point_at(x1a)
                fx1 = eval_f(yx1); evals += 1
            else:
                loA = x1a
                x1a, fx1 = x2a, fx2
                x2a = loA + phi * (hiA - loA)
                yx2 = point_at(x2a)
                fx2 = eval_f(yx2); evals += 1

            if abs(hiA - loA) <= 1e-10:
                break

        # pick best among {0, x1a, x2a, b}
        candidates = [(f, 0.0), (fb, b), (fx1, x1a), (fx2, x2a)]
        candidates.sort(key=lambda t: t[0])
        best_f, best_alpha = candidates[0]
        return best_f, point_at(best_alpha), evals

    def polish(x0, f0, max_evals):
        x = x0[:]
        f = f0
        evals = 0

        # direction set: start with coordinate axes, then learned directions
        dirs = []
        for j in range(dim):
            d = [0.0] * dim
            d[j] = 1.0
            dirs.append(d)

        # base step: smaller than before to reduce bouncing; adapted per dim
        base = 0.18 if dim <= 6 else (0.12 if dim <= 20 else 0.08)
        step0 = base * (sorted(span)[len(span)//2])

        # coordinate exploratory steps
        coord_step = [0.22 * s for s in span]
        min_step = [1e-12 * s for s in span]

        last_x = x[:]
        last_f = f

        while evals < max_evals and time.time() < deadline:
            improved = False

            # 1) Powell-like line searches over direction set
            for d in dirs:
                if evals >= max_evals or time.time() >= deadline:
                    break
                f2, x2, e = line_search_reflect(x, f, d, step0, max_evals - evals)
                evals += e
                if f2 < f:
                    x, f = x2, f2
                    improved = True

            # 2) Add a "net movement" direction (Powell update)
            move = [x[i] - last_x[i] for i in range(dim)]
            if sum(mi*mi for mi in move) > 1e-24:
                dirs = dirs[1:] + [move]  # drop oldest, add new

            last_x, last_f = x[:], f

            # 3) Small coordinate pattern search to catch narrow valleys
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if evals >= max_evals or time.time() >= deadline:
                    break
                sj = coord_step[j]
                if sj <= min_step[j]:
                    continue
                lo, hi = bounds[j]
                xj = x[j]

                xp = x[:]
                xp[j] = reflect_1d(xj + sj, lo, hi)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                else:
                    xm = x[:]
                    xm[j] = reflect_1d(xj - sj, lo, hi)
                    fm = eval_f(xm); evals += 1
                    if fm < f:
                        x, f = xm, fm
                        improved = True
                    else:
                        coord_step[j] *= 0.75

                if improved and evals + 2 <= max_evals and time.time() < deadline:
                    # quick quadratic refine along coordinate
                    c = x[j]
                    a = 0.35 * sj
                    xL = x[:]; xL[j] = reflect_1d(c - a, lo, hi)
                    fL = eval_f(xL); evals += 1
                    xR = x[:]; xR[j] = reflect_1d(c + a, lo, hi)
                    fR = eval_f(xR); evals += 1
                    fC = f
                    denom = (fL - 2.0 * fC + fR)
                    if abs(denom) > 1e-30:
                        delta = 0.5 * a * (fL - fR) / denom
                        if abs(delta) <= 2.0 * a:
                            xQ = x[:]
                            xQ[j] = reflect_1d(c + delta, lo, hi)
                            fQ = eval_f(xQ); evals += 1
                            if fQ < f:
                                x, f = xQ, fQ

            if improved:
                step0 *= 1.10
                if step0 > 0.5 * max(span):
                    step0 = 0.5 * max(span)
                for j in range(dim):
                    coord_step[j] *= 1.05
                    if coord_step[j] > 0.5 * span[j]:
                        coord_step[j] = 0.5 * span[j]
            else:
                step0 *= 0.70
                tiny = True
                for j in range(dim):
                    coord_step[j] *= 0.70
                    if coord_step[j] < min_step[j]:
                        coord_step[j] = min_step[j]
                    if coord_step[j] > 12.0 * min_step[j]:
                        tiny = False
                if tiny or step0 < 1e-14 * max(span):
                    break

        return f, x

    # ============================================================
    # Main optimizer: Multi-strategy DE + pbest + "trig" mix + heavy-tail inject
    # ============================================================

    # ---- initialization with LDS + opposition + small gaussian jitter ----
    init_budget = max(600, 190 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best

        if random.random() < 0.82:
            x = halton_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()

        # slight jitter (helps when Halton hits structure)
        if random.random() < 0.55:
            for d in range(dim):
                lo, hi = bounds[d]
                sd = 0.01 * span[d]
                x[d] = reflect_1d(x[d] + random.gauss(0.0, 1.0) * sd, lo, hi)

        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_archive(fx, x)

        if random.random() < 0.30 and time.time() < deadline:
            xo = [bounds[d][0] + bounds[d][1] - x[d] for d in range(dim)]
            fo = eval_f(xo)
            if fo < best:
                best, best_x = fo, xo[:]
            push_archive(fo, xo)

    # early polish of top elites
    for i in range(min(10, len(archive))):
        if time.time() >= deadline:
            return best
        f0, x0 = archive[i]
        f2, x2 = polish(x0, f0, max_evals=max(120, 24 * dim))
        if f2 < best:
            best, best_x = f2, x2[:]
        push_archive(f2, x2)

    # ---- DE population ----
    NP = max(36, min(160, 16 * dim))
    NPmin = max(14, 4 * dim)

    pop, pop_f = [], []
    F, CR = [], []
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i < len(archive) and random.random() < 0.80:
            x = archive[i][1][:]
        elif i % 2 == 0:
            x = halton_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()

        fx = eval_f(x)
        pop.append(x); pop_f.append(fx)
        F.append(0.45 + 0.35 * random.random())
        CR.append(0.10 + 0.80 * random.random())
        if fx < best:
            best, best_x = fx, x[:]

    last_improve_t = time.time()
    no_improve_gens = 0

    def sample_indices(n, banned):
        r = random.randrange(n)
        while r in banned:
            r = random.randrange(n)
        return r

    while True:
        if time.time() >= deadline:
            return best

        # rank
        order = sorted(range(len(pop)), key=lambda i: pop_f[i])
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            last_improve_t = time.time()
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        n = len(pop)
        new_pop = [None] * n
        new_pop_f = [None] * n

        # pbest pool (JADE-style)
        p = 0.12 if dim <= 10 else 0.18
        pcount = max(2, int(p * n))
        pbest_ids = order[:pcount]

        for i in range(n):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = pop_f[i]

            # jDE adaptation
            Fi, CRi = F[i], CR[i]
            if random.random() < 0.10:
                Fi = 0.1 + 0.9 * random.random()
            if random.random() < 0.10:
                CRi = random.random()

            # pick pbest
            pb = pop[random.choice(pbest_ids)]

            # choose r1, r2 distinct
            r1 = sample_indices(n, {i})
            r2 = sample_indices(n, {i, r1})

            x1 = pop[r1]
            x2 = pop[r2]

            # Strategy mix:
            #  - mostly current-to-pbest/1
            #  - sometimes trig mutation (diversity)
            use_trig = (random.random() < 0.12)

            v = [0.0] * dim
            if not use_trig:
                for d in range(dim):
                    v[d] = xi[d] + Fi * (pb[d] - xi[d]) + Fi * (x1[d] - x2[d])
            else:
                # trig mutation on three vectors (xi, x1, x2)
                f1, f2v, f3 = fi, pop_f[r1], pop_f[r2]
                s = abs(f1) + abs(f2v) + abs(f3) + 1e-12
                p1 = abs(f1) / s
                p2 = abs(f2v) / s
                p3 = abs(f3) / s
                for d in range(dim):
                    v[d] = (p1 * xi[d] + p2 * x1[d] + p3 * x2[d]) + \
                           Fi * ((x1[d] - x2[d]) + (x2[d] - xi[d]) + (xi[d] - x1[d]))

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    lo, hi = bounds[d]
                    u[d] = reflect_1d(v[d], lo, hi)
                else:
                    u[d] = xi[d]

            fu = eval_f(u)

            # selection
            if fu <= fi:
                new_pop[i] = u
                new_pop_f[i] = fu
                F[i] = Fi
                CR[i] = CRi
                push_archive(fu, u)
                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = time.time()
                    no_improve_gens = 0
            else:
                new_pop[i] = xi
                new_pop_f[i] = fi

        pop, pop_f = new_pop, new_pop_f

        # periodic heavy-tail injection around elites (escapes local minima)
        time_left = deadline - time.time()
        frac_left = max(0.0, time_left / max(1e-9, max_time))
        inject_prob = 0.10 + 0.35 * (1.0 - frac_left)
        if random.random() < inject_prob and archive:
            k_rep = max(2, len(pop) // 6)
            worst = sorted(range(len(pop)), key=lambda i: pop_f[i])[-k_rep:]
            for idx in worst:
                if time.time() >= deadline:
                    return best
                base = archive[random.randrange(min(len(archive), 10))][1]
                xnew = base[:]
                # Cauchy-like step: tan(pi*(u-0.5))
                for d in range(dim):
                    lo, hi = bounds[d]
                    scale = (0.01 + 0.08 * random.random()) * span[d]
                    u = random.random()
                    cstep = math.tan(math.pi * (u - 0.5))
                    xnew[d] = reflect_1d(xnew[d] + scale * cstep, lo, hi)
                fnew = eval_f(xnew)
                pop[idx] = xnew
                pop_f[idx] = fnew
                push_archive(fnew, xnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    last_improve_t = time.time()
                    no_improve_gens = 0

        # increasingly aggressive polishing near the end (exploitation)
        if random.random() < (0.18 + 0.65 * (1.0 - frac_left)):
            budget = max(80, int((14 + 34 * (1.0 - frac_left)) * dim))
            f2, x2 = polish(best_x, best, max_evals=budget)
            if f2 < best:
                best, best_x = f2, x2[:]
                push_archive(f2, x2)

        # stagnation restart + population focusing
        if (time.time() - last_improve_t > 0.22 * max_time) or (no_improve_gens >= 14):
            order = sorted(range(len(pop)), key=lambda i: pop_f[i])
            k_rep = max(3, len(pop) // 4)

            for kk in range(k_rep):
                if time.time() >= deadline:
                    return best
                idx = order[-1 - kk]
                if random.random() < 0.80 and archive:
                    base = archive[random.randrange(min(len(archive), 8))][1]
                    xnew = [0.0] * dim
                    for d in range(dim):
                        lo, hi = bounds[d]
                        sd = (0.015 + 0.10 * random.random()) * span[d]
                        xnew[d] = reflect_1d(base[d] + random.gauss(0.0, 1.0) * sd, lo, hi)
                else:
                    xnew = halton_vec(k_hal); k_hal += 1

                fnew = eval_f(xnew)
                pop[idx] = xnew
                pop_f[idx] = fnew
                F[idx] = 0.45 + 0.35 * random.random()
                CR[idx] = 0.10 + 0.80 * random.random()
                push_archive(fnew, xnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    last_improve_t = time.time()
                    no_improve_gens = 0

            # shrink population to focus compute later
            if len(pop) > NPmin:
                newN = max(NPmin, int(0.87 * len(pop)))
                order = sorted(range(len(pop)), key=lambda i: pop_f[i])
                keep = order[:newN]
                pop = [pop[i] for i in keep]
                pop_f = [pop_f[i] for i in keep]
                F = [F[i] for i in keep]
                CR = [CR[i] for i in keep]

            last_improve_t = time.time()
            no_improve_gens = 0
