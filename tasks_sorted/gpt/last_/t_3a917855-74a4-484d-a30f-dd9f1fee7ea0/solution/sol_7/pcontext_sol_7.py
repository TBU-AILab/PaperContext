import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def reflect_1d(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # ---------- Halton (scrambled) ----------
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
    def halton_scrambled_vec(k):
        x = []
        for i in range(dim):
            b = primes[i]
            u = halton_scrambled(k, b, digit_perm[b])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------------- elite archive ----------------
    archive = []  # list of (f, x)
    archive_cap = 32

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

    # ---------------- best so far ----------------
    best_x = rand_vec()
    best = eval_f(best_x)
    push_archive(best, best_x)

    # ============================================================
    # Stronger local search: adaptive coordinate search + occasional
    # random direction + quadratic 1D refinement.
    # ============================================================
    def polish(x0, f0, max_evals):
        x = x0[:]
        f = f0
        evals = 0

        # start steps relative to span
        step = [0.25 * s for s in span]
        if dim <= 6:
            step = [0.35 * s for s in span]
        min_step = [1e-12 * s for s in span]

        # random permutation per call
        perm = list(range(dim))
        random.shuffle(perm)

        while evals < max_evals and time.time() < deadline:
            improved = False

            # coordinate pattern search
            for j in perm:
                if evals >= max_evals or time.time() >= deadline:
                    break

                sj = step[j]
                if sj <= min_step[j]:
                    continue
                lo, hi = bounds[j]
                xj = x[j]

                # try +/- step
                xp = x[:]
                xp[j] = reflect_1d(xj + sj, lo, hi)
                fp = eval_f(xp); evals += 1

                xm = None
                fm = float('inf')
                if evals < max_evals and time.time() < deadline:
                    xm = x[:]
                    xm[j] = reflect_1d(xj - sj, lo, hi)
                    fm = eval_f(xm); evals += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    improved = True

                    # quadratic refinement around current point along coordinate j
                    if evals + 2 <= max_evals and time.time() < deadline:
                        c = x[j]
                        a = 0.5 * sj

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
                                    improved = True
                else:
                    step[j] *= 0.72  # shrink if unproductive

            # random direction probes (small set)
            if evals < max_evals and time.time() < deadline:
                probes = 2 if dim <= 10 else 3
                for _ in range(probes):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    nnz = 1 if dim == 1 else (2 if dim < 12 else 3)
                    idxs = random.sample(range(dim), nnz)
                    med_step = sorted(step)[len(step)//2]
                    alpha = (0.35 + 0.9 * random.random()) * med_step
                    xt = x[:]
                    for j in idxs:
                        lo, hi = bounds[j]
                        xt[j] = reflect_1d(xt[j] + alpha * (1.0 if random.random() < 0.5 else -1.0), lo, hi)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved = True

            if improved:
                # gentle expansion
                for j in range(dim):
                    step[j] *= 1.08
                    if step[j] > 0.55 * span[j]:
                        step[j] = 0.55 * span[j]
            else:
                # global shrink, stop if tiny
                tiny = True
                for j in range(dim):
                    step[j] *= 0.68
                    if step[j] < min_step[j]:
                        step[j] = min_step[j]
                    if step[j] > 12.0 * min_step[j]:
                        tiny = False
                if tiny:
                    break

        return f, x

    # ============================================================
    # NEW main optimizer: Differential Evolution + optional "opposition"
    # + periodic restart + stronger exploitation schedule.
    # (Keeps it self-contained; no numpy.)
    # ============================================================

    # --- initialization with LDS + opposition ---
    init_budget = max(500, 160 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best
        if random.random() < 0.85:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()

        f = eval_f(x)
        if f < best:
            best, best_x = f, x[:]
        push_archive(f, x)

        # opposition point sometimes helps early
        if random.random() < 0.35 and time.time() < deadline:
            xo = []
            for d in range(dim):
                lo, hi = bounds[d]
                xo.append(lo + hi - x[d])
            fo = eval_f(xo)
            if fo < best:
                best, best_x = fo, xo[:]
            push_archive(fo, xo)

    # early intensive polish of a few elites
    for i in range(min(8, len(archive))):
        if time.time() >= deadline:
            return best
        f0, x0 = archive[i]
        f2, x2 = polish(x0, f0, max_evals=max(80, 18 * dim))
        if f2 < best:
            best, best_x = f2, x2
        push_archive(f2, x2)

    # ---- DE settings (jDE style self-adaptation, plus current-to-best) ----
    NP = max(30, min(140, 14 * dim))
    NPmin = max(12, 4 * dim)

    pop = []
    pop_f = []
    F = []
    CR = []

    # seed population from archive + halton/random
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i < len(archive) and random.random() < 0.75:
            x = archive[i][1][:]
        elif i % 2 == 0:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        pop_f.append(fx)
        F.append(0.5 + 0.3 * random.random())
        CR.append(0.2 + 0.6 * random.random())
        if fx < best:
            best, best_x = fx, x[:]

    last_best = best
    last_improve_t = time.time()

    while True:
        if time.time() >= deadline:
            return best

        # rank indices by fitness
        order = sorted(range(len(pop)), key=lambda i: pop_f[i])
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            last_improve_t = time.time()

        new_pop = [None] * len(pop)
        new_pop_f = [None] * len(pop)

        for i in range(len(pop)):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = pop_f[i]

            # jDE adaptation
            Fi = F[i]
            CRi = CR[i]
            if random.random() < 0.12:
                Fi = 0.1 + 0.9 * random.random()
            if random.random() < 0.12:
                CRi = random.random()

            # choose r1, r2, r3 distinct from i
            n = len(pop)
            r1 = i
            while r1 == i:
                r1 = random.randrange(n)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(n)
            r3 = i
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(n)

            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]

            # current-to-best/1 with a small random component:
            # v = xi + Fi*(best - xi) + Fi*(x1 - x2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (best_x[d] - xi[d]) + Fi * (x1[d] - x2[d])

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

            if fu <= fi:
                new_pop[i] = u
                new_pop_f[i] = fu
                F[i] = Fi
                CR[i] = CRi
                push_archive(fu, u)
                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = time.time()
            else:
                new_pop[i] = xi
                new_pop_f[i] = fi

        pop, pop_f = new_pop, new_pop_f

        # periodic polishing with increasing intensity as time passes
        time_left = deadline - time.time()
        frac_left = max(0.0, time_left / max(1e-9, max_time))
        if random.random() < (0.20 + 0.55 * (1.0 - frac_left)):
            budget = max(60, int((10 + 25 * (1.0 - frac_left)) * dim))
            f2, x2 = polish(best_x, best, max_evals=budget)
            if f2 < best:
                best, best_x = f2, x2
                push_archive(f2, x2)

        # restart / injection if stagnating
        if time.time() - last_improve_t > 0.18 * max_time:
            # replace worst 25% by samples around elites + a few global
            order = sorted(range(len(pop)), key=lambda i: pop_f[i])
            k_rep = max(2, len(pop) // 4)
            for kk in range(k_rep):
                if time.time() >= deadline:
                    return best
                idx = order[-1 - kk]

                if random.random() < 0.8 and archive:
                    base = archive[random.randrange(min(len(archive), 8))][1]
                    xnew = [0.0] * dim
                    for d in range(dim):
                        lo, hi = bounds[d]
                        sd = (0.02 + 0.10 * random.random()) * span[d]
                        xnew[d] = reflect_1d(base[d] + random.gauss(0.0, 1.0) * sd, lo, hi)
                else:
                    xnew = halton_scrambled_vec(k_hal); k_hal += 1

                fnew = eval_f(xnew)
                pop[idx] = xnew
                pop_f[idx] = fnew
                F[idx] = 0.5 + 0.3 * random.random()
                CR[idx] = 0.2 + 0.6 * random.random()
                push_archive(fnew, xnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    last_improve_t = time.time()

            # shrink population a bit after stagnation (focus compute)
            if len(pop) > NPmin:
                newN = max(NPmin, int(0.85 * len(pop)))
                order = sorted(range(len(pop)), key=lambda i: pop_f[i])
                keep = order[:newN]
                pop = [pop[i] for i in keep]
                pop_f = [pop_f[i] for i in keep]
                F = [F[i] for i in keep]
                CR = [CR[i] for i in keep]

            last_improve_t = time.time()
            last_best = best
