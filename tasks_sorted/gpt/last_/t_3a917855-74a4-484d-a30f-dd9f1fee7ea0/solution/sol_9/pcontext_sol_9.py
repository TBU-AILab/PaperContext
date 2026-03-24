import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ----------------- helpers -----------------
    def eval_f(x):
        return float(func(x))

    def reflect_1d(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    span = []
    loB = []
    hiB = []
    for lo, hi in bounds:
        loB.append(lo); hiB.append(hi)
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def rand_vec():
        return [random.uniform(loB[i], hiB[i]) for i in range(dim)]

    # ----------------- Sobol-ish / Halton LDS (scrambled) -----------------
    # (Halton scrambled remains good and self-contained)
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
            x.append(loB[i] + u * (hiB[i] - loB[i]))
        return x

    # ----------------- archive -----------------
    archive = []  # list of (f, x)
    archive_cap = 64

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
        # keep best + diversity
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 1e-4:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    # ----------------- initial best -----------------
    best_x = rand_vec()
    best = eval_f(best_x)
    push_archive(best, best_x)

    # ============================================================
    # NEW: stronger local search = adaptive (1+1)-ES + occasional coordinate quad
    # Very cheap per iteration, great anytime behavior.
    # ============================================================
    def local_es(x0, f0, max_evals, sigma0):
        x = x0[:]
        f = f0
        evals = 0

        # per-dimension sigma
        sig = [max(1e-15 * span[i], sigma0 * span[i]) for i in range(dim)]
        # success rule parameters
        succ = 0
        tries = 0

        # cache for occasional coord refine
        coord_order = list(range(dim))

        while evals < max_evals and time.time() < deadline:
            # propose
            y = x[:]
            for i in range(dim):
                y[i] = reflect_1d(y[i] + random.gauss(0.0, 1.0) * sig[i], loB[i], hiB[i])

            fy = eval_f(y)
            evals += 1
            tries += 1

            if fy < f:
                x, f = y, fy
                succ += 1
                push_archive(f, x)
                # mild expand on success
                for i in range(dim):
                    sig[i] *= 1.08
            else:
                # shrink slightly on failure
                for i in range(dim):
                    sig[i] *= 0.985

            # 1/5th success adaptation every window
            if tries >= 24:
                rate = succ / float(tries)
                if rate > 0.22:
                    for i in range(dim):
                        sig[i] *= 1.18
                elif rate < 0.18:
                    for i in range(dim):
                        sig[i] *= 0.82
                succ = 0
                tries = 0

            # occasional coordinate quadratic touch (helps separable-ish problems)
            if evals + 3 <= max_evals and random.random() < 0.10 and time.time() < deadline:
                random.shuffle(coord_order)
                j = coord_order[0]
                c = x[j]
                a = max(1e-14 * span[j], 0.6 * sig[j])
                xL = x[:]; xL[j] = reflect_1d(c - a, loB[j], hiB[j])
                xR = x[:]; xR[j] = reflect_1d(c + a, loB[j], hiB[j])
                fL = eval_f(xL); fR = eval_f(xR)
                evals += 2
                fC = f
                denom = (fL - 2.0 * fC + fR)
                if abs(denom) > 1e-30:
                    delta = 0.5 * a * (fL - fR) / denom
                    if abs(delta) <= 2.0 * a:
                        xQ = x[:]
                        xQ[j] = reflect_1d(c + delta, loB[j], hiB[j])
                        fQ = eval_f(xQ); evals += 1
                        if fQ < f:
                            x, f = xQ, fQ
                            push_archive(f, x)

            # stop if sigmas are extremely tiny
            tiny = True
            for i in range(dim):
                if sig[i] > 1e-12 * span[i]:
                    tiny = False
                    break
            if tiny:
                break

        return f, x

    # ============================================================
    # Main: L-SHADE-like DE with "current-to-pbest/1" + external archive (A)
    # + linear population reduction + periodic ES polish
    # ============================================================

    # ---------- initialization ----------
    init_budget = max(500, 140 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best

        if random.random() < 0.85:
            x = halton_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()

        # opposition sometimes
        if random.random() < 0.28:
            xo = [loB[i] + hiB[i] - x[i] for i in range(dim)]
            fo = eval_f(xo)
            push_archive(fo, xo)
            if fo < best:
                best, best_x = fo, xo[:]

        # jitter
        if random.random() < 0.55:
            for i in range(dim):
                x[i] = reflect_1d(x[i] + random.gauss(0.0, 1.0) * (0.01 * span[i]), loB[i], hiB[i])

        fx = eval_f(x)
        push_archive(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

    # early polish of a few bests (cheap, strong)
    for i in range(min(8, len(archive))):
        if time.time() >= deadline:
            return best
        f0, x0 = archive[i]
        f2, x2 = local_es(x0, f0, max_evals=max(80, 16 * dim), sigma0=0.08)
        if f2 < best:
            best, best_x = f2, x2[:]
        push_archive(f2, x2)

    # ---------- DE settings ----------
    NP0 = max(40, min(180, 18 * dim))
    NPmin = max(12, 4 * dim)

    # success-history memories (SHADE)
    H = 10
    MF = [0.5] * H
    MCR = [0.5] * H
    k_mem = 0

    pop = []
    pop_f = []
    for i in range(NP0):
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
        if fx < best:
            best, best_x = fx, x[:]

    # external archive for difference vectors (as in JADE/SHADE)
    A = []
    Amax = NP0

    def rand_cauchy(loc, scale):
        # loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def ensure_in_bounds(vec):
        for i in range(dim):
            vec[i] = reflect_1d(vec[i], loB[i], hiB[i])

    last_best = best
    last_improve_t = time.time()

    while time.time() < deadline:
        n = len(pop)
        # linear population size reduction with time
        frac = (time.time() - t0) / max(1e-9, max_time)
        targetN = int(round(NP0 - (NP0 - NPmin) * frac))
        if n > targetN:
            # drop worst
            order = sorted(range(n), key=lambda i: pop_f[i])
            keep = order[:targetN]
            pop = [pop[i] for i in keep]
            pop_f = [pop_f[i] for i in keep]
            n = len(pop)
            if len(A) > Amax:
                A[:] = A[:Amax]

        order = sorted(range(n), key=lambda i: pop_f[i])
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            last_improve_t = time.time()

        # p-best set
        p = 0.11 + 0.14 * min(1.0, dim / 30.0)  # slightly larger p in higher dims
        pcount = max(2, int(p * n))
        pbest_ids = order[:pcount]

        new_pop = [None] * n
        new_pop_f = [None] * n

        SF = []
        SCR = []
        dF = []  # fitness improvements as weights

        union = pop + [a[1] for a in A]
        union_f = pop_f + [a[0] for a in A]

        for i in range(n):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = pop_f[i]

            r = random.randrange(H)
            # sample CR ~ N(MCR, 0.1), clipped
            CRi = MCR[r] + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0
            # sample F ~ Cauchy(MF, 0.1), resample until >0
            Fi = rand_cauchy(MF[r], 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = rand_cauchy(MF[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            # choose pbest
            xpbest = pop[random.choice(pbest_ids)]

            # choose r1 != i from pop
            r1 = random.randrange(n - 1)
            if r1 >= i:
                r1 += 1
            xr1 = pop[r1]

            # choose r2 from union, not equal to i or r1 if possible
            U = len(union)
            r2 = random.randrange(U)
            # a few attempts to avoid duplicates
            for _ in range(6):
                if r2 < n and (r2 == i or r2 == r1):
                    r2 = random.randrange(U)
                else:
                    break
            xr2 = union[r2]

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]
            ensure_in_bounds(u)

            fu = eval_f(u)

            if fu <= fi:
                new_pop[i] = u
                new_pop_f[i] = fu
                push_archive(fu, u)
                # add defeated parent to external archive
                A.append((fi, xi[:]))
                if len(A) > Amax:
                    # random removal
                    A.pop(random.randrange(len(A)))

                SF.append(Fi)
                SCR.append(CRi)
                dF.append(fi - fu)

                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve_t = time.time()
            else:
                new_pop[i] = xi
                new_pop_f[i] = fi

        pop, pop_f = new_pop, new_pop_f

        # update memories (weighted Lehmer mean for F)
        if SF:
            wsum = sum(dF) + 1e-30
            # for CR: weighted arithmetic mean
            mcr = 0.0
            for crv, w in zip(SCR, dF):
                mcr += (w / wsum) * crv
            # for F: weighted Lehmer mean
            num = 0.0
            den = 0.0
            for fv, w in zip(SF, dF):
                ww = w / wsum
                num += ww * fv * fv
                den += ww * fv
            mf = (num / (den + 1e-30))
            if mf <= 0.0: mf = 0.5
            if mf > 1.0: mf = 1.0

            MCR[k_mem] = mcr
            MF[k_mem] = mf
            k_mem = (k_mem + 1) % H

        # periodic ES polish, more often near end or on stagnation
        time_left = deadline - time.time()
        frac_left = max(0.0, time_left / max(1e-9, max_time))
        stagnating = (time.time() - last_improve_t) > 0.18 * max_time

        if (random.random() < (0.12 + 0.55 * (1.0 - frac_left))) or stagnating:
            # pick a seed: best or a near-best diverse elite
            if archive and random.random() < 0.45:
                seed = archive[random.randrange(min(10, len(archive)))][1][:]
                fseed = eval_f(seed)
            else:
                seed = best_x[:]
                fseed = best

            budget = max(60, int((10 + 26 * (1.0 - frac_left)) * dim))
            sigma0 = 0.10 if not stagnating else 0.18
            f2, x2 = local_es(seed, fseed, max_evals=budget, sigma0=sigma0)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()
            push_archive(f2, x2)

        # mild restart of a few worst if hard stagnation
        if (time.time() - last_improve_t) > 0.30 * max_time:
            n = len(pop)
            if n >= 6:
                order = sorted(range(n), key=lambda i: pop_f[i])
                krep = max(2, n // 6)
                for kk in range(krep):
                    if time.time() >= deadline:
                        return best
                    idx = order[-1 - kk]
                    if archive and random.random() < 0.8:
                        base = archive[random.randrange(min(8, len(archive)))][1]
                        xnew = base[:]
                        for d in range(dim):
                            xnew[d] = reflect_1d(xnew[d] + random.gauss(0.0, 1.0) * (0.06 * span[d]),
                                                loB[d], hiB[d])
                    else:
                        xnew = halton_vec(k_hal); k_hal += 1
                    fnew = eval_f(xnew)
                    pop[idx] = xnew
                    pop_f[idx] = fnew
                    push_archive(fnew, xnew)
                    if fnew < best:
                        best, best_x = fnew, xnew[:]
                        last_improve_t = time.time()

        last_best = best

    return best
