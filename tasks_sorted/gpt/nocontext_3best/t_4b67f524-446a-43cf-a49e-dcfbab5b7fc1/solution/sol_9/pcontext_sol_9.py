import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Goal: beat the current best (~29.0015) more reliably by combining:
      1) Strong global coverage (scrambled Halton + opposition + elite archive).
      2) A stronger main optimizer than the provided DE variant: L-SHADE / JADE-style DE
         with:
            - current-to-pbest/1
            - external archive
            - SUCCESS-HISTORY adaptation of F and CR (Lehmer mean for F)
            - linear population size reduction (L-SHADE idea) for faster late-stage refine
      3) Short, evaluation-efficient local search slices on the best elite:
         - adaptive coordinate + pair pattern search (trust-region)
      4) Restart logic when stagnating: reseed worst individuals from elites + Halton.

    Assumptions:
      - func accepts a Python list (or array-like) of length dim and returns float.
      - bounds is a list of (low, high) per dimension.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    fixed = [spans[i] <= 0.0 for i in range(dim)]
    span_max = max([s for s in spans] + [0.0])

    def now():
        return time.time()

    def evalf(x):
        return float(func(x))

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def reflect_into_bounds(x):
        # repeated reflection, works better than clamp for DE steps
        y = list(x)
        for i in range(dim):
            if fixed[i]:
                y[i] = lows[i]
                continue
            lo, hi = lows[i], highs[i]
            v = y[i]
            if lo == hi:
                y[i] = lo
                continue
            # reflect repeatedly if needed
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
            y[i] = v
        return y

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lows[i] if fixed[i] else lows[i] + random.random() * spans[i]
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = lows[i] if fixed[i] else (lows[i] + highs[i] - x[i])
        return y

    # ---------- scrambled Halton ----------
    def first_primes(k):
        primes = []
        c = 2
        while len(primes) < k:
            is_p = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    bases = first_primes(max(1, dim))
    scr = [random.random() for _ in range(dim)]

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton(idx):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = (vdc(idx + 1, bases[i]) + scr[i]) % 1.0
                x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite archive with coarse diversity hashing ----------
    elite_k = max(18, min(90, 24 + 2 * dim))
    elites = []  # list of (f, x)
    q_levels = 16 if dim <= 30 else (12 if dim <= 80 else 10)
    seen_cells = set()

    def cell_key(x):
        key = []
        for i in range(dim):
            if fixed[i] or spans[i] <= 0.0:
                key.append(0)
            else:
                u = (x[i] - lows[i]) / spans[i]
                b = int(u * q_levels)
                if b < 0:
                    b = 0
                elif b >= q_levels:
                    b = q_levels - 1
                key.append(b)
        return tuple(key)

    def push_elite(f, x):
        nonlocal elites
        ck = cell_key(x)
        if ck in seen_cells and elites:
            # skip near-duplicates unless competitive with best
            if f > elites[0][0] * 1.01 + 1e-12:
                return
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]
        seen_cells.clear()
        for ff, xx in elites:
            seen_cells.add(cell_key(xx))

    # ---------- local search: short trust-region pattern search ----------
    eps_floor = 1e-14 + 1e-12 * (span_max + 1.0)

    def local_search(x0, f0, time_end):
        x = x0[:]
        fx = f0

        frac = (now() - t0) / max(1e-12, float(max_time))
        base = (0.14 * (1.0 - 0.85 * frac) + 0.015)
        delta = [0.0] * dim
        for i in range(dim):
            delta[i] = 0.0 if fixed[i] else max(eps_floor, base * spans[i])

        if dim <= 60:
            active = list(range(dim))
        else:
            active = random.sample(range(dim), max(24, min(90, 12 + dim // 6)))

        def try_move(xc, idxs, steps):
            xn = xc[:]
            for k, i in enumerate(idxs):
                if not fixed[i]:
                    xn[i] += steps[k]
            xn = clamp(xn)
            fn = evalf(xn)
            return fn, xn

        it = 0
        while now() < time_end and it < (7 if dim <= 40 else 5):
            it += 1
            improved = False
            random.shuffle(active)

            # coordinate steps
            for i in active:
                if now() >= time_end:
                    break
                if fixed[i]:
                    continue
                di = delta[i]
                if di <= eps_floor:
                    continue

                fn, xn = try_move(x, [i], [di])
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    delta[i] = min(0.45 * spans[i], di * 1.22)
                    continue

                if now() >= time_end:
                    break
                fn, xn = try_move(x, [i], [-di])
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    delta[i] = min(0.45 * spans[i], di * 1.22)
                else:
                    delta[i] = max(eps_floor, di * 0.70)

            # random pairs
            pair_tries = 2 if dim <= 50 else 1
            for _ in range(pair_tries):
                if now() >= time_end or len(active) < 2:
                    break
                i, j = random.sample(active, 2)
                if fixed[i] and fixed[j]:
                    continue
                si = 0.0 if fixed[i] else delta[i]
                sj = 0.0 if fixed[j] else delta[j]
                if si <= eps_floor and sj <= eps_floor:
                    continue
                combos = [(si, sj), (si, -sj), (-si, sj), (-si, -sj)]
                random.shuffle(combos)
                for a, b in combos:
                    if now() >= time_end:
                        break
                    fn, xn = try_move(x, [i, j], [a, b])
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        if not fixed[i]:
                            delta[i] = min(0.45 * spans[i], max(delta[i], abs(a)) * 1.10)
                        if not fixed[j]:
                            delta[j] = min(0.45 * spans[j], max(delta[j], abs(b)) * 1.10)
                        break

            if not improved:
                small = 0
                for i in active:
                    if not fixed[i] and delta[i] <= eps_floor * 8.0:
                        small += 1
                if small >= max(1, int(0.7 * len(active))):
                    break

        return fx, x

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    hidx = 0

    init_n = max(90, min(1800, 120 + 40 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best
        x = rand_point() if (j % 19 == 0) else halton(hidx)
        if j % 19 != 0:
            hidx += 1
        x = clamp(x)
        fx = evalf(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        if now() >= deadline:
            return best
        xo = opposition(x)
        fo = evalf(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        best_x = rand_point()
        best = evalf(best_x)
        push_elite(best, best_x)

    # ---------- L-SHADE / JADE-like DE ----------
    def pick_distinct(exclude_set, n, pool_size):
        res = []
        tries = 0
        while len(res) < n and tries < 4000:
            tries += 1
            r = random.randrange(pool_size)
            if r in exclude_set or r in res:
                continue
            res.append(r)
        if len(res) < n:
            for r in range(pool_size):
                if r in exclude_set or r in res:
                    continue
                res.append(r)
                if len(res) == n:
                    break
        return res

    # population sizing
    NP0 = max(24, min(140, 18 + 6 * int(math.sqrt(dim)) + dim // 2))
    NPmin = max(6, min(25, 6 + dim // 10))

    # build initial pop using elites + halton/random
    pop = []
    pop_f = []
    while len(pop) < NP0 and now() < deadline:
        if elites and random.random() < 0.75:
            x = random.choice(elites)[1][:]
        else:
            x = halton(hidx) if random.random() < 0.75 else rand_point()
            hidx += 1
        x = clamp(x)
        fx = evalf(x)
        pop.append(x)
        pop_f.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

    if not pop:
        return best

    # external archive
    archive = []
    arch_max = len(pop)

    # success-history memory (SHADE)
    H = 8 if dim < 60 else 10
    MCR = [0.85] * H
    MF = [0.55] * H
    mem_idx = 0

    no_improve = 0
    restart_after = 140 + 20 * dim
    next_local = now() + 0.02

    gen = 0
    while now() < deadline:
        gen += 1
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # linear population reduction (L-SHADE)
        NP_target = int(round(NPmin + (NP0 - NPmin) * (1.0 - frac)))
        if NP_target < NPmin:
            NP_target = NPmin
        if NP_target < len(pop):
            # remove worst to shrink
            order = list(range(len(pop)))
            order.sort(key=lambda i: pop_f[i])
            keep = order[:NP_target]
            keep_set = set(keep)
            pop = [pop[i] for i in keep]
            pop_f = [pop_f[i] for i in keep]
            # shrink archive cap accordingly
            arch_max = max(8, len(pop))
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]

        NP = len(pop)
        if NP < 4:
            break

        # periodic local search slice
        if t >= next_local and elites and (deadline - t) > 1e-6:
            slice_len = min(deadline - t, (0.028 + 0.0015 * dim) * (1.15 - 0.55 * frac))
            time_end = t + max(0.0, slice_len)
            seed = elites[0][1][:]
            fseed = evalf(seed)
            if fseed < best:
                best, best_x = fseed, seed[:]
            fL, xL = local_search(seed, fseed, time_end)
            push_elite(fL, xL)
            if fL < best:
                best, best_x = fL, xL[:]
                no_improve = 0
            else:
                no_improve += 1
            next_local = now() + (0.06 + 0.0035 * dim) * (0.35 + 0.95 * frac)
            if now() >= deadline:
                break

        # pbest fraction
        p = 0.18 - 0.10 * frac
        if p < 0.06:
            p = 0.06
        pbest_count = max(2, int(p * NP))

        idxs = list(range(NP))
        idxs.sort(key=lambda i: pop_f[i])

        union = pop + archive
        union_size = len(union)

        # store successful parameters for memory update
        S_CR = []
        S_F = []
        dF = []  # fitness improvements for weights

        for i in range(NP):
            if now() >= deadline:
                break

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # generate CR ~ N(mu, 0.1) clipped
            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() + random.random() - 2.0) / 2.0
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # generate F ~ Cauchy(mu, 0.1) truncated to (0,1]
            # simple Cauchy via tan
            F = None
            for _ in range(12):
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))
                ftry = mu_f + 0.1 * c
                if ftry > 0.0:
                    F = ftry
                    break
            if F is None:
                F = mu_f if mu_f > 0.0 else 0.5
            if F > 1.0:
                F = 1.0

            # choose pbest
            pbest_idx = idxs[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pb = pop[pbest_idx]

            # r1 from pop
            r1 = pick_distinct({i, pbest_idx}, 1, NP)[0]

            # r2 from union (pop+archive), distinct where applicable
            r2 = None
            for _ in range(40):
                cand = random.randrange(union_size)
                if cand < NP:
                    if cand == i or cand == r1 or cand == pbest_idx:
                        continue
                r2 = cand
                break
            if r2 is None:
                r2 = random.randrange(union_size)

            x_r1 = pop[r1]
            x_r2 = union[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if fixed[d]:
                    v[d] = lows[d]
                else:
                    v[d] = x_i[d] + F * (x_pb[d] - x_i[d]) + F * (x_r1[d] - x_r2[d])
            v = reflect_into_bounds(v)

            # binomial crossover
            uvec = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if fixed[d]:
                    uvec[d] = lows[d]
                else:
                    if random.random() < CR or d == jrand:
                        uvec[d] = v[d]
            uvec = clamp(uvec)
            fu = evalf(uvec)

            if fu <= pop_f[i]:
                # archive update
                if len(archive) < arch_max:
                    archive.append(pop[i][:])
                else:
                    archive[random.randrange(arch_max)] = pop[i][:]

                # store success stats
                diff = pop_f[i] - fu
                if diff < 0.0:
                    diff = 0.0
                S_CR.append(CR)
                S_F.append(F)
                dF.append(diff)

                pop[i] = uvec
                pop_f[i] = fu
                push_elite(fu, uvec)

                if fu < best:
                    best, best_x = fu, uvec[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # update success-history memory (SHADE)
        if S_F:
            # weights proportional to improvement (fallback to uniform if all zero)
            wsum = sum(dF)
            if wsum <= 1e-300:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            # weighted mean for CR (arithmetic)
            mcr = 0.0
            for w, cr in zip(weights, S_CR):
                mcr += w * cr

            # weighted Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            mf = num / den if den > 1e-300 else MF[mem_idx]

            # smooth to avoid collapse
            if mcr < 0.0:
                mcr = 0.0
            if mcr > 1.0:
                mcr = 1.0
            if mf <= 0.0:
                mf = 0.1
            if mf > 1.0:
                mf = 1.0

            MCR[mem_idx] = 0.9 * MCR[mem_idx] + 0.1 * mcr
            MF[mem_idx] = 0.9 * MF[mem_idx] + 0.1 * mf
            mem_idx = (mem_idx + 1) % H

        # keep archive in check
        if len(archive) > arch_max:
            random.shuffle(archive)
            archive = archive[:arch_max]

        # stagnation restart: reseed worst fraction
        if no_improve >= restart_after and now() < deadline:
            no_improve = 0
            if random.random() < 0.5:
                archive = []

            NP = len(pop)
            order = list(range(NP))
            order.sort(key=lambda k: pop_f[k], reverse=True)
            k_reseed = max(2, NP // 3)

            for kk in range(k_reseed):
                if now() >= deadline:
                    break
                ii = order[kk]

                if elites and random.random() < 0.88:
                    base = (elites[0][1] if random.random() < 0.6 else random.choice(elites)[1])[:]
                    jitter = (0.16 * (1.0 - 0.85 * frac) + 0.02)
                    xnew = base[:]
                    for d in range(dim):
                        if fixed[d]:
                            xnew[d] = lows[d]
                        else:
                            xnew[d] += (random.random() * 2.0 - 1.0) * jitter * spans[d]
                    xnew = clamp(xnew)
                else:
                    xnew = halton(hidx) if random.random() < 0.75 else rand_point()
                    hidx += 1
                    xnew = clamp(xnew)

                fnew = evalf(xnew)
                pop[ii] = xnew
                pop_f[ii] = fnew
                push_elite(fnew, xnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

    return best
