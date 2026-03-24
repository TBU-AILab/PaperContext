import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Stricter improvement over the current best (~49.03): a time-bounded,
    self-contained hybrid that is usually stronger on hard bounded problems:

      - Global phase: low-discrepancy (scrambled Halton) + opposition + elite archive.
      - Main optimizer: "DE/JADE-like" Differential Evolution with:
          * current-to-pbest/1 mutation (fast convergence)
          * external archive (improves diversity / avoids premature convergence)
          * jDE-style self-adaptation of F and CR per-individual
          * bound handling via reflection (better than pure clamping for DE)
      - Local phase (short slices): robust coordinate/pair pattern search trust-region,
        only on best elite; very evaluation-efficient.
      - Restarts: when stagnating, reseed part of population from elites/halton.

    No external libraries required.
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
    span_max = max(spans) if spans else 0.0

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
        # reflection tends to preserve DE step directions better than clamp
        y = list(x)
        for i in range(dim):
            if fixed[i]:
                y[i] = lows[i]
                continue
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # repeated reflect if big overshoot
            # reflect around bounds: lo..hi
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
    elite_k = max(16, min(72, 18 + dim))
    elites = []  # list of (f, x)
    q_levels = 16 if dim <= 30 else 10
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
            # avoid duplicates unless close to best
            if f > elites[0][0] * 1.015 + 1e-12:
                return
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]
        seen_cells.clear()
        for ff, xx in elites:
            seen_cells.add(cell_key(xx))

    # ---------- initialization (coverage + opposition) ----------
    best = float("inf")
    best_x = None
    hidx = 0

    # Make init size moderate; DE needs a population anyway.
    init_n = max(80, min(1400, 140 + 35 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best
        if j % 17 == 0:
            x = rand_point()
        else:
            x = halton(hidx)
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

    # ---------- local search: short trust-region pattern search ----------
    eps_floor = 1e-14 + 1e-12 * (span_max + 1.0)

    def local_search(x0, f0, time_end):
        x = x0[:]
        fx = f0

        frac = (now() - t0) / max(1e-12, float(max_time))
        base = (0.16 * (1.0 - 0.85 * frac) + 0.02)
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
        while now() < time_end and it < (6 if dim <= 40 else 4):
            it += 1
            improved = False
            random.shuffle(active)
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
                    delta[i] = min(0.45 * spans[i], di * 1.25)
                    continue
                if now() >= time_end:
                    break
                fn, xn = try_move(x, [i], [-di])
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    delta[i] = min(0.45 * spans[i], di * 1.25)
                else:
                    delta[i] = max(eps_floor, di * 0.70)

            # a couple random pairs
            pair_tries = 2 if dim <= 40 else 1
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

    # ---------- Differential Evolution (JADE-ish + jDE self adaptation) ----------
    # Population size
    NP = max(18, min(80, 12 + 5 * int(math.sqrt(dim)) + dim // 4))
    # Keep NP even-ish
    if NP < 4:
        NP = 4

    # Build initial population mostly from elites + halton/random
    pop = []
    pop_f = []
    used = 0
    while len(pop) < NP and now() < deadline:
        if elites and random.random() < 0.70:
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
        used += 1

    if not pop:
        return best

    # Per-individual parameters (jDE)
    F_i = [0.5 for _ in range(NP)]
    CR_i = [0.9 for _ in range(NP)]
    tau1 = 0.10
    tau2 = 0.10

    # External archive A for diversity (classic JADE)
    archive = []
    arch_max = NP

    def pick_distinct(exclude, n, pool_size):
        # returns list of n indices in [0,pool_size) distinct and not in exclude set
        res = []
        tries = 0
        while len(res) < n and tries < 2000:
            tries += 1
            r = random.randrange(pool_size)
            if r in exclude:
                continue
            if r in res:
                continue
            res.append(r)
        if len(res) < n:
            # fallback deterministic fill
            for r in range(pool_size):
                if r in exclude or r in res:
                    continue
                res.append(r)
                if len(res) == n:
                    break
        return res

    next_local = now() + 0.03
    no_improve = 0
    restart_after = 120 + 18 * dim

    gen = 0
    while now() < deadline:
        gen += 1
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # periodic local search on the best elite (short slice)
        if t >= next_local and (deadline - t) > 1e-6:
            slice_len = min(deadline - t, (0.030 + 0.0016 * dim) * (1.10 - 0.55 * frac))
            time_end = t + max(0.0, slice_len)
            seed = elites[0][1][:] if elites else best_x[:]
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
            next_local = now() + (0.07 + 0.004 * dim) * (0.35 + 0.95 * frac)
            if now() >= deadline:
                break

        # build p-best set fraction (JADE): smaller late for exploitation
        p = 0.22 - 0.12 * frac
        if p < 0.08:
            p = 0.08
        pbest_count = max(2, int(p * NP))

        # create ranking indices
        idxs = list(range(NP))
        idxs.sort(key=lambda i: pop_f[i])

        # union pool for "r2" from pop + archive
        union = pop + archive
        union_size = len(union)

        # iterate individuals
        for i in range(NP):
            if now() >= deadline:
                break

            # jDE self-adapt
            if random.random() < tau1:
                # F in (0.1, 0.9) skewed to moderate
                F_i[i] = 0.1 + 0.8 * random.random()
            if random.random() < tau2:
                CR_i[i] = random.random()

            Fi = F_i[i]
            CR = CR_i[i]

            # choose pbest
            pbest_idx = idxs[random.randrange(pbest_count)]
            x_i = pop[i]
            x_pb = pop[pbest_idx]

            # choose r1 from pop (not i, not pbest ideally)
            r1 = pick_distinct({i, pbest_idx}, 1, NP)[0]

            # choose r2 from union (pop+archive), not i and not r1 and not pbest if in pop indices
            # if r2 points into pop, ensure distinct indices
            # easiest: retry random selection
            r2 = None
            for _ in range(30):
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
                    v[d] = x_i[d] + Fi * (x_pb[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])

            v = reflect_into_bounds(v)

            # binomial crossover
            u = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if fixed[d]:
                    u[d] = lows[d]
                    continue
                if random.random() < CR or d == jrand:
                    u[d] = v[d]

            u = clamp(u)
            fu = evalf(u)

            # selection + archive update
            if fu <= pop_f[i]:
                # add replaced to archive
                if len(archive) < arch_max:
                    archive.append(pop[i][:])
                else:
                    # random replace in archive
                    archive[random.randrange(arch_max)] = pop[i][:]
                pop[i] = u
                pop_f[i] = fu
                push_elite(fu, u)
                if fu < best:
                    best, best_x = fu, u[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # shrink archive if over
        if len(archive) > arch_max:
            random.shuffle(archive)
            archive = archive[:arch_max]

        # stagnation restart: reseed worst part of population
        if no_improve >= restart_after and now() < deadline:
            no_improve = 0
            # refresh archive too
            if random.random() < 0.5:
                archive = []

            # reseed worst 1/3
            idxs = list(range(NP))
            idxs.sort(key=lambda k: pop_f[k], reverse=True)
            k_reseed = max(2, NP // 3)
            for k in range(k_reseed):
                if now() >= deadline:
                    break
                ii = idxs[k]
                if elites and random.random() < 0.85:
                    # jitter around a good elite
                    base = (elites[0][1] if random.random() < 0.6 else random.choice(elites)[1])[:]
                    jitter = (0.18 * (1.0 - 0.85 * frac) + 0.02)
                    xnew = base[:]
                    for d in range(dim):
                        if fixed[d]:
                            xnew[d] = lows[d]
                        else:
                            xnew[d] += (random.random() * 2.0 - 1.0) * jitter * spans[d]
                    xnew = clamp(xnew)
                else:
                    xnew = halton(hidx) if random.random() < 0.7 else rand_point()
                    hidx += 1
                    xnew = clamp(xnew)
                fnew = evalf(xnew)
                pop[ii] = xnew
                pop_f[ii] = fnew
                push_elite(fnew, xnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

                # reset its params
                F_i[ii] = 0.3 + 0.6 * random.random()
                CR_i[ii] = random.random()

        # mild time-aware tightening of F (later => smaller steps)
        if gen % 7 == 0:
            targetF = 0.55 * (1.0 - 0.75 * frac) + 0.15
            for i in range(NP):
                if F_i[i] > targetF and random.random() < 0.25:
                    F_i[i] = 0.7 * F_i[i] + 0.3 * targetF

    return best
