import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Design goal: outperform the prior CMA/NM hybrids by adding:
      - A robust global engine: Differential Evolution (DE) with
        *current-to-pbest/1* mutation, sinusoidal dither, and archive (JADE/SHADE-like),
        which is strong on bounded black-box problems.
      - A small elite archive + occasional "opposition" injection for diversity.
      - A cheap late-stage local polish: bounded coordinate pattern search.
      - Strict time checks.

    Returns:
      best objective value found within max_time.
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
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # ---- bounds handling: mirror (better than clip) ----
    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def repair(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---- Halton (for initial coverage) ----
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---- elite archive (small) ----
    elite_size = max(12, min(50, 14 + int(4.0 * math.sqrt(dim))))
    elites = []  # sorted list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    def best_from_elites():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # ---- initialization: Halton + random + opposition ----
    best = float("inf")
    best_x = None

    # population size for DE (time-friendly but robust)
    NP = max(18, min(90, 12 + 6 * int(math.sqrt(dim)) + 2 * dim))
    # keep NP reasonable for larger dims
    NP = min(NP, max(22, 5 * dim))

    pop = []
    fit = []

    init_n = NP
    for i in range(init_n):
        if now() >= deadline:
            return best

        if random.random() < 0.80:
            x = halton_point(hal_k)
            hal_k += 1
        else:
            x = rand_uniform_point()

        fx = evaluate(x)
        pop.append(x)
        fit.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition evaluation for ~half of initial points (time-aware)
        if now() >= deadline:
            return best
        if random.random() < 0.55:
            xo = [lows[d] + highs[d] - x[d] for d in range(dim)]
            repair(xo)
            fo = evaluate(xo)
            push_elite(fo, xo)
            if fo < best:
                best, best_x = fo, xo[:]
            # if opposition is better, replace in population sometimes
            if fo < fx and random.random() < 0.7:
                pop[-1] = xo
                fit[-1] = fo

    # ---- external archive for JADE-style diversity ----
    A = []          # list of vectors
    Amax = NP

    # ---- JADE-like parameter memories ----
    mu_F = 0.55
    mu_CR = 0.55

    def cauchy_rand(loc, scale):
        # loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def normal_rand(loc, scale):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return loc + scale * z

    def pick_index_excluding(n, exclude_set):
        # n small-ish; simple loop ok
        while True:
            r = random.randrange(n)
            if r not in exclude_set:
                return r

    # ---- cheap local polish (coordinate pattern) ----
    def polish(x, fx, rounds=2, max_coords=14):
        if dim == 0:
            return fx, x
        xbest = x[:]
        fbest = fx

        idxs = list(range(dim))
        idxs.sort(key=lambda i: spans_nz[i], reverse=True)
        idxs = idxs[:max(1, min(dim, max_coords))]

        # step schedule based on span (not sigma-dependent)
        base = 0.06
        for r in range(rounds):
            improved = False
            step_mul = base * (0.55 ** r)
            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = min(0.25 * spans_nz[i], max(1e-12, step_mul * spans_nz[i]))

                xp = xbest[:]
                xp[i] += delta
                repair(xp)
                fp = evaluate(xp)
                if fp < fbest:
                    # pattern
                    step = xp[i] - xbest[i]
                    xbest, fbest = xp, fp
                    xpp = xbest[:]
                    xpp[i] += step
                    repair(xpp)
                    fpp = evaluate(xpp)
                    if fpp < fbest:
                        xbest, fbest = xpp, fpp
                    improved = True
                    continue

                xm = xbest[:]
                xm[i] -= delta
                repair(xm)
                fm = evaluate(xm)
                if fm < fbest:
                    step = xm[i] - xbest[i]
                    xbest, fbest = xm, fm
                    xmm = xbest[:]
                    xmm[i] += step
                    repair(xmm)
                    fmm = evaluate(xmm)
                    if fmm < fbest:
                        xbest, fbest = xmm, fmm
                    improved = True

            if not improved:
                break
        return fbest, xbest

    # ---- main DE loop ----
    gen = 0
    no_best = 0
    last_best = best

    while now() < deadline:
        gen += 1

        # time-aware intensification trigger
        time_left = deadline - now()
        if time_left <= 0:
            break
        endgame = (time_left / float(max_time)) < 0.22 if max_time > 0 else True

        # choose p-best fraction (JADE)
        p = 0.10 + 0.10 * random.random()  # in [0.10, 0.20]
        p_num = max(2, int(p * NP))

        S_F = []
        S_CR = []

        # precompute best indices list for pbest selection
        idxs_sorted = list(range(NP))
        idxs_sorted.sort(key=lambda i: fit[i])
        pbest_pool = idxs_sorted[:p_num]

        for i in range(NP):
            if now() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            # sample control parameters
            # CR ~ N(mu_CR, 0.1) clipped [0,1]
            CR = normal_rand(mu_CR, 0.10)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # F ~ Cauchy(mu_F, 0.1) resample until >0, then cap at 1
            F = cauchy_rand(mu_F, 0.10)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = cauchy_rand(mu_F, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # slight sinusoidal dither (helps)
            F *= (0.90 + 0.20 * math.sin(2.0 * math.pi * random.random()))
            if F < 0.05:
                F = 0.05
            elif F > 1.0:
                F = 1.0

            # pick pbest
            pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
            x_pbest = pop[pbest_idx]

            # pick r1, r2 from P union A, excluding i and pbest for r1/r2 indices within pop
            # Implement by picking vectors directly:
            r1 = pick_index_excluding(NP, {i, pbest_idx})
            x_r1 = pop[r1]

            # union pool for r2
            union = pop + A
            union_n = len(union)
            # ensure union_n >= 2
            if union_n <= 1:
                r2_vec = pop[pick_index_excluding(NP, {i, pbest_idx, r1})]
            else:
                # try to avoid choosing xi itself by identity check
                for _ in range(10):
                    v = union[random.randrange(union_n)]
                    if v is not xi and v is not x_r1:
                        r2_vec = v
                        break
                else:
                    r2_vec = union[random.randrange(union_n)]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + F * (x_pbest[d] - xi[d]) + F * (x_r1[d] - r2_vec[d])

            # crossover: binomial, ensure at least one from v
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
            repair(u)

            fu = evaluate(u)

            if fu <= fi:
                # selection
                if len(A) < Amax:
                    A.append(xi[:])
                else:
                    A[random.randrange(Amax)] = xi[:]

                pop[i] = u
                fit[i] = fu
                push_elite(fu, u)

                S_F.append(F)
                S_CR.append(CR)

                if fu < best:
                    best = fu
                    best_x = u[:]
            else:
                # mild occasional opposition injection on failures (cheap diversity)
                if random.random() < 0.015 and now() < deadline:
                    xo = [lows[d] + highs[d] - xi[d] for d in range(dim)]
                    repair(xo)
                    fo = evaluate(xo)
                    if fo < fit[i]:
                        if len(A) < Amax:
                            A.append(pop[i][:])
                        else:
                            A[random.randrange(Amax)] = pop[i][:]
                        pop[i] = xo
                        fit[i] = fo
                        push_elite(fo, xo)
                        if fo < best:
                            best = fo
                            best_x = xo[:]

        # update parameter means (Lehmer mean for F; mean for CR)
        if S_F:
            # Lehmer mean: sum(F^2)/sum(F)
            s1 = sum(f * f for f in S_F)
            s2 = sum(S_F)
            if s2 > 0.0:
                mu_F = 0.9 * mu_F + 0.1 * (s1 / s2)
                if mu_F < 0.05:
                    mu_F = 0.05
                elif mu_F > 0.95:
                    mu_F = 0.95
        if S_CR:
            mu_CR = 0.9 * mu_CR + 0.1 * (sum(S_CR) / float(len(S_CR)))
            if mu_CR < 0.0:
                mu_CR = 0.0
            elif mu_CR > 1.0:
                mu_CR = 1.0

        # stagnation logic + endgame polish
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            no_best = 0
        else:
            no_best += 1

        if endgame or (gen % (6 + int(math.sqrt(dim))) == 0):
            if best_x is not None and now() < deadline:
                fp, xp = polish(best_x, best, rounds=1 if not endgame else 2, max_coords=14)
                push_elite(fp, xp)
                if fp < best:
                    best, best_x = fp, xp[:]
                    last_best = best
                    no_best = 0

        # if strongly stagnated, re-seed a fraction of worst individuals from elites / Halton
        if no_best >= (10 + int(2.0 * math.sqrt(dim))):
            no_best = 0
            # refresh worst q individuals
            q = max(2, NP // 6)
            idxs_sorted = list(range(NP))
            idxs_sorted.sort(key=lambda i: fit[i], reverse=True)
            worst = idxs_sorted[:q]

            for k in worst:
                if now() >= deadline:
                    break
                # pick seed: mostly from elites, else Halton
                if elites and random.random() < 0.85:
                    # biased to top
                    top = min(len(elites), 14)
                    idx = int((random.random() ** 2) * top)
                    xseed = elites[idx][1][:]
                    # jitter seed slightly
                    for d in range(dim):
                        if spans[d] > 0.0 and random.random() < 0.6:
                            xseed[d] += (0.08 * spans_nz[d]) * (2.0 * random.random() - 1.0)
                    repair(xseed)
                else:
                    xseed = halton_point(hal_k) if random.random() < 0.7 else rand_uniform_point()
                    hal_k += 1

                fseed = evaluate(xseed)
                pop[k] = xseed
                fit[k] = fseed
                push_elite(fseed, xseed)
                if fseed < best:
                    best, best_x = fseed, xseed[:]
                    last_best = best

    return best
