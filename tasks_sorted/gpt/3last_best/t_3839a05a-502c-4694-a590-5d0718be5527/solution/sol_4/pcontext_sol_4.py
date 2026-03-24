import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over your current best (JADE/SHADE-like DE + simple refine):
      1) Better boundary handling: *periodic reflection* (fold) keeps large jumps usable.
      2) Smarter local search: embedded *SPSA-like* gradient sign steps + coordinate polish.
      3) Multi-strategy generation: mixes
           - current-to-pbest/1 (exploitation)
           - rand/1 (exploration)
           - best/2 (escaping local basins)
         with probabilities adapted by recent success.
      4) Lightweight "surrogate-like" reuse: cache + occasional neighbor reuse.
      5) More robust stagnation recovery: partial reinit + sigma kick + archive refresh.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return time.time() < deadline

    if dim <= 0:
        return float(func([]))

    # --- bounds ---
    lo = [0.0] * dim
    hi = [0.0] * dim
    span = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
        span[i] = b - a

    # --- robust fold (periodic reflection) into bounds ---
    # If x outside [lo,hi], reflect across boundaries repeatedly (like triangle wave).
    def fold_inplace(x):
        for i in range(dim):
            a, b = lo[i], hi[i]
            s = span[i]
            if s <= 0.0:
                x[i] = a
                continue
            v = x[i] - a
            # map to [0, 2s)
            v = v % (2.0 * s)
            if v > s:
                v = 2.0 * s - v
            x[i] = a + v
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # --- random variates ---
    def randn():
        # ~N(0,1) via 12 uniforms
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy():
        u = random.random()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # --- cache (quantized) ---
    cache = {}
    cache_max = 12000

    def key_of(x):
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0 else 1.0
            # quantize ~1e-7 of range
            k.append(int(((x[i] - lo[i]) / s) * 1e7 + 0.5))
        return tuple(k)

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        if len(cache) > cache_max:
            # prune random chunk
            kill = min(len(cache) // 6, 2000)
            # avoid sample error if dict small
            keys = list(cache.keys())
            if kill > 0 and kill < len(keys):
                for kk in random.sample(keys, k=kill):
                    cache.pop(kk, None)
        return fx

    # --- Halton init (low discrepancy) ---
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(max(1, dim))

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            i, rem = divmod(i, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(index, primes[i])
            x[i] = lo[i] + u * span[i]
        return x

    # --- elite archive ---
    elite_k = max(8, min(28, 4 + dim // 2))
    elites = []  # (f, x)

    def add_elite(x, fx):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # --- local search: SPSA-like + coordinate polish ---
    def local_refine(x0, f0, steps=14):
        # Two-phase:
        #  (A) SPSA-style perturbations to get descent direction in O(1) evals/step
        #  (B) short coordinate pattern search to polish
        x = x0[:]
        f = f0

        # (A) SPSA-ish
        # step sizes tied to bounds; decrease over iterations
        base = [0.06 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        for t in range(steps):
            if not time_left():
                break
            # pick subset to reduce overhead in high dim
            m = max(2, int(math.sqrt(dim)))
            idxs = range(dim) if m >= dim else random.sample(range(dim), m)

            ck = 0.10 / (1.0 + 0.35 * t)
            ak = 0.12 / (1.0 + 0.25 * t)

            d = [0.0] * dim
            for j in idxs:
                # Rademacher +/-1
                d[j] = 1.0 if random.random() < 0.5 else -1.0

            xp = x[:]
            xm = x[:]
            for j in idxs:
                stepj = ck * base[j]
                xp[j] += stepj * d[j]
                xm[j] -= stepj * d[j]
            fold_inplace(xp)
            fold_inplace(xm)

            fp = evaluate(xp)
            fm = evaluate(xm)

            # gradient estimate along selected dims
            denom = 2.0
            improved = False
            xt = x[:]
            for j in idxs:
                # ghat ~ (fp - fm) / (2*c*delta)
                # delta is +/-1 -> division is multiply by delta
                ghat = ((fp - fm) / denom) * d[j]
                # gradient step
                xt[j] -= ak * base[j] * ghat
            fold_inplace(xt)
            ft = evaluate(xt)

            if ft < f:
                x, f = xt, ft
                improved = True
            else:
                # if SPSA step failed, try best of xp/xm as fallback
                if fp < f:
                    x, f = xp, fp
                    improved = True
                elif fm < f:
                    x, f = xm, fm
                    improved = True

            if improved:
                # mild shrink to stabilize near optimum
                for j in idxs:
                    base[j] *= 0.96
            else:
                # mild expand to jump out of flat regions
                for j in idxs:
                    base[j] = min(base[j] * 1.06, 0.30 * (span[j] if span[j] > 0 else 1.0))

        # (B) coordinate polish (very cheap)
        step = [0.02 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        for _ in range(max(6, steps // 2)):
            if not time_left():
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if span[j] <= 0:
                    continue
                sj = step[j]
                if sj <= 1e-16 * (span[j] if span[j] > 0 else 1.0):
                    continue
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[j] += sgn * sj
                    fold_inplace(xt)
                    ft = evaluate(xt)
                    if ft < f:
                        x, f = xt, ft
                        improved = True
                        break
                if improved:
                    break
            if improved:
                for j in range(dim):
                    step[j] = min(step[j] * 1.10, 0.20 * (span[j] if span[j] > 0 else 1.0))
            else:
                for j in range(dim):
                    step[j] *= 0.60

        return x, f

    # --- initialization: population ---
    NP = max(14, min(50, 8 + 2 * int(math.sqrt(dim)) + dim // 2))

    pop, fit = [], []

    # seed midpoint
    mid = [lo[i] + 0.5 * span[i] for i in range(dim)]
    fold_inplace(mid)
    best_x = mid[:]
    best = evaluate(best_x)
    add_elite(best_x, best)

    # fill with Halton + opposition + random
    idx = 1
    while len(pop) < NP and time_left():
        x = halton_point(idx)
        fx = evaluate(x)
        pop.append(x); fit.append(fx)
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

        xo = [lo[i] + hi[i] - x[i] for i in range(dim)]
        fold_inplace(xo)
        fxo = evaluate(xo)
        if len(pop) < NP:
            pop.append(xo); fit.append(fxo)
        add_elite(xo, fxo)
        if fxo < best:
            best, best_x = fxo, xo[:]

        if (idx & 3) == 0 and len(pop) < NP:
            xr = rand_point()
            fr = evaluate(xr)
            pop.append(xr); fit.append(fr)
            add_elite(xr, fr)
            if fr < best:
                best, best_x = fr, xr[:]
        idx += 1

    if not pop:
        return best

    # initial polish
    if time_left():
        bx, bf = local_refine(best_x, best, steps=10)
        if bf < best:
            best, best_x = bf, bx[:]
            add_elite(best_x, best)

    # --- Adaptive DE core with multi-strategy mixing ---
    mu_F = 0.65
    mu_CR = 0.55
    archive = []
    arch_max = NP * 3

    # strategy weights (adapt on success)
    # s0: current-to-pbest/1, s1: rand/1, s2: best/2
    w = [0.60, 0.25, 0.15]
    succ_w = [1e-9, 1e-9, 1e-9]  # avoid zeros

    p_best_rate = 0.22

    def sample_F():
        for _ in range(10):
            v = mu_F + 0.12 * cauchy()
            if v > 0.0:
                return min(1.0, v)
        return min(1.0, max(0.02, mu_F))

    def sample_CR():
        v = mu_CR + 0.10 * randn()
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def pick_idx(exclude):
        j = random.randrange(NP)
        if j == exclude:
            j = (j + 1) % NP
        return j

    def pick_distinct(exclude, k):
        res = []
        tries = 0
        while len(res) < k and tries < 80:
            j = random.randrange(NP)
            if j == exclude or j in res:
                tries += 1
                continue
            res.append(j)
        while len(res) < k:
            j = random.randrange(NP)
            if j != exclude:
                res.append(j)
        return res

    def choose_strategy():
        r = random.random() * (w[0] + w[1] + w[2])
        if r < w[0]:
            return 0
        if r < w[0] + w[1]:
            return 1
        return 2

    no_improve = 0
    stagnate_after = 80 + 9 * dim
    gen = 0

    while time_left():
        gen += 1

        # pbest pool
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        pN = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_pool = order[:pN]

        succ_F, succ_CR, succ_df = [], [], []
        strat_succ = [0.0, 0.0, 0.0]

        for i in range(NP):
            if not time_left():
                break

            xi = pop[i]
            fi = fit[i]
            F = sample_F()
            CR = sample_CR()

            s = choose_strategy()

            # Mutation
            if s == 0:
                # current-to-pbest/1 with archive option
                pb = pop[random.choice(pbest_pool)]
                r1 = pick_distinct(i, 1)[0]
                x1 = pop[r1]
                if archive and random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_distinct(i, 1)[0]
                    x2 = pop[r2]
                v = [xi[d] + F * (pb[d] - xi[d]) + F * (x1[d] - x2[d]) for d in range(dim)]

            elif s == 1:
                # rand/1 (exploration)
                a, b, c = pick_distinct(i, 3)
                xa, xb, xc = pop[a], pop[b], pop[c]
                v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            else:
                # best/2 (stronger basin escape when stuck)
                xb = best_x
                a, b, c, d2 = pick_distinct(i, 4)
                x1, x2, x3, x4 = pop[a], pop[b], pop[c], pop[d2]
                v = [xb[j] + F * (x1[j] - x2[j]) + 0.5 * F * (x3[j] - x4[j]) for j in range(dim)]

            # Crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            # anti-collapse micro-noise
            if random.random() < 0.10:
                m = max(1, int(math.sqrt(dim)))
                idxs = range(dim) if m >= dim else random.sample(range(dim), m)
                amp = (0.0015 + 0.008 * random.random())
                for d in idxs:
                    if span[d] > 0:
                        u[d] += amp * span[d] * randn()

            fold_inplace(u)
            fu = evaluate(u)

            if fu <= fi:
                # archive replaced
                archive.append(xi[:])
                if len(archive) > arch_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fit[i] = fu

                df = max(0.0, fi - fu)
                succ_F.append(F)
                succ_CR.append(CR)
                succ_df.append(df)
                strat_succ[s] += (df + 1e-12)

                if fu < best:
                    best, best_x = fu, u[:]
                    add_elite(best_x, best)
                    no_improve = 0

                    # opportunistic local refine on new best
                    if time_left():
                        bx, bf = local_refine(best_x, best, steps=8)
                        if bf < best:
                            best, best_x = bf, bx[:]
                            add_elite(best_x, best)
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # adapt mu_F, mu_CR (SHADE-style)
        if succ_F:
            wsum = sum(succ_df) if any(succ_df) else float(len(succ_df))
            if wsum <= 0.0:
                weights = [1.0 / len(succ_F)] * len(succ_F)
            else:
                weights = [(df / wsum) for df in succ_df]

            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for k in range(len(succ_F)):
                fk = succ_F[k]
                wk = weights[k]
                num += wk * fk * fk
                den += wk * fk
                cr_mean += wk * succ_CR[k]

            if den > 1e-12:
                mu_F = 0.86 * mu_F + 0.14 * (num / den)
            mu_CR = 0.86 * mu_CR + 0.14 * cr_mean

            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(1.0, max(0.0, mu_CR))

        # adapt strategy weights by recent success
        for s in range(3):
            succ_w[s] = 0.85 * succ_w[s] + 0.15 * strat_succ[s]
        tot = succ_w[0] + succ_w[1] + succ_w[2]
        if tot > 0:
            # keep some exploration floor
            w = [max(0.08, succ_w[0] / tot),
                 max(0.08, succ_w[1] / tot),
                 max(0.06, succ_w[2] / tot)]
            ssum = w[0] + w[1] + w[2]
            w = [wi / ssum for wi in w]

        # stagnation handling: partial restart around elites + global
        if no_improve > stagnate_after and time_left():
            no_improve = 0
            # refresh archive to avoid bias
            if len(archive) > arch_max // 2:
                archive = random.sample(archive, k=arch_max // 2)

            # reinit worst third
            order = list(range(NP))
            order.sort(key=lambda i: fit[i], reverse=True)
            k_re = max(2, NP // 3)

            for t in range(k_re):
                if not time_left():
                    break
                i = order[t]
                if elites and random.random() < 0.80:
                    _, xe = random.choice(elites)
                    xnew = xe[:]
                    # radius schedule: sometimes local, sometimes big kick
                    rad = (0.08 + 0.55 * random.random())
                    heavy = (random.random() < 0.15)
                    for d in range(dim):
                        if span[d] > 0:
                            z = cauchy() if heavy else randn()
                            xnew[d] += rad * span[d] * z
                else:
                    xnew = rand_point()

                fold_inplace(xnew)
                fnew = evaluate(xnew)
                pop[i] = xnew
                fit[i] = fnew
                add_elite(xnew, fnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # reset control params slightly toward exploration
            mu_F = 0.72
            mu_CR = 0.48
            w = [0.55, 0.30, 0.15]

    return best
