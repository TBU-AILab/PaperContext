import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Main upgrades vs your best (#3):
      - Stronger early coverage: scrambled Halton + LHS + opposition.
      - Better core: L-SHADE-style DE/current-to-pbest/1 with success-history memories + archive.
      - More reliable local exploitation: (1+1)-ES with 1/5 rule + short coordinate/pattern polish.
      - Time-aware scheduling: exploration early, exploitation late; restart when stagnating.
      - Safer objective evaluation: NaN/inf guard + midpoint pull; bounded by periodic reflection fold.
      - Faster evaluations: quantized cache to avoid duplicate calls.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return time.time() < deadline

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # ---------------- bounds ----------------
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
    mid = [lo[i] + 0.5 * span[i] for i in range(dim)]

    # periodic reflection (triangle-wave fold)
    def fold_inplace(x):
        for i in range(dim):
            s = span[i]
            if s <= 0.0:
                x[i] = lo[i]
                continue
            a = lo[i]
            v = x[i] - a
            v = v % (2.0 * s)
            if v > s:
                v = 2.0 * s - v
            x[i] = a + v
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---------------- RNG helpers ----------------
    def randn():
        # ~N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy():
        u = random.random()
        if u < 1e-12:
            u = 1e-12
        elif u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------------- caching / safe eval ----------------
    cache = {}
    cache_max = 16000
    q = 1_000_000  # quantization bins in [0,1]

    def key_of(x):
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0.0 else 1.0
            u = (x[i] - lo[i]) / s
            if u <= 0.0:
                k.append(0)
            elif u >= 1.0:
                k.append(q)
            else:
                k.append(int(u * q + 0.5))
        return tuple(k)

    def safe_eval(x):
        # x assumed folded already
        try:
            fx = float(func(x))
        except Exception:
            return 1e300
        if fx != fx or fx == float("inf") or fx == float("-inf"):
            # pull towards mid and retry once
            xt = [(x[i] + mid[i]) * 0.5 for i in range(dim)]
            fold_inplace(xt)
            try:
                fx2 = float(func(xt))
            except Exception:
                return 1e300
            if fx2 == fx2 and fx2 not in (float("inf"), float("-inf")):
                return fx2
            return 1e300
        return fx

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = safe_eval(x)
        cache[k] = fx
        if len(cache) > cache_max:
            # prune a random chunk
            kill = min(len(cache) // 6, 2800)
            keys = list(cache.keys())
            if 0 < kill < len(keys):
                for kk in random.sample(keys, k=kill):
                    cache.pop(kk, None)
        return fx

    # ---------------- low discrepancy init ----------------
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
    halton_shift = [random.random() for _ in range(dim)]

    def van_der_corput(i, base):
        v = 0.0
        denom = 1.0
        while i > 0:
            i, rem = divmod(i, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index):
        x = [0.0] * dim
        for d in range(dim):
            u = van_der_corput(index, primes[d])
            u = u + halton_shift[d]
            u -= math.floor(u)
            x[d] = lo[d] + u * span[d]
        return x

    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                s = span[d]
                if s <= 0.0:
                    x[d] = lo[d]
                else:
                    u = (perms[d][i] + random.random()) / float(n)
                    x[d] = lo[d] + u * s
            pts.append(x)
        return pts

    # ---------------- elites ----------------
    elite_k = max(10, min(40, 8 + dim // 2))
    elites = []  # (f, x)

    def add_elite(x, fx):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # ---------------- local refine: (1+1)-ES + coordinate polish ----------------
    def refine(x0, f0, budget=24):
        x = x0[:]
        f = f0

        # per-dim sigma
        sigma = [0.08 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        sig_min = [1e-14 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        sig_max = [0.35 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

        # Phase A: (1+1)-ES with 1/5th rule on subsets
        success = 0
        trials = 0
        itA = max(10, budget // 2)
        for _ in range(itA):
            if not time_left():
                break
            m = max(2, int(math.sqrt(dim)))
            idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)

            xt = x[:]
            heavy = (random.random() < 0.15)
            for j in idxs:
                if span[j] > 0.0:
                    z = cauchy() if heavy else randn()
                    xt[j] += sigma[j] * z
            fold_inplace(xt)
            ft = evaluate(xt)

            trials += 1
            if ft < f:
                x, f = xt, ft
                success += 1
                for j in idxs:
                    sigma[j] = min(sig_max[j], sigma[j] * 1.20)
            else:
                for j in idxs:
                    sigma[j] = max(sig_min[j], sigma[j] * 0.86)

            if trials >= 10:
                rate = success / float(trials)
                if rate > 0.2:
                    for j in range(dim):
                        sigma[j] = min(sig_max[j], sigma[j] * 1.06)
                else:
                    for j in range(dim):
                        sigma[j] = max(sig_min[j], sigma[j] * 0.92)
                trials = 0
                success = 0

        # Phase B: coordinate/pattern polish
        step = [0.02 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        itB = max(8, budget // 2)
        for _ in range(itB):
            if not time_left():
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if not time_left():
                    break
                if span[j] <= 0.0:
                    continue
                sj = step[j]
                if sj <= 1e-16 * (span[j] if span[j] > 0 else 1.0):
                    continue
                # try +/- step
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
                    step[j] = min(step[j] * 1.12, 0.22 * (span[j] if span[j] > 0 else 1.0))
            else:
                for j in range(dim):
                    step[j] *= 0.60

        return x, f

    # ---------------- initialization ----------------
    best_x = mid[:]
    fold_inplace(best_x)
    best = evaluate(best_x)
    add_elite(best_x, best)

    seed_pool = []

    # Halton + opposition
    hal_n = max(12, min(40, 10 + int(2.2 * math.sqrt(dim))))
    idx = 1
    for _ in range(hal_n):
        if not time_left():
            break
        x = halton_point(idx)
        fold_inplace(x)
        fx = evaluate(x)
        seed_pool.append((fx, x))
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

        xo = [lo[i] + hi[i] - x[i] for i in range(dim)]
        fold_inplace(xo)
        fxo = evaluate(xo)
        seed_pool.append((fxo, xo))
        add_elite(xo, fxo)
        if fxo < best:
            best, best_x = fxo, xo[:]
        idx += 1

    # LHS
    lhs_n = max(10, min(28, 8 + int(math.sqrt(dim))))
    for x in lhs_points(lhs_n):
        if not time_left():
            break
        fold_inplace(x)
        fx = evaluate(x)
        seed_pool.append((fx, x))
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

    seed_pool.sort(key=lambda t: t[0])

    # quick initial refine
    if time_left():
        bx, bf = refine(best_x, best, budget=18)
        if bf < best:
            best, best_x = bf, bx[:]
            add_elite(best_x, best)

    # ---------------- L-SHADE-ish DE ----------------
    NP = max(18, min(64, 12 + 2 * int(math.sqrt(dim)) + dim // 2))
    pop = []
    fit = []

    # seed from best seeds + jitter
    take = min(len(seed_pool), max(8, NP // 2))
    jrad = 0.07
    for i in range(take):
        if not time_left():
            break
        _, xb = seed_pool[i]
        x = xb[:]
        m = max(2, int(math.sqrt(dim)))
        idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)
        heavy = (random.random() < 0.15)
        for d in idxs:
            if span[d] > 0.0:
                z = cauchy() if heavy else randn()
                x[d] += jrad * span[d] * z
        fold_inplace(x)
        fx = evaluate(x)
        pop.append(x); fit.append(fx)
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

    while len(pop) < NP and time_left():
        x = rand_point()
        fold_inplace(x)
        fx = evaluate(x)
        pop.append(x); fit.append(fx)
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

    if not pop:
        return best

    archive = []
    arch_max = NP * 3

    # success-history memories
    H = max(6, min(24, 6 + dim // 8))
    M_F = [0.6] * H
    M_CR = [0.5] * H
    k_mem = 0

    p_best_rate = 0.20

    def sample_F(memF):
        for _ in range(12):
            v = memF + 0.10 * cauchy()
            if v > 0.0:
                return 1.0 if v > 1.0 else v
        return max(0.03, min(1.0, memF))

    def sample_CR(memCR):
        v = memCR + 0.10 * randn()
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def pick_distinct(exclude, k):
        res = []
        tries = 0
        while len(res) < k and tries < 120:
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

    no_improve = 0
    stagnate_after = 70 + 8 * dim
    refine_cooldown = 0

    while time_left():
        # pbest pool
        order = list(range(NP))
        order.sort(key=lambda ii: fit[ii])
        pN = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_pool = order[:pN]

        S_F, S_CR, S_df = [], [], []

        for ii in range(NP):
            if not time_left():
                break

            xi = pop[ii]
            fi = fit[ii]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            pb = pop[random.choice(pbest_pool)]
            r1 = pick_distinct(ii, 1)[0]
            x1 = pop[r1]

            if archive and random.random() < 0.5:
                x2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_distinct(ii, 1)[0]
                x2 = pop[r2]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (pb[d] - xi[d]) + F * (x1[d] - x2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            # anti-collapse micro-noise
            if random.random() < 0.10:
                m = max(1, int(math.sqrt(dim)))
                idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)
                amp = 0.0010 + 0.0040 * random.random()
                for d in idxs:
                    if span[d] > 0.0:
                        u[d] += amp * span[d] * randn()

            fold_inplace(u)
            fu = evaluate(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > arch_max:
                    del archive[random.randrange(len(archive))]

                pop[ii] = u
                fit[ii] = fu

                df = fi - fu
                if df < 0.0:
                    df = 0.0
                S_F.append(F)
                S_CR.append(CR)
                S_df.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    add_elite(best_x, best)
                    no_improve = 0

                    if refine_cooldown <= 0 and time_left():
                        bx, bf = refine(best_x, best, budget=22)
                        if bf < best:
                            best, best_x = bf, bx[:]
                            add_elite(best_x, best)
                        refine_cooldown = 10 + dim // 3
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if refine_cooldown > 0:
                refine_cooldown -= 1

        # update memories
        if S_F:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                weights = [1.0 / len(S_F)] * len(S_F)
            else:
                weights = [df / wsum for df in S_df]

            num = 0.0
            den = 0.0
            cr_m = 0.0
            for j in range(len(S_F)):
                wj = weights[j]
                fj = S_F[j]
                num += wj * fj * fj
                den += wj * fj
                cr_m += wj * S_CR[j]
            if den > 1e-18:
                M_F[k_mem] = num / den
            M_CR[k_mem] = cr_m

            # clamp
            if M_F[k_mem] < 0.05: M_F[k_mem] = 0.05
            if M_F[k_mem] > 0.95: M_F[k_mem] = 0.95
            if M_CR[k_mem] < 0.0: M_CR[k_mem] = 0.0
            if M_CR[k_mem] > 1.0: M_CR[k_mem] = 1.0

            k_mem = (k_mem + 1) % H

        # stagnation restart (reseed worst half around elites)
        if no_improve > stagnate_after and time_left():
            no_improve = 0

            if len(archive) > arch_max // 2:
                archive = random.sample(archive, k=arch_max // 2)

            sources = [x for (_, x) in elites[:min(len(elites), elite_k)]]
            order = list(range(NP))
            order.sort(key=lambda ii: fit[ii], reverse=True)

            k_re = max(3, NP // 2)
            for t in range(k_re):
                if not time_left():
                    break
                ii = order[t]
                if sources and random.random() < 0.88:
                    xnew = random.choice(sources)[:]
                    rad = 0.10 + 0.55 * random.random()
                    heavy = (random.random() < 0.18)
                    for d in range(dim):
                        if span[d] > 0.0:
                            z = cauchy() if heavy else randn()
                            xnew[d] += rad * span[d] * z
                else:
                    xnew = rand_point()

                fold_inplace(xnew)
                fnew = evaluate(xnew)
                pop[ii] = xnew
                fit[ii] = fnew
                add_elite(xnew, fnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # reset memories slightly exploratory
            for i in range(H):
                M_F[i] = 0.68
                M_CR[i] = 0.45
            refine_cooldown = 0

    return best
