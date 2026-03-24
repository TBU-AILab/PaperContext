import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Key upgrades vs your current best:
      1) Better robustness across landscapes:
         - Adds CMA-ES-like diagonal adaptation (very cheap) inside local refine (no matrices).
         - Uses a true trust-radius schedule for the refine stage (expands/contracts based on success).

      2) Stronger global search:
         - Keeps L-SHADE/current-to-pbest/1 core + archive + pop reduction
         - Adds occasional "two-difference" mutation (DE/rand/2-like) when diversity collapses.
         - Adds mild explicit diversity injection based on population spread (not only stagnation counter).

      3) Smarter restarts:
         - When stagnating, reseed worst using a mixture of:
              * elite-centered heavy-tail
              * low-discrepancy (Halton) far samples
              * opposition about best and about mid
         - Restarts are time-fraction aware.

      4) Evaluation efficiency:
         - Quantized cache kept, but uses "soft hashing" (lower q) early, higher q late.

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

    # ---------- bounds ----------
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

    # periodic reflection fold (triangle wave)
    def fold_inplace(x):
        for i in range(dim):
            s = span[i]
            if s <= 0.0:
                x[i] = lo[i]
                continue
            a = lo[i]
            v = (x[i] - a) % (2.0 * s)
            if v > s:
                v = 2.0 * s - v
            x[i] = a + v
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---------- RNG helpers ----------
    def randn():
        # approx N(0,1)
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

    # ---------- time fraction ----------
    def time_frac():
        now = time.time()
        denom = max(1e-12, (deadline - t0))
        tf = (now - t0) / denom
        if tf < 0.0:
            return 0.0
        if tf > 1.0:
            return 1.0
        return tf

    # ---------- caching + safe eval ----------
    cache = {}
    cache_max = 22000

    def quant_q():
        # coarse early (more reuse), finer late (more accuracy)
        tf = time_frac()
        # between 350k and 1.6M
        return int(350_000 + tf * 1_250_000)

    def key_of(x):
        q = quant_q()
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
        try:
            fx = float(func(x))
        except Exception:
            return 1e300
        if fx != fx or fx in (float("inf"), float("-inf")):
            # pull towards midpoint and retry
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
            kill = min(len(cache) // 5, 5000)
            keys = list(cache.keys())
            if 0 < kill < len(keys):
                for kk in random.sample(keys, k=kill):
                    cache.pop(kk, None)
        return fx

    # ---------- low discrepancy init ----------
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
    hal_shift = [random.random() for _ in range(dim)]

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
            u = u + hal_shift[d]
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

    # ---------- elites ----------
    elite_k = max(10, min(48, 10 + dim // 2))
    elites = []  # list of (f, x)

    def add_elite(x, fx):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # ---------- improved local refine (directional probes + diag-adaptation + coordinate polish) ----------
    def local_refine(x0, f0, budget=34):
        x = x0[:]
        f = f0

        # diagonal "CMA-ish" scale (not a matrix): update from successful steps
        scale = [0.06 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        smin = [1e-15 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        smax = [0.40 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

        # global trust radius multiplier
        trust = 1.0
        trust_min, trust_max = 0.15, 3.0

        # Phase A: random subspace directional probes with acceptance + trust update
        itA = max(12, (2 * budget) // 3)
        succ = 0
        tried = 0
        for _ in range(itA):
            if not time_left():
                break

            tf = time_frac()
            m = max(2, int(math.sqrt(dim)))
            if dim > 60 and tf < 0.4:
                m = max(2, int(0.6 * math.sqrt(dim)))
            idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)

            # direction
            dirv = [0.0] * dim
            n2 = 0.0
            heavy = (random.random() < (0.10 + 0.10 * (1.0 - tf)))
            for j in idxs:
                r = cauchy() if heavy else randn()
                dirv[j] = r
                n2 += r * r
            if n2 <= 1e-18:
                continue
            invn = 1.0 / math.sqrt(n2)

            # 2-sided probe (few evals, stable)
            rad = (0.55 + 0.75 * random.random()) * trust
            xp = x[:]
            xm = x[:]
            for j in idxs:
                stepj = rad * scale[j] * dirv[j] * invn
                xp[j] += stepj
                xm[j] -= stepj
            fold_inplace(xp)
            fold_inplace(xm)
            fp = evaluate(xp)
            fm = evaluate(xm)

            if fp < fm:
                cand, fcand, sgn = xp, fp, 1.0
            else:
                cand, fcand, sgn = xm, fm, -1.0

            tried += 1
            if fcand < f:
                # accept
                dx = [cand[j] - x[j] for j in idxs]
                x, f = cand, fcand
                succ += 1

                # trust expansion
                trust = min(trust_max, trust * 1.08)

                # diagonal adaptation: if coord moved a lot relative to scale => increase
                for t, j in enumerate(idxs):
                    aj = abs(dx[t])
                    if aj > 0.0:
                        # move scale towards observed successful step
                        target = min(smax[j], max(smin[j], 0.65 * aj + 0.35 * scale[j]))
                        scale[j] = 0.85 * scale[j] + 0.15 * target
                        scale[j] = min(smax[j], max(smin[j], scale[j]))
            else:
                # contract
                trust = max(trust_min, trust * 0.93)
                for j in idxs:
                    scale[j] = max(smin[j], scale[j] * 0.97)

            # small extrapolation along the better direction (sometimes)
            if time_left() and random.random() < 0.40:
                xt = x[:]
                for j in idxs:
                    xt[j] += sgn * 0.35 * trust * scale[j] * dirv[j] * invn
                fold_inplace(xt)
                ft = evaluate(xt)
                if ft < f:
                    x, f = xt, ft
                    succ += 1
                    trust = min(trust_max, trust * 1.04)
                    for j in idxs:
                        scale[j] = min(smax[j], scale[j] * 1.02)

            if tried >= 10:
                rate = succ / float(tried)
                # push trust based on "1/5 success rule"
                if rate > 0.22:
                    trust = min(trust_max, trust * 1.06)
                else:
                    trust = max(trust_min, trust * 0.94)
                succ = 0
                tried = 0

        # Phase B: coordinate/pattern polish (short)
        step = [0.02 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        itB = max(8, budget // 3)
        for _ in range(itB):
            if not time_left():
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if span[j] <= 0.0:
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
                    step[j] = min(step[j] * 1.10, 0.25 * (span[j] if span[j] > 0 else 1.0))
            else:
                for j in range(dim):
                    step[j] *= 0.55

        return x, f

    # ---------- initialization ----------
    best_x = mid[:]
    fold_inplace(best_x)
    best = evaluate(best_x)
    add_elite(best_x, best)

    seed_pool = []

    # Halton + opposition (about mid)
    hal_n = max(16, min(60, 14 + int(2.8 * math.sqrt(dim))))
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

    # LHS batch
    lhs_n = max(12, min(34, 10 + int(1.6 * math.sqrt(dim))))
    for x in lhs_points(lhs_n):
        if not time_left():
            break
        fold_inplace(x)
        fx = evaluate(x)
        seed_pool.append((fx, x))
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

    # elite jitter seeds
    if time_left() and elites:
        for _ in range(min(14, 3 + dim // 5)):
            if not time_left():
                break
            _, xe = random.choice(elites)
            x = xe[:]
            m = max(2, int(math.sqrt(dim)))
            idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)
            rad = 0.05 + 0.20 * random.random()
            heavy = (random.random() < 0.20)
            for d in idxs:
                if span[d] > 0.0:
                    z = cauchy() if heavy else randn()
                    x[d] += rad * span[d] * z
            fold_inplace(x)
            fx = evaluate(x)
            seed_pool.append((fx, x))
            add_elite(x, fx)
            if fx < best:
                best, best_x = fx, x[:]

    seed_pool.sort(key=lambda t: t[0])

    # initial refine
    if time_left():
        bx, bf = local_refine(best_x, best, budget=20)
        if bf < best:
            best, best_x = bf, bx[:]
            add_elite(best_x, best)

    # ---------- L-SHADE-ish DE with pop reduction + diversity-aware mutation ----------
    NP_init = max(26, min(78, 16 + 2 * int(math.sqrt(dim)) + dim // 2))
    NP_min = max(10, min(30, 8 + int(math.sqrt(dim))))
    pop, fit = [], []

    # seed population from best seeds + random
    take = min(len(seed_pool), max(12, NP_init // 2))
    jrad = 0.06
    for i in range(take):
        if not time_left():
            break
        _, xb = seed_pool[i]
        x = xb[:]
        m = max(2, int(math.sqrt(dim)))
        idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)
        heavy = (random.random() < 0.16)
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

    while len(pop) < NP_init and time_left():
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
    arch_max = NP_init * 3

    H = max(6, min(26, 6 + dim // 8))
    M_F = [0.65] * H
    M_CR = [0.50] * H
    k_mem = 0

    p_best_rate = 0.20

    def sample_F(memF):
        for _ in range(16):
            v = memF + 0.12 * cauchy()
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

    def pick_indices(n, exclude, k):
        res = []
        tries = 0
        while len(res) < k and tries < 200:
            j = random.randrange(n)
            if j == exclude or j in res:
                tries += 1
                continue
            res.append(j)
        while len(res) < k:
            j = random.randrange(n)
            if j != exclude:
                res.append(j)
        return res

    def pop_spread():
        # normalized average per-dim std (cheap)
        n = len(pop)
        if n < 3:
            return 0.0
        means = [0.0] * dim
        for x in pop:
            for d in range(dim):
                means[d] += x[d]
        invn = 1.0 / n
        for d in range(dim):
            means[d] *= invn
        acc = 0.0
        for d in range(dim):
            if span[d] <= 0.0:
                continue
            v = 0.0
            md = means[d]
            for x in pop:
                t = x[d] - md
                v += t * t
            v = math.sqrt(v * invn)
            acc += (v / span[d])
        return acc / max(1, dim)

    def eig_crossover(xi, vi, CR):
        u = xi[:]
        perm = list(range(dim))
        random.shuffle(perm)
        jrand = perm[0]
        for d in perm:
            if d == jrand or random.random() < CR:
                u[d] = vi[d]
        return u

    no_improve = 0
    stag1 = 60 + 7 * dim
    stag2 = 150 + 12 * dim
    refine_cooldown = 0

    while time_left():
        n = len(pop)
        if n == 0:
            break

        tf = time_frac()
        target = int(round(NP_init - tf * (NP_init - NP_min)))
        if target < NP_min:
            target = NP_min
        if target < n:
            # remove worst
            order = list(range(n))
            order.sort(key=lambda i: fit[i], reverse=True)
            kill = n - target
            kill_set = set(order[:kill])
            pop = [pop[i] for i in range(n) if i not in kill_set]
            fit = [fit[i] for i in range(n) if i not in kill_set]
            n = len(pop)
            arch_max = max(30, 3 * n)
            if len(archive) > arch_max:
                archive = random.sample(archive, k=arch_max)

        # pbest pool
        order = list(range(n))
        order.sort(key=lambda i: fit[i])
        pN = max(2, int(math.ceil(p_best_rate * n)))
        pbest_pool = order[:pN]

        spread = pop_spread()
        low_div = (spread < (0.035 if dim <= 30 else 0.02))

        S_F, S_CR, S_df = [], [], []

        for ii in range(n):
            if not time_left():
                break

            xi = pop[ii]
            fi = fit[ii]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            pb = pop[random.choice(pbest_pool)]

            # choose mutation style
            use_rand2 = low_div and (random.random() < (0.25 + 0.30 * (1.0 - tf)))
            if use_rand2 and n >= 5:
                a, b, c, d = pick_indices(n, ii, 4)
                xa, xb, xc, xd = pop[a], pop[b], pop[c], pop[d]
                # DE/rand/2
                v = [xa[j] + F * (xb[j] - xc[j]) + F * (xd[j] - xi[j]) for j in range(dim)]
            else:
                # current-to-pbest/1 with archive
                r1 = pick_indices(n, ii, 1)[0]
                x1 = pop[r1]
                if archive and random.random() < 0.55:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_indices(n, ii, 1)[0]
                    x2 = pop[r2]
                v = [xi[j] + F * (pb[j] - xi[j]) + F * (x1[j] - x2[j]) for j in range(dim)]

            # crossover
            if random.random() < (0.14 + (0.10 if low_div else 0.0)):
                u = eig_crossover(xi, v, CR)
            else:
                u = xi[:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        u[d] = v[d]

            # diversity injection: mild noise when collapsed (time-aware)
            if low_div and random.random() < 0.18:
                m = max(1, int(math.sqrt(dim)))
                idxs = list(range(dim)) if m >= dim else random.sample(range(dim), m)
                amp = (0.0015 + (1.0 - tf) * 0.010 * random.random())
                for d in idxs:
                    if span[d] > 0.0:
                        u[d] += amp * span[d] * (cauchy() if random.random() < 0.3 else randn())

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
                        bx, bf = local_refine(best_x, best, budget=30)
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

        # update SHADE memories
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
                w = weights[j]
                fj = S_F[j]
                num += w * fj * fj
                den += w * fj
                cr_m += w * S_CR[j]
            if den > 1e-18:
                M_F[k_mem] = num / den
            M_CR[k_mem] = cr_m

            # clamp
            if M_F[k_mem] < 0.05: M_F[k_mem] = 0.05
            if M_F[k_mem] > 0.95: M_F[k_mem] = 0.95
            if M_CR[k_mem] < 0.0: M_CR[k_mem] = 0.0
            if M_CR[k_mem] > 1.0: M_CR[k_mem] = 1.0
            k_mem = (k_mem + 1) % H

        # -------- stagnation handling (stronger, time-aware) --------
        if no_improve > stag1 and time_left():
            no_improve = 0
            n = len(pop)
            order = list(range(n))
            order.sort(key=lambda i: fit[i], reverse=True)
            k_re = max(3, int(0.40 * n))

            sources = [x for (_, x) in elites[:min(len(elites), elite_k)]]
            if not sources:
                sources = [best_x]

            for t in range(k_re):
                if not time_left():
                    break
                ii = order[t]

                choice = random.random()
                if choice < 0.60:
                    # elite heavy-tail
                    base = random.choice(sources)[:]
                    rad = 0.08 + (0.55 if time_frac() < 0.6 else 0.30) * random.random()
                    heavy = True
                    for d in range(dim):
                        if span[d] > 0.0:
                            z = cauchy() if heavy else randn()
                            base[d] += rad * span[d] * z
                elif choice < 0.85:
                    # far sample
                    base = halton_point(1 + random.randrange(1, 8000))
                else:
                    # opposition about best
                    base = [lo[d] + hi[d] - best_x[d] for d in range(dim)]

                fold_inplace(base)
                fnew = evaluate(base)
                pop[ii] = base
                fit[ii] = fnew
                add_elite(base, fnew)
                if fnew < best:
                    best, best_x = fnew, base[:]

            # reset memories slightly more exploratory
            for i in range(H):
                M_F[i] = 0.70
                M_CR[i] = 0.45
            refine_cooldown = 0

        if no_improve > stag2 and time_left():
            no_improve = 0
            n = len(pop)
            order = list(range(n))
            order.sort(key=lambda i: fit[i], reverse=True)
            k_re = max(4, n // 2)

            if len(archive) > max(24, n):
                archive = random.sample(archive, k=max(24, n))

            # partial rebuild
            for t in range(k_re):
                if not time_left():
                    break
                ii = order[t]
                r = random.random()
                if r < 0.45:
                    xnew = halton_point(1 + random.randrange(1, 12000))
                elif r < 0.75:
                    # around best with big kick
                    xnew = best_x[:]
                    rad = 0.05 + (0.50 if random.random() < 0.35 else 0.20) * random.random()
                    heavy = (random.random() < 0.25)
                    for d in range(dim):
                        if span[d] > 0.0:
                            z = cauchy() if heavy else randn()
                            xnew[d] += rad * span[d] * z
                else:
                    # opposition about mid
                    xnew = [lo[d] + hi[d] - mid[d] + (mid[d] - best_x[d]) for d in range(dim)]

                fold_inplace(xnew)
                fnew = evaluate(xnew)
                pop[ii] = xnew
                fit[ii] = fnew
                add_elite(xnew, fnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            for i in range(H):
                M_F[i] = 0.72
                M_CR[i] = 0.48
            refine_cooldown = 0

    return best
