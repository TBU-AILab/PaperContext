import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimization (no external libraries).

    Upgrades over the provided DE/JADE-like baseline:
      - Better boundary handling: bounce-back + randomized repair (reduces corner trapping).
      - Explicit population size schedule (larger early, smaller later) with fast rebuild.
      - "Best-of-rand/1/bin" injection mix + current-to-pbest/1 (improves robustness).
      - Stronger stagnation response: partial reinit using LHS + archive reset.
      - Two-stage local search:
           * adaptive coordinate pattern search
           * occasional small random-direction line search
      - Evaluation caching of rounded points (helps when func is noisy/expensive? still safe).
        (Cache is bounded and uses coarse rounding to avoid huge memory.)

    Returns:
      best fitness found (float).
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---- preprocess bounds ----
    lows = [0.0] * dim
    highs = [0.0] * dim
    spans = [0.0] * dim
    for i in range(dim):
        lo, hi = float(bounds[i][0]), float(bounds[i][1])
        if hi < lo:
            lo, hi = hi, lo
        lows[i], highs[i] = lo, hi
        s = hi - lo
        spans[i] = s if s > 0.0 else 0.0

    # ---- utilities ----
    def clip(v, d):
        lo = lows[d]; hi = highs[d]
        if v < lo: return lo
        if v > hi: return hi
        return v

    def bounce(v, d):
        """Reflect into bounds; if still out due to huge step, random-repair."""
        lo = lows[d]; hi = highs[d]
        if spans[d] == 0.0:
            return lo
        # a few reflections
        for _ in range(3):
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            else:
                return v
        # still outside -> random repair inside
        if v < lo or v > hi:
            return lo + spans[d] * random.random()
        return v

    def rand_point():
        return [lows[i] + spans[i] * random.random() if spans[i] > 0.0 else lows[i] for i in range(dim)]

    def lhs_points(n):
        if n <= 0:
            return []
        perms = []
        for d in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)
        inv = 1.0 / n
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    x[d] = lows[d]
                else:
                    u = (perms[d][k] + random.random()) * inv
                    x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    def safe_float(v):
        try:
            v = float(v)
        except Exception:
            return float("inf")
        if v != v or v == float("inf") or v == float("-inf"):
            return float("inf")
        return v

    # coarse cache (bounded)
    cache = {}
    cache_keys = []
    cache_max = 6000
    # rounding granularity based on span (avoid too coarse for tiny ranges)
    gran = []
    for d in range(dim):
        s = spans[d]
        if s <= 0:
            gran.append(0.0)
        else:
            gran.append(s / 2000.0)  # about 2k buckets per dimension (very coarse in high-d)

    def key_of(x):
        k = []
        for d in range(dim):
            g = gran[d]
            if g == 0.0:
                k.append(0)
            else:
                k.append(int((x[d] - lows[d]) / g + 0.5))
        return tuple(k)

    def eval_x(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        val = safe_float(func(x))
        cache[k] = val
        cache_keys.append(k)
        if len(cache_keys) > cache_max:
            old = cache_keys.pop(0)
            cache.pop(old, None)
        return val

    # sampling distributions
    def cauchy(loc, scale):
        u = random.random() - 0.5
        return loc + scale * math.tan(math.pi * u)

    def normal(loc, scale):
        u1 = max(1e-300, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return loc + scale * z

    # ---- population schedule ----
    pop_hi = max(28, min(160, 14 + 9 * dim))
    pop_lo = max(18, min(80, 10 + 5 * dim))

    def current_pop_size():
        # linearly shrink with time
        now = time.time()
        if deadline <= t0:
            return pop_lo
        a = (now - t0) / (deadline - t0 + 1e-12)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        return int(pop_hi + (pop_lo - pop_hi) * a)

    # ---- initialize ----
    pop = current_pop_size()
    X = lhs_points(min(pop, max(10, pop // 2)))
    while len(X) < pop:
        X.append(rand_point())

    Fx = [float("inf")] * pop
    best = float("inf")
    best_x = None
    for i in range(pop):
        if time.time() >= deadline:
            return best
        Fx[i] = eval_x(X[i])
        if Fx[i] < best:
            best = Fx[i]
            best_x = X[i][:]

    # ---- DE/JADE-like params ----
    pbest_rate = 0.18
    min_pbest = 2
    mu_F = 0.55
    mu_CR = 0.85
    c_adapt = 0.12

    # archive
    A = []
    Amax = pop_hi

    # stagnation logic
    stagnation = 0
    last_best = best
    it = 0

    # ---- local search (pattern + random direction) ----
    def coord_pattern_search(x0, f0, budget, step_frac):
        x = x0[:]
        fx = f0
        step = [step_frac * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] == 0.0:
                step[d] = 0.0
        shrink = 0.5
        tiny = 1e-14
        used = 0
        while used < budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                s = step[d]
                if s <= tiny:
                    continue
                # + move
                xp = x[:]
                xp[d] = clip(xp[d] + s, d)
                fp = eval_x(xp); used += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    if used >= budget or time.time() >= deadline:
                        break
                    continue
                if used >= budget or time.time() >= deadline:
                    break
                # - move
                xm = x[:]
                xm[d] = clip(xm[d] - s, d)
                fm = eval_x(xm); used += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    if used >= budget or time.time() >= deadline:
                        break
            if not improved:
                m = 0.0
                for d in range(dim):
                    step[d] *= shrink
                    if step[d] > m:
                        m = step[d]
                if m <= tiny:
                    break
        return x, fx

    def rand_dir_line_search(x0, f0, budget, step_frac):
        if budget <= 0 or time.time() >= deadline:
            return x0, f0
        # random direction normalized by spans
        dvec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        # scale by span to avoid tiny dims dominating
        norm = 0.0
        for i in range(dim):
            s = spans[i] if spans[i] > 0.0 else 1.0
            norm += (dvec[i] * s) * (dvec[i] * s)
        if norm <= 0.0:
            return x0, f0
        norm = math.sqrt(norm)
        for i in range(dim):
            dvec[i] /= norm

        # try a few step sizes (geometric)
        xbest = x0[:]
        fbest = f0
        alpha = step_frac * (sum(spans) / max(1, dim))
        if alpha <= 0.0:
            return x0, f0
        used = 0
        for k in range(6):
            if used >= budget or time.time() >= deadline:
                break
            a = alpha * (0.6 ** k)
            # forward/backward
            for sign in (1.0, -1.0):
                if used >= budget or time.time() >= deadline:
                    break
                xt = x0[:]
                for i in range(dim):
                    if spans[i] == 0.0:
                        xt[i] = lows[i]
                    else:
                        xt[i] = bounce(xt[i] + sign * a * dvec[i] * spans[i], i)
                ft = eval_x(xt); used += 1
                if ft < fbest:
                    xbest, fbest = xt, ft
        return xbest, fbest

    # ---- main loop ----
    while time.time() < deadline:
        it += 1

        # resize population gradually (remove worst if needed)
        target_pop = current_pop_size()
        if target_pop < pop:
            # drop worst individuals
            order = sorted(range(pop), key=lambda i: Fx[i], reverse=True)
            drop = pop - target_pop
            keep = set(order[drop:])
            X = [X[i] for i in range(pop) if i in keep]
            Fx = [Fx[i] for i in range(pop) if i in keep]
            pop = target_pop

        # best tracking
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagnation = max(0, stagnation - 4)
        else:
            stagnation += 1

        if best >= last_best - 1e-15:
            pass
        else:
            last_best = best
            stagnation = max(0, stagnation - 8)

        # stagnation response: partial restart with LHS + archive reset
        if stagnation > 90 and time.time() < deadline:
            k = max(2, pop // 5)
            # replace k worst with LHS points + a few near-best jitters
            order = sorted(range(pop), key=lambda i: Fx[i], reverse=True)
            repl = order[:k]
            pts = lhs_points(max(1, k // 2))
            while len(pts) < k:
                # mix: random global or near-best
                if best_x is not None and random.random() < 0.6:
                    x = best_x[:]
                    for d in range(dim):
                        if spans[d] > 0.0:
                            x[d] = bounce(x[d] + (random.random() - 0.5) * 0.25 * spans[d], d)
                    pts.append(x)
                else:
                    pts.append(rand_point())
            for idx, xnew in zip(repl, pts):
                X[idx] = xnew
                Fx[idx] = eval_x(xnew)
                if Fx[idx] < best:
                    best = Fx[idx]
                    best_x = xnew[:]
            A = []
            stagnation = 40

        # sort for pbest sampling
        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])
        top = max(min_pbest, int(pbest_rate * pop))
        if top > pop:
            top = pop

        union = X + A
        union_n = len(union)

        S_F, S_CR, delta_f = [], [], []

        # immigrant trick (tiny)
        if random.random() < 0.15:
            j = random.randrange(pop)
            xj = rand_point()
            fj = eval_x(xj)
            if fj < Fx[j]:
                if len(A) < Amax:
                    A.append(X[j])
                X[j], Fx[j] = xj, fj
                if fj < best:
                    best, best_x = fj, xj[:]

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi = X[i]
            fi = Fx[i]

            Fi = cauchy(mu_F, 0.12)
            tries = 0
            while Fi <= 0.0 and tries < 8:
                Fi = cauchy(mu_F, 0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.08
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(mu_CR, 0.12)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # choose pbest
            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # choose r1,r2
            forbidden = {i, pbest}
            # r1 from pop
            while True:
                r1 = random.randrange(pop)
                if r1 not in forbidden:
                    break
            forbidden.add(r1)
            # r2 from union
            if union_n <= 1:
                r2u = r1
            else:
                while True:
                    r2u = random.randrange(union_n)
                    if r2u >= pop:
                        break
                    if r2u not in forbidden:
                        break

            xr1 = X[r1]
            xr2 = union[r2u]

            # strategy mix:
            # 70% current-to-pbest/1, 30% best-of-rand/1 (helps when population collapses)
            if random.random() < 0.70:
                v = [0.0] * dim
                for d in range(dim):
                    if spans[d] == 0.0:
                        v[d] = lows[d]
                    else:
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            else:
                # best-of-rand/1: best + F*(r1-r2)
                xb = X[sorted_idx[0]]
                v = [0.0] * dim
                for d in range(dim):
                    if spans[d] == 0.0:
                        v[d] = lows[d]
                    else:
                        v[d] = xb[d] + Fi * (xr1[d] - xr2[d])

            # binomial crossover with bounce + occasional random coordinate
            jrand = random.randrange(dim) if dim > 0 else 0
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = bounce(v[d], d)
                if spans[d] > 0.0 and random.random() < 0.001:
                    u[d] = lows[d] + spans[d] * random.random()

            fu = eval_x(u)

            if fu <= fi:
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))
                X[i] = u
                Fx[i] = fu

                S_F.append(Fi)
                S_CR.append(CRi)
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                delta_f.append(df)

                if fu < best:
                    best = fu
                    best_x = u[:]
                    stagnation = max(0, stagnation - 6)

        # parameter adaptation
        if S_F:
            wsum = sum(delta_f) + 1e-300
            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, delta_f):
                ww = w / wsum
                num += ww * (fval * fval)
                den += ww * fval
            F_lehmer = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))

            cr_mean = 0.0
            for crval, w in zip(S_CR, delta_f):
                cr_mean += (w / wsum) * crval

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * cr_mean
            if mu_F < 0.05: mu_F = 0.05
            if mu_F > 0.95: mu_F = 0.95
            if mu_CR < 0.05: mu_CR = 0.05
            if mu_CR > 0.95: mu_CR = 0.95
        else:
            # no successes -> slightly increase exploration
            mu_F = min(0.95, mu_F + 0.03)
            mu_CR = max(0.10, mu_CR - 0.04)

        # local search scheduling (more near end, also when stagnating)
        if best_x is not None and (it % 14 == 0 or stagnation > 110):
            tl = deadline - time.time()
            if tl < 0.30:
                b1 = 110
                step = 0.02
            elif stagnation > 110:
                b1 = 70
                step = 0.05
            else:
                b1 = 45
                step = 0.07

            rx, rf = coord_pattern_search(best_x, best, budget=b1, step_frac=step)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject to worst
                widx = max(range(pop), key=lambda i: Fx[i])
                X[widx], Fx[widx] = best_x[:], best
                stagnation = 0

            # occasional line search
            if time.time() < deadline and (tl < 0.50 or stagnation > 120):
                rx2, rf2 = rand_dir_line_search(best_x, best, budget=18, step_frac=0.05)
                if rf2 < best:
                    best, best_x = rf2, rx2[:]
                    widx = max(range(pop), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best
                    stagnation = 0

    return best
