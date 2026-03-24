import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improved over the provided DE/JADE-like hybrid by adding:
      - 2-phase search: strong global (SHADE-like DE/current-to-pbest/1 + archive)
        + aggressive late-stage local search (multi-start coordinate + 1D quadratic fit)
      - Success-history adaptation with *per-individual* F/CR sampling from memories
      - Robust bound handling (reflect + parent-backoff)
      - Population size reduction over time (more exploration early, exploitation late)
      - Stagnation detectors triggering: partial restart around best + random immigrants
      - Optional mirrored opposition points during init for better coverage
      - Safe evaluation + small cache

    Returns: best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---------- bounds ----------
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

    def clip(v, d):
        lo, hi = lows[d], highs[d]
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect(v, d):
        lo, hi = lows[d], highs[d]
        s = spans[d]
        if s <= 0.0:
            return lo
        # reflect a few times, then clip
        for _ in range(4):
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            else:
                return v
        return clip(v, d)

    # ---------- safe eval + small cache ----------
    cache = {}
    cache_max = 7000

    def key_of(x):
        # mild quantization relative to span (helps local search repeats)
        q = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                q.append(0)
            else:
                q.append(int((x[d] - lows[d]) / (s * 1e-12 + 1e-30)))
        return tuple(q)

    def safe_eval(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        try:
            v = float(func(x))
            if v != v or v in (float("inf"), float("-inf")):
                v = float("inf")
        except Exception:
            v = float("inf")
        if len(cache) < cache_max:
            cache[k] = v
        return v

    # ---------- RNG helpers ----------
    def rand_point():
        return [lows[d] + spans[d] * random.random() for d in range(dim)]

    def lhs_block(n):
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
                u = (perms[d][k] + random.random()) * inv
                x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    def cauchy(loc, scale):
        u = random.random() - 0.5
        return loc + scale * math.tan(math.pi * u)

    def normal(loc, scale):
        u1 = max(1e-300, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return loc + scale * z

    # ---------- init pop size / reduction schedule ----------
    NP0 = max(30, min(220, 16 + 10 * dim))
    NPmin = max(12, min(60, 8 + 4 * dim))

    def current_NP(tf):
        # linear reduction
        n = int(round(NP0 - tf * (NP0 - NPmin)))
        if n < NPmin:
            n = NPmin
        if n > NP0:
            n = NP0
        return n

    # ---------- SHADE-like memories ----------
    H = 8 + int(2 * math.sqrt(max(1, dim)))
    M_F = [0.55] * H
    M_CR = [0.85] * H
    k_mem = 0
    pmin, pmax = 0.08, 0.35

    # archive
    A = []
    Amax = NP0

    # ---------- initialization with LHS + opposite points ----------
    X, Fx = [], []
    best = float("inf")
    best_x = None

    n_lhs = max(12, NP0 // 2)
    init = lhs_block(min(NP0, n_lhs))
    while len(init) < NP0:
        init.append(rand_point())

    # opposition injection (mirrored around center)
    # Replace some random individuals with their opposite if it's better (cheap diversity boost)
    for x in init:
        if time.time() >= deadline:
            return best
        f = safe_eval(x)
        X.append(x)
        Fx.append(f)
        if f < best:
            best, best_x = f, x[:]

    for _ in range(min(NP0 // 3, 40)):
        if time.time() >= deadline:
            return best
        i = random.randrange(NP0)
        xi = X[i]
        xo = [0.0] * dim
        for d in range(dim):
            xo[d] = lows[d] + highs[d] - xi[d]
            xo[d] = clip(xo[d], d)
        fo = safe_eval(xo)
        if fo < Fx[i]:
            X[i], Fx[i] = xo, fo
            if fo < best:
                best, best_x = fo, xo[:]

    # ---------- local search: coordinate + 1D quadratic fit ----------
    def local_refine_quad(x0, f0, eval_budget, step_frac):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        steps = [step_frac * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] <= 0.0:
                steps[d] = 0.0

        shrink = 0.5
        min_step = 1e-14
        evals = 0

        while evals < eval_budget and time.time() < deadline:
            improved = False
            # random coordinate order helps in noisy/ill-conditioned cases
            order = list(range(dim))
            random.shuffle(order)

            for d in order:
                if evals >= eval_budget or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= min_step:
                    continue

                x0d = x[d]

                # sample three points along coordinate: -s, 0, +s
                xm = x[:]; xm[d] = clip(x0d - s, d)
                xp = x[:]; xp[d] = clip(x0d + s, d)

                fm = safe_eval(xm); evals += 1
                if evals >= eval_budget or time.time() >= deadline:
                    if fm < fx:
                        x, fx = xm, fm
                    break
                fp = safe_eval(xp); evals += 1

                # take best among +/- directly (fast)
                if fm < fx or fp < fx:
                    if fm <= fp:
                        x, fx = xm, fm
                    else:
                        x, fx = xp, fp
                    improved = True
                    continue

                # attempt a 1D quadratic fit around current point if no direct improvement:
                # Use points: (x0-s, fm), (x0, fx), (x0+s, fp)
                # Parabola vertex offset t = s*(fm - fp) / (2*(fm - 2*fx + fp))
                denom = (fm - 2.0 * fx + fp)
                if abs(denom) > 1e-30:
                    t = 0.5 * s * (fm - fp) / denom
                    # clamp the proposed step to [-s, s]
                    if t > s:
                        t = s
                    elif t < -s:
                        t = -s
                    if abs(t) > 0.1 * s:
                        xq = x[:]
                        xq[d] = clip(x0d + t, d)
                        fq = safe_eval(xq); evals += 1
                        if fq < fx:
                            x, fx = xq, fq
                            improved = True

            if not improved:
                # shrink steps
                mx = 0.0
                for d in range(dim):
                    steps[d] *= shrink
                    if steps[d] > mx:
                        mx = steps[d]
                if mx <= min_step:
                    break

        return x, fx

    # ---------- stagnation controls ----------
    stagn = 0
    last_best = best
    last_improve_t = time.time()

    def inject_around_best(k, radius_frac):
        nonlocal best, best_x
        if best_x is None:
            return
        # replace worst k
        idxs = sorted(range(len(X)), key=lambda i: Fx[i], reverse=True)[:k]
        for idx in idxs:
            if time.time() >= deadline:
                return
            xnew = best_x[:]
            for d in range(dim):
                s = spans[d]
                if s > 0.0:
                    # uniform perturbation in a box
                    xnew[d] = clip(xnew[d] + (random.random() * 2.0 - 1.0) * radius_frac * s, d)
            fnew = safe_eval(xnew)
            X[idx], Fx[idx] = xnew, fnew
            if fnew < best:
                best, best_x = fnew, xnew[:]

    # ---------- main loop ----------
    it = 0
    while time.time() < deadline:
        it += 1

        # time fraction
        tf = (time.time() - t0) / max(1e-12, (deadline - t0))
        if tf < 0.0:
            tf = 0.0
        elif tf > 1.0:
            tf = 1.0

        # pop-size reduction
        NP = current_NP(tf)
        if len(X) > NP:
            # remove worst to match NP
            worst = sorted(range(len(X)), key=lambda i: Fx[i], reverse=True)
            remove = worst[:(len(X) - NP)]
            remove_set = set(remove)
            X = [x for j, x in enumerate(X) if j not in remove_set]
            Fx = [f for j, f in enumerate(Fx) if j not in remove_set]
            # also trim archive
            if len(A) > Amax:
                A = A[-Amax:]

        # update best
        bidx = min(range(len(X)), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
            last_improve_t = time.time()
        else:
            stagn += 1

        # p-best fraction schedule
        p = pmax - (pmax - pmin) * tf
        top = max(2, int(p * len(X)))

        # stagnation actions
        if stagn > 120 and time.time() < deadline:
            # small immigrants + around-best injection
            k = max(1, len(X) // 12)
            inject_around_best(k, radius_frac=0.08 * (1.0 - tf) + 0.01)
            for _ in range(max(1, len(X) // 25)):
                if time.time() >= deadline:
                    break
                j = max(range(len(X)), key=lambda i: Fx[i])
                xnew = rand_point()
                fnew = safe_eval(xnew)
                X[j], Fx[j] = xnew, fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
            stagn = 70  # prevent immediate retrigger

        # DE generation
        idx_sorted = sorted(range(len(X)), key=lambda i: Fx[i])
        union = X + A
        union_n = len(union)

        S_F, S_CR, dF = [], [], []

        for i in range(len(X)):
            if time.time() >= deadline:
                break

            xi, fi = X[i], Fx[i]

            r = random.randrange(H)
            Fi = cauchy(M_F[r], 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 8:
                Fi = cauchy(M_F[r], 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(M_CR[r], 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            pbest = idx_sorted[random.randrange(top)]
            xpbest = X[pbest]

            # pick r1 != i, pbest
            forbidden = {i, pbest}
            while True:
                r1 = random.randrange(len(X))
                if r1 not in forbidden:
                    break
            forbidden.add(r1)

            # pick r2 from union, if from pop ensure not forbidden
            if union_n <= 1:
                r2u = r1
            else:
                while True:
                    r2u = random.randrange(union_n)
                    if r2u >= len(X):
                        break
                    if r2u not in forbidden:
                        break

            xr1 = X[r1]
            xr2 = union[r2u]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] <= 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover + repair (reflect + parent-backoff)
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ud = reflect(v[d], d)
                    # if huge excursion, pull back toward parent a bit
                    if random.random() < 0.03:
                        ud = 0.8 * ud + 0.2 * xi[d]
                        ud = clip(ud, d)
                    u[d] = ud

            fu = safe_eval(u)

            if fu <= fi:
                # archive old
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))

                X[i], Fx[i] = u, fu

                S_F.append(Fi)
                S_CR.append(CRi)
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                dF.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    last_best = best
                    last_improve_t = time.time()
                    stagn = max(0, stagn - 10)

        # memory update (weighted)
        if S_F:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F
            num = 0.0
            den = 0.0
            crw = 0.0
            for fval, crval, w in zip(S_F, S_CR, dF):
                ww = w / wsum
                num += ww * (fval * fval)
                den += ww * fval
                crw += ww * crval
            new_MF = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))
            new_MCR = crw

            M_F[k_mem] = min(0.95, max(0.05, new_MF))
            M_CR[k_mem] = min(0.95, max(0.05, new_MCR))
            k_mem = (k_mem + 1) % H

        # late-stage local search (more aggressive near the end)
        tl = deadline - time.time()
        if best_x is not None and tl > 0.0:
            # trigger periodically or if no improvement for a while
            if (it % 10 == 0 and tf > 0.35) or (time.time() - last_improve_t > 0.35 * max_time):
                # budget increases as time runs out
                if tl < 0.25:
                    budget = 220
                    stepf = 0.010
                elif tl < 0.8:
                    budget = 140
                    stepf = 0.020
                else:
                    budget = 70
                    stepf = 0.05 * (1.0 - 0.6 * tf)

                rx, rf = local_refine_quad(best_x, best, eval_budget=budget, step_frac=stepf)
                if rf < best:
                    best, best_x = rf, rx[:]
                    last_best = best
                    last_improve_t = time.time()
                    # inject into worst
                    widx = max(range(len(X)), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best
                    stagn = 0

    return best
