import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs previous hybrids:
      - Stronger global exploration: scrambled Halton + LHS micro-batches + elite
        "DE/current-to-best/1" variation.
      - More robust local refinement: bounded BOBYQA-like trust-region coordinate
        model is too heavy; instead we use:
          * adaptive (1+λ)-ES on normalized space
          * plus a very cheap quadratic 1D line fit per coordinate (3-point)
            when near an elite (acts like curvature exploitation)
      - Explicit evaluation caching (helps when reflect creates duplicates)
      - Time-aware budgeting and stagnation-driven restarts

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # -------------------- helpers --------------------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scales = [s if s > 0.0 else 1.0 for s in spans]
    inv_scales = [1.0 / s for s in scales]

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        if w <= 0.0:
            return lo
        z = (v - lo) % (2.0 * w)
        if z > w:
            z = 2.0 * w - z
        return lo + z

    def fix(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect(x[i], lo, hi)
        return y

    # evaluation cache (rounded key)
    cache = {}
    def key_of(x):
        # quantize lightly to merge numerically identical reflections
        return tuple(int((x[i] - bounds[i][0]) * (1e12 * inv_scales[i])) for i in range(dim))

    def evaluate(x):
        x = fix(x)
        k = key_of(x)
        if k in cache:
            return cache[k], x
        try:
            v = func(x)
            if v is None:
                v = float("inf")
            v = float(v)
            if not math.isfinite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v, x

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def dist_norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            s += abs(a[i] - b[i]) * inv_scales[i]
        return s

    def elite_insert(elites, x, fx, max_elites, dedup_eps=1e-10):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        for f, p in elites:
            ok = True
            for _, q in out:
                if dist_norm_l1(p, q) < dedup_eps:
                    ok = False
                    break
            if ok:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # -------------------- scrambled Halton --------------------
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
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        f = 1.0
        r = 0.0
        i = index
        perm = digit_perm[base]
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            u = halton_scrambled(k, primes[i])
            x[i] = lo + u * (hi - lo)
        return x

    # -------------------- small LHS batches --------------------
    def lhs_batch(n):
        per_dim = []
        for i in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            per_dim.append(arr)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                x[i] = lo + per_dim[i][k] * (hi - lo)
            pts.append(x)
        return pts

    # -------------------- normalized-space operators --------------------
    def to_u(x):
        u = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            if hi == lo:
                u[i] = 0.5
            else:
                u[i] = (x[i] - lo) / (hi - lo)
                if u[i] < 0.0: u[i] = 0.0
                if u[i] > 1.0: u[i] = 1.0
        return u

    def from_u(u):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo + u[i] * (hi - lo)
        return x

    # -------------------- local: (1+λ)-ES in normalized space --------------------
    def es_local(x0, f0, sigma_u, lam, iters):
        # sigma_u is in normalized [0,1] units
        u = to_u(x0)
        fu = f0
        success = 0
        for t in range(iters):
            if time.time() >= deadline:
                break
            best_cand_u = None
            best_cand_f = fu
            for _ in range(lam):
                if time.time() >= deadline:
                    break
                cand = [0.0] * dim
                for i in range(dim):
                    # gaussian mutation on [0,1], reflect into [0,1]
                    v = u[i] + random.gauss(0.0, sigma_u)
                    # reflect into [0,1]
                    if v < 0.0 or v > 1.0:
                        v = (v % 2.0)
                        if v > 1.0:
                            v = 2.0 - v
                    cand[i] = v
                fx, xfix = evaluate(from_u(cand))
                if fx < best_cand_f:
                    best_cand_f = fx
                    best_cand_u = to_u(xfix)
            if best_cand_u is not None and best_cand_f < fu:
                u, fu = best_cand_u, best_cand_f
                success += 1
                sigma_u *= 1.08
            else:
                sigma_u *= 0.92
            if sigma_u < 1e-6:
                sigma_u = 1e-6
            if sigma_u > 0.5:
                sigma_u = 0.5
        return from_u(u), fu, success

    # -------------------- local: cheap 1D quadratic fit per coordinate --------------------
    def quad_coord_polish(x0, f0, step_frac, max_passes=1):
        x = list(x0)
        fx = f0
        for _ in range(max_passes):
            if time.time() >= deadline:
                break
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for i in order:
                if time.time() >= deadline:
                    break
                lo, hi = bounds[i]
                if hi == lo:
                    continue
                h = step_frac * (hi - lo)
                if h <= 0.0:
                    continue
                xi = x[i]
                x1 = list(x); x1[i] = reflect(xi - h, lo, hi)
                x2 = list(x); x2[i] = xi
                x3 = list(x); x3[i] = reflect(xi + h, lo, hi)
                f1, x1 = evaluate(x1)
                f2 = fx
                f3, x3 = evaluate(x3)

                # fit parabola through (-h,f1),(0,f2),(+h,f3): min at t* = h*(f1-f3)/(2*(f1-2f2+f3))
                denom = (f1 - 2.0 * f2 + f3)
                if denom != 0.0 and math.isfinite(denom):
                    tstar = 0.5 * h * (f1 - f3) / denom
                    # clamp within [-h, h]
                    if tstar < -h: tstar = -h
                    if tstar >  h: tstar =  h
                    xt = list(x)
                    xt[i] = reflect(xi + tstar, lo, hi)
                    ft, xt = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                else:
                    # fallback: take best of three
                    if f1 < fx:
                        x, fx = x1, f1
                        improved = True
                    if f3 < fx:
                        x, fx = x3, f3
                        improved = True
            if not improved:
                break
        return x, fx

    # -------------------- global variation: DE/current-to-best/1 on elites --------------------
    def de_trial(x, bestx, a, b, F, CR):
        # x + F*(best-x) + F*(a-b) with binomial crossover
        trial = list(x)
        jrand = random.randrange(dim)
        for j in range(dim):
            if random.random() < CR or j == jrand:
                v = x[j] + F * (bestx[j] - x[j]) + F * (a[j] - b[j])
                lo, hi = bounds[j]
                trial[j] = reflect(v, lo, hi)
        return trial

    # -------------------- initialization --------------------
    best = float("inf")
    best_x = None
    elites = []
    max_elites = max(14, min(90, 6 * dim))

    hal_start = 1 + random.randrange(512)
    n_hal = max(30, min(220, 14 * dim))
    for k in range(n_hal):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = halton_point(hal_start + k)
        fx, x = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

        # opposition injection occasionally
        if (k % 3) == 0:
            xo = [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]
            fxo, xo = evaluate(xo)
            if fxo < best:
                best, best_x = fxo, xo
            elites = elite_insert(elites, xo, fxo, max_elites)

    # small LHS burst
    for x in lhs_batch(max(10, min(70, 4 * dim))):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        fx, x = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

    if best_x is None:
        x = rand_uniform()
        fx, x = evaluate(x)
        best, best_x = fx, x
        elites = [(best, best_x)]

    # -------------------- main loop --------------------
    no_improve = 0
    phase = 0
    while time.time() < deadline:
        phase += 1

        if not elites:
            elites = [(best, best_x)]
        m = len(elites)

        # pick a base elite (best-biased)
        r = random.random()
        idx = int((r * r) * m)
        fx0, x0 = elites[idx]

        # 1) DE step if we have enough elites
        if m >= 4 and time.time() < deadline:
            # choose two distinct random elites (not idx)
            ia = random.randrange(m)
            ib = random.randrange(m)
            while ib == ia:
                ib = random.randrange(m)
            a = elites[ia][1]
            b = elites[ib][1]

            # parameters adapt with stagnation
            stagn = 1.0 + min(3.0, no_improve / 18.0)
            F = 0.4 + 0.25 * random.random()
            CR = 0.2 + 0.6 * random.random()
            F *= min(2.0, stagn)

            xt = de_trial(x0, best_x, a, b, F=F, CR=CR)
            ft, xt = evaluate(xt)
            elites = elite_insert(elites, xt, ft, max_elites)
            if ft < best:
                best, best_x = ft, xt
                no_improve = 0
            else:
                no_improve += 1

        if time.time() >= deadline:
            break

        # 2) Local ES around a top elite / best
        topk = min(m, max(3, dim))
        pick = random.randrange(topk)
        fxL, xL = elites[pick]

        # sigma in normalized space: start moderate, shrink with time, boost with stagnation
        time_decay = 1.0 / (1.0 + 0.03 * phase)
        stagn = 1.0 + min(3.0, no_improve / 20.0)
        sigma_u = (0.12 * time_decay) * stagn
        lam = min(30, 6 + dim // 2)
        iters = min(12, 3 + dim // 6)

        x1, f1, _ = es_local(xL, fxL, sigma_u=sigma_u, lam=lam, iters=iters)
        elites = elite_insert(elites, x1, f1, max_elites)
        if f1 < best:
            best, best_x = f1, x1
            no_improve = 0
        else:
            no_improve += 1

        if time.time() >= deadline:
            break

        # 3) Quadratic coordinate polish occasionally on best
        if (phase % 3) == 0:
            step_frac = 0.06 * time_decay
            x2, f2 = quad_coord_polish(best_x, best, step_frac=step_frac, max_passes=1)
            elites = elite_insert(elites, x2, f2, max_elites)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0

        if time.time() >= deadline:
            break

        # 4) Global injections / restarts if stuck
        if (phase % 8) == 0:
            xg = halton_point(hal_start + n_hal + phase * 5 + random.randrange(32))
            fg, xg = evaluate(xg)
            elites = elite_insert(elites, xg, fg, max_elites)
            if fg < best:
                best, best_x = fg, xg
                no_improve = 0

        if no_improve >= 28 and time.time() < deadline:
            # LHS micro-batch restart
            for x in lhs_batch(max(8, min(26, 2 * dim))):
                if time.time() >= deadline:
                    break
                fx, x = evaluate(x)
                elites = elite_insert(elites, x, fx, max_elites)
                if fx < best:
                    best, best_x = fx, x
            no_improve = 0

    return best
