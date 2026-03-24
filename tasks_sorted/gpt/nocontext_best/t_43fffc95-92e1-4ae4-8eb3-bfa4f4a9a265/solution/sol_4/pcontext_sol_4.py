import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improved hybrid over the provided JADE-like DE:
      - Better bound handling: reflection + "repair-to-parent" fallback (prevents edge-traps)
      - Rank-based selection pressure + p-best scheduling (more exploration early, exploitation late)
      - External archive + "rand-to-best" occasional mutation (escapes stagnation)
      - Cheap surrogate-like intensification: adaptive pattern/coordinate search around best
      - Periodic partial population refresh using LHS blocks (diversity without full restart)
      - Evaluation caching (tiny, safe) to avoid re-evaluating identical points

    Returns: best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---------------- bounds ----------------
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
        if v < lows[d]:
            return lows[d]
        if v > highs[d]:
            return highs[d]
        return v

    def reflect(v, d):
        """Reflect into bounds; if still out after a few reflections, clip."""
        lo, hi = lows[d], highs[d]
        if spans[d] == 0.0:
            return lo
        for _ in range(3):
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            else:
                return v
        return clip(v, d)

    # ---------------- safe eval + tiny cache ----------------
    # Cache helps when the algorithm re-creates same vectors during refinement / repair.
    cache = {}
    cache_max = 5000

    def key_of(x):
        # quantize to make cache effective but safe-ish
        # (quantization is mild; doesn't change what we pass to func)
        q = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                q.append(0)
            else:
                # 1e-12 relative span bucket
                q.append(int((x[d] - lows[d]) / (s * 1e-12 + 1e-30)))
        return tuple(q)

    def safe_eval(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        try:
            v = float(func(x))
            if v != v or v == float("inf") or v == float("-inf"):
                v = float("inf")
        except Exception:
            v = float("inf")
        if len(cache) < cache_max:
            cache[k] = v
        return v

    # ---------------- RNG helpers ----------------
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

    # ---------------- population sizing ----------------
    pop = max(24, min(160, 14 + 8 * dim))
    immigrants = max(1, pop // 20)

    # JADE-style parameter memories
    mu_F = 0.55
    mu_CR = 0.85
    c_adapt = 0.12

    # archive
    A = []
    Amax = pop

    # best tracking
    best = float("inf")
    best_x = None

    # ---------------- init ----------------
    X, Fx = [], []
    init = lhs_block(min(pop, max(12, pop // 2)))
    while len(init) < pop:
        init.append(rand_point())

    for x in init:
        if time.time() >= deadline:
            return best
        f = safe_eval(x)
        X.append(x)
        Fx.append(f)
        if f < best:
            best, best_x = f, x[:]

    # ---------------- refinement: adaptive pattern + coordinate search ----------------
    def local_refine(x0, f0, eval_budget, step_frac):
        x = x0[:]
        fx = f0
        if x is None:
            return x0, f0

        # initial step sizes per-dimension
        steps = [step_frac * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] == 0.0:
                steps[d] = 0.0

        shrink = 0.6
        min_step = 1e-12

        evals = 0
        while evals < eval_budget and time.time() < deadline:
            improved = False

            # pattern step: try a random direction first (cheap)
            if evals < eval_budget and time.time() < deadline:
                dirn = [0.0] * dim
                for d in range(dim):
                    if steps[d] > 0.0 and random.random() < 0.25:
                        dirn[d] = steps[d] * (1.0 if random.random() < 0.5 else -1.0)
                if any(v != 0.0 for v in dirn):
                    xt = x[:]
                    for d in range(dim):
                        if dirn[d] != 0.0:
                            xt[d] = clip(xt[d] + dirn[d], d)
                    ft = safe_eval(xt); evals += 1
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True

            # coordinate moves
            for d in range(dim):
                if evals >= eval_budget or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= min_step:
                    continue

                # +s
                xp = x[:]
                xp[d] = clip(xp[d] + s, d)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= eval_budget or time.time() >= deadline:
                    break

                # -s
                xm = x[:]
                xm[d] = clip(xm[d] - s, d)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                # shrink steps
                m = 0.0
                for d in range(dim):
                    steps[d] *= shrink
                    if steps[d] > m:
                        m = steps[d]
                if m <= min_step:
                    break

        return x, fx

    # ---------------- main loop ----------------
    stagnation = 0
    it = 0

    while time.time() < deadline:
        it += 1

        # time fraction for scheduling
        tf = (time.time() - t0) / max(1e-12, (deadline - t0))
        if tf < 0.0: tf = 0.0
        if tf > 1.0: tf = 1.0

        # p-best schedule: explore early (bigger top set), exploit late (smaller top set)
        pbest_rate = 0.35 - 0.25 * tf   # 0.35 -> 0.10
        if pbest_rate < 0.08:
            pbest_rate = 0.08
        min_pbest = 2

        # update current best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagnation = max(0, stagnation - 4)
        else:
            stagnation += 1

        # periodic partial refresh (keeps diversity without discarding everything)
        if stagnation > 120 and time.time() < deadline:
            k = max(immigrants, pop // 10)
            refresh_pts = lhs_block(k)
            # replace worst k
            worst = sorted(range(pop), key=lambda i: Fx[i], reverse=True)[:k]
            for idx, xnew in zip(worst, refresh_pts):
                if time.time() >= deadline:
                    break
                fnew = safe_eval(xnew)
                X[idx], Fx[idx] = xnew, fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
            # keep some stagnation to avoid repeated refresh thrash
            stagnation = 80

        # sort once for pbest
        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])
        top = max(min_pbest, int(pbest_rate * pop))

        # union for r2
        union = X + A
        union_n = len(union)

        # collect successes
        S_F, S_CR, dF = [], [], []

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi, fi = X[i], Fx[i]

            # sample parameters
            Fi = cauchy(mu_F, 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(mu_CR, 0.1)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # choose pbest
            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # choose r1 from pop excluding i,pbest
            forbidden = {i, pbest}
            while True:
                r1 = random.randrange(pop)
                if r1 not in forbidden:
                    break
            forbidden.add(r1)

            # choose r2 from union excluding forbidden if from pop
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

            # occasional "rand-to-best" mix when stagnating (escape local minima)
            use_best_pull = (stagnation > 60 and random.random() < 0.15)

            # mutation: current-to-pbest/1 (+ optional best pull)
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    base = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                    if use_best_pull and best_x is not None:
                        base = base + 0.35 * Fi * (best_x[d] - xi[d])
                    v[d] = base

            # crossover + repair
            jrand = random.randrange(dim)
            u = xi[:]
            changed = False
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ud = v[d]
                    # reflect then (rarely) repair towards parent to avoid bad boundary bouncing
                    ud = reflect(ud, d)
                    if random.random() < 0.02:
                        ud = 0.75 * ud + 0.25 * xi[d]
                        ud = clip(ud, d)
                    u[d] = ud
                    changed = True

                # tiny random reset to avoid collapse
                if spans[d] > 0.0 and random.random() < 0.001:
                    u[d] = lows[d] + spans[d] * random.random()
                    changed = True

            if not changed:
                # ensure at least one dimension changes
                d = jrand
                u[d] = lows[d] + spans[d] * random.random()

            fu = safe_eval(u)

            if fu <= fi:
                # archive update
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))

                X[i], Fx[i] = u, fu

                S_F.append(Fi)
                S_CR.append(CRi)
                df = fi - fu
                if df < 0.0: df = 0.0
                dF.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    stagnation = max(0, stagnation - 6)

        # adapt memories
        if S_F:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, dF):
                ww = w / wsum
                num += ww * (fval * fval)
                den += ww * fval
            F_lehmer = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))

            # weighted mean for CR
            cr_mean = 0.0
            for crval, w in zip(S_CR, dF):
                cr_mean += (w / wsum) * crval

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * cr_mean

            if mu_F < 0.05: mu_F = 0.05
            if mu_F > 0.95: mu_F = 0.95
            if mu_CR < 0.05: mu_CR = 0.05
            if mu_CR > 0.95: mu_CR = 0.95
        else:
            # no progress -> more exploration
            mu_F = min(0.95, mu_F + 0.03)
            mu_CR = max(0.05, mu_CR - 0.04)

        # intensify near the end or after strong stagnation
        if best_x is not None and (it % 16 == 0 or stagnation > 140):
            tl = max(0.0, deadline - time.time())
            # allocate refine effort increasing towards the end
            if tl < 0.20:
                budget = 140
                step_frac = 0.015
            elif tl < 0.60:
                budget = 90
                step_frac = 0.025
            else:
                budget = 45
                step_frac = 0.06 * (0.35 + 0.65 * (1.0 - tf))
            rx, rf = local_refine(best_x, best, eval_budget=budget, step_frac=step_frac)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject into worst
                widx = max(range(pop), key=lambda i: Fx[i])
                X[widx], Fx[widx] = best_x[:], best
                stagnation = 0

    return best
