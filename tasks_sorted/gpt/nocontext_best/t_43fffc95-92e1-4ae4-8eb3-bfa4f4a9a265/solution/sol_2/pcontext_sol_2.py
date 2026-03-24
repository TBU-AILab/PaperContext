import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Hybrid algorithm:
      - LHS-like initialization
      - DE (current-to-pbest/1) with *archive* (JADE-like) for stronger exploration
      - Success-history style parameter adaptation (lightweight)
      - Opposition restart on stagnation + random immigrants
      - Budgeted local coordinate search around best near the end

    Returns best fitness found (float).
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

    def clip(xi, i):
        lo = lows[i]
        hi = highs[i]
        if xi < lo:
            return lo
        if xi > hi:
            return hi
        return xi

    # ---- safe evaluation ----
    def safe_eval(x):
        try:
            v = float(func(x))
            if v != v or v == float("inf") or v == float("-inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---- sampling ----
    def rand_point():
        return [lows[i] + spans[i] * random.random() for i in range(dim)]

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
                u = (perms[d][k] + random.random()) * inv
                x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    # ---- parameters ----
    pop = max(20, min(120, 12 + 7 * dim))  # slightly larger than before tends to help robustness
    pbest_rate = 0.2                       # choose from top p% as pbest
    min_pbest = 2
    immigrants = max(1, pop // 25)

    # success-history means (JADE-like)
    mu_F = 0.55
    mu_CR = 0.85
    c_adapt = 0.12   # adaptation speed

    # archive for replaced individuals (diversity)
    A = []
    Amax = pop

    # local refine configuration
    refine_interval = 18
    refine_min_budget = 25
    refine_max_budget = 80

    # ---- init population ----
    X = []
    Fx = []
    best = float("inf")
    best_x = None

    init_n = min(pop, max(10, pop // 2))
    pts = lhs_points(init_n)
    while len(pts) < pop:
        pts.append(rand_point())

    for x in pts:
        if time.time() >= deadline:
            return best
        f = safe_eval(x)
        X.append(x)
        Fx.append(f)
        if f < best:
            best, best_x = f, x[:]

    # ---- helpers ----
    def cauchy(loc, scale):
        # Cauchy via inverse CDF
        u = random.random() - 0.5
        return loc + scale * math.tan(math.pi * u)

    def normal(loc, scale):
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(max(1e-300, u1))) * math.cos(2.0 * math.pi * u2)
        return loc + scale * z

    def pick_index_excluding(n, forbidden_set):
        # small n: rejection sampling is fine
        while True:
            r = random.randrange(n)
            if r not in forbidden_set:
                return r

    def get_pbest_index():
        # choose from top pbest_rate fraction
        order = sorted(range(pop), key=lambda i: Fx[i])
        top = max(min_pbest, int(pbest_rate * pop))
        return order[random.randrange(top)]

    def reflect_to_bounds(val, d):
        # reflect a couple times then clip (better than hard clip for DE)
        lo = lows[d]; hi = highs[d]
        if spans[d] == 0.0:
            return lo
        for _ in range(2):
            if val < lo:
                val = lo + (lo - val)
            elif val > hi:
                val = hi - (val - hi)
            else:
                return val
        return clip(val, d)

    # ---- local refinement (coordinate search with shrinking steps) ----
    def refine(best_x, best_f, budget, step_frac):
        x = best_x[:]
        fx = best_f
        steps = [step_frac * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] == 0.0:
                steps[d] = 0.0

        shrink = 0.55
        tiny = 1e-12
        evals = 0
        while evals < budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                if evals >= budget or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= tiny:
                    continue

                # + step
                xp = x[:]
                xp[d] = clip(xp[d] + s, d)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= budget or time.time() >= deadline:
                    break

                # - step
                xm = x[:]
                xm[d] = clip(xm[d] - s, d)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                m = 0.0
                for d in range(dim):
                    steps[d] *= shrink
                    if steps[d] > m:
                        m = steps[d]
                if m <= tiny:
                    break
        return x, fx

    # ---- main loop ----
    stagnation = 0
    it = 0

    while time.time() < deadline:
        it += 1

        # track best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagnation = max(0, stagnation - 3)
        else:
            stagnation += 1

        # stagnation handling: opposition jump + immigrants
        if stagnation > 80 and time.time() < deadline:
            # opposition around mid-point
            for _ in range(immigrants):
                j = random.randrange(pop)
                xj = X[j]
                xo = [0.0] * dim
                for d in range(dim):
                    if spans[d] == 0.0:
                        xo[d] = lows[d]
                    else:
                        # opposition
                        xo[d] = lows[d] + highs[d] - xj[d]
                        # small jitter
                        xo[d] = reflect_to_bounds(xo[d] + 0.02 * spans[d] * (random.random() - 0.5), d)
                fo = safe_eval(xo)
                X[j], Fx[j] = xo, fo
                if fo < best:
                    best, best_x = fo, xo[:]
            stagnation = 55

        # generate control params per individual; collect successes
        S_F = []
        S_CR = []
        delta_f = []

        # sort indices for pbest selection efficiency
        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])

        # create union size for r2 selection
        # X U A (for r2)
        union = X + A
        union_n = len(union)

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi = X[i]
            fi = Fx[i]

            # sample Fi from Cauchy, CRi from Normal (JADE-like)
            Fi = cauchy(mu_F, 0.1)
            # resample until > 0
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(mu_CR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose pbest among top p%
            top = max(min_pbest, int(pbest_rate * pop))
            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # choose r1 from population, r2 from union (population + archive)
            forbidden = {i, pbest}
            r1 = pick_index_excluding(pop, forbidden)
            forbidden.add(r1)

            # r2 from union, mapped to either pop or archive
            # ensure not the same vector index when r2 happens to be from pop
            if union_n <= 1:
                r2u = r1
            else:
                # forbid pop indices i,pbest,r1, but allow archive indices freely
                while True:
                    r2u = random.randrange(union_n)
                    if r2u >= pop:
                        break
                    if r2u not in forbidden:
                        break
            xr1 = X[r1]
            xr2 = union[r2u]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = reflect_to_bounds(v[d], d)

                # very small random coordinate reset to prevent collapse
                if spans[d] > 0.0 and random.random() < 0.0015:
                    u[d] = lows[d] + spans[d] * random.random()

            fu = safe_eval(u)

            # selection + archive update
            if fu <= fi:
                # add replaced parent to archive (if different)
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        # random removal
                        A.pop(random.randrange(len(A)))

                X[i] = u
                Fx[i] = fu

                # record successful parameters
                S_F.append(Fi)
                S_CR.append(CRi)
                df = fi - fu
                if df < 0.0:
                    df = 0.0
                delta_f.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    stagnation = max(0, stagnation - 5)

        # update mu_F, mu_CR based on successes (weighted)
        if S_F:
            wsum = sum(delta_f) + 1e-300
            # Lehmer mean for F (promotes larger successful F)
            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, delta_f):
                ww = w / wsum
                num += ww * (fval * fval)
                den += ww * fval
            F_lehmer = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))

            # weighted arithmetic mean for CR
            cr_mean = 0.0
            for crval, w in zip(S_CR, delta_f):
                cr_mean += (w / wsum) * crval

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * cr_mean

            # keep in reasonable bounds
            if mu_F < 0.05: mu_F = 0.05
            if mu_F > 0.95: mu_F = 0.95
            if mu_CR < 0.05: mu_CR = 0.05
            if mu_CR > 0.95: mu_CR = 0.95
        else:
            # if no improvements, nudge towards exploration
            mu_F = min(0.95, mu_F + 0.02)
            mu_CR = max(0.10, mu_CR - 0.03)

        # periodic local refinement; spend more near the end
        if best_x is not None and (it % refine_interval == 0 or stagnation > 110):
            tl = max(0.0, deadline - time.time())
            # allocate small eval budget; increase when close to deadline
            if tl < 0.25:
                budget = refine_max_budget
                step_frac = 0.02
            else:
                budget = refine_min_budget
                step_frac = 0.06 * max(0.25, 1.0 - stagnation / 200.0)

            rx, rf = refine(best_x, best, budget=budget, step_frac=step_frac)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject into worst
                widx = max(range(pop), key=lambda i: Fx[i])
                X[widx], Fx[widx] = best_x[:], best
                stagnation = 0

    return best
