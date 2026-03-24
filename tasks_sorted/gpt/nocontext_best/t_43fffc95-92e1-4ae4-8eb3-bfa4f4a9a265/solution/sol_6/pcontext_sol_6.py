import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (self-contained, no external libs).

    Improvements over the provided hybrid-DE:
      - Adds global step-size annealing + heavy-tail restarts near-best (more robust escapes)
      - Uses a cleaner SHADE-like parameter memory (multiple memories instead of single mu_F/mu_CR)
      - Stronger, safer bound handling (bounce-back + parent-mix repair)
      - Stagnation detection triggers: (a) partial restart via LHS, (b) "best-bias" mutation burst
      - Adds an explicit trust-region local search (adaptive coordinate + random subspace)
      - Lighter caching (hash of rounded coords), avoids pathological huge-keys

    Returns: best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---------------- bounds prep ----------------
    lows, highs, spans = [0.0]*dim, [0.0]*dim, [0.0]*dim
    for d in range(dim):
        lo, hi = float(bounds[d][0]), float(bounds[d][1])
        if hi < lo:
            lo, hi = hi, lo
        lows[d], highs[d] = lo, hi
        spans[d] = max(0.0, hi - lo)

    def clip(x, d):
        if x < lows[d]: return lows[d]
        if x > highs[d]: return highs[d]
        return x

    def bounce(x, d):
        """Reflect/bounce into bounds."""
        lo, hi = lows[d], highs[d]
        s = spans[d]
        if s <= 0.0:
            return lo
        # reflect repeatedly, then clip as final safety
        for _ in range(6):
            if x < lo:
                x = lo + (lo - x)
            elif x > hi:
                x = hi - (x - hi)
            else:
                return x
        return clip(x, d)

    # ---------------- eval + cache ----------------
    # Cache uses rounding to 12 significant-ish digits relative to span to be effective
    cache = {}
    cache_max = 6000

    def key_of(x):
        k = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                k.append(0)
            else:
                # map to integer grid; ~1e-10 of span resolution
                k.append(int((x[d] - lows[d]) / (s * 1e-10 + 1e-30)))
        return tuple(k)

    def safe_eval(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        try:
            v = float(func(x))
            if not (v == v) or v == float("inf") or v == float("-inf"):
                v = float("inf")
        except Exception:
            v = float("inf")
        if len(cache) < cache_max:
            cache[k] = v
        return v

    # ---------------- RNG helpers ----------------
    def rand_point():
        return [lows[d] + spans[d] * random.random() if spans[d] > 0 else lows[d] for d in range(dim)]

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
                if spans[d] <= 0:
                    x[d] = lows[d]
                else:
                    u = (perms[d][k] + random.random()) * inv
                    x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    def normal(mu, sigma):
        u1 = max(1e-300, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def cauchy(mu, gamma):
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    # ---------------- population + SHADE memory ----------------
    pop = max(28, min(180, 16 + 9 * dim))
    Amax = pop
    immigrants = max(2, pop // 12)

    # SHADE-style memories
    H = 8  # number of memory slots
    M_F = [0.6] * H
    M_CR = [0.85] * H
    mem_idx = 0

    # archive
    A = []

    # best
    best = float("inf")
    best_x = None

    # ---------------- init population ----------------
    X, Fx = [], []
    init = lhs_block(min(pop, max(14, pop // 2)))
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

    # ---------------- local search (trust region) ----------------
    def local_trust_search(x0, f0, budget, base_step):
        """
        Adaptive trust-region coordinate + random subspace search.
        base_step is fraction of span.
        """
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        # trust radii per dimension
        r = [base_step * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] <= 0.0:
                r[d] = 0.0

        min_r = 1e-14
        grow = 1.25
        shrink = 0.55

        evals = 0
        while evals < budget and time.time() < deadline:
            improved = False

            # random-subspace trial (cheap direction)
            if evals < budget and time.time() < deadline:
                xt = x[:]
                anychg = False
                # choose ~sqrt(dim) dims
                k = int(math.sqrt(dim) + 0.5)
                if k < 1: k = 1
                for _ in range(k):
                    d = random.randrange(dim)
                    if r[d] > 0:
                        xt[d] = clip(xt[d] + (2.0 * random.random() - 1.0) * r[d], d)
                        anychg = True
                if anychg:
                    ft = safe_eval(xt); evals += 1
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True

            # coordinate polling with best-improving move
            for d in range(dim):
                if evals >= budget or time.time() >= deadline:
                    break
                if r[d] <= min_r:
                    continue

                # try +r, -r (pick best)
                xp = x[:]; xp[d] = clip(xp[d] + r[d], d)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= budget or time.time() >= deadline:
                    break

                xm = x[:]; xm[d] = clip(xm[d] - r[d], d)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            # trust region update
            if improved:
                for d in range(dim):
                    r[d] *= grow
                    # cap radius
                    if r[d] > 0.35 * spans[d]:
                        r[d] = 0.35 * spans[d]
            else:
                maxr = 0.0
                for d in range(dim):
                    r[d] *= shrink
                    if r[d] > maxr: maxr = r[d]
                if maxr <= min_r:
                    break

        return x, fx

    # ---------------- main loop ----------------
    stagn = 0
    it = 0
    last_best = best

    while time.time() < deadline:
        it += 1
        now = time.time()
        tf = (now - t0) / max(1e-12, (deadline - t0))
        if tf < 0.0: tf = 0.0
        if tf > 1.0: tf = 1.0

        # update best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagn = max(0, stagn - 5)
        else:
            stagn += 1

        # pbest schedule
        pbest_rate = 0.40 - 0.30 * tf  # 0.40 -> 0.10
        if pbest_rate < 0.08:
            pbest_rate = 0.08
        top = max(2, int(pbest_rate * pop))

        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])

        # union for archive usage
        union = X + A
        union_n = len(union)

        # success records
        S_F, S_CR, dF = [], [], []

        # annealed global scaling (more exploration early)
        global_scale = 0.9 - 0.55 * tf  # 0.9 -> 0.35
        if global_scale < 0.25:
            global_scale = 0.25

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi, fi = X[i], Fx[i]

            # pick memory slot
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            Fi = cauchy(muF, 0.12)
            tries = 0
            while Fi <= 0.0 and tries < 8:
                Fi = cauchy(muF, 0.12)
                tries += 1
            if Fi <= 0.0: Fi = 0.08
            if Fi > 1.0: Fi = 1.0
            Fi *= global_scale
            if Fi < 0.02: Fi = 0.02

            CRi = normal(muCR, 0.10)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # r1 from pop excluding i,pbest
            forbidden = {i, pbest}
            while True:
                r1 = random.randrange(pop)
                if r1 not in forbidden:
                    break
            forbidden.add(r1)

            # r2 from union excluding forbidden if from pop
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

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            # occasional stronger best-bias burst when stagnating
            best_pull = 0.0
            if stagn > 70 and best_x is not None and random.random() < 0.22:
                best_pull = 0.45 * Fi

            for d in range(dim):
                if spans[d] <= 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                    if best_pull != 0.0:
                        v[d] += best_pull * (best_x[d] - xi[d])

            # binomial crossover + repair
            u = xi[:]
            jrand = random.randrange(dim)
            changed = False
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ud = bounce(v[d], d)
                    # mix with parent a bit if on/near boundary to reduce edge-trapping
                    if spans[d] > 0.0:
                        if ud <= lows[d] or ud >= highs[d] or random.random() < 0.015:
                            ud = 0.70 * ud + 0.30 * xi[d]
                            ud = clip(ud, d)
                    u[d] = ud
                    changed = True

            if not changed:
                d = jrand
                u[d] = lows[d] + spans[d] * random.random() if spans[d] > 0 else lows[d]

            # heavy-tail near-best restart (rare, helps jump basins)
            if best_x is not None and (stagn > 90) and random.random() < 0.03:
                for d in range(dim):
                    if spans[d] > 0.0 and random.random() < 0.35:
                        # Student-t-ish using cauchy then clamp
                        step = 0.08 * spans[d] * math.tan(math.pi * (random.random() - 0.5))
                        u[d] = clip(best_x[d] + step, d)

            fu = safe_eval(u)

            if fu <= fi:
                # archive update
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))

                X[i], Fx[i] = u, fu

                S_F.append(Fi / max(1e-12, global_scale))  # store un-annealed-ish for memory stability
                S_CR.append(CRi)
                df = fi - fu
                if df < 0.0: df = 0.0
                dF.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    stagn = max(0, stagn - 8)

        # memory update (SHADE): weighted Lehmer mean for F, weighted mean for CR
        if S_F:
            wsum = sum(dF) + 1e-300

            # CR
            cr_new = 0.0
            for crv, w in zip(S_CR, dF):
                cr_new += (w / wsum) * crv

            # F (Lehmer)
            num = 0.0
            den = 0.0
            for fv, w in zip(S_F, dF):
                ww = w / wsum
                num += ww * (fv * fv)
                den += ww * fv
            f_new = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))

            # update cyclic memory
            M_F[mem_idx] = min(0.98, max(0.06, f_new))
            M_CR[mem_idx] = min(0.98, max(0.02, cr_new))
            mem_idx = (mem_idx + 1) % H
        else:
            # no success -> encourage exploration by nudging memories
            j = random.randrange(H)
            M_F[j] = min(0.98, M_F[j] + 0.05)
            M_CR[j] = max(0.02, M_CR[j] - 0.06)

        # stagnation actions: partial restart with LHS (replace worst)
        if stagn > 140 and time.time() < deadline:
            k = max(immigrants, pop // 9)
            pts = lhs_block(k)
            worst = sorted(range(pop), key=lambda i: Fx[i], reverse=True)[:k]
            for idx, xnew in zip(worst, pts):
                if time.time() >= deadline:
                    break
                fnew = safe_eval(xnew)
                X[idx], Fx[idx] = xnew, fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    stagn = 0
            stagn = 95  # avoid immediate repeated restarts

        # periodic local trust-region search around current best
        if best_x is not None and (it % 14 == 0 or (stagn > 110 and it % 7 == 0)):
            tl = deadline - time.time()
            if tl > 0.0:
                if tl < 0.25:
                    budget = 160
                    step = 0.010
                elif tl < 0.70:
                    budget = 110
                    step = 0.020
                else:
                    budget = 70
                    step = 0.045 * (0.35 + 0.65 * (1.0 - tf))
                rx, rf = local_trust_search(best_x, best, budget, step)
                if rf < best:
                    best, best_x = rf, rx[:]
                    # inject into worst
                    widx = max(range(pop), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best
                    stagn = 0

        # if still no improvement for a long time, do a tiny "micro-mutation" on a few individuals
        if (best >= last_best - 1e-18):
            if stagn > 200 and time.time() < deadline and best_x is not None:
                m = max(2, pop // 12)
                for _ in range(m):
                    if time.time() >= deadline:
                        break
                    idx = random.randrange(pop)
                    x = X[idx][:]
                    # jitter a couple dims around best
                    for __ in range(2):
                        d = random.randrange(dim)
                        if spans[d] > 0:
                            x[d] = clip(best_x[d] + (random.random() - 0.5) * 0.12 * spans[d], d)
                    f = safe_eval(x)
                    if f < Fx[idx]:
                        X[idx], Fx[idx] = x, f
                        if f < best:
                            best, best_x = f, x[:]
                            stagn = 0
                stagn = 140
        last_best = best

    return best
