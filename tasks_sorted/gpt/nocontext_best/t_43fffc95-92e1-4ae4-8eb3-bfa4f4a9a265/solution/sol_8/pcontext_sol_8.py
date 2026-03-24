import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (self-contained, no external libs).

    Improvements over the provided code (still DE-family, but stronger in practice):
      1) Multi-strategy DE per individual (jDE-style self-adaptive F/CR):
         - randomly mixes: current-to-pbest/1, rand/1, best/1 (rare), plus optional "best-pull"
         - each individual carries its own F and CR that evolve, reducing manual tuning sensitivity
      2) Proper bound handling using "bounce-back toward parent" (keeps feasibility and locality)
      3) Stagnation control:
         - lightweight restarts of worst individuals using LHS
         - shrinking "sigma" Gaussian sampling around best late in the run (intensification)
      4) Cheap local improvement: small coordinate search around best on stagnation/late time
      5) Evaluation cache (quantized) to avoid wasting time on identical proposals

    Returns: best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # -------- bounds ----------
    lows = [0.0] * dim
    highs = [0.0] * dim
    spans = [0.0] * dim
    for d in range(dim):
        lo, hi = float(bounds[d][0]), float(bounds[d][1])
        if hi < lo:
            lo, hi = hi, lo
        lows[d], highs[d] = lo, hi
        spans[d] = max(0.0, hi - lo)

    def clip(v, d):
        if v < lows[d]:
            return lows[d]
        if v > highs[d]:
            return highs[d]
        return v

    def repair_to_parent(trial, parent):
        # "bounce back" toward parent if out of bounds: keeps feasibility and prevents sticking on edges
        x = trial[:]
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
                continue
            if x[d] < lows[d] or x[d] > highs[d]:
                # random convex combo between parent and nearest bound-projected point
                projected = clip(x[d], d)
                lam = random.random()
                x[d] = lam * parent[d] + (1.0 - lam) * projected
                # very small safety clip
                x[d] = clip(x[d], d)
        return x

    # -------- cache ----------
    cache = {}
    cache_max = 6000

    def key_of(x):
        # quantize relative to span; modest coarseness to avoid collisions too much
        k = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                k.append(0)
            else:
                # about 1e-10 span resolution
                k.append(int((x[d] - lows[d]) / (s * 1e-10 + 1e-30)))
        return tuple(k)

    def safe_eval(x):
        kk = key_of(x)
        if kk in cache:
            return cache[kk]
        try:
            v = float(func(x))
            if v != v or v in (float("inf"), float("-inf")):
                v = float("inf")
        except Exception:
            v = float("inf")
        if len(cache) < cache_max:
            cache[kk] = v
        return v

    # -------- sampling helpers ----------
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
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][i] + random.random()) * inv
                x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    def gauss(mu, sigma):
        u1 = max(1e-300, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    # -------- local search around best ----------
    def coord_search(x0, f0, budget, step_frac):
        x = x0[:]
        fx = f0
        if x is None:
            return x0, f0
        steps = [step_frac * spans[d] for d in range(dim)]
        min_step = 1e-12
        shrink = 0.6
        evals = 0
        while evals < budget and time.time() < deadline:
            improved = False
            # try a random order of coordinates
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if evals >= budget or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= min_step or spans[d] == 0.0:
                    continue

                xp = x[:]
                xp[d] = clip(xp[d] + s, d)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                if evals >= budget or time.time() >= deadline:
                    break

                xm = x[:]
                xm[d] = clip(xm[d] - s, d)
                fm = safe_eval(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            if not improved:
                mx = 0.0
                for d in range(dim):
                    steps[d] *= shrink
                    if steps[d] > mx:
                        mx = steps[d]
                if mx <= min_step:
                    break
        return x, fx

    # -------- population size ----------
    pop = max(28, min(180, 18 + 9 * dim))
    # self-adaptive per-individual parameters (jDE flavor)
    F = [0.5 + 0.3 * random.random() for _ in range(pop)]
    CR = [0.6 + 0.35 * random.random() for _ in range(pop)]

    # strategy rates
    tau1 = 0.10   # chance to reset F
    tau2 = 0.10   # chance to reset CR

    # archive (for diversity in difference vectors)
    A = []
    Amax = pop

    # -------- init ----------
    X = []
    Fx = []
    init = lhs_block(min(pop, max(12, pop // 2)))
    while len(init) < pop:
        init.append(rand_point())

    best = float("inf")
    best_x = None

    for x in init:
        if time.time() >= deadline:
            return best
        f = safe_eval(x)
        X.append(x)
        Fx.append(f)
        if f < best:
            best = f
            best_x = x[:]

    stagn = 0
    it = 0

    # -------- main loop ----------
    while time.time() < deadline:
        it += 1

        # time fraction
        tf = (time.time() - t0) / max(1e-12, (deadline - t0))
        if tf < 0.0: tf = 0.0
        if tf > 1.0: tf = 1.0

        # update best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagn = max(0, stagn - 6)
        else:
            stagn += 1

        # pbest size schedule
        p_rate = 0.30 - 0.22 * tf   # 0.30 -> 0.08
        if p_rate < 0.08:
            p_rate = 0.08

        # partial restart on stagnation
        if stagn > 110 and time.time() < deadline:
            k = max(2, pop // 10)
            worst = sorted(range(pop), key=lambda i: Fx[i], reverse=True)[:k]
            fresh = lhs_block(k)
            for idx, xnew in zip(worst, fresh):
                if time.time() >= deadline:
                    break
                fnew = safe_eval(xnew)
                X[idx], Fx[idx] = xnew, fnew
                # reset self-adaptive params
                F[idx] = 0.5 + 0.3 * random.random()
                CR[idx] = 0.6 + 0.35 * random.random()
                if fnew < best:
                    best, best_x = fnew, xnew[:]
            stagn = 70  # reduce thrash

        # sort for pbest
        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])
        top = max(2, int(p_rate * pop))

        union = X + A
        union_n = len(union)

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi = X[i]
            fi = Fx[i]

            # jDE self-adaptation
            if random.random() < tau1:
                F[i] = 0.1 + 0.9 * random.random()
            if random.random() < tau2:
                CR[i] = random.random()

            Fi = F[i]
            CRi = CR[i]

            # pick pbest
            pbest = sorted_idx[random.randrange(top)]
            xp = X[pbest]

            # choose r1, r2
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop)
            r2u = random.randrange(union_n) if union_n > 0 else r1

            xr1 = X[r1]
            xr2 = union[r2u] if union_n > 0 else xr1

            # multi-strategy mutation
            # - mostly current-to-pbest/1 (good balance)
            # - sometimes rand/1 (diversity)
            # - rarely best/1 (strong exploitation late)
            strat = random.random()
            v = [0.0] * dim

            if strat < 0.70:
                # current-to-pbest/1
                for d in range(dim):
                    v[d] = xi[d] + Fi * (xp[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            elif strat < 0.95:
                # rand/1
                a = random.randrange(pop)
                b = random.randrange(pop)
                c = random.randrange(pop)
                while b == a:
                    b = random.randrange(pop)
                while c == a or c == b:
                    c = random.randrange(pop)
                xa, xb, xc = X[a], X[b], X[c]
                for d in range(dim):
                    v[d] = xa[d] + Fi * (xb[d] - xc[d])
            else:
                # best/1 (prefer later)
                if best_x is None:
                    best_x = xi
                a = random.randrange(pop)
                b = random.randrange(pop)
                while b == a:
                    b = random.randrange(pop)
                xa, xb = X[a], X[b]
                for d in range(dim):
                    v[d] = best_x[d] + Fi * (xa[d] - xb[d])

            # optional "best pull" when stagnating / late
            if best_x is not None and (stagn > 60 or tf > 0.65) and random.random() < (0.10 + 0.20 * tf):
                pull = 0.25 + 0.35 * random.random()
                for d in range(dim):
                    v[d] = v[d] + pull * Fi * (best_x[d] - xi[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            changed = False
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
                    changed = True
            if not changed:
                u[jrand] = v[jrand]

            # repair
            u = repair_to_parent(u, xi)

            # rare re-randomize one coordinate to avoid collapse
            if random.random() < 0.002:
                d = random.randrange(dim)
                if spans[d] > 0.0:
                    u[d] = lows[d] + spans[d] * random.random()

            fu = safe_eval(u)

            if fu <= fi:
                # archive
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))
                X[i], Fx[i] = u, fu
                if fu < best:
                    best, best_x = fu, u[:]
                    stagn = max(0, stagn - 8)

        # Intensification: sample around best with shrinking sigma late
        if best_x is not None and time.time() < deadline:
            if (it % 12 == 0 and tf > 0.35) or stagn > 130:
                # a few Gaussian samples around best
                sigma_frac = max(1e-4, 0.08 * (1.0 - tf) + 0.005)
                tries = 6 if tf < 0.8 else 10
                for _ in range(tries):
                    if time.time() >= deadline:
                        break
                    cand = best_x[:]
                    for d in range(dim):
                        if spans[d] > 0.0:
                            cand[d] = clip(gauss(best_x[d], sigma_frac * spans[d]), d)
                    fc = safe_eval(cand)
                    if fc < best:
                        best, best_x = fc, cand[:]
                        # insert into worst
                        widx = max(range(pop), key=lambda i: Fx[i])
                        X[widx], Fx[widx] = best_x[:], best
                        stagn = 0

        # Local search near the end or deep stagnation
        if best_x is not None and time.time() < deadline:
            if (tf > 0.80 and it % 8 == 0) or stagn > 150:
                tl = deadline - time.time()
                if tl < 0.20:
                    budget, step = 140, 0.012
                elif tl < 0.60:
                    budget, step = 90, 0.020
                else:
                    budget, step = 45, 0.05
                rx, rf = coord_search(best_x, best, budget=budget, step_frac=step)
                if rf < best:
                    best, best_x = rf, rx[:]
                    widx = max(range(pop), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best
                    stagn = 0

    return best
