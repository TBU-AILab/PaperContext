import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improvements over the provided code (kept self-contained and robust):
      1) Better global search: multi-armed mutation mix (pbest-DE + rand/1 + best/1) chosen adaptively
      2) Stronger stagnation handling: partial restart around best + random immigrants + archive refresh
      3) Much better local search: bounded Nelder–Mead-style simplex (small, cheap) + coordinate polish
      4) More reliable bound handling: reflection + convex pull-back to parent on repeated violations
      5) Safer caching: exact float tuple (rounded) rather than span-scaled huge integers
      6) Early-stop in inner loops based on remaining time (no overruns)

    Returns: best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    if dim <= 0:
        return float("inf")

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

    def clip(xd, d):
        if xd < lows[d]:
            return lows[d]
        if xd > highs[d]:
            return highs[d]
        return xd

    def reflect_into(xd, d):
        lo, hi = lows[d], highs[d]
        s = spans[d]
        if s <= 0.0:
            return lo
        # reflect a few times then clip
        for _ in range(4):
            if xd < lo:
                xd = lo + (lo - xd)
            elif xd > hi:
                xd = hi - (xd - hi)
            else:
                return xd
        return clip(xd, d)

    def repair_vec(u, parent, max_tries=3):
        # if some dims far out, pull towards parent (convex) after reflection
        # (helps when reflection bounces repeatedly)
        for _ in range(max_tries):
            ok = True
            for d in range(dim):
                if u[d] < lows[d] - 1e-15 or u[d] > highs[d] + 1e-15:
                    ok = False
                    break
            if ok:
                return u
            a = 0.5  # pull factor
            for d in range(dim):
                u[d] = clip(a * parent[d] + (1.0 - a) * reflect_into(u[d], d), d)
        # final clip
        for d in range(dim):
            u[d] = clip(u[d], d)
        return u

    # ---------------- eval + cache ----------------
    cache = {}
    cache_max = 12000

    def cache_key(x):
        # round to stabilize key; still passes original x to func
        # 1e-12 relative to span; fallback absolute if span is 0
        key = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                key.append(0.0)
            else:
                key.append(round((x[d] - lows[d]) / s, 12))
        return tuple(key)

    def safe_eval(x):
        k = cache_key(x)
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

    # ---------------- sampling helpers ----------------
    def rand_point():
        return [lows[d] + spans[d] * random.random() for d in range(dim)]

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
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][i] + random.random()) * inv
                x[d] = lows[d] + spans[d] * u
            pts.append(x)
        return pts

    def normal(mu, sig):
        u1 = max(1e-300, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sig * z

    def cauchy(mu, gam):
        u = random.random() - 0.5
        return mu + gam * math.tan(math.pi * u)

    # ---------------- local search: coordinate polish ----------------
    def coord_polish(x0, f0, budget, step_frac):
        x = x0[:]
        fx = f0
        if budget <= 0:
            return x, fx
        steps = [step_frac * spans[d] for d in range(dim)]
        shrink = 0.5
        min_step = 1e-12
        evals = 0

        while evals < budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                if evals >= budget or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= min_step or spans[d] <= 0.0:
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

    # ---------------- local search: small bounded Nelder-Mead ----------------
    def nelder_mead(xbest, fbest, budget, init_scale):
        if budget <= 0 or xbest is None:
            return xbest, fbest
        n = dim
        # build simplex: xbest + scaled basis moves
        simplex = [xbest[:]]
        values = [fbest]
        scale = init_scale
        for i in range(n):
            xi = xbest[:]
            if spans[i] > 0.0:
                xi[i] = clip(xi[i] + scale * spans[i], i)
            simplex.append(xi)
        # evaluate remaining points (respect budget)
        evals = 0
        for k in range(1, len(simplex)):
            if evals >= budget or time.time() >= deadline:
                return xbest, fbest
            fk = safe_eval(simplex[k]); evals += 1
            values.append(fk)
            if fk < fbest:
                fbest = fk
                xbest = simplex[k][:]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while evals < budget and time.time() < deadline:
            # sort
            idx = sorted(range(n + 1), key=lambda i: values[i])
            simplex = [simplex[i] for i in idx]
            values = [values[i] for i in idx]

            if values[0] < fbest:
                fbest = values[0]
                xbest = simplex[0][:]

            # centroid of best n
            centroid = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += simplex[j][i]
                centroid[i] = s / n

            worst = simplex[-1]
            # reflection
            xr = [0.0] * n
            for i in range(n):
                xr[i] = centroid[i] + alpha * (centroid[i] - worst[i])
                xr[i] = reflect_into(xr[i], i)
            xr = repair_vec(xr, worst)
            fr = safe_eval(xr); evals += 1
            if fr < values[0]:
                # expansion
                xe = [0.0] * n
                for i in range(n):
                    xe[i] = centroid[i] + gamma * (xr[i] - centroid[i])
                    xe[i] = reflect_into(xe[i], i)
                xe = repair_vec(xe, xr)
                fe = safe_eval(xe); evals += 1
                if fe < fr:
                    simplex[-1], values[-1] = xe, fe
                else:
                    simplex[-1], values[-1] = xr, fr
            elif fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            else:
                # contraction
                xc = [0.0] * n
                if fr < values[-1]:
                    # outside
                    for i in range(n):
                        xc[i] = centroid[i] + rho * (xr[i] - centroid[i])
                        xc[i] = reflect_into(xc[i], i)
                else:
                    # inside
                    for i in range(n):
                        xc[i] = centroid[i] - rho * (centroid[i] - worst[i])
                        xc[i] = reflect_into(xc[i], i)
                xc = repair_vec(xc, worst)
                fc = safe_eval(xc); evals += 1
                if fc < values[-1]:
                    simplex[-1], values[-1] = xc, fc
                else:
                    # shrink
                    bestp = simplex[0]
                    for j in range(1, n + 1):
                        if evals >= budget or time.time() >= deadline:
                            break
                        xj = simplex[j]
                        for i in range(n):
                            xj[i] = clip(bestp[i] + sigma * (xj[i] - bestp[i]), i)
                        fj = safe_eval(xj); evals += 1
                        simplex[j] = xj
                        values[j] = fj

        # return best in simplex
        bi = min(range(n + 1), key=lambda i: values[i])
        return simplex[bi], values[bi]

    # ---------------- DE main ----------------
    pop = max(26, min(180, 18 + 7 * dim))
    immigrants = max(1, pop // 18)
    A = []
    Amax = pop

    mu_F = 0.55
    mu_CR = 0.85
    c_adapt = 0.12

    # mutation arm weights (adaptive)
    # 0: current-to-pbest/1 (JADE)
    # 1: rand/1
    # 2: best/1
    w = [0.62, 0.23, 0.15]
    succ = [1.0, 1.0, 1.0]  # success counters (start nonzero)

    best = float("inf")
    best_x = None

    # init population via LHS + random
    X, Fx = [], []
    init = lhs_points(min(pop, max(14, pop // 2)))
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

    stagnation = 0
    it = 0

    while time.time() < deadline:
        it += 1
        now = time.time()
        tf = (now - t0) / max(1e-12, (deadline - t0))
        if tf < 0.0:
            tf = 0.0
        elif tf > 1.0:
            tf = 1.0

        # update best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagnation = max(0, stagnation - 5)
        else:
            stagnation += 1

        # pbest schedule
        pbest_rate = max(0.08, 0.34 - 0.26 * tf)
        top = max(2, int(pbest_rate * pop))
        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])

        # stagnation actions
        if stagnation > 90 and time.time() < deadline:
            # replace worst few with samples around best + some pure random
            k = max(immigrants, pop // 10)
            worst = sorted(range(pop), key=lambda i: Fx[i], reverse=True)[:k]
            for j, idx in enumerate(worst):
                if time.time() >= deadline:
                    break
                if best_x is not None and j < (2 * k) // 3:
                    xnew = best_x[:]
                    # gaussian perturbation scaled by time (smaller late)
                    scale = (0.18 * (1.0 - 0.75 * tf))  # fraction of span
                    for d in range(dim):
                        if spans[d] > 0.0:
                            xnew[d] = reflect_into(xnew[d] + normal(0.0, 1.0) * scale * spans[d], d)
                    xnew = repair_vec(xnew, best_x)
                else:
                    xnew = rand_point()
                fnew = safe_eval(xnew)
                X[idx], Fx[idx] = xnew, fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
            # reduce archive too (freshen)
            if len(A) > Amax // 2:
                random.shuffle(A)
                A = A[:Amax // 2]
            stagnation = 55  # prevent thrashing

        # union for r2
        union = X + A
        union_n = len(union)

        S_F, S_CR, dF = [], [], []
        arm_succ = [0, 0, 0]
        arm_try = [0, 0, 0]

        # iterate individuals
        for i in range(pop):
            if time.time() >= deadline:
                break

            xi, fi = X[i], Fx[i]

            # sample DE params (JADE-like)
            Fi = cauchy(mu_F, 0.1)
            for _ in range(6):
                if Fi > 0.0:
                    break
                Fi = cauchy(mu_F, 0.1)
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(mu_CR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose mutation arm by roulette
            r = random.random() * (w[0] + w[1] + w[2])
            if r < w[0]:
                arm = 0
            elif r < w[0] + w[1]:
                arm = 1
            else:
                arm = 2
            arm_try[arm] += 1

            # indices helpers
            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # pick distinct indices
            forbidden = {i}
            if arm == 0:
                forbidden.add(pbest)

            def pick_pop(excl):
                while True:
                    rj = random.randrange(pop)
                    if rj not in excl:
                        return rj

            def pick_union(excl):
                if union_n <= 1:
                    return 0
                while True:
                    rj = random.randrange(union_n)
                    if rj >= pop:
                        return rj
                    if rj not in excl:
                        return rj

            # build mutant v
            if arm == 0:
                r1 = pick_pop(forbidden); forbidden.add(r1)
                r2u = pick_union(forbidden)
                xr1 = X[r1]
                xr2 = union[r2u]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            elif arm == 1:
                r0 = pick_pop(forbidden); forbidden.add(r0)
                r1 = pick_pop(forbidden); forbidden.add(r1)
                r2u = pick_union(forbidden)
                x0 = X[r0]
                x1 = X[r1]
                x2 = union[r2u]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = x0[d] + Fi * (x1[d] - x2[d])
            else:
                # best/1 with archive difference
                if best_x is None:
                    best_base = xi
                else:
                    best_base = best_x
                r1 = pick_pop(forbidden); forbidden.add(r1)
                r2u = pick_union(forbidden)
                xr1 = X[r1]
                xr2 = union[r2u]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = best_base[d] + Fi * (xr1[d] - xr2[d])

            # crossover -> trial u
            jrand = random.randrange(dim)
            u = xi[:]
            changed = False
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ud = reflect_into(v[d], d)
                    u[d] = ud
                    changed = True
            if not changed:
                d = jrand
                u[d] = reflect_into(v[d], d)

            # repair (rarely needed but helps stability)
            u = repair_vec(u, xi)

            fu = safe_eval(u)

            if fu <= fi:
                # success: update archive, accept, record for adaptation
                if Amax > 0:
                    A.append(xi)
                    if len(A) > Amax:
                        A.pop(random.randrange(len(A)))

                X[i], Fx[i] = u, fu

                S_F.append(Fi)
                S_CR.append(CRi)
                imp = fi - fu
                if imp < 0.0:
                    imp = 0.0
                dF.append(imp)

                arm_succ[arm] += 1

                if fu < best:
                    best, best_x = fu, u[:]
                    stagnation = max(0, stagnation - 8)

        # adapt DE memories
        if S_F:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for fval, ww in zip(S_F, dF):
                wgt = ww / wsum
                num += wgt * (fval * fval)
                den += wgt * fval
            F_leh = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))
            # weighted mean for CR
            CRm = 0.0
            for cr, ww in zip(S_CR, dF):
                CRm += (ww / wsum) * cr

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_leh
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * CRm
            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.95, max(0.05, mu_CR))
        else:
            mu_F = min(0.95, mu_F + 0.02)
            mu_CR = max(0.05, mu_CR - 0.03)

        # adapt arm weights (soft)
        for a in range(3):
            succ[a] = 0.85 * succ[a] + 0.15 * (arm_succ[a] + 1.0) / (arm_try[a] + 1.0)
        ssum = succ[0] + succ[1] + succ[2]
        w = [succ[0] / ssum, succ[1] / ssum, succ[2] / ssum]
        # keep some minimum diversity
        w = [max(0.08, w[0]), max(0.08, w[1]), max(0.06, w[2])]
        ssum = w[0] + w[1] + w[2]
        w = [w[0] / ssum, w[1] / ssum, w[2] / ssum]

        # periodic intensification (NM + coordinate)
        if best_x is not None and (it % 14 == 0 or stagnation > 120):
            tl = deadline - time.time()
            if tl > 0.05:
                # budgets scale with remaining time; keep bounded
                nm_budget = 40 if tl > 0.8 else 65 if tl > 0.25 else 90
                nm_scale = 0.08 * (0.35 + 0.65 * (1.0 - tf))  # smaller late
                xb, fb = nelder_mead(best_x, best, budget=nm_budget, init_scale=nm_scale)
                if fb < best:
                    best, best_x = fb, xb[:]
                    stagnation = 0
                    # inject into worst
                    widx = max(range(pop), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best

            tl = deadline - time.time()
            if best_x is not None and tl > 0.03:
                pol_budget = 60 if tl > 0.6 else 90 if tl > 0.2 else 130
                step_frac = 0.03 if tl > 0.6 else 0.02 if tl > 0.2 else 0.012
                xb, fb = coord_polish(best_x, best, budget=pol_budget, step_frac=step_frac)
                if fb < best:
                    best, best_x = fb, xb[:]
                    stagnation = 0
                    widx = max(range(pop), key=lambda i: Fx[i])
                    X[widx], Fx[widx] = best_x[:], best

    return best
