import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (self-contained, no external libs).

    Improvements vs provided hybrid DE:
      - Stronger global-to-local strategy: CMA-ES-like diagonal adaptation + DE/current-to-best
      - Robust restarts with adaptive sampling radius around incumbent (IRace-ish without overhead)
      - Better local refinement: Powell-lite (pattern search) + coordinate steps + parabolic 1D tweak
      - Smarter bound handling: reflect + convex pull-back to feasible parent
      - Tiny evaluation cache (quantized) for refinement loops
      - Time-aware scheduling of exploration/exploitation
    Returns: best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # ---------- bounds ----------
    lows = [0.0] * dim
    highs = [0.0] * dim
    spans = [0.0] * dim
    for d in range(dim):
        lo, hi = float(bounds[d][0]), float(bounds[d][1])
        if hi < lo:
            lo, hi = hi, lo
        lows[d], highs[d] = lo, hi
        s = hi - lo
        spans[d] = s if s > 0.0 else 0.0

    def clip(v, d):
        if v < lows[d]:
            return lows[d]
        if v > highs[d]:
            return highs[d]
        return v

    def reflect(v, d):
        lo, hi = lows[d], highs[d]
        if spans[d] == 0.0:
            return lo
        # reflect a few times then clip
        for _ in range(4):
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
            else:
                return v
        return clip(v, d)

    # ---------- eval + cache ----------
    cache = {}
    cache_max = 8000

    def key_of(x):
        # quantize mildly relative to span
        k = []
        for d in range(dim):
            s = spans[d]
            if s <= 0.0:
                k.append(0)
            else:
                # 1e-11 span buckets
                k.append(int((x[d] - lows[d]) / (s * 1e-11 + 1e-30)))
        return tuple(k)

    def safe_eval(x):
        kk = key_of(x)
        if kk in cache:
            return cache[kk]
        try:
            v = float(func(x))
            if v != v or v == float("inf") or v == float("-inf"):
                v = float("inf")
        except Exception:
            v = float("inf")
        if len(cache) < cache_max:
            cache[kk] = v
        return v

    # ---------- sampling helpers ----------
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

    def gauss():
        # Box-Muller standard normal
        u1 = max(1e-300, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def time_frac():
        if deadline <= t0:
            return 1.0
        tf = (time.time() - t0) / (deadline - t0)
        if tf < 0.0:
            return 0.0
        if tf > 1.0:
            return 1.0
        return tf

    # ---------- local refinement (Powell-lite + coord + tiny parabola) ----------
    def local_refine(x0, f0, max_evals, init_step_frac):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        # per-dim steps
        steps = [init_step_frac * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] == 0.0:
                steps[d] = 0.0

        shrink = 0.62
        min_step = 1e-14
        evals = 0

        # directions for Powell-like move (coordinate basis initially)
        dirs = []
        for d in range(dim):
            v = [0.0] * dim
            if spans[d] > 0.0:
                v[d] = 1.0
            dirs.append(v)

        def try_move(dir_vec, alpha):
            xt = x[:]
            for d in range(dim):
                if dir_vec[d] != 0.0 and spans[d] > 0.0:
                    xt[d] = clip(xt[d] + alpha * dir_vec[d], d)
            return xt

        while evals < max_evals and time.time() < deadline:
            improved = False
            x_start = x[:]
            f_start = fx

            # line-ish search along each direction using 2-3 probes
            for dir_vec in dirs:
                if evals >= max_evals or time.time() >= deadline:
                    break

                # choose a scalar step based on dominant axis
                # (keep it cheap: just use average step magnitude)
                s = 0.0
                cnt = 0
                for d in range(dim):
                    if dir_vec[d] != 0.0:
                        s += steps[d]
                        cnt += 1
                if cnt == 0:
                    continue
                s = s / cnt
                if s <= min_step:
                    continue

                # probe -s, +s
                xm = try_move(dir_vec, -s)
                fm = safe_eval(xm); evals += 1
                xp = try_move(dir_vec, +s)
                fp = safe_eval(xp); evals += 1

                # take best of the three (x, xm, xp)
                best_local = fx
                best_x_local = None
                if fm < best_local:
                    best_local = fm
                    best_x_local = xm
                if fp < best_local:
                    best_local = fp
                    best_x_local = xp

                # tiny parabolic tweak if both sides better/worse pattern exists
                # We attempt parabola through (-s,fm),(0,fx),(+s,fp)
                if evals < max_evals and time.time() < deadline:
                    denom = (fm - 2.0 * fx + fp)
                    if abs(denom) > 1e-300:
                        alpha = 0.5 * s * (fm - fp) / denom
                        if abs(alpha) <= 2.5 * s:
                            xq = try_move(dir_vec, alpha)
                            fq = safe_eval(xq); evals += 1
                            if fq < best_local:
                                best_local = fq
                                best_x_local = xq

                if best_x_local is not None and best_local < fx:
                    x, fx = best_x_local, best_local
                    improved = True

            # Powell-like extra direction: from start to end
            if evals < max_evals and time.time() < deadline:
                dir_new = [x[d] - x_start[d] for d in range(dim)]
                norm = math.sqrt(sum(v * v for v in dir_new))
                if norm > 0.0:
                    invn = 1.0 / norm
                    for d in range(dim):
                        dir_new[d] *= invn
                    # try one step along new direction
                    # step size: average steps
                    s = 0.0
                    cnt = 0
                    for d in range(dim):
                        if spans[d] > 0.0:
                            s += steps[d]
                            cnt += 1
                    s = (s / max(1, cnt))
                    if s > min_step:
                        xn = try_move(dir_new, s)
                        fn = safe_eval(xn); evals += 1
                        if fn < fx:
                            x, fx = xn, fn
                            improved = True
                            # rotate directions: drop oldest, add new
                            if dirs:
                                dirs.pop(0)
                            dirs.append(dir_new)

            # coordinate quick nudges (cheap polishing)
            for d in range(dim):
                if evals >= max_evals or time.time() >= deadline:
                    break
                s = steps[d]
                if s <= min_step or spans[d] == 0.0:
                    continue
                # +/-
                xp = x[:]; xp[d] = clip(xp[d] + s, d)
                fp = safe_eval(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue
                xm = x[:]; xm[d] = clip(xm[d] - s, d)
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

            # accept only if improved, otherwise keep fx (already)
            if fx >= f_start and not improved:
                x, fx = x_start, f_start

        return x, fx

    # ---------- initial population ----------
    # keep some DE population but add diagonal-adaptive sampling around best
    pop = max(26, min(180, 18 + 8 * dim))
    X = []
    Fx = []

    init = lhs_block(min(pop, max(14, pop // 2)))
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
            best, best_x = f, x[:]

    # ---------- DE params + diagonal adaptation state ----------
    mu_F, mu_CR = 0.55, 0.85
    c_adapt = 0.10

    # diagonal "sigma" for sampling around best (like separable CMA-ES)
    sigma = [0.30 * spans[d] for d in range(dim)]
    for d in range(dim):
        if spans[d] == 0.0:
            sigma[d] = 0.0

    # archive for DE/current-to-pbest
    A = []
    Amax = pop

    stagn = 0
    it = 0

    # restart control
    restart_no_improve = 140  # will scale by dim/time dynamically

    while time.time() < deadline:
        it += 1
        tf = time_frac()

        # update best
        bidx = min(range(pop), key=lambda i: Fx[i])
        if Fx[bidx] < best:
            best = Fx[bidx]
            best_x = X[bidx][:]
            stagn = max(0, stagn - 5)
        else:
            stagn += 1

        # schedule pbest pool: larger early, smaller late
        pbest_rate = 0.40 - 0.30 * tf  # 0.40 -> 0.10
        if pbest_rate < 0.08:
            pbest_rate = 0.08
        top = max(2, int(pbest_rate * pop))

        sorted_idx = sorted(range(pop), key=lambda i: Fx[i])

        # occasional diagonal sampling around best (exploitation) + around random elite (exploration)
        # adapt sigma: shrink when improving, inflate when stagnating
        if best_x is not None:
            if stagn > 0 and stagn % 25 == 0:
                inflate = 1.10
                for d in range(dim):
                    sigma[d] = min(0.50 * spans[d], sigma[d] * inflate)
            if it % 12 == 0:
                shrink = 0.90
                for d in range(dim):
                    sigma[d] = max(1e-15 * (spans[d] + 1.0), sigma[d] * shrink)

        # restart/refresh chunk on stagnation
        dynamic_restart = restart_no_improve + 8 * max(1, dim)
        if stagn > dynamic_restart and time.time() < deadline:
            k = max(2, pop // 6)
            # half LHS globally, half gaussian around best
            new_pts = lhs_block(k // 2)
            while len(new_pts) < k:
                if best_x is None:
                    new_pts.append(rand_point())
                else:
                    x = best_x[:]
                    # sample separably
                    for d in range(dim):
                        if spans[d] > 0.0:
                            x[d] = reflect(best_x[d] + sigma[d] * gauss(), d)
                    new_pts.append(x)
            worst = sorted(range(pop), key=lambda i: Fx[i], reverse=True)[:k]
            for idx, xnew in zip(worst, new_pts):
                if time.time() >= deadline:
                    break
                fnew = safe_eval(xnew)
                X[idx], Fx[idx] = xnew, fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
            # also reset archive partly
            if len(A) > Amax // 2:
                random.shuffle(A)
                A = A[:Amax // 2]
            stagn = dynamic_restart // 3

        # DE generation with success-history adaptation
        S_F, S_CR, dF = [], [], []
        union = X + A
        union_n = len(union)

        for i in range(pop):
            if time.time() >= deadline:
                break

            xi, fi = X[i], Fx[i]

            # sample F, CR
            # cauchy for F (heavy tail), normal-ish for CR using gauss
            Fi = mu_F + 0.10 * math.tan(math.pi * (random.random() - 0.5))
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = mu_F + 0.10 * math.tan(math.pi * (random.random() - 0.5))
                tries += 1
            if Fi <= 0.0:
                Fi = 0.08
            if Fi > 1.0:
                Fi = 1.0

            CRi = mu_CR + 0.10 * gauss()
            if CRi < 0.0:
                CRi = 0.0
            if CRi > 1.0:
                CRi = 1.0

            pbest = sorted_idx[random.randrange(top)]
            xpbest = X[pbest]

            # r1 from population excluding i,pbest
            forbidden = {i, pbest}
            r1 = random.randrange(pop)
            while r1 in forbidden:
                r1 = random.randrange(pop)
            forbidden.add(r1)

            # r2 from union excluding forbidden if from pop
            if union_n <= 1:
                r2u = r1
            else:
                r2u = random.randrange(union_n)
                while (r2u < pop and r2u in forbidden) and union_n > 1:
                    r2u = random.randrange(union_n)

            xr1 = X[r1]
            xr2 = union[r2u]

            # mutation: current-to-pbest/1 + (r1-r2)
            # plus a mild pull to global best if stagnating
            best_pull = 0.0
            if best_x is not None:
                if stagn > 60:
                    best_pull = 0.25
                if stagn > 140:
                    best_pull = 0.40

            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    base = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                    if best_pull > 0.0:
                        base += best_pull * Fi * (best_x[d] - xi[d])
                    v[d] = base

            # crossover + repair
            jrand = random.randrange(dim)
            u = xi[:]
            changed = False
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    ud = reflect(v[d], d)
                    # convex pull-back to parent (helps if reflection bounces in tight bounds)
                    if random.random() < 0.04:
                        ud = 0.85 * ud + 0.15 * xi[d]
                        ud = clip(ud, d)
                    u[d] = ud
                    changed = True

            if not changed:
                d = jrand
                if spans[d] > 0.0:
                    u[d] = lows[d] + spans[d] * random.random()
                else:
                    u[d] = lows[d]

            # occasional diagonal gaussian sample around best (inject)
            if best_x is not None and random.random() < (0.03 + 0.05 * tf):
                for d in range(dim):
                    if spans[d] > 0.0 and random.random() < 0.35:
                        u[d] = reflect(best_x[d] + sigma[d] * gauss(), d)

            fu = safe_eval(u)

            if fu <= fi:
                # archive
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
                    best = fu
                    best_x = u[:]
                    stagn = max(0, stagn - 7)

        # adapt mu_F and mu_CR
        if S_F:
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F, weighted mean for CR
            num = 0.0
            den = 0.0
            crm = 0.0
            for fval, crval, w in zip(S_F, S_CR, dF):
                ww = w / wsum
                num += ww * (fval * fval)
                den += ww * fval
                crm += ww * crval
            F_lehmer = (num / den) if den > 1e-300 else (sum(S_F) / len(S_F))
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * crm
            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.95, max(0.05, mu_CR))
        else:
            # no successes -> explore
            mu_F = min(0.95, mu_F + 0.03)
            mu_CR = max(0.05, mu_CR - 0.04)

        # periodic local refinement (more aggressive near end)
        if best_x is not None and (it % 10 == 0 or (stagn > 120 and it % 4 == 0)):
            tl = deadline - time.time()
            if tl <= 0.0:
                break
            # budget & step schedule
            if tl < 0.20:
                budget = 220
                step_frac = 0.012
            elif tl < 0.70:
                budget = 140
                step_frac = 0.022
            else:
                budget = 70
                step_frac = 0.06 * (0.40 + 0.60 * (1.0 - tf))
            rx, rf = local_refine(best_x, best, max_evals=budget, init_step_frac=step_frac)
            if rf < best:
                best, best_x = rf, rx[:]
                # inject into worst individual
                widx = max(range(pop), key=lambda i: Fx[i])
                X[widx], Fx[widx] = best_x[:], best
                stagn = 0
                # shrink sigma after confirmed improvement
                for d in range(dim):
                    sigma[d] = max(1e-15 * (spans[d] + 1.0), sigma[d] * 0.85)

    return best
