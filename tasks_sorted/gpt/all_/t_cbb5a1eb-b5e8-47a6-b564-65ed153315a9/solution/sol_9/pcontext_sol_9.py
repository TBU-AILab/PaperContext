import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Upgrade over your best (JADE-DE + SPSA + coord pattern):
      - Keeps JADE/current-to-pbest/1 DE global search (fast, robust).
      - Adds *CMA-ES style local "trust region"* but in a time-cheap diagonal form:
        rank-µ update of per-dimension stds + evolution-path drift. This is much more stable than
        coordinate steps on curved valleys, yet far cheaper than full-cov CMA.
      - Uses *bandit scheduling* between (DE / diag-CMA / SPSA / coordinate pattern) based on
        recent ROI (improvement per evaluation), so time is spent on what works for THIS func.
      - Uses *whitened SPSA*: gradient steps scaled by current diag stds, making SPSA far less fragile.
      - More careful time/eval budgeting and fewer heavy operations per loop.

    Returns:
        best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # ---------- bounds handling: mirror ----------
    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def repair(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- Halton for init / reseeds ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- RNG helpers ----------
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        z0 = r * math.cos(t)
        z1 = r * math.sin(t)
        _spare = z1
        _has_spare = True
        return z0

    def normal_rand(mu, sigma):
        return mu + sigma * randn()

    def cauchy_rand(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # ---------- elite archive ----------
    elite_size = max(12, min(48, 14 + int(4.0 * math.sqrt(dim))))
    elites = []  # sorted list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    # ---------- cheap coordinate/pattern ----------
    def coord_pattern(x, fx, rounds=2, max_coords=14, base=0.05):
        if dim == 0:
            return fx, x
        xbest, fbest = x[:], fx
        idxs = list(range(dim))
        idxs.sort(key=lambda i: spans_nz[i], reverse=True)
        idxs = idxs[:max(1, min(dim, max_coords))]

        for r in range(rounds):
            if now() >= deadline:
                break
            improved = False
            step_mul = base * (0.55 ** r)
            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = min(0.25 * spans_nz[i], max(1e-12, step_mul * spans_nz[i]))

                xp = xbest[:]
                xp[i] += delta
                repair(xp)
                fp = evaluate(xp)
                if fp < fbest:
                    step = xp[i] - xbest[i]
                    xbest, fbest = xp, fp
                    # pattern
                    xpp = xbest[:]
                    xpp[i] += step
                    repair(xpp)
                    fpp = evaluate(xpp)
                    if fpp < fbest:
                        xbest, fbest = xpp, fpp
                    improved = True
                    continue

                xm = xbest[:]
                xm[i] -= delta
                repair(xm)
                fm = evaluate(xm)
                if fm < fbest:
                    step = xm[i] - xbest[i]
                    xbest, fbest = xm, fm
                    xmm = xbest[:]
                    xmm[i] += step
                    repair(xmm)
                    fmm = evaluate(xmm)
                    if fmm < fbest:
                        xbest, fbest = xmm, fmm
                    improved = True
            if not improved:
                break
        return fbest, xbest

    # ---------- diag-CMA local engine (cheap trust-region) ----------
    def diag_cma_refine(x, fx, steps=10):
        """
        Small (µ,λ)-style diagonal CMA-ish refinement around x.
        Uses very small lambda and only diagonal std updates => good ROI under tight time.
        """
        if dim == 0:
            return fx, x
        # local population sizes
        lam = max(8, min(18, 8 + int(2.0 * math.sqrt(dim))))
        mu = max(3, lam // 2)
        # log weights
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        sw = sum(w)
        w = [wi / sw for wi in w]

        # initial diag stds proportional to span (but small: local)
        avg_span = sum(spans_nz) / float(dim)
        d = [max(1e-12, 0.08 * spans_nz[i]) for i in range(dim)]
        # path (drift)
        p = [0.0] * dim

        xb, fb = x[:], fx

        for g in range(steps):
            if now() >= deadline:
                break

            # sample
            pop = []
            for _ in range(lam):
                if now() >= deadline:
                    break
                y = [randn() for _ in range(dim)]
                xc = xb[:]
                # inject drift by p
                for i in range(dim):
                    if spans[i] == 0.0:
                        xc[i] = lows[i]
                    else:
                        xc[i] += d[i] * y[i] + 0.15 * p[i]
                repair(xc)
                fc = evaluate(xc)
                pop.append((fc, xc, y))

            if len(pop) < mu:
                break
            pop.sort(key=lambda t: t[0])

            # update best
            if pop[0][0] < fb:
                xb, fb = pop[0][1][:], pop[0][0]

            # recombine mean (local)
            x_new = [0.0] * dim
            for j in range(mu):
                wi = w[j]
                xj = pop[j][1]
                for i in range(dim):
                    x_new[i] += wi * xj[i]
            repair(x_new)
            f_new = evaluate(x_new)
            if f_new < fb:
                xb, fb = x_new[:], f_new

            # update path based on movement
            for i in range(dim):
                if spans[i] == 0.0:
                    p[i] = 0.0
                else:
                    step = (x_new[i] - x[i]) / max(1e-12, d[i])
                    p[i] = 0.75 * p[i] + 0.25 * (step * d[i])

            # update diag stds from selected z^2
            for i in range(dim):
                if spans[i] == 0.0:
                    d[i] = 1e-12
                    continue
                ez2 = 0.0
                for j in range(mu):
                    wi = w[j]
                    zj = pop[j][2][i]
                    ez2 += wi * (zj * zj)
                # mild update toward ez2=1
                fac = math.exp(0.25 * (ez2 - 1.0))
                di = d[i] * fac
                di = min(0.35 * spans_nz[i], max(1e-12, di))
                d[i] = di

            x = xb[:]  # shift anchor
        return fb, xb

    # ---------- whitened SPSA ----------
    def spsa_refine(x, fx, steps=18, scale_vec=None):
        if dim == 0:
            return fx, x
        xb, fb = x[:], fx
        # scale vector (e.g., diag stds); fallback to spans
        if scale_vec is None:
            scale_vec = spans_nz[:]
        avg_scale = sum(scale_vec) / float(dim)

        a0 = 0.10 * avg_scale
        c0 = 0.08 * avg_scale

        for k in range(1, steps + 1):
            if now() >= deadline:
                break
            ak = a0 / (k ** 0.602)
            ck = c0 / (k ** 0.101)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            x_plus = xb[:]
            x_minus = xb[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    x_plus[i] = lows[i]
                    x_minus[i] = lows[i]
                else:
                    ci = ck * (scale_vec[i] / (avg_scale if avg_scale > 0 else 1.0))
                    x_plus[i] += ci * delta[i]
                    x_minus[i] -= ci * delta[i]
            repair(x_plus)
            repair(x_minus)

            f_plus = evaluate(x_plus)
            if now() >= deadline:
                if f_plus < fb:
                    xb, fb = x_plus, f_plus
                break
            f_minus = evaluate(x_minus)

            if f_plus < fb:
                xb, fb = x_plus, f_plus
            if f_minus < fb:
                xb, fb = x_minus, f_minus

            denom = max(1e-18, 2.0 * ck)
            x_new = xb[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    x_new[i] = lows[i]
                else:
                    ghat = (f_plus - f_minus) / (denom * delta[i])
                    # whiten/normalize by scale_vec: more stable across dims
                    x_new[i] -= ak * (ghat / max(1e-12, scale_vec[i]))
            repair(x_new)
            f_new = evaluate(x_new)
            if f_new < fb:
                xb, fb = x_new, f_new

        return fb, xb

    # ---------- Initialization ----------
    best = float("inf")
    best_x = None

    # DE population
    NP = max(18, min(80, 10 + 3 * int(math.sqrt(dim)) + 2 * dim))
    NP = min(NP, max(22, 4 * dim))

    pop, fit = [], []
    init_n = NP
    for _ in range(init_n):
        if now() >= deadline:
            return best
        x = halton_point(hal_k) if random.random() < 0.82 else rand_uniform_point()
        hal_k += 1 if x is not None else 0
        fx = evaluate(x)
        pop.append(x)
        fit.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        if now() >= deadline:
            return best
        if random.random() < 0.50:
            xo = [lows[d] + highs[d] - x[d] for d in range(dim)]
            repair(xo)
            fo = evaluate(xo)
            push_elite(fo, xo)
            if fo < best:
                best, best_x = fo, xo[:]
            if fo < fx and random.random() < 0.7:
                pop[-1] = xo
                fit[-1] = fo

    # JADE-like archive
    A = []
    Amax = NP
    mu_F = 0.55
    mu_CR = 0.55

    def pick_index_excluding(n, exclude):
        while True:
            r = random.randrange(n)
            if r not in exclude:
                return r

    # ---------- bandit scheduler ----------
    arms = ["DE", "CMA", "SPSA", "COORD"]
    score = {a: 1.0 for a in arms}  # pseudo-reward
    cost = {a: 1.0 for a in arms}   # pseudo-cost
    last_best = best
    no_best = 0
    gen = 0

    def choose_arm():
        # epsilon-greedy with soft bias to higher ROI = score/cost
        if random.random() < 0.18:
            return arms[random.randrange(len(arms))]
        best_a = arms[0]
        best_roi = -1e99
        for a in arms:
            roi = score[a] / max(1e-12, cost[a])
            if roi > best_roi:
                best_roi = roi
                best_a = a
        return best_a

    def update_arm(a, improvement, evals_used):
        # improvement >=0; we want improvement per eval, but keep smooth
        score[a] = 0.90 * score[a] + 0.10 * max(0.0, improvement)
        cost[a] = 0.90 * cost[a] + 0.10 * max(1.0, float(evals_used))

    # ---------- Main loop ----------
    while now() < deadline:
        gen += 1
        time_left = deadline - now()
        frac_left = time_left / float(max_time) if max_time > 0 else 0.0
        endgame = frac_left < 0.22

        arm = choose_arm()

        if arm == "DE":
            # one DE generation-ish sweep (but time-aware)
            idxs_sorted = list(range(NP))
            idxs_sorted.sort(key=lambda i: fit[i])
            p = 0.10 + 0.10 * random.random()
            p_num = max(2, int(p * NP))
            pbest_pool = idxs_sorted[:p_num]

            S_F, S_CR = [], []
            union = pop + A
            union_n = len(union)

            evals_used = 0
            before = best

            for i in range(NP):
                if now() >= deadline:
                    break

                xi = pop[i]
                fi = fit[i]

                CR = normal_rand(mu_CR, 0.10)
                CR = 0.0 if CR < 0.0 else (1.0 if CR > 1.0 else CR)

                F = cauchy_rand(mu_F, 0.10)
                tries = 0
                while F <= 0.0 and tries < 8:
                    F = cauchy_rand(mu_F, 0.10)
                    tries += 1
                if F <= 0.0:
                    F = 0.1
                if F > 1.0:
                    F = 1.0
                F *= (0.92 + 0.16 * math.sin(2.0 * math.pi * random.random()))
                F = min(1.0, max(0.05, F))

                pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
                x_pbest = pop[pbest_idx]

                r1 = pick_index_excluding(NP, {i, pbest_idx})
                x_r1 = pop[r1]

                if union_n <= 1:
                    r2_vec = pop[pick_index_excluding(NP, {i, pbest_idx, r1})]
                else:
                    r2_vec = union[random.randrange(union_n)]
                    for _ in range(6):
                        if (r2_vec is not xi) and (r2_vec is not x_r1):
                            break
                        r2_vec = union[random.randrange(union_n)]

                v = [0.0] * dim
                for d in range(dim):
                    if spans[d] == 0.0:
                        v[d] = lows[d]
                    else:
                        v[d] = xi[d] + F * (x_pbest[d] - xi[d]) + F * (x_r1[d] - r2_vec[d])

                jrand = random.randrange(dim)
                u = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        u[d] = v[d]
                repair(u)

                fu = evaluate(u)
                evals_used += 1

                if fu <= fi:
                    if len(A) < Amax:
                        A.append(xi[:])
                    else:
                        A[random.randrange(Amax)] = xi[:]
                    pop[i] = u
                    fit[i] = fu
                    push_elite(fu, u)
                    S_F.append(F)
                    S_CR.append(CR)
                    if fu < best:
                        best, best_x = fu, u[:]
                else:
                    # tiny diversity injection
                    if random.random() < (0.010 if not endgame else 0.004) and now() < deadline:
                        xn = halton_point(hal_k) if random.random() < 0.7 else rand_uniform_point()
                        hal_k += 1
                        fn = evaluate(xn)
                        evals_used += 1
                        if fn < fit[i]:
                            if len(A) < Amax:
                                A.append(pop[i][:])
                            else:
                                A[random.randrange(Amax)] = pop[i][:]
                            pop[i] = xn
                            fit[i] = fn
                            push_elite(fn, xn)
                            if fn < best:
                                best, best_x = fn, xn[:]

            if S_F:
                s1 = sum(f * f for f in S_F)
                s2 = sum(S_F)
                if s2 > 0.0:
                    mu_F = 0.90 * mu_F + 0.10 * (s1 / s2)
                    mu_F = min(0.95, max(0.05, mu_F))
            if S_CR:
                mu_CR = 0.90 * mu_CR + 0.10 * (sum(S_CR) / float(len(S_CR)))
                mu_CR = min(1.0, max(0.0, mu_CR))

            update_arm("DE", max(0.0, before - best), max(1, evals_used))

        else:
            # local phases operate on current best
            if best_x is None:
                x = rand_uniform_point()
                fx = evaluate(x)
                push_elite(fx, x)
                if fx < best:
                    best, best_x = fx, x[:]

            before = best

            # allocate tiny budgets depending on time
            if arm == "CMA":
                steps = 6 if not endgame else 12
                fb, xb = diag_cma_refine(best_x, best, steps=steps)
                push_elite(fb, xb)
                if fb < best:
                    best, best_x = fb, xb[:]
                update_arm("CMA", max(0.0, before - best), steps * 12)

            elif arm == "SPSA":
                # use scale from spans; in endgame, smaller steps
                steps = 10 if not endgame else 22
                fb, xb = spsa_refine(best_x, best, steps=steps, scale_vec=spans_nz)
                push_elite(fb, xb)
                if fb < best:
                    best, best_x = fb, xb[:]
                update_arm("SPSA", max(0.0, before - best), steps * 3)

            else:  # COORD
                fb, xb = coord_pattern(best_x, best, rounds=1 if not endgame else 2,
                                       max_coords=14, base=0.05 if endgame else 0.06)
                push_elite(fb, xb)
                if fb < best:
                    best, best_x = fb, xb[:]
                update_arm("COORD", max(0.0, before - best), 60)

        # stagnation tracking and reseed worst DE individuals
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            no_best = 0
        else:
            no_best += 1

        if no_best >= (10 + int(2.0 * math.sqrt(dim))):
            no_best = 0
            q = max(2, NP // 6)
            idxs_worst = list(range(NP))
            idxs_worst.sort(key=lambda i: fit[i], reverse=True)
            worst = idxs_worst[:q]
            for k in worst:
                if now() >= deadline:
                    break
                if elites and random.random() < 0.90:
                    top = min(len(elites), 16)
                    idx = int((random.random() ** 2) * top)
                    xseed = elites[idx][1][:]
                    for d in range(dim):
                        if spans[d] > 0.0 and random.random() < 0.65:
                            xseed[d] += (0.12 * spans_nz[d]) * (2.0 * random.random() - 1.0)
                    repair(xseed)
                else:
                    xseed = halton_point(hal_k) if random.random() < 0.75 else rand_uniform_point()
                    hal_k += 1
                fseed = evaluate(xseed)
                pop[k] = xseed
                fit[k] = fseed
                push_elite(fseed, xseed)
                if fseed < best:
                    best, best_x = fseed, xseed[:]
                    last_best = best

    return best
