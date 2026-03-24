import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimization (no external libraries).

    Changes vs previous:
      - Switch core to L-SHADE-like DE with:
          * current-to-pbest/1 + archive
          * success-history memories for F/CR (cyclic), not just single mu
          * weighted updates by fitness gain
          * linear population size reduction (LPSR): big early, smaller late
      - Robust boundary handling: bounce-back (reflection) + final clamp
      - Better initialization: LHS + opposition + a few "centered jitter" points
      - Smarter stagnation recovery: re-inject around best with shrinking radius
      - Lightweight final-phase local search (coordinate + random directions)

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    lows = [float(bounds[i][0]) for i in range(dim)]
    highs = [float(bounds[i][1]) for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    def now():
        return time.time()

    def eval_f(x):
        return float(func(x))

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    # Box-Muller normal sampler
    _spare = [None]
    def randn():
        if _spare[0] is not None:
            z = _spare[0]
            _spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare[0] = z1
        return z0

    # Better boundary handling than pure clamp:
    # reflect (bounce) any out-of-bounds coordinate, then clamp as safety.
    def reflect_bounds(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo or x[i] > hi:
                # reflect repeatedly in case step is huge
                if x[i] < lo:
                    x[i] = lo + (lo - x[i])
                if x[i] > hi:
                    x[i] = hi - (x[i] - hi)
                # if still out (can happen if far), fold using modulo reflection
                if x[i] < lo or x[i] > hi:
                    w = hi - lo
                    if w <= 0.0:
                        x[i] = lo
                    else:
                        y = x[i] - lo
                        m = y % (2.0 * w)
                        x[i] = lo + (m if m <= w else (2.0 * w - m))
            # final safety clamp
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opp_vec(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def center_jitter(scale):
        # sample around center with gaussian jitter
        x = []
        for i in range(dim):
            c = 0.5 * (lows[i] + highs[i])
            x.append(c + scale * spans[i] * randn())
        return reflect_bounds(x)

    # LHS-ish points
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + random.random()) / n
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # small local search used late / on new best
    def local_search(xb, fb, max_evals, radius_frac):
        # coordinate + a few random directions, shrinking radius
        evals = 0
        x = xb[:]
        bestx = xb[:]
        bestf = fb

        rad = max(1e-12, radius_frac)  # fraction of span
        coord_step = [rad * spans[i] for i in range(dim)]

        while evals < max_evals and now() < deadline:
            improved = False

            # coordinate moves
            for d in range(dim):
                if evals >= max_evals or now() >= deadline:
                    break
                sd = coord_step[d]
                if sd <= 1e-18 * spans[d]:
                    continue

                xp = bestx[:]
                xp[d] += sd
                reflect_bounds(xp)
                fp = eval_f(xp); evals += 1
                if fp < bestf:
                    bestx, bestf = xp, fp
                    improved = True
                    continue

                if evals >= max_evals or now() >= deadline:
                    break

                xm = bestx[:]
                xm[d] -= sd
                reflect_bounds(xm)
                fm = eval_f(xm); evals += 1
                if fm < bestf:
                    bestx, bestf = xm, fm
                    improved = True

            # random direction probes (cheap)
            if evals < max_evals and now() < deadline:
                trials = 2 if dim <= 10 else 1
                for _ in range(trials):
                    if evals >= max_evals or now() >= deadline:
                        break
                    dirn = [randn() for _ in range(dim)]
                    nrm = math.sqrt(sum(v*v for v in dirn)) + 1e-12
                    step = rad
                    xr = [bestx[i] + (step * spans[i]) * (dirn[i] / nrm) for i in range(dim)]
                    reflect_bounds(xr)
                    fr = eval_f(xr); evals += 1
                    if fr < bestf:
                        bestx, bestf = xr, fr
                        improved = True

            if improved:
                rad *= 1.10
                if rad > 0.25:
                    rad = 0.25
                for d in range(dim):
                    coord_step[d] = rad * spans[d]
            else:
                rad *= 0.5
                for d in range(dim):
                    coord_step[d] = rad * spans[d]
                if rad < 1e-12:
                    break

        return bestx, bestf

    # ----------- L-SHADE style setup -----------
    # time-aware sizing
    NP_max = max(24, min(140, 12 * dim))
    NP_min = max(10, min(40, 4 * dim))

    if max_time <= 0.15:
        NP_max = max(12, min(NP_max, 28))
        NP_min = max(8, min(NP_min, 16))

    # initialize population
    pop = []
    fit = []

    # mix of LHS, opposition, and a few center-jitter points
    n_lhs = min(NP_max, max(8, int(math.sqrt(NP_max) + 7)))
    lhs = lhs_points(n_lhs)
    for x in lhs:
        if now() >= deadline:
            return float("inf") if not fit else min(fit)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

        if len(pop) < NP_max and now() < deadline:
            xo = reflect_bounds(opp_vec(x))
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)

    # centered jitter points (good for many bounded problems)
    for _ in range(min(6, max(2, dim // 3))):
        if len(pop) >= NP_max or now() >= deadline:
            break
        x = center_jitter(0.15)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    while len(pop) < NP_max and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    NP = len(pop)

    best_i = min(range(NP), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    if now() >= deadline:
        return best

    # SHADE memories
    H = 8 if dim <= 15 else 12
    M_F = [0.6] * H
    M_CR = [0.5] * H
    k = 0

    # Archive
    archive = []
    arch_max = NP_max

    # misc
    last_best = best
    stagn = 0
    last_ls_time = 0.0

    def pick_r(exclude_set, n):
        j = random.randrange(n)
        while j in exclude_set:
            j = random.randrange(n)
        return j

    # main loop
    gen = 0
    while now() < deadline:
        gen += 1

        # population reduction (linear with time)
        frac = (now() - t0) / max(1e-12, max_time)
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0

        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        # shrink population by removing worst (keep best)
        if NP > target_NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            # adjust archive size too
            arch_max = max(arch_max, NP)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # sort for pbest selection
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # p-best fraction schedule (more exploit later)
        pfrac = 0.35 * (1.0 - frac) + 0.08  # 0.43 -> 0.08
        pcount = max(2, int(pfrac * NP))

        # store successful parameters
        S_F = []
        S_CR = []
        S_w = []

        improved_gen = False

        union = pop + archive
        unionN = len(union)

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick memory index r
            r = random.randrange(H)
            mF = M_F[r]
            mCR = M_CR[r]

            # F from Cauchy-like around mF (approx via tan(pi(u-0.5)))
            # retry to keep F>0
            F = -1.0
            for _ in range(8):
                u = random.random()
                cauchy = math.tan(math.pi * (u - 0.5))
                F = mF + 0.1 * cauchy
                if F > 0.0:
                    break
            if F <= 0.0:
                F = mF
            F = clamp(F, 0.05, 1.0)

            # CR from normal around mCR
            CR = clamp01(mCR + 0.1 * randn())

            # choose pbest among top pcount
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # r1 from pop excluding i and pbest
            r1 = pick_r({i, pbest_idx}, NP)
            xr1 = pop[r1]

            # r2 from union excluding i and r1 if in pop part (archive has no restriction)
            # simple resampling
            r2u = random.randrange(unionN)
            for _ in range(12):
                if r2u < NP and (r2u == i or r2u == r1):
                    r2u = random.randrange(unionN)
                else:
                    break
            xr2 = union[r2u]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
            reflect_bounds(v)

            # binomial crossover
            jrand = random.randrange(dim)
            uvec = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    uvec[d] = v[d]

            fu = eval_f(uvec)

            if fu <= fi:
                # update archive
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                gain = fi - fu
                if gain < 0.0:
                    gain = 0.0
                w = gain + 1e-12  # weight
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

                if fu < best:
                    best = fu
                    bestx = uvec[:]
                    improved_gen = True
            # refresh union occasionally if archive grows (cheap-ish)
        # update best index after generation
        bi = min(range(NP), key=lambda j: fit[j])
        if fit[bi] < best:
            best = fit[bi]
            bestx = pop[bi][:]

        # update memories
        if S_F:
            wsum = sum(S_w) + 1e-30
            # weighted Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for w, F, CR in zip(S_w, S_F, S_CR):
                num += w * F * F
                den += w * F
                cr_mean += w * CR
            F_new = num / (den + 1e-30)
            CR_new = cr_mean / wsum

            M_F[k] = clamp(F_new, 0.05, 1.0)
            M_CR[k] = clamp01(CR_new)
            k = (k + 1) % H

        # stagnation logic
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # targeted re-injection if stagnating
        if stagn >= max(6, int(2.0 * math.sqrt(NP))) and now() < deadline:
            stagn = 0
            # replace worst ~20% with samples around best (shrinking radius over time)
            krep = max(2, NP // 5)
            worst = sorted(range(NP), key=lambda j: fit[j], reverse=True)
            rad = 0.25 * (1.0 - frac) + 0.03  # 0.28 -> 0.03
            rep = 0
            for idx in worst:
                if rep >= krep or now() >= deadline:
                    break
                # never replace the current best individual
                if pop[idx] is bestx:
                    continue
                x = bestx[:]
                for d in range(dim):
                    x[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
                reflect_bounds(x)
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                rep += 1
                if fx < best:
                    best = fx
                    bestx = x[:]

        # late-phase local search (throttled)
        # do it mainly in the last ~35% time or when a generation improved best
        if now() < deadline:
            time_left = deadline - now()
            if (frac > 0.65 or improved_gen) and (now() - last_ls_time) > 0.05:
                # small budget and radius depending on phase
                budget = max(8, min(180, 6 * dim))
                radius = 0.08 * (1.0 - frac) + 0.01
                x2, f2 = local_search(bestx, best, budget, radius)
                if f2 < best:
                    best = f2
                    bestx = x2[:]
                last_ls_time = now()

        # keep archive bounded
        if len(archive) > arch_max:
            archive = archive[-arch_max:]

    return best
