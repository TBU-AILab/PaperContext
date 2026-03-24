import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libraries).

    Improvements over the provided DE/L-SHADE-like code:
      1) Proper L-SHADE selection/reduction:
         - Population reduction removes worst by fitness (keeps best index safe).
         - Archive size tracks NP (|A| = NP) as in common SHADE variants.
      2) Better boundary handling for trial vectors:
         - "Midpoint repair" for violated coordinates (common DE repair),
           plus final clamp. This often beats reflection on rugged functions.
      3) Add an inexpensive surrogate of "eigen/random-subspace" exploitation:
         - occasional random subspace crossover mask (speeds high-dim).
      4) Stronger final exploitation:
         - a compact Powell-like directional pattern search around best
           with adaptive step, only late or upon improvements.
      5) Evaluation batching awareness:
         - avoids too-frequent time checks inside tight inner loops, but still safe.

    Returns:
        best (float): best objective found within max_time
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    lows = [float(bounds[i][0]) for i in range(dim)]
    highs = [float(bounds[i][1]) for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def now():
        return time.time()

    def eval_f(x):
        return float(func(x))

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # LHS-ish initializer
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / n
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][i] + random.random()) * invn
                x[d] = lows[d] + u * spans[d]
            pts.append(x)
        return pts

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Repair used for trial vectors: midpoint with parent if out of bounds
    def repair_midpoint(u, x_parent):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if u[i] < lo:
                u[i] = 0.5 * (lo + x_parent[i])
            elif u[i] > hi:
                u[i] = 0.5 * (hi + x_parent[i])
            # safety clamp
            if u[i] < lo:
                u[i] = lo
            elif u[i] > hi:
                u[i] = hi
        return u

    # compact directional search around best (Powell-ish, but tiny budget)
    def pattern_search(bestx, bestf, budget, step_frac):
        # directions: coordinate basis + a few random directions
        x = bestx[:]
        f = bestf
        step = max(1e-12, float(step_frac))
        coord_steps = [step * spans[i] for i in range(dim)]

        # build a small direction set
        dirs = []
        for i in range(dim):
            d = [0.0] * dim
            d[i] = 1.0
            dirs.append(d)
        # add a couple random directions for coupling
        extra = 2 if dim <= 20 else 1
        for _ in range(extra):
            v = [randn() for _ in range(dim)]
            nrm = math.sqrt(sum(t*t for t in v)) + 1e-12
            dirs.append([t / nrm for t in v])

        evals = 0
        while evals < budget and now() < deadline:
            improved = False
            for dvec in dirs:
                if evals >= budget or now() >= deadline:
                    break
                # try + step
                xp = x[:]
                for i in range(dim):
                    xp[i] += dvec[i] * coord_steps[i]
                repair_midpoint(xp, x)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= budget or now() >= deadline:
                    break
                # try - step
                xm = x[:]
                for i in range(dim):
                    xm[i] -= dvec[i] * coord_steps[i]
                repair_midpoint(xm, x)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            if improved:
                # mild step expansion
                step = min(0.25, step * 1.25)
            else:
                # shrink step
                step *= 0.5
                if step < 1e-12:
                    break
            coord_steps = [step * spans[i] for i in range(dim)]

        return x, f

    # ---- L-SHADE-ish DE core ----
    if max_time <= 0.0:
        # At least do one eval
        x = rand_vec()
        return eval_f(x)

    NP_max = max(30, min(160, 14 * dim))
    NP_min = max(10, min(50, 4 * dim))
    if max_time <= 0.2:
        NP_max = max(14, min(NP_max, 32))
        NP_min = max(8, min(NP_min, 16))

    # init population: LHS + opposition + a few center jitters
    pop, fit = [], []
    n_lhs = min(NP_max, max(10, int(2 * math.sqrt(NP_max) + 6)))
    for x in lhs_points(n_lhs):
        if now() >= deadline: break
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if len(pop) < NP_max and now() < deadline:
            xo = opposition(x)
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)

    # add some center jitter points
    c = [0.5 * (lows[i] + highs[i]) for i in range(dim)]
    for _ in range(min(6, max(2, dim // 3))):
        if len(pop) >= NP_max or now() >= deadline: break
        x = [c[i] + 0.12 * spans[i] * randn() for i in range(dim)]
        x = [clamp(x[i], lows[i], highs[i]) for i in range(dim)]
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    while len(pop) < NP_max and now() < deadline:
        x = rand_vec()
        pop.append(x); fit.append(eval_f(x))

    if not pop:
        return float("inf")

    NP = len(pop)
    best_i = min(range(NP), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # SHADE memories
    H = 10 if dim <= 20 else 14
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_k = 0

    # archive (size tracks NP)
    archive = []
    arch_max = NP

    def pick_excluding(n, excl):
        j = random.randrange(n)
        while j in excl:
            j = random.randrange(n)
        return j

    last_best = best
    stagn = 0
    last_ls = t0

    # reduce time checks overhead in inner loop
    check_every = 8

    while True:
        t = now()
        if t >= deadline:
            return best

        frac = (t - t0) / max(1e-12, max_time)
        frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)

        # linear pop reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min: target_NP = NP_min
        if NP > target_NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            # adjust archive capacity to NP (common choice)
            arch_max = NP
            if len(archive) > arch_max:
                # keep random subset
                random.shuffle(archive)
                archive = archive[:arch_max]

        # indices sorted for pbest
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        pfrac = 0.35 * (1.0 - frac) + 0.08
        pcount = max(2, int(pfrac * NP))

        S_F, S_CR, S_w = [], [], []
        improved_gen = False

        union = pop + archive
        unionN = len(union)

        for i in range(NP):
            if (i % check_every) == 0 and now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mF = M_F[r]
            mCR = M_CR[r]

            # Cauchy around mF
            F = -1.0
            for _ in range(10):
                u = random.random()
                F = mF + 0.1 * math.tan(math.pi * (u - 0.5))
                if F > 0.0:
                    break
            if F <= 0.0:
                F = mF
            F = clamp(F, 0.05, 1.0)

            # Normal around mCR
            CR = clamp01(mCR + 0.1 * randn())

            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            r1 = pick_excluding(NP, {i, pbest_idx})
            xr1 = pop[r1]

            # r2 from union, avoiding i/r1 if from pop portion
            r2u = random.randrange(unionN)
            for _ in range(12):
                if r2u < NP and (r2u == i or r2u == r1):
                    r2u = random.randrange(unionN)
                else:
                    break
            xr2 = union[r2u]

            # mutation current-to-pbest/1
            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            # crossover: binomial + occasional random subspace mask
            jrand = random.randrange(dim)
            uvec = xi[:]

            if dim >= 12 and random.random() < 0.25:
                # random subspace: update about ~sqrt(dim) coords
                ksub = max(2, int(math.sqrt(dim)))
                chosen = set()
                while len(chosen) < ksub:
                    chosen.add(random.randrange(dim))
                chosen.add(jrand)
                for d in chosen:
                    if random.random() < CR or d == jrand:
                        uvec[d] = v[d]
            else:
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

            repair_midpoint(uvec, xi)
            fu = eval_f(uvec)

            if fu <= fi:
                # archive gets replaced individual
                if arch_max > 0:
                    if len(archive) < arch_max:
                        archive.append(xi[:])
                    else:
                        archive[random.randrange(arch_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                gain = fi - fu
                w = (gain if gain > 0.0 else 0.0) + 1e-12
                S_F.append(F); S_CR.append(CR); S_w.append(w)

                if fu < best:
                    best = fu
                    bestx = uvec[:]
                    improved_gen = True

        # memory update
        if S_F:
            wsum = sum(S_w) + 1e-30
            num = 0.0
            den = 0.0
            cr_acc = 0.0
            for w, F, CR in zip(S_w, S_F, S_CR):
                num += w * F * F
                den += w * F
                cr_acc += w * CR
            M_F[mem_k] = clamp(num / (den + 1e-30), 0.05, 1.0)
            M_CR[mem_k] = clamp01(cr_acc / wsum)
            mem_k = (mem_k + 1) % H

        # keep archive bounded
        if len(archive) > arch_max:
            random.shuffle(archive)
            archive = archive[:arch_max]

        # stagnation + restart injection
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if stagn >= max(7, int(2.2 * math.sqrt(NP))) and now() < deadline:
            stagn = 0
            # replace worst 20% with best-centered samples
            krep = max(2, NP // 5)
            worst = sorted(range(NP), key=lambda j: fit[j], reverse=True)
            rad = 0.22 * (1.0 - frac) + 0.02
            rep = 0
            for idx in worst:
                if rep >= krep or now() >= deadline:
                    break
                x = [bestx[d] + (random.random() * 2.0 - 1.0) * rad * spans[d] for d in range(dim)]
                x = [clamp(x[d], lows[d], highs[d]) for d in range(dim)]
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                rep += 1
                if fx < best:
                    best = fx
                    bestx = x[:]

        # late exploitation with pattern search (throttled)
        t = now()
        if t < deadline:
            if (frac > 0.70 or improved_gen) and (t - last_ls) > 0.06:
                budget = max(10, min(220, 7 * dim))
                step_frac = 0.06 * (1.0 - frac) + 0.008
                x2, f2 = pattern_search(bestx, best, budget, step_frac)
                if f2 < best:
                    best = f2
                    bestx = x2[:]
                    last_best = best
                last_ls = now()
