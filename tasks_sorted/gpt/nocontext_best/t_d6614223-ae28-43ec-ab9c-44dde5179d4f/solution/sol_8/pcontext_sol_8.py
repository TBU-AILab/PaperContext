import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improvements over provided code:
      - Uses a more faithful L-SHADE/JADE-style DE core:
          * current-to-pbest/1 + external archive
          * SHADE success-history memories for F/CR (weighted Lehmer mean for F)
          * Linear population size reduction (LPSR)
      - Evaluation caching (rounded key) to avoid re-evaluating identical points
      - Stronger, simpler boundary repair (reflect with period folding)
      - Better "stagnation kick": partial restarts mixing best-centered and uniform
      - Late phase: small adaptive pattern search around best (deterministic coords)
      - Minor speed tweaks (local bindings, fewer sorts, stable bookkeeping)

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    T = float(max_time)
    if T <= 0.0:
        # must return something; try a single midpoint eval if possible
        x = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")
    deadline = t0 + T

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    # ---------- helpers ----------
    rnd = random.random
    rndi = random.randrange
    tan = math.tan
    pi = math.pi
    sqrt = math.sqrt
    log = math.log
    cos = math.cos
    sin = math.sin

    def now():
        return time.time()

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # Box-Muller
    spare = [None]
    def randn():
        z = spare[0]
        if z is not None:
            spare[0] = None
            return z
        u1 = max(1e-12, rnd())
        u2 = rnd()
        r = sqrt(-2.0 * log(u1))
        th = 2.0 * pi * u2
        z0 = r * cos(th)
        z1 = r * sin(th)
        spare[0] = z1
        return z0

    # Reflect with period folding (always ends inside [lo,hi])
    def repair_inplace(x):
        for i in range(dim):
            lo = lows[i]; hi = highs[i]
            if x[i] < lo or x[i] > hi:
                w = hi - lo
                if w <= 0.0:
                    x[i] = lo
                else:
                    y = x[i] - lo
                    m = y % (2.0 * w)
                    x[i] = lo + (m if m <= w else (2.0 * w - m))
        return x

    def rand_vec():
        return [lows[i] + rnd() * spans[i] for i in range(dim)]

    def center_vec(jitter_frac):
        x = []
        for i in range(dim):
            c = 0.5 * (lows[i] + highs[i])
            x.append(c + (jitter_frac * spans[i]) * randn())
        return repair_inplace(x)

    def lhs_points(n):
        # light LHS: per-dim permutation stratification
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / n
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + rnd()) * invn
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # Evaluation cache with rounding to reduce collisions but still useful
    # (helps on plateaus / discrete-ish objectives / frequent boundary repairs).
    cache = {}
    # choose rounding precision based on span magnitude
    # (keep modest to prevent huge dictionary for continuous problems)
    prec = []
    for i in range(dim):
        s = spans[i]
        if s <= 0:
            prec.append(12)
        else:
            # about 1e-10 relative, capped
            p = int(max(6, min(12, 10 - math.floor(math.log10(s + 1e-30)))))
            prec.append(p)

    def key_of(x):
        return tuple(round(x[i], prec[i]) for i in range(dim))

    def eval_f(x):
        k = key_of(x)
        v = cache.get(k)
        if v is None:
            v = float(func(x))
            cache[k] = v
        return v

    # small deterministic pattern search around best (late phase)
    def pattern_search(bestx, bestf, budget, rad_frac):
        x = bestx[:]
        f = bestf
        rad = max(1e-12, rad_frac)
        # coordinate steps
        steps = [rad * spans[i] for i in range(dim)]
        evals = 0
        while evals < budget and now() < deadline:
            improved = False
            for d in range(dim):
                if evals >= budget or now() >= deadline:
                    break
                sd = steps[d]
                if sd <= 0.0:
                    continue
                xp = x[:]
                xp[d] += sd
                repair_inplace(xp)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue
                if evals >= budget or now() >= deadline:
                    break
                xm = x[:]
                xm[d] -= sd
                repair_inplace(xm)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True
            if improved:
                rad = min(0.35, rad * 1.25)
            else:
                rad *= 0.5
                if rad < 1e-12:
                    break
            for d in range(dim):
                steps[d] = rad * spans[d]
        return x, f

    # ---------- algorithm parameters ----------
    # Population sizing: slightly larger early than your code when time allows
    NP_max = max(28, min(180, 14 * dim))
    NP_min = max(10, min(50, 4 * dim))

    if T <= 0.2:
        NP_max = max(14, min(NP_max, 32))
        NP_min = max(8, min(NP_min, 18))

    H = 10 if dim <= 15 else 14
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_k = 0

    # ---------- init population ----------
    pop = []
    fit = []

    # Mix: LHS + opposition + centered jitter + random fill
    n_lhs = min(NP_max, max(10, int(sqrt(NP_max) + 10)))
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return min(fit) if fit else float("inf")
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if len(pop) < NP_max and now() < deadline:
            xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
            repair_inplace(xo)
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)

    for _ in range(min(8, max(2, dim // 2))):
        if len(pop) >= NP_max or now() >= deadline:
            break
        x = center_vec(0.12)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    while len(pop) < NP_max and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    NP = len(pop)
    if NP == 0:
        return float("inf")

    best_i = min(range(NP), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # Archive (for xr2)
    archive = []
    arch_max = NP_max

    # track stagnation
    last_best = best
    stagn = 0
    last_ps = t0

    # Utility: pick index not in excluded
    def pick_excluding(excl, n):
        j = rndi(n)
        while j in excl:
            j = rndi(n)
        return j

    gen = 0
    while now() < deadline:
        gen += 1
        t = now()
        frac = (t - t0) / T
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0

        # Linear population size reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min
        if NP > target_NP:
            # keep best target_NP individuals
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # p-best fraction: more exploit late
        pfrac = 0.40 * (1.0 - frac) + 0.06  # ~0.46 -> 0.06
        pcount = max(2, int(pfrac * NP))

        # Indices sorted for pbest selection
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # success sets for SHADE update
        S_F = []
        S_CR = []
        S_w = []

        union = pop + archive
        unionN = len(union)

        improved_gen = False

        # main per-individual update
        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # Sample memory index
            r = rndi(H)
            mF = M_F[r]
            mCR = M_CR[r]

            # Cauchy for F (resample)
            F = -1.0
            for _ in range(10):
                u = rnd()
                F = mF + 0.1 * tan(pi * (u - 0.5))
                if F > 0.0:
                    break
            if F <= 0.0:
                F = mF
            F = clamp(F, 0.05, 1.0)

            # Normal for CR
            CR = clamp01(mCR + 0.1 * randn())

            # Choose pbest from top pcount
            pbest_idx = idx_sorted[rndi(pcount)]
            xpbest = pop[pbest_idx]

            # r1 from pop (exclude i, pbest)
            r1 = pick_excluding({i, pbest_idx}, NP)
            xr1 = pop[r1]

            # r2 from union (avoid i/r1 if refers to pop-part)
            r2u = rndi(unionN)
            for _ in range(12):
                if r2u < NP and (r2u == i or r2u == r1):
                    r2u = rndi(unionN)
                else:
                    break
            xr2 = union[r2u]

            # Mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
            repair_inplace(v)

            # Binomial crossover
            jrand = rndi(dim)
            uvec = xi[:]  # base
            for d in range(dim):
                if d == jrand or rnd() < CR:
                    uvec[d] = v[d]

            fu = eval_f(uvec)

            if fu <= fi:
                # archive update with replaced parent
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[rndi(arch_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                gain = fi - fu
                w = gain + 1e-12  # ensure positive
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

                if fu < best:
                    best = fu
                    bestx = uvec[:]
                    improved_gen = True

        # Update best from population
        bi = min(range(NP), key=lambda j: fit[j])
        if fit[bi] < best:
            best = fit[bi]
            bestx = pop[bi][:]
            improved_gen = True

        # SHADE memory update
        if S_F:
            wsum = sum(S_w) + 1e-30
            num = 0.0
            den = 0.0
            crw = 0.0
            for w, F, CR in zip(S_w, S_F, S_CR):
                num += w * F * F
                den += w * F
                crw += w * CR
            F_new = num / (den + 1e-30)
            CR_new = crw / wsum
            M_F[mem_k] = clamp(F_new, 0.05, 1.0)
            M_CR[mem_k] = clamp01(CR_new)
            mem_k = (mem_k + 1) % H

        # stagnation detection
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # Stagnation kick / partial restart
        if stagn >= max(7, int(2.2 * sqrt(NP))) and now() < deadline:
            stagn = 0
            # Replace worst portion
            krep = max(2, NP // 4)
            worst = sorted(range(NP), key=lambda j: fit[j], reverse=True)
            # time-adaptive radius and mix
            rad = 0.30 * (1.0 - frac) + 0.02
            for c, idx in enumerate(worst):
                if c >= krep or now() >= deadline:
                    break
                # keep current best individual safe by index equality
                if idx == bi:
                    continue
                if rnd() < 0.65:
                    # best-centered perturbation
                    x = bestx[:]
                    for d in range(dim):
                        x[d] += (2.0 * rnd() - 1.0) * rad * spans[d]
                    repair_inplace(x)
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    bestx = x[:]
                    bi = idx

        # Late-phase local refinement (pattern search)
        # Throttle by time and only when close to end or when improved.
        tcur = now()
        if tcur < deadline:
            if (frac > 0.70 or improved_gen) and (tcur - last_ps) > 0.06:
                budget = max(10, min(220, 7 * dim))
                radius = 0.10 * (1.0 - frac) + 0.006
                bx, bf = pattern_search(bestx, best, budget, radius)
                if bf < best:
                    best = bf
                    bestx = bx[:]
                last_ps = now()

        # keep archive bounded
        if len(archive) > arch_max:
            archive = archive[-arch_max:]

    return best
