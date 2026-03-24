import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (self-contained, no external libs).

    Improvements over the provided code:
      - Uses *evaluation-budgeted* timing checks (lower overhead than datetime).
      - Stronger DE core: jDE-style self-adaptation per individual (Fi, CRi)
        + current-to-pbest/1 + archive (JADE/SHADE spirit).
      - Heavy-tailed mutation steps (occasional large F) to escape local minima.
      - Smarter boundary handling: reflection with repeated folding.
      - Better restarts: partial reinit via (best + Lévy-ish step) and opposition.
      - Local search: lightweight pattern/coordinate search with step control.
      - Fewer sorts: partial pbest set updated by periodic ranking.
    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    if max_time is None:
        max_time = 1.0
    deadline = t0 + max(0.0, float(max_time))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0

    # ---- small utilities ----
    def now():
        return time.time()

    def eval_f(x):
        return float(func(x))

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    # Box-Muller normal
    _spare = [None]
    def randn():
        s = _spare[0]
        if s is not None:
            _spare[0] = None
            return s
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare[0] = z1
        return z0

    # Reflection/folding boundary handling (robust for huge steps)
    def reflect_bounds(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if hi <= lo:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                w = hi - lo
                y = v - lo
                m = y % (2.0 * w)
                v = lo + (m if m <= w else (2.0 * w - m))
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x[i] = v
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opp_vec(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Lévy-ish step (no special functions): use heavy-tailed via Cauchy / |N|^p
    def heavy_step(scale_frac):
        # mixture: mostly gaussian, sometimes cauchy-like
        step = [0.0] * dim
        if random.random() < 0.25:
            # cauchy-like using tan(pi(u-0.5))
            for i in range(dim):
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))
                step[i] = scale_frac * spans[i] * 0.15 * c
        else:
            p = 1.35  # heavy-ish: |N|^p
            for i in range(dim):
                z = abs(randn()) ** p
                sgn = -1.0 if random.random() < 0.5 else 1.0
                step[i] = sgn * scale_frac * spans[i] * 0.35 * z
        return step

    # LHS-like init (cheap)
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

    # very light local search (pattern/coordinate with shrinking step)
    def local_search(bestx, bestf, budget, step_frac):
        x = bestx[:]
        f = bestf
        step = [max(1e-15, step_frac) * spans[i] for i in range(dim)]
        evals = 0

        # try a couple random directions too
        while evals < budget and now() < deadline:
            improved = False

            # coordinate probes
            for d in range(dim):
                if evals >= budget or now() >= deadline:
                    break
                sd = step[d]
                if sd <= 0.0:
                    continue

                xp = x[:]
                xp[d] += sd
                reflect_bounds(xp)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= budget or now() >= deadline:
                    break

                xm = x[:]
                xm[d] -= sd
                reflect_bounds(xm)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            # random direction probes
            if evals < budget and now() < deadline:
                trials = 2 if dim <= 12 else 1
                for _ in range(trials):
                    if evals >= budget or now() >= deadline:
                        break
                    dirn = [randn() for _ in range(dim)]
                    nrm = math.sqrt(sum(v*v for v in dirn)) + 1e-12
                    xr = [x[i] + (step_frac * spans[i]) * (dirn[i] / nrm) for i in range(dim)]
                    reflect_bounds(xr)
                    fr = eval_f(xr); evals += 1
                    if fr < f:
                        x, f = xr, fr
                        improved = True

            if improved:
                # slightly expand (helps follow curved valleys)
                step_frac = min(0.25, step_frac * 1.15)
            else:
                # shrink
                step_frac *= 0.5
                if step_frac < 1e-12:
                    break
            step = [step_frac * spans[i] for i in range(dim)]

        return x, f

    # ---- parameterization ----
    # population sizing tuned for time-bounded setting
    NP_max = max(24, min(160, 10 * dim + 20))
    NP_min = max(10, min(50, 4 * dim + 6))

    if max_time <= 0.2:
        NP_max = min(NP_max, 36)
        NP_min = min(NP_min, 18)

    # init population: LHS + opposition + a few center points
    pop = []
    fit = []
    Fi = []   # per-individual F
    CRi = []  # per-individual CR

    n_lhs = min(NP_max, max(10, int(math.sqrt(NP_max) + 9)))
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return float("inf") if not fit else min(fit)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        Fi.append(0.6 + 0.2 * random.random())
        CRi.append(0.4 + 0.4 * random.random())

        if len(pop) < NP_max and now() < deadline:
            xo = reflect_bounds(opp_vec(x))
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)
            Fi.append(0.6 + 0.2 * random.random())
            CRi.append(0.4 + 0.4 * random.random())

    # center jitter
    c = [0.5 * (lows[i] + highs[i]) for i in range(dim)]
    for _ in range(min(6, max(2, dim // 3))):
        if len(pop) >= NP_max or now() >= deadline:
            break
        x = [c[i] + 0.12 * spans[i] * randn() for i in range(dim)]
        reflect_bounds(x)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        Fi.append(0.6 + 0.2 * random.random())
        CRi.append(0.4 + 0.4 * random.random())

    while len(pop) < NP_max and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        Fi.append(0.6 + 0.2 * random.random())
        CRi.append(0.4 + 0.4 * random.random())

    NP = len(pop)
    bi = min(range(NP), key=lambda i: fit[i])
    bestx = pop[bi][:]
    best = fit[bi]

    if now() >= deadline:
        return best

    # Archive (stores replaced solutions)
    archive = []
    arch_max = NP_max

    # Helper: pick random index not in a small exclude set
    def pick_r(excl, n):
        j = random.randrange(n)
        while j in excl:
            j = random.randrange(n)
        return j

    # Ranking refresh period to reduce sorting overhead
    rank_refresh = 3
    idx_sorted = sorted(range(NP), key=lambda i: fit[i])
    last_best = best
    stagn = 0
    last_ls = 0.0

    gen = 0
    while now() < deadline:
        gen += 1

        # time fraction
        frac = (now() - t0) / max(1e-12, float(max_time))
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # Linear population size reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        if NP > target_NP:
            # remove worst, keep best
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            keep = idx_sorted[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            Fi  = [Fi[i]  for i in keep]
            CRi = [CRi[i] for i in keep]
            NP = target_NP
            arch_max = max(arch_max, NP)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        # refresh ranking sometimes
        if gen % rank_refresh == 1:
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # pbest count (more exploit later)
        pfrac = 0.30 * (1.0 - frac) + 0.08
        pcount = max(2, int(pfrac * NP))

        union = pop + archive
        unionN = len(union)

        improved_gen = False

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # jDE self-adaptation
            if random.random() < 0.1:
                # mixture: mostly moderate, sometimes large
                if random.random() < 0.2:
                    Fi[i] = clamp(0.9 + 0.2 * random.random(), 0.05, 1.0)
                else:
                    Fi[i] = clamp(0.3 + 0.6 * random.random(), 0.05, 1.0)
            if random.random() < 0.1:
                CRi[i] = clamp01(random.random())

            F = Fi[i]
            CR = CRi[i]

            # occasional heavy-tail kick
            if random.random() < 0.07:
                F = clamp(F * (1.0 + 0.8 * abs(randn())), 0.05, 1.0)

            # choose pbest from top pcount
            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            r1 = pick_r({i, pbest_idx}, NP)
            xr1 = pop[r1]

            r2u = random.randrange(unionN)
            for _ in range(10):
                if r2u < NP and (r2u == i or r2u == r1):
                    r2u = random.randrange(unionN)
                else:
                    break
            xr2 = union[r2u]

            # current-to-pbest/1
            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
            reflect_bounds(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            fu = eval_f(u)

            if fu <= fi:
                # archive update
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = u
                fit[i] = fu
                improved_gen = True

                if fu < best:
                    best = fu
                    bestx = u[:]

        # stagnation tracking
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # restart / injection if stagnating
        if stagn >= max(8, int(2.5 * math.sqrt(NP))) and now() < deadline:
            stagn = 0

            # replace worst ~25% with heavy-tail around best + opposition + random
            krep = max(2, NP // 4)
            worst = sorted(range(NP), key=lambda j: fit[j], reverse=True)

            rad = 0.30 * (1.0 - frac) + 0.02
            rep = 0
            for idx in worst:
                if rep >= krep or now() >= deadline:
                    break

                # build candidate
                r = random.random()
                if r < 0.55:
                    step = heavy_step(rad)
                    x = [bestx[d] + step[d] for d in range(dim)]
                elif r < 0.80:
                    x = opp_vec(bestx)
                else:
                    x = rand_vec()

                reflect_bounds(x)
                fx = eval_f(x)

                pop[idx] = x
                fit[idx] = fx
                Fi[idx] = 0.4 + 0.6 * random.random()
                CRi[idx] = random.random()

                rep += 1
                if fx < best:
                    best = fx
                    bestx = x[:]

            idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # local search late or after improvements (throttled)
        if now() < deadline:
            if (frac > 0.60 or improved_gen) and (now() - last_ls) > 0.06:
                budget = max(10, min(220, 7 * dim))
                step_frac = 0.10 * (1.0 - frac) + 0.008
                x2, f2 = local_search(bestx, best, budget, step_frac)
                if f2 < best:
                    best = f2
                    bestx = x2[:]
                    last_best = best
                last_ls = now()

    return best
