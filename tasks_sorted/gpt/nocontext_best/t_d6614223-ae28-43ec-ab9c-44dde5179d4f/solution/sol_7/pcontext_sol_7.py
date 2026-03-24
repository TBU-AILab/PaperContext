import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs provided code:
      - More reliable stagnation escape: multi-start "micro-restarts" with
        adaptive radius + occasional full random immigrants.
      - Stronger late exploitation: pattern-search + random-orthogonal probes
        (still derivative-free and cheap).
      - Cleaner / safer boundary reflection.
      - Slightly better p-best schedule and parameter sampling stability.
      - Ensures evaluation effort shifts from exploration -> exploitation as time runs out.

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

    # --- RNG helpers (no numpy) ---
    _spare = [None]
    def randn():
        z = _spare[0]
        if z is not None:
            _spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        _spare[0] = r * math.sin(th)
        return z0

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    # Robust reflection into bounds (works even for very large steps)
    def reflect_bounds(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            w = hi - lo
            y = x[i] - lo
            # map to [0, 2w) then reflect to [0, w]
            m = y % (2.0 * w)
            if m < 0.0:
                m += 2.0 * w
            if m <= w:
                x[i] = lo + m
            else:
                x[i] = lo + (2.0 * w - m)
            # safety clamp
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
        x = []
        for i in range(dim):
            c = 0.5 * (lows[i] + highs[i])
            x.append(c + scale * spans[i] * randn())
        return reflect_bounds(x)

    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / max(1, n)
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + random.random()) * invn
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # Late-phase local exploitation: pattern search + a few random directions
    def local_search(bestx, bestf, max_evals, radius_frac):
        x = bestx[:]
        f = bestf
        evals = 0

        rad = max(1e-12, float(radius_frac))
        step = [rad * spans[i] for i in range(dim)]

        while evals < max_evals and now() < deadline:
            improved = False

            # Pattern search on coordinates
            for d in range(dim):
                if evals >= max_evals or now() >= deadline:
                    break
                sd = step[d]
                if sd <= 1e-18 * spans[d]:
                    continue

                xp = x[:]
                xp[d] += sd
                reflect_bounds(xp)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= max_evals or now() >= deadline:
                    break

                xm = x[:]
                xm[d] -= sd
                reflect_bounds(xm)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            # A couple random direction probes (helps in rotated landscapes)
            if evals < max_evals and now() < deadline:
                trials = 3 if dim <= 12 else 2 if dim <= 40 else 1
                for _ in range(trials):
                    if evals >= max_evals or now() >= deadline:
                        break
                    v = [randn() for _ in range(dim)]
                    nrm = math.sqrt(sum(vi * vi for vi in v)) + 1e-12
                    # try both + and - along direction
                    for sgn in (1.0, -1.0):
                        if evals >= max_evals or now() >= deadline:
                            break
                        xr = [x[i] + sgn * rad * spans[i] * (v[i] / nrm) for i in range(dim)]
                        reflect_bounds(xr)
                        fr = eval_f(xr); evals += 1
                        if fr < f:
                            x, f = xr, fr
                            improved = True
                            break

            if improved:
                rad = min(0.35, rad * 1.20)
            else:
                rad *= 0.55
                if rad < 1e-12:
                    break
            for d in range(dim):
                step[d] = rad * spans[d]

        return x, f

    # ---- L-SHADE-ish DE setup ----
    # Population size: a bit larger early, reduce with time
    NP_max = max(28, min(180, 14 * dim))
    NP_min = max(10, min(50, 4 * dim))

    if max_time <= 0.2:
        NP_max = max(14, min(NP_max, 36))
        NP_min = max(8, min(NP_min, 18))

    # Initialize population with mixture
    pop, fit = [], []

    n_lhs = min(NP_max, max(10, int(math.sqrt(NP_max) + 8)))
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return float("inf") if not fit else min(fit)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)
        if len(pop) < NP_max and now() < deadline:
            xo = reflect_bounds(opp_vec(x))
            fxo = eval_f(xo)
            pop.append(xo); fit.append(fxo)

    for _ in range(min(8, max(3, dim // 3))):
        if len(pop) >= NP_max or now() >= deadline:
            break
        x = center_jitter(0.18)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    while len(pop) < NP_max and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    NP = len(pop)
    bi = min(range(NP), key=lambda i: fit[i])
    bestx = pop[bi][:]
    best = fit[bi]

    if now() >= deadline:
        return best

    # Success-history memories
    H = 10 if dim <= 20 else 14
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_k = 0

    archive = []
    arch_max = NP_max

    def pick_r(exclude, n):
        j = random.randrange(n)
        while j in exclude:
            j = random.randrange(n)
        return j

    last_best = best
    stagn = 0
    last_ls_time = 0.0
    last_restart_time = 0.0
    restarts = 0

    # Main loop
    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))
        frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)

        # Linear population size reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        if NP > target_NP:
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = order[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = target_NP
            arch_max = max(arch_max, NP)
            if len(archive) > arch_max:
                archive = archive[-arch_max:]

        idx_sorted = sorted(range(NP), key=lambda i: fit[i])

        # p-best fraction: higher early, tighter late
        pfrac = 0.40 * (1.0 - frac) + 0.06  # 0.46 -> 0.06
        pcount = max(2, int(pfrac * NP))

        S_F, S_CR, S_w = [], [], []
        improved_gen = False

        union = pop + archive
        unionN = len(union)

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mF, mCR = M_F[r], M_CR[r]

            # F: cauchy-ish sampling, stabilized
            F = -1.0
            for _ in range(10):
                u = random.random()
                c = math.tan(math.pi * (u - 0.5))
                F = mF + 0.08 * c
                if F > 0.0:
                    break
            if F <= 0.0:
                F = mF
            F = clamp(F, 0.04, 1.0)

            # CR: normal
            CR = clamp01(mCR + 0.10 * randn())

            pbest_idx = idx_sorted[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            r1 = pick_r({i, pbest_idx}, NP)
            xr1 = pop[r1]

            # pick from union excluding invalid pop indices
            r2u = random.randrange(unionN)
            for _ in range(16):
                if r2u < NP and (r2u == i or r2u == r1 or r2u == pbest_idx):
                    r2u = random.randrange(unionN)
                else:
                    break
            xr2 = union[r2u]

            # current-to-pbest/1
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
                # archive update (store replaced parent)
                if len(archive) < arch_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(arch_max)] = xi[:]

                pop[i] = uvec
                fit[i] = fu

                gain = fi - fu
                if gain < 0.0:
                    gain = 0.0
                w = gain + 1e-12
                S_F.append(F); S_CR.append(CR); S_w.append(w)

                if fu < best:
                    best = fu
                    bestx = uvec[:]
                    improved_gen = True

        # Best refresh
        bi = min(range(NP), key=lambda j: fit[j])
        if fit[bi] < best:
            best = fit[bi]
            bestx = pop[bi][:]

        # Update memories
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
            M_F[mem_k] = clamp(F_new, 0.04, 1.0)
            M_CR[mem_k] = clamp01(CR_new)
            mem_k = (mem_k + 1) % H

        # Stagnation tracking
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # --- Micro-restarts / immigrants if stagnating ---
        # Increase aggressiveness later and with repeated restarts.
        if now() < deadline:
            trig = max(7, int(2.2 * math.sqrt(NP)))
            if stagn >= trig and (now() - last_restart_time) > 0.05:
                stagn = 0
                last_restart_time = now()
                restarts += 1

                # Replace worst portion
                worst_k = max(2, NP // (4 if frac < 0.6 else 3))
                worst = sorted(range(NP), key=lambda j: fit[j], reverse=True)

                # Radius around best shrinks with time but expands with restart count a bit
                rad = (0.30 * (1.0 - frac) + 0.02) * (1.0 + 0.15 * min(6, restarts))
                rad = min(0.60, rad)

                # Also inject a few full-random immigrants sometimes
                n_random = 0
                if frac < 0.5 and restarts % 2 == 0:
                    n_random = max(1, worst_k // 5)

                rep = 0
                for idx in worst:
                    if rep >= worst_k or now() >= deadline:
                        break

                    # don't overwrite current best vector
                    if fit[idx] == best:
                        continue

                    if n_random > 0:
                        x = rand_vec()
                        n_random -= 1
                    else:
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

        # --- Late-phase local search (more consistent, but throttled) ---
        if now() < deadline:
            # start earlier than before but with small budget;
            # heavier near the end.
            if (frac > 0.55 or improved_gen) and (now() - last_ls_time) > 0.04:
                # Budget ramps up near the end, still bounded
                base = 5 * dim
                ramp = int((frac ** 2) * 10 * dim)
                budget = int(clamp(base + ramp, 10, 260))
                radius = 0.10 * (1.0 - frac) + 0.006
                x2, f2 = local_search(bestx, best, budget, radius)
                if f2 < best:
                    best, bestx = f2, x2[:]
                last_ls_time = now()

        # Keep archive bounded
        if len(archive) > arch_max:
            archive = archive[-arch_max:]

    return best
