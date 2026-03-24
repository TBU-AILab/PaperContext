import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (stdlib only).

    Improvements vs the best prior (JADE/DE + local TR):
      - Adds a fast surrogate-free "line-search on random directions" (RDSA-style)
        around the incumbent with adaptive step + backtracking.
      - Uses a stronger restart scheme: multi-armed restarts (global random / elite
        centroid / best-perturb) with radius reset.
      - Uses small elite-based recombination proposals each generation.
      - Fixes/strengthens success-history update (mu_F, mu_CR) and uses per-individual
        memories (lightweight SHADE-ish).
      - Better time budgeting: always reserves end-game time for aggressive local polish.

    Returns: best fitness (float)
    """
    t0 = time.time()
    T = float(max_time)
    deadline = t0 + T

    # --------- guards / bounds ----------
    if dim <= 0:
        try:
            v = float(func([]))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    for i in range(dim):
        if highs[i] < lows[i]:
            lows[i], highs[i] = highs[i], lows[i]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def ensure_bounds(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            # reflect repeatedly if outside
            while y[i] < lo or y[i] > hi:
                if y[i] < lo:
                    y[i] = lo + (lo - y[i])
                if y[i] > hi:
                    y[i] = hi - (y[i] - hi)
            if y[i] < lo: y[i] = lo
            if y[i] > hi: y[i] = hi
        return y

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def rand_dir():
        # random direction ~ N(0,1) normalized (cheap)
        s2 = 0.0
        d = [0.0] * dim
        for i in range(dim):
            r = random.gauss(0.0, 1.0)
            d[i] = r
            s2 += r * r
        if s2 <= 1e-30:
            # fallback: axis
            j = random.randrange(dim)
            d = [0.0] * dim
            d[j] = 1.0
            return d
        inv = 1.0 / math.sqrt(s2)
        for i in range(dim):
            d[i] *= inv
        return d

    def opposition(x):
        y = [(lows[i] + highs[i]) - x[i] for i in range(dim)]
        return ensure_bounds(y)

    # --------- elite store ----------
    elite_cap = max(10, min(50, 3 * dim + 10))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_cap:
            elites.pop()

    # --------- initialization: LHS-ish + opposition ----------
    # keep it fast but diverse
    pop_size = max(18, min(90, 10 + 2 * dim + 4 * int(math.sqrt(dim))))
    if T <= 1.0:
        pop_size = max(14, min(pop_size, 28))
    elif T <= 3.0:
        pop_size = max(16, min(pop_size, 48))

    bins = []
    for d in range(dim):
        n = pop_size
        if spans[d] <= 0:
            bins.append([lows[d]] * n)
        else:
            pts = [(k + random.random()) / n for k in range(n)]
            random.shuffle(pts)
            bins.append([lows[d] + p * spans[d] for p in pts])

    pop, fit = [], []
    best = float("inf")
    best_x = rand_vec()

    # Seed with time cap
    seed_deadline = t0 + min(0.18, 0.07 + 0.0025 * dim) * T
    i = 0
    while i < pop_size and time.time() < seed_deadline:
        x = ensure_bounds([bins[d][i] for d in range(dim)])
        f = safe_eval(x)
        pop.append(x); fit.append(f)
        push_elite(f, x)
        if f < best:
            best, best_x = f, x[:]

        xo = opposition(x)
        fo = safe_eval(xo)
        pop.append(xo); fit.append(fo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]
        i += 1

    while len(pop) < pop_size and time.time() < deadline:
        x = rand_vec()
        f = safe_eval(x)
        pop.append(x); fit.append(f)
        push_elite(f, x)
        if f < best:
            best, best_x = f, x[:]

    if len(pop) > pop_size:
        order = sorted(range(len(pop)), key=lambda k: fit[k])[:pop_size]
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]

    # --------- SHADE-lite DE with per-individual memories ----------
    H = max(6, min(20, dim))  # history size
    M_F = [0.6] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []
    archive_max = pop_size

    def clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    last_improve_t = time.time()
    stall_restart_seconds = max(0.25, 0.22 * T)

    # trust radii for local search
    tr = [0.22 * s if s > 0 else 0.0 for s in spans]

    # --------- local directional line search (main new component) ----------
    def directional_search(x0, f0, frac):
        """
        Random-direction pattern search with backtracking, reflection-safe.
        Spend very few evals but often yields big gains late.
        """
        if time.time() >= deadline:
            return x0, f0

        # step size relative to span; shrink over time but not too small
        base = 0.18 * (1.0 - frac) + 0.02
        # also tie to current trust region
        step = [max(1e-15, base * (spans[i] if spans[i] > 0 else 0.0) + 0.35 * tr[i]) for i in range(dim)]

        # number of directions: small (time-bounded)
        ndirs = 2 if dim > 25 else 3
        best_loc_x, best_loc_f = x0[:], f0

        for _ in range(ndirs):
            if time.time() >= deadline:
                break
            d = rand_dir()

            # try +/- and a short backtracking sequence on the better side
            cand_p = ensure_bounds([x0[i] + d[i] * step[i] for i in range(dim)])
            fp = safe_eval(cand_p)
            push_elite(fp, cand_p)

            cand_m = ensure_bounds([x0[i] - d[i] * step[i] for i in range(dim)])
            fm = safe_eval(cand_m)
            push_elite(fm, cand_m)

            if fp < best_loc_f or fm < best_loc_f:
                if fp <= fm:
                    x1, f1, sign = cand_p, fp, 1.0
                else:
                    x1, f1, sign = cand_m, fm, -1.0

                best_loc_x, best_loc_f = x1[:], f1

                # backtracking/forward tracking along same dir (2 extra tries max)
                alpha = 1.0
                for _bt in range(2):
                    if time.time() >= deadline:
                        break
                    alpha *= 1.6
                    cand2 = ensure_bounds([x0[i] + sign * d[i] * (alpha * step[i]) for i in range(dim)])
                    f2 = safe_eval(cand2)
                    push_elite(f2, cand2)
                    if f2 < best_loc_f:
                        best_loc_x, best_loc_f = cand2[:], f2
                    else:
                        break  # stop extending if not improving

        return best_loc_x, best_loc_f

    # --------- restart helper ----------
    def restart(frac):
        nonlocal pop, fit, archive, M_F, M_CR, h_idx, tr, last_improve_t
        # keep a few elites
        keep = max(3, pop_size // 6)
        base = elites[:min(len(elites), keep)]
        if not base:
            base = [(best, best_x[:])]

        # choose restart "mode"
        r = random.random()
        if r < 0.45:
            center = rand_vec()
        elif r < 0.75:
            center = base[0][1][:]  # best
        else:
            # elite centroid of top few
            k = min(len(base), max(3, keep))
            center = [0.0] * dim
            for j in range(k):
                xj = base[j][1]
                for d in range(dim):
                    center[d] += xj[d]
            inv = 1.0 / k
            for d in range(dim):
                center[d] *= inv
            center = ensure_bounds(center)

        # widen trust region
        for d in range(dim):
            if spans[d] > 0:
                tr[d] = max(0.18 * spans[d], tr[d])

        # reset history mildly (don’t fully forget)
        M_F = [0.65] * H
        M_CR = [0.5] * H
        h_idx = 0
        archive = []

        new_pop, new_fit = [], []
        # inject kept elites first
        for f, x in base:
            new_pop.append(x[:])
            new_fit.append(safe_eval(x))

        # fill remainder with mix: random, around center, around best
        while len(new_pop) < pop_size and time.time() < deadline:
            u = random.random()
            if u < 0.45:
                x = rand_vec()
            else:
                if u < 0.75:
                    c = center
                    s = (0.45 * (1.0 - frac) + 0.12)
                else:
                    c = best_x
                    s = (0.35 * (1.0 - frac) + 0.10)
                x = c[:]
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, s * spans[d])
                x = ensure_bounds(x)

            f = safe_eval(x)
            new_pop.append(x); new_fit.append(f)
            push_elite(f, x)

        pop, fit = new_pop, new_fit
        last_improve_t = time.time()

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best
        gen += 1
        frac = min(1.0, (now - t0) / max(1e-12, T))

        # end-game reserve: prioritize local improvement
        if frac > 0.82:
            x2, f2 = directional_search(best_x, best, frac)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()
            # a couple quick coordinate pokes
            for _ in range(2):
                if time.time() >= deadline:
                    return best
                d = random.randrange(dim)
                if spans[d] <= 0:
                    continue
                step = max(1e-15, (0.015 + 0.05 * (1.0 - frac)) * spans[d])
                cand = best_x[:]
                cand[d] += random.choice([-1.0, 1.0]) * step
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                push_elite(fc, cand)
                if fc < best:
                    best, best_x = fc, cand[:]
                    last_improve_t = time.time()
            continue

        # pbest pressure increases over time
        p_min, p_max = 0.05, 0.35
        p_frac = p_min + (p_max - p_min) * (0.20 + 0.80 * frac)
        p_cnt = max(2, int(math.ceil(p_frac * pop_size)))

        order = sorted(range(pop_size), key=lambda k: fit[k])

        S_F, S_CR, dF = [], [], []
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            # sample Fi ~ cauchy(M_F[r], 0.1), CRi ~ normal(M_CR[r], 0.1)
            Fi = None
            for _t in range(10):
                u = random.random()
                val = M_F[r] + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < val <= 1.0:
                    Fi = val
                    break
            if Fi is None:
                Fi = max(0.05, min(1.0, M_F[r]))
            CRi = clip01(random.gauss(M_CR[r], 0.1))

            # later: slightly smaller Fi to refine
            Fi *= (0.95 - 0.15 * frac)
            Fi = max(0.05, min(1.0, Fi))

            pbest_idx = order[random.randrange(p_cnt)]
            xpb = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            union = pop + archive
            union_n = len(union)
            x2 = None
            for _t in range(25):
                j = random.randrange(union_n)
                if j < pop_size and (j == i or j == r1):
                    continue
                x2 = union[j]
                break
            if x2 is None:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x2 = pop[r2]

            xr1 = pop[r1]
            xr2 = x2

            v = [xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]
            v = ensure_bounds(v)

            jrand = random.randrange(dim)
            uvec = [v[d] if (random.random() < CRi or d == jrand) else xi[d] for d in range(dim)]
            uvec = ensure_bounds(uvec)

            fu = safe_eval(uvec)
            push_elite(fu, uvec)

            # elite recombination proposal (cheap, once in a while)
            if (i == 0) and elites and (gen % 2 == 0) and time.time() < deadline:
                # blend best with another elite + small noise
                e = elites[random.randrange(min(len(elites), 8))][1]
                mix = 0.65 + 0.25 * random.random()
                cand = [mix * best_x[d] + (1.0 - mix) * e[d] for d in range(dim)]
                for d in range(dim):
                    if spans[d] > 0:
                        cand[d] += random.gauss(0.0, 0.02 * spans[d] * (1.0 - frac))
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                push_elite(fc, cand)
                if fc < fu:
                    uvec, fu = cand, fc

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = uvec
                fit[i] = fu

                if fu < fi:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    dF.append(fi - fu)

                if fu < best:
                    best, best_x = fu, uvec[:]
                    last_improve_t = time.time()

        # update history (SHADE style)
        if dF:
            s = sum(dF)
            wts = [di / s for di in dF] if s > 0 else [1.0 / len(dF)] * len(dF)

            # weighted mean for CR
            mcr = 0.0
            for w, cr in zip(wts, S_CR):
                mcr += w * cr

            # weighted Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, f in zip(wts, S_F):
                num += w * f * f
                den += w * f
            mf = M_F[h_idx]
            if den > 1e-12:
                mf = num / den

            M_CR[h_idx] = clip01(mcr)
            M_F[h_idx] = max(0.05, min(1.0, mf))
            h_idx = (h_idx + 1) % H

        # interleaved local improvement around best (directional)
        if (gen % 2 == 0) and time.time() < deadline:
            x2, f2 = directional_search(best_x, best, frac)
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()

            # shrink trust region slowly
            for d in range(dim):
                if spans[d] > 0:
                    tr[d] = max(0.004 * spans[d], tr[d] * (0.88 + 0.06 * (1.0 - frac)))

        # restart on stall
        if time.time() - last_improve_t > stall_restart_seconds:
            restart(frac)

        # occasional forced best perturb (very cheap)
        if (gen % 3 == 0) and time.time() < deadline:
            x = best_x[:]
            scale = (0.18 * (1.0 - frac) + 0.03)
            for d in range(dim):
                if spans[d] > 0 and random.random() < (0.25 if dim > 25 else 0.35):
                    x[d] += random.gauss(0.0, scale * spans[d])
            x = ensure_bounds(x)
            fx = safe_eval(x)
            push_elite(fx, x)
            if fx < best:
                best, best_x = fx, x[:]
                last_improve_t = time.time()
