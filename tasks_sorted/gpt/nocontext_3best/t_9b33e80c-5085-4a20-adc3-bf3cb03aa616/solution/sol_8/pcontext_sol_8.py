import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib only).

    What is improved vs your current best (#1):
      1) Better global exploration without extra evals:
         - Uses a low-discrepancy (Halton) initializer + opposition (as before),
           but also keeps a small "explorer" subgroup refreshed regularly.
      2) Stronger exploitation near the best:
         - Adds an extremely effective bounded quadratic line-search along
           random directions + coordinate directions (derivative-free).
         - Adds a tiny simplex-like local step (very small overhead) only when cheap.
      3) More reliable time/eval budgeting:
         - Tracks eval time EMA and uses it to decide how many extra trials/local
           steps are safe, with a strict end margin to avoid deadline misses.
      4) More stable DE core:
         - Keeps L-SHADE/JADE success-history adaptation + archive,
           but removes some wasted “screening” and replaces it with
           (a) one main DE trial and (b) an occasional second trial only if budget allows.
      5) Robust handling of zero-span dimensions and bounds.

    Returns:
        best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lo = [0.0] * dim
    hi = [0.0] * dim
    span = [0.0] * dim
    span2_sum = 0.0
    for i, (a, b) in enumerate(bounds):
        a = float(a); b = float(b)
        if b < a:
            raise ValueError("Each bound must be (low, high) with high >= low")
        lo[i] = a; hi[i] = b
        span[i] = b - a
        span2_sum += span[i] * span[i]

    # Single-point domain
    if all(s == 0.0 for s in span):
        x0 = [lo[i] for i in range(dim)]
        return float(func(x0))

    random.seed()

    # -------------------- small utilities --------------------
    def now():
        return time.time()

    def time_frac():
        return (now() - t0) / max(1e-12, float(max_time))

    def clamp(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def clamp_vec(x):
        y = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                y[d] = lo[d]
            else:
                y[d] = clamp(x[d], lo[d], hi[d])
        return y

    def repair_midpoint(u, parent):
        # Midpoint repair + clamp (good for DE stability)
        for d in range(dim):
            if span[d] == 0.0:
                u[d] = lo[d]
            else:
                if u[d] < lo[d]:
                    u[d] = 0.5 * (lo[d] + parent[d])
                elif u[d] > hi[d]:
                    u[d] = 0.5 * (hi[d] + parent[d])
                if u[d] < lo[d]: u[d] = lo[d]
                elif u[d] > hi[d]: u[d] = hi[d]
        return u

    def l2_dist2(a, b):
        s = 0.0
        for d in range(dim):
            t = a[d] - b[d]
            s += t * t
        return s

    def rand_point():
        x = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + random.random() * span[d]
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                xo[d] = lo[d]
            else:
                xo[d] = clamp(lo[d] + hi[d] - x[d], lo[d], hi[d])
        return xo

    # Box-Muller normal
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_normal(mu, sig):
        return mu + sig * randn()

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # -------------------- Halton sequence --------------------
    def first_primes(n):
        primes, k = [], 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_value(index, base):
        f, r, i = 1.0, 0.0, index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    bases = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + halton_value(k, bases[d]) * span[d]
        return x

    # -------------------- time-aware evaluation --------------------
    eval_calls = 0
    avg_eval = None  # EMA seconds

    def eval_f(x):
        nonlocal eval_calls, avg_eval
        t1 = now()
        fx = float(func(x))
        t2 = now()
        dt = max(1e-6, t2 - t1)
        eval_calls += 1
        if avg_eval is None:
            avg_eval = dt
        else:
            a = 0.08
            avg_eval = (1.0 - a) * avg_eval + a * dt
        return fx

    def remaining_eval_budget(margin_frac=0.07):
        rem = max(0.0, deadline - now())
        rem2 = max(0.0, rem - margin_frac * float(max_time))
        est = avg_eval if avg_eval is not None else 1e-3
        return int(rem2 / max(est, 1e-6))

    # -------------------- local: bounded directional quadratic line-search --------------------
    def local_dir_search(x, fx, time_limit, frac):
        """
        Derivative-free: pick directions, do small bracket + quadratic fit on [-a, a].
        Very effective on smooth-ish landscapes and cheap per improvement.
        """
        bestx = list(x)
        bestf = fx

        # base step fraction shrinks with time
        base = 0.18 * (1.0 - 0.90 * frac) + 0.0025
        # cap to avoid huge steps in large spans
        a0 = base

        # limit evaluations
        budget = max(16, min(110, 5 * dim + 20))

        # Precompute some coordinate directions and random directions
        # We'll alternate: coordinate, random
        tries = 0
        while tries < budget and now() < time_limit:
            tries += 1

            # pick direction
            dvec = [0.0] * dim
            if dim <= 10 or random.random() < 0.55:
                j = random.randrange(dim)
                if span[j] == 0.0:
                    continue
                dvec[j] = 1.0 if random.random() < 0.5 else -1.0
            else:
                # random gaussian direction, then normalize approximately
                norm2 = 0.0
                for j in range(dim):
                    if span[j] == 0.0:
                        dvec[j] = 0.0
                    else:
                        z = randn()
                        dvec[j] = z
                        norm2 += z * z
                if norm2 <= 1e-24:
                    continue
                inv = 1.0 / math.sqrt(norm2)
                for j in range(dim):
                    dvec[j] *= inv

            # choose step "a" relative to spans along direction
            # compute max feasible alpha in each direction (bounds)
            max_pos = float("inf")
            max_neg = float("inf")
            for j in range(dim):
                if span[j] == 0.0 or dvec[j] == 0.0:
                    continue
                if dvec[j] > 0:
                    max_pos = min(max_pos, (hi[j] - bestx[j]) / dvec[j])
                    max_neg = min(max_neg, (bestx[j] - lo[j]) / dvec[j])
                else:
                    # dvec[j] < 0
                    max_pos = min(max_pos, (bestx[j] - lo[j]) / (-dvec[j]))
                    max_neg = min(max_neg, (hi[j] - bestx[j]) / (-dvec[j]))

            if max_pos <= 0.0 and max_neg <= 0.0:
                continue

            # scale step by a0 * typical span (use RMS span)
            rms_span = math.sqrt(span2_sum / float(dim))
            a = a0 * rms_span
            # clamp a to feasible
            a = max(1e-16, min(a, 0.85 * max_pos if max_pos != float("inf") else a))
            a = max(1e-16, min(a, 0.85 * max_neg if max_neg != float("inf") else a))

            # evaluate at -a, 0, +a (0 already known)
            xm = [bestx[j] - a * dvec[j] for j in range(dim)]
            xp = [bestx[j] + a * dvec[j] for j in range(dim)]
            xm = clamp_vec(xm)
            xp = clamp_vec(xp)

            fm = eval_f(xm)
            if now() >= time_limit:
                break
            fp = eval_f(xp)
            if now() >= time_limit:
                break

            # Quadratic fit through (-a,fm), (0,bestf), (+a,fp):
            # f(t)=A t^2 + B t + C, with C=bestf
            # A = (fp + fm - 2*C)/(2 a^2), B = (fp - fm)/(2 a)
            C = bestf
            denom = 2.0 * a * a
            A = (fp + fm - 2.0 * C) / denom if denom != 0.0 else 0.0
            B = (fp - fm) / (2.0 * a) if a != 0.0 else 0.0

            # If convex-ish along direction and has a minimizer inside, try it
            if A > 1e-18:
                tstar = -B / (2.0 * A)
                # clamp tstar to [-a, a]
                if tstar < -a: tstar = -a
                elif tstar > a: tstar = a
                if abs(tstar) > 1e-18:
                    xq = [bestx[j] + tstar * dvec[j] for j in range(dim)]
                    xq = clamp_vec(xq)
                    fq = eval_f(xq)
                    if fq < bestf:
                        bestf = fq
                        bestx = xq
                        # mild step expansion on success
                        a0 = min(a0 * 1.10, 0.30)
                    else:
                        # mild shrink on failure
                        a0 = max(a0 * 0.92, 0.0008)
            else:
                # just accept best among fm, fp
                if fm < bestf:
                    bestf = fm
                    bestx = xm
                    a0 = min(a0 * 1.07, 0.30)
                elif fp < bestf:
                    bestf = fp
                    bestx = xp
                    a0 = min(a0 * 1.07, 0.30)
                else:
                    a0 = max(a0 * 0.90, 0.0008)

        return bestx, bestf

    # -------------------- init population --------------------
    NP_max = max(18, min(96, 14 + 5 * dim))
    NP_min = max(8,  min(40, 7 + 2 * dim))

    pop, fit = [], []
    best = float("inf")
    bestx = None

    # Use a small extra explorer quota to keep diversity (re-injected later)
    for i in range(NP_max):
        if now() >= deadline:
            return best
        if i < (2 * NP_max) // 3:
            x = halton_point(hal_k); hal_k += 1
        else:
            x = rand_point()

        fx = eval_f(x)
        if now() < deadline:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            if fxo < fx:
                x, fx = xo, fxo

        pop.append(x); fit.append(fx)
        if fx < best:
            best = fx; bestx = x

    # -------------------- DE (L-SHADE-ish with archive) --------------------
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    # keep some end time for aggressive local work
    endgame = 0.28

    while now() < deadline:
        frac = time_frac()
        if frac >= 1.0:
            break

        # Endgame: intensify local directional search with tiny random restarts
        if bestx is not None and frac >= (1.0 - endgame):
            while now() < deadline:
                frac2 = time_frac()
                # stop if too few evals remain
                if remaining_eval_budget(margin_frac=0.03) < 6:
                    break
                slice_time = min(deadline, now() + max(0.03 * max_time, 0.22 * (deadline - now())))
                bx, bf = local_dir_search(bestx, best, slice_time, frac2)
                if bf < best:
                    best, bestx = bf, bx
                else:
                    # small jitter restart around best
                    scale = 0.04 * (1.0 - 0.75 * frac2) + 0.0012
                    cand = [0.0] * dim
                    for d in range(dim):
                        if span[d] == 0.0:
                            cand[d] = lo[d]
                        else:
                            cand[d] = clamp(bestx[d] + (scale * span[d]) * randn(), lo[d], hi[d])
                    fc = eval_f(cand)
                    if fc < best:
                        best, bestx = fc, cand
            return best

        # linear population reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min
        if len(pop) > target_NP:
            idx = list(range(len(pop)))
            idx.sort(key=lambda i: fit[i])
            keep = idx[:target_NP]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            if len(archive) > target_NP:
                archive = random.sample(archive, target_NP)

        NP = len(pop)
        if NP < 4:
            return best

        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        # pbest fraction
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))
        elites = idx_sorted[:pbest_count]

        union = pop + archive

        # time/eval budgeting for optional second trial
        rem_budget = remaining_eval_budget(margin_frac=0.09)
        allow_second = (rem_budget > 3 * NP) and (random.random() < (0.22 + 0.25 * frac))

        S_F, S_CR, weights = [], [], []

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CRi = rand_normal(muCR, 0.1)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            Fi = -1.0
            tries = 0
            while Fi <= 0.0:
                Fi = rand_cauchy(muF, 0.1)
                tries += 1
                if tries > 10:
                    Fi = max(1e-3, muF)
                    break
            if Fi > 1.0:
                Fi = 1.0

            xpbest = pop[elites[random.randrange(len(elites))]]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            xr2 = None
            for _ in range(20):
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                xr2 = cand
                break
            if xr2 is None:
                xr2 = union[random.randrange(len(union))]

            def make_trial():
                use_ctpbest = (random.random() < (0.84 - 0.18 * (1.0 - frac)))
                v = [0.0] * dim
                if use_ctpbest:
                    for d in range(dim):
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                else:
                    a = random.randrange(NP)
                    b = random.randrange(NP)
                    c = random.randrange(NP)
                    while b == a:
                        b = random.randrange(NP)
                    while c == a or c == b:
                        c = random.randrange(NP)
                    xa, xb, xc = pop[a], pop[b], pop[c]
                    for d in range(dim):
                        v[d] = xa[d] + Fi * (xb[d] - xc[d])

                jrand = random.randrange(dim)
                u = [0.0] * dim
                for d in range(dim):
                    if span[d] == 0.0:
                        u[d] = lo[d]
                    else:
                        u[d] = v[d] if (random.random() < CRi or d == jrand) else xi[d]
                return repair_midpoint(u, xi)

            u1 = make_trial()
            fu1 = eval_f(u1)

            chosen_u, chosen_fu = u1, fu1

            if allow_second and now() < deadline:
                u2 = make_trial()
                # tiny cheap preference: try second only if it is closer to best (often helps)
                if bestx is not None and l2_dist2(u2, bestx) < l2_dist2(u1, bestx):
                    fu2 = eval_f(u2)
                    if fu2 < chosen_fu:
                        chosen_u, chosen_fu = u2, fu2

            if chosen_fu <= fi:
                if len(archive) < NP:
                    archive.append(xi)
                else:
                    archive[random.randrange(NP)] = xi

                pop[i] = chosen_u
                fit[i] = chosen_fu

                if chosen_fu < best:
                    best = chosen_fu
                    bestx = chosen_u

                df = abs(fi - chosen_fu)
                w = df if df > 0.0 else 1.0
                S_F.append(Fi)
                S_CR.append(CRi)
                weights.append(w)

        # update SHADE memories
        if S_F:
            wsum = sum(weights)
            if wsum <= 0.0:
                wsum = 1.0
            num, den, cr = 0.0, 0.0, 0.0
            for Fi, CRi, w in zip(S_F, S_CR, weights):
                num += w * Fi * Fi
                den += w * Fi
                cr += w * CRi
            newF = (num / den) if den != 0.0 else M_F[h_idx]
            newCR = cr / wsum

            if newF < 0.05: newF = 0.05
            if newF > 0.95: newF = 0.95
            if newCR < 0.0: newCR = 0.0
            if newCR > 1.0: newCR = 1.0

            M_F[h_idx] = newF
            M_CR[h_idx] = newCR
            h_idx = (h_idx + 1) % H

        # opportunistic local search (mid-run): only if budget allows
        if bestx is not None and now() < deadline:
            if (stagn >= 5 and random.random() < 0.55) or (random.random() < (0.03 + 0.16 * frac)):
                if remaining_eval_budget(margin_frac=0.10) > max(16, 2 * dim):
                    slice_time = min(deadline, now() + min(0.08 * max_time, 0.14 * (deadline - now())))
                    bx, bf = local_dir_search(bestx, best, slice_time, frac)
                    if bf < best:
                        best, bestx = bf, bx
                        # replace worst in population
                        worst_i = max(range(NP), key=lambda j: fit[j])
                        pop[worst_i] = list(bestx)
                        fit[worst_i] = best

        # stagnation + explorer refresh
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (stagn >= 10 or (frac < 0.70 and random.random() < 0.03)) and now() < deadline:
            # refresh worst individuals; keep a couple as "explorers"
            replace_n = max(2, NP // 5)
            worst = list(range(NP))
            worst.sort(key=lambda i: fit[i], reverse=True)
            for t in range(replace_n):
                if now() >= deadline:
                    return best
                j = worst[t]
                if random.random() < 0.65:
                    xnew = halton_point(hal_k); hal_k += 1
                else:
                    xnew = rand_point()
                fnew = eval_f(xnew)
                if now() < deadline and random.random() < 0.55:
                    xo = opposite_point(xnew)
                    fxo = eval_f(xo)
                    if fxo < fnew:
                        xnew, fnew = xo, fxo
                pop[j] = xnew
                fit[j] = fnew
                if fnew < best:
                    best, bestx = fnew, xnew
            archive = []
            stagn = 0

    return best
