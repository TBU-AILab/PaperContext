import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved limited-time minimizer (stdlib only).

    Core idea (upgrade vs provided #1):
      - Keep a strong L-SHADE/JADE-style DE global search (current-to-pbest/1 + archive).
      - Add *much stronger, more consistent* exploitation with a bounded
        trust-region local search around the incumbent:
            * random-orthant coordinate moves (fast)
            * random-direction parabolic (quadratic) line search (powerful on smooth parts)
            * adaptive step-size (1/5 success rule style)
      - Add a cheap multi-start policy near the end: several short local bursts
        started from best and a few elite points (not only best), plus tiny jitters.
      - More disciplined time budgeting: we continuously estimate eval cost and
        reserve an end slice for local refinement.

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
        lo[i] = a
        hi[i] = b
        s = b - a
        span[i] = s
        span2_sum += s * s

    if all(s == 0.0 for s in span):
        x0 = [lo[i] for i in range(dim)]
        return float(func(x0))

    random.seed()

    # ---------------- utilities ----------------
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
        # Stable DE repair: midpoint then clamp
        for d in range(dim):
            if span[d] == 0.0:
                u[d] = lo[d]
            else:
                if u[d] < lo[d]:
                    u[d] = 0.5 * (lo[d] + parent[d])
                elif u[d] > hi[d]:
                    u[d] = 0.5 * (hi[d] + parent[d])
                if u[d] < lo[d]:
                    u[d] = lo[d]
                elif u[d] > hi[d]:
                    u[d] = hi[d]
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
            x[d] = lo[d] if span[d] == 0.0 else (lo[d] + random.random() * span[d])
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for d in range(dim):
            if span[d] == 0.0:
                xo[d] = lo[d]
            else:
                xo[d] = clamp(lo[d] + hi[d] - x[d], lo[d], hi[d])
        return xo

    # low discrepancy init / injection: Halton
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

    # RNG helpers
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_normal(mu, sig):
        return mu + sig * randn()

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # ---------------- time-aware evaluation ----------------
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

    def remaining_eval_budget(margin_frac=0.08):
        rem = max(0.0, deadline - now())
        rem2 = max(0.0, rem - margin_frac * float(max_time))
        est = avg_eval if avg_eval is not None else 1e-3
        return int(rem2 / max(est, 1e-6))

    # ---------------- Local trust-region search (key improvement) ----------------
    def local_trust_search(x0, f0, time_limit, frac, max_evals_hint):
        """
        Bounded local search around incumbent with adaptive trust region.
        Mixes coordinate moves + random-direction quadratic line search.
        """
        x = list(x0)
        fx = f0

        # Trust radius as fraction of RMS span
        rms_span = math.sqrt(span2_sum / float(max(1, dim)))
        # start larger earlier, smaller later
        rad = (0.20 * (1.0 - 0.90 * frac) + 0.0025) * rms_span
        rad_min = 1e-14 * max(1.0, rms_span)
        rad_max = 0.50 * max(1e-12, rms_span)

        # evaluation budget for this local call
        budget = max(10, min(int(max_evals_hint), 180))
        used = 0
        succ = 0
        att = 0

        def try_point(xcand):
            nonlocal x, fx, used, succ
            if now() >= time_limit:
                return False
            fc = eval_f(xcand)
            used += 1
            if fc < fx:
                x, fx = xcand, fc
                succ += 1
                return True
            return False

        while used < budget and now() < time_limit:
            att += 1

            # Choose a move type: coord (cheap) vs random-dir quadratic (strong)
            use_quad = (random.random() < (0.35 + 0.30 * frac)) and (used + 3 <= budget)

            if not use_quad:
                # Coordinate / small subspace move (1-3 coords)
                k = 1 if dim <= 10 else (2 if random.random() < 0.70 else 3)
                coords = random.sample(range(dim), min(k, dim))

                cand = list(x)
                moved = False
                for d in coords:
                    if span[d] == 0.0:
                        cand[d] = lo[d]
                        continue
                    step = rad
                    # scale by per-d span to avoid wasting on tiny dimensions
                    # (use fraction of span)
                    step = min(step, 0.35 * span[d])
                    if step <= 0.0:
                        continue
                    direction = -1.0 if random.random() < 0.5 else 1.0
                    cand[d] = clamp(cand[d] + direction * step, lo[d], hi[d])
                    moved = True

                if moved:
                    improved = try_point(cand)
                    # mild adaptation
                    if improved:
                        rad = min(rad * 1.10, rad_max)
                    else:
                        rad = max(rad * 0.92, rad_min)
                continue

            # Random-direction quadratic line search
            # direction
            dvec = [0.0] * dim
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
            invn = 1.0 / math.sqrt(norm2)
            for j in range(dim):
                dvec[j] *= invn

            # feasible alpha bounds
            max_pos = float("inf")
            max_neg = float("inf")
            for j in range(dim):
                dj = dvec[j]
                if span[j] == 0.0 or dj == 0.0:
                    continue
                if dj > 0:
                    max_pos = min(max_pos, (hi[j] - x[j]) / dj)
                    max_neg = min(max_neg, (x[j] - lo[j]) / dj)
                else:
                    max_pos = min(max_pos, (x[j] - lo[j]) / (-dj))
                    max_neg = min(max_neg, (hi[j] - x[j]) / (-dj))

            if max_pos <= 0.0 and max_neg <= 0.0:
                continue

            a = rad
            if max_pos != float("inf"):
                a = min(a, 0.85 * max_pos)
            if max_neg != float("inf"):
                a = min(a, 0.85 * max_neg)
            a = max(1e-16, a)

            xm = clamp_vec([x[j] - a * dvec[j] for j in range(dim)])
            xp = clamp_vec([x[j] + a * dvec[j] for j in range(dim)])

            fm = eval_f(xm); used += 1
            if now() >= time_limit or used >= budget:
                break
            fp = eval_f(xp); used += 1
            if now() >= time_limit or used >= budget:
                break

            # quadratic fit through (-a,fm), (0,fx), (+a,fp)
            C = fx
            denom = 2.0 * a * a
            A = (fp + fm - 2.0 * C) / denom if denom != 0.0 else 0.0
            B = (fp - fm) / (2.0 * a) if a != 0.0 else 0.0

            improved = False
            if A > 1e-18:
                tstar = -B / (2.0 * A)
                if tstar < -a: tstar = -a
                elif tstar > a: tstar = a
                xq = clamp_vec([x[j] + tstar * dvec[j] for j in range(dim)])
                fq = eval_f(xq); used += 1
                if fq < fx:
                    x, fx = xq, fq
                    improved = True
            else:
                # accept best of endpoints
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                elif fp < fx:
                    x, fx = xp, fp
                    improved = True

            if improved:
                rad = min(rad * 1.18, rad_max)
            else:
                rad = max(rad * 0.85, rad_min)

            # occasional stronger shrink if repeatedly failing
            if att % 12 == 0:
                rate = succ / float(max(1, att))
                if rate < 0.12:
                    rad = max(rad * 0.70, rad_min)

        return x, fx

    # ---------------- initialization ----------------
    NP_max = max(20, min(110, 16 + 5 * dim))
    NP_min = max(8,  min(45, 7 + 2 * dim))

    pop, fit = [], []
    best = float("inf")
    bestx = None

    # init: Halton + random + opposition
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

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            bestx = x

    # ---------------- DE (L-SHADE/JADE-ish) ----------------
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    # reserve end portion for local multi-start refinement
    endgame = 0.26

    while now() < deadline:
        frac = time_frac()
        if frac >= 1.0:
            break

        # endgame: multi-start local search from best + a few elites
        if bestx is not None and frac >= (1.0 - endgame):
            # pick a few elite seeds
            idx = list(range(len(pop)))
            idx.sort(key=lambda i: fit[i])
            elite_count = min(len(pop), max(3, len(pop) // 6))
            elites = [pop[i] for i in idx[:elite_count]]
            # cycle through elites + best with jitter
            e = 0
            while now() < deadline and remaining_eval_budget(margin_frac=0.03) > 6:
                seed = bestx if (e % 3 == 0) else elites[e % elite_count]
                e += 1
                # tiny jitter to escape flat local minima
                jitter = (0.020 * (1.0 - 0.75 * frac) + 0.0010)
                x0 = [0.0] * dim
                for d in range(dim):
                    if span[d] == 0.0:
                        x0[d] = lo[d]
                    else:
                        x0[d] = clamp(seed[d] + jitter * span[d] * randn(), lo[d], hi[d])
                f0 = eval_f(x0)
                if f0 < best:
                    best, bestx = f0, x0

                slice_time = min(deadline, now() + max(0.03 * max_time, 0.22 * (deadline - now())))
                bx, bf = local_trust_search(x0, f0, slice_time, time_frac(),
                                            max_evals_hint=max(25, 4 * dim + 25))
                if bf < best:
                    best, bestx = bf, bx
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

        # pbest fraction increases a bit over time
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))
        elites_idx = idx_sorted[:pbest_count]

        union = pop + archive

        rem_budget = remaining_eval_budget(margin_frac=0.10)
        allow_second = (rem_budget > 3 * NP) and (random.random() < (0.20 + 0.20 * frac))

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

            xpbest = pop[elites_idx[random.randrange(len(elites_idx))]]

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
                use_ctpbest = (random.random() < (0.85 - 0.18 * (1.0 - frac)))
                if use_ctpbest:
                    v = [xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d]) for d in range(dim)]
                else:
                    a = random.randrange(NP)
                    b = random.randrange(NP)
                    c = random.randrange(NP)
                    while b == a:
                        b = random.randrange(NP)
                    while c == a or c == b:
                        c = random.randrange(NP)
                    xa, xb, xc = pop[a], pop[b], pop[c]
                    v = [xa[d] + Fi * (xb[d] - xc[d]) for d in range(dim)]

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
                # evaluate second only if it is "more promising" (closer to incumbent best)
                if bestx is None or l2_dist2(u2, bestx) <= l2_dist2(u1, bestx):
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
            wsum = sum(weights) or 1.0
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

        # opportunistic local trust search mid-run (more consistent than before)
        if bestx is not None and now() < deadline:
            do_local = (stagn >= 5 and random.random() < 0.60) or (random.random() < (0.02 + 0.14 * frac))
            if do_local and remaining_eval_budget(margin_frac=0.12) > max(20, 3 * dim):
                slice_time = min(deadline, now() + min(0.08 * max_time, 0.14 * (deadline - now())))
                bx, bf = local_trust_search(bestx, best, slice_time, frac,
                                            max_evals_hint=max(25, 3 * dim + 35))
                if bf < best:
                    best, bestx = bf, bx
                    # replace worst with polished best
                    worst_i = max(range(NP), key=lambda j: fit[j])
                    pop[worst_i] = list(bestx)
                    fit[worst_i] = best

        # stagnation + diversity injection
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (stagn >= 10 or (frac < 0.70 and random.random() < 0.03)) and now() < deadline:
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
