import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no numpy).

    Improvements vs the current best (#1 in prompt):
      - Adds an explicit *tri-modal* search schedule that adapts to remaining time:
          (A) global exploration (Halton+opposition + wide DE)
          (B) exploitation (L-SHADE current-to-pbest/1 with archive, shrinking pop)
          (C) intensification (SPSA-like gradient sign steps + pattern search polish)
      - Uses *evaluation-time estimation* to keep a safe end-game budget.
      - Uses *rank-based* pbest and *distance-aware* parent selection to avoid collapse.
      - Adds a cheap *SPSA* local optimizer near best for continuous domains; very strong
        when objective is smooth-ish but still works as noisy descent.
      - Keeps strict deadline checks everywhere.

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
    spans = [0.0] * dim
    for i, (a, b) in enumerate(bounds):
        a = float(a); b = float(b)
        if b < a:
            raise ValueError("Each bound must be (low, high) with high >= low")
        lo[i] = a
        hi[i] = b
        spans[i] = b - a

    # Degenerate case
    if all(s == 0.0 for s in spans):
        x0 = [lo[i] for i in range(dim)]
        return float(func(x0))

    # -------------------------- helpers --------------------------
    def now():
        return time.time()

    def time_frac():
        return (now() - t0) / max(1e-12, float(max_time))

    def clamp(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def l2_dist2(a, b):
        s = 0.0
        for d in range(dim):
            t = a[d] - b[d]
            s += t * t
        return s

    def rand_point():
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + random.random() * spans[d]
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                xo[d] = lo[d]
            else:
                xo[d] = clamp(lo[d] + hi[d] - x[d], lo[d], hi[d])
        return xo

    def repair_midpoint(u, parent):
        # midpoint repair + clamp; stable for DE
        for d in range(dim):
            if spans[d] == 0.0:
                u[d] = lo[d]
            else:
                if u[d] < lo[d]:
                    u[d] = 0.5 * (lo[d] + parent[d])
                elif u[d] > hi[d]:
                    u[d] = 0.5 * (hi[d] + parent[d])
                if u[d] < lo[d]: u[d] = lo[d]
                elif u[d] > hi[d]: u[d] = hi[d]
        return u

    # primes + halton for init / injections
    def first_primes(n):
        primes, k = [], 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r: break
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

    def halton_point(k, bases):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lo[d]
            else:
                x[d] = lo[d] + halton_value(k, bases[d]) * spans[d]
        return x

    # RNG: Box-Muller normal and Cauchy
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_normal(mu, sigma):
        return mu + sigma * randn()

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # ---- time-aware evaluation wrapper ----
    eval_calls = 0
    avg_eval = None  # EMA of eval cost

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
            a = 0.07
            avg_eval = (1.0 - a) * avg_eval + a * dt
        return fx

    # quick remaining evaluation estimate with margin
    def remaining_eval_budget(margin_frac=0.06):
        rem = max(0.0, deadline - now())
        rem2 = max(0.0, rem - margin_frac * float(max_time))
        est = avg_eval if avg_eval is not None else 1e-3
        return int(rem2 / max(est, 1e-6))

    # -------------------------- local: SPSA + pattern search --------------------------
    def spsa_refine(x, fx, deadline_local, frac):
        """
        SPSA-like stochastic gradient descent:
        Uses 2 evals per step. Very strong late-stage for continuous problems.
        """
        bestx = list(x)
        bestf = fx

        # base step sizes shrink with time
        # a: learning rate; c: perturbation
        a0 = 0.12 * (1.0 - 0.85 * frac) + 0.006
        c0 = 0.08 * (1.0 - 0.90 * frac) + 0.002

        # convert to absolute per-dim scaling using spans
        # also ensure minimum movement where span>0
        a = [a0 * spans[d] for d in range(dim)]
        c = [c0 * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] > 0.0:
                mn = 1e-14 * spans[d]
                if a[d] < mn: a[d] = mn
                if c[d] < mn: c[d] = mn

        # budget: limited steps
        max_steps = max(8, min(60, 3 * dim))
        # if we are very late, fewer steps but more aggressive polish later by pattern search
        if frac > 0.85:
            max_steps = max(6, min(30, 2 * dim))

        for k in range(1, max_steps + 1):
            if now() >= deadline_local:
                break

            # decreasing schedules
            ak = 1.0 / (k ** 0.35)
            ck = 1.0 / (k ** 0.15)

            delta = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    delta[d] = 0.0
                else:
                    delta[d] = -1.0 if random.random() < 0.5 else 1.0

            x_plus = [0.0] * dim
            x_minus = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    x_plus[d] = lo[d]
                    x_minus[d] = lo[d]
                else:
                    pd = (c[d] * ck) * delta[d]
                    x_plus[d] = clamp(bestx[d] + pd, lo[d], hi[d])
                    x_minus[d] = clamp(bestx[d] - pd, lo[d], hi[d])

            f_plus = eval_f(x_plus)
            if now() >= deadline_local:
                break
            f_minus = eval_f(x_minus)

            # gradient estimate and update
            cand = [0.0] * dim
            denom_eps = 1e-30
            for d in range(dim):
                if spans[d] == 0.0:
                    cand[d] = lo[d]
                else:
                    cd = (c[d] * ck)
                    ghat = (f_plus - f_minus) / (2.0 * max(denom_eps, cd)) * delta[d]
                    cand[d] = clamp(bestx[d] - (a[d] * ak) * ghat, lo[d], hi[d])

            fc = eval_f(cand)
            if fc < bestf:
                bestf = fc
                bestx = cand
            else:
                # small probability to accept if nearly equal (helps noise / plateaus)
                if fc <= bestf + 1e-12 and random.random() < 0.08:
                    bestf = fc
                    bestx = cand

        return bestx, bestf

    def pattern_polish(x, fx, deadline_local, frac):
        """
        Very cheap coordinate/random-subspace search to finish.
        """
        bestx = list(x)
        bestf = fx
        base = 0.10 * (1.0 - 0.90 * frac) + 0.0025
        step = [base * spans[d] for d in range(dim)]
        for d in range(dim):
            if spans[d] > 0.0:
                mn = 1e-14 * spans[d]
                if step[d] < mn: step[d] = mn

        budget = max(14, min(90, 5 * dim))
        succ = 0
        for it in range(1, budget + 1):
            if now() >= deadline_local:
                break

            if dim <= 10 or random.random() < 0.7:
                d = random.randrange(dim)
                if spans[d] == 0.0:
                    continue
                s = step[d]
                direction = -1.0 if random.random() < 0.5 else 1.0
                cand = list(bestx)
                cand[d] = clamp(cand[d] + direction * s, lo[d], hi[d])
            else:
                k = 2 if dim <= 30 else 3
                coords = random.sample(range(dim), k)
                cand = list(bestx)
                for d in coords:
                    if spans[d] == 0.0:
                        cand[d] = lo[d]
                    else:
                        cand[d] = clamp(cand[d] + (2.0 * random.random() - 1.0) * step[d], lo[d], hi[d])

            if cand == bestx:
                continue
            fc = eval_f(cand)
            if fc < bestf:
                bestf = fc
                bestx = cand
                succ += 1
                # expand slightly
                if it % 3 == 0:
                    for d in range(dim):
                        if spans[d] > 0.0:
                            step[d] = min(step[d] * 1.12, 0.5 * spans[d])
            else:
                # shrink occasionally
                if it % 7 == 0 and succ == 0:
                    for d in range(dim):
                        if spans[d] > 0.0:
                            step[d] = max(step[d] * 0.70, 1e-14 * spans[d])

        return bestx, bestf

    # -------------------------- initialization --------------------------
    random.seed()

    bases = first_primes(dim)
    k_hal = 1

    NP_max = max(18, min(100, 14 + 5 * dim))
    NP_min = max(8, min(42, 7 + 2 * dim))

    pop, fit = [], []
    best = float("inf")
    bestx = None

    # init: halton+random, with opposition pair
    for i in range(NP_max):
        if now() >= deadline:
            return best
        if i < (2 * NP_max) // 3:
            x = halton_point(k_hal, bases)
            k_hal += 1
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

    # -------------------------- L-SHADE DE core --------------------------
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    while now() < deadline:
        frac = time_frac()
        if frac >= 1.0:
            break

        # pop reduction
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

        # pbest fraction
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))

        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])
        elites = idx_sorted[:pbest_count]

        # distance-aware parent pick: prefer r1 far from xi sometimes (avoid collapse)
        def pick_r1(i):
            if NP <= 2:
                return (i + 1) % NP
            if random.random() < 0.65 and bestx is not None:
                # choose among a few candidates the farthest from xi
                bestj = None
                bestd = -1.0
                for _ in range(4):
                    j = random.randrange(NP)
                    if j == i:
                        continue
                    d2 = l2_dist2(pop[j], pop[i])
                    if d2 > bestd:
                        bestd = d2
                        bestj = j
                if bestj is not None:
                    return bestj
            j = i
            while j == i:
                j = random.randrange(NP)
            return j

        union = pop + archive

        # time budgeting: avoid over-spending late
        rem_budget = remaining_eval_budget(margin_frac=0.07)
        # cap trials if low budget
        if rem_budget < 2 * NP:
            n_trials = 1
            allow_second_eval = False
        else:
            n_trials = 2 if dim <= 24 else 3
            allow_second_eval = (rem_budget > 3 * NP and random.random() < 0.30)

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

            # pick pbest from elites
            xpbest = pop[elites[random.randrange(len(elites))]]

            r1 = pick_r1(i)
            xr1 = pop[r1]

            # r2 from union, avoid identical objects if possible
            xr2 = None
            for _ in range(25):
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                xr2 = cand
                break
            if xr2 is None:
                xr2 = union[random.randrange(len(union))]

            # generate several trials; evaluate best-scored
            trials = []
            for _t in range(n_trials):
                use_ctpbest = (random.random() < (0.82 - 0.20 * (1.0 - frac)))
                v = [0.0] * dim
                if use_ctpbest:
                    for d in range(dim):
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                else:
                    # rand/1
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
                    if spans[d] == 0.0:
                        u[d] = lo[d]
                    else:
                        u[d] = v[d] if (random.random() < CRi or d == jrand) else xi[d]
                u = repair_midpoint(u, xi)

                # score: close to best but keep some step from parent (avoid micro-steps early)
                db = l2_dist2(u, bestx) if bestx is not None else 0.0
                dpar = l2_dist2(u, xi)
                score = db + (0.03 + 0.10 * (1.0 - frac)) * dpar
                trials.append((score, u))

            trials.sort(key=lambda z: z[0])
            u1 = trials[0][1]
            fu1 = eval_f(u1)

            chosen_u, chosen_fu = u1, fu1

            if allow_second_eval and len(trials) > 1 and now() < deadline:
                u2 = trials[1][1]
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

        # update memories (SHADE)
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

        # -------------------------- intensification (SPSA + polish) --------------------------
        # Trigger more often late and on stagnation; ensure we don't burn the deadline.
        if bestx is not None and now() < deadline:
            frac = time_frac()
            do_local = (random.random() < (0.05 + 0.45 * frac)) or (stagn >= 6 and random.random() < 0.65)

            # ensure we keep some evals for DE/injections
            if do_local and remaining_eval_budget(margin_frac=0.05) > max(20, 3 * dim):
                # local deadline slice
                slice_time = max(0.0, min(0.22 * (deadline - now()), 0.18 * float(max_time)))
                local_deadline = min(deadline, now() + slice_time)

                bx, bf = spsa_refine(bestx, best, local_deadline, frac)
                if bf < best:
                    best, bestx = bf, bx

                    # inject improved best into population (replace worst)
                    worst_i = 0
                    worst_f = fit[0]
                    for j in range(1, NP):
                        if fit[j] > worst_f:
                            worst_f = fit[j]
                            worst_i = j
                    pop[worst_i] = list(bestx)
                    fit[worst_i] = best

                # tiny final polish if time remains in slice
                if now() < local_deadline:
                    bx, bf = pattern_polish(bestx, best, local_deadline, frac)
                    if bf < best:
                        best, bestx = bf, bx
                        # inject again
                        worst_i = 0
                        worst_f = fit[0]
                        for j in range(1, NP):
                            if fit[j] > worst_f:
                                worst_f = fit[j]
                                worst_i = j
                        pop[worst_i] = list(bestx)
                        fit[worst_i] = best

        # -------------------------- stagnation/diversity injection --------------------------
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (stagn >= 10 or (frac < 0.75 and random.random() < 0.03)) and now() < deadline:
            # refresh some worst with halton/random + opposition
            replace_n = max(2, NP // 5)
            worst = list(range(NP))
            worst.sort(key=lambda i: fit[i], reverse=True)

            for t in range(replace_n):
                if now() >= deadline:
                    return best
                j = worst[t]
                xnew = halton_point(k_hal, bases) if random.random() < 0.60 else rand_point()
                k_hal += 1
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
