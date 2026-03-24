import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved limited-time minimizer (self-contained, no numpy).

    Main upgrades vs your current best (#1):
      - Proper evaluation-budgeting: estimates eval cost online and chooses how many
        trial vectors to *actually evaluate* each generation.
      - Hybrid global search: L-SHADE/JADE-style DE (current-to-pbest/1 + archive)
        + periodic CMA-ES-like diagonal Gaussian sampling around the incumbent.
      - Better local search: adaptive pattern search (coordinate + random subspace)
        with 1+1 step-size control.
      - More robust stagnation handling: diversity injection based on spread, not
        just "gens since improvement".
      - Deadline-safe everywhere.

    Returns:
        best fitness (float)
    """

    # -------------------------- basic checks --------------------------
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
        if b < a:
            raise ValueError("Each bound must be (low, high) with high >= low")
        lo[i] = float(a)
        hi[i] = float(b)
        spans[i] = hi[i] - lo[i]

    if all(s == 0.0 for s in spans):
        x0 = [lo[i] for i in range(dim)]
        return float(func(x0))

    # -------------------------- helpers --------------------------
    def clamp(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def repair_midpoint(u, parent):
        # midpoint repair + clamp
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
                u = halton_value(k, bases[d])
                x[d] = lo[d] + u * spans[d]
        return x

    # Box-Muller normal
    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_normal(mu, sigma):
        return mu + sigma * randn()

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # time-aware evaluation wrapper (tracks eval time)
    eval_calls = 0
    avg_eval = None  # exponential moving average

    def eval_f(x):
        nonlocal eval_calls, avg_eval
        t1 = time.time()
        fx = float(func(x))
        t2 = time.time()
        dt = t2 - t1
        eval_calls += 1
        if avg_eval is None:
            avg_eval = max(1e-6, dt)
        else:
            # stable EMA
            a = 0.08
            avg_eval = (1.0 - a) * avg_eval + a * max(1e-6, dt)
        return fx

    def time_frac():
        return (time.time() - t0) / max(1e-12, float(max_time))

    # -------------------------- local search (stronger, still cheap) --------------------------
    def local_pattern_search(x, fx, deadline):
        """
        Adaptive coordinate + random subspace pattern search with 1+1 step-size control.
        Uses only a small number of evaluations; good as a late-stage polisher.
        """
        bestx = list(x)
        bestf = fx

        frac = time_frac()
        base = 0.18 * (1.0 - 0.85 * frac) + 0.004  # late -> tiny steps
        step = [base * spans[d] for d in range(dim)]
        # minimum step to allow movement
        for d in range(dim):
            if spans[d] > 0.0:
                m = 1e-14 * spans[d]
                if step[d] < m: step[d] = m

        # modest budget; scaled by dim but bounded
        budget = max(18, min(120, 6 * dim))
        success = 0
        attempts = 0

        while attempts < budget and time.time() < deadline:
            attempts += 1

            # choose move type: coordinate or random subspace
            if dim <= 8 or random.random() < 0.65:
                # coordinate move
                d = random.randrange(dim)
                if spans[d] == 0.0:
                    continue
                s = step[d]
                if s <= 0.0:
                    continue
                direction = -1.0 if random.random() < 0.5 else 1.0
                cand = list(bestx)
                cand[d] = clamp(cand[d] + direction * s, lo[d], hi[d])
            else:
                # random 2-3D subspace move
                k = 2 if dim <= 25 else 3
                coords = random.sample(range(dim), k)
                cand = list(bestx)
                for d in coords:
                    if spans[d] == 0.0:
                        cand[d] = lo[d]
                    else:
                        s = step[d]
                        cand[d] = clamp(cand[d] + (2.0 * random.random() - 1.0) * s, lo[d], hi[d])

            if cand == bestx:
                continue
            fc = eval_f(cand)

            if fc < bestf:
                bestf = fc
                bestx = cand
                success += 1
                # expand steps slightly where possible
                if random.random() < 0.8:
                    for d in range(dim):
                        if spans[d] > 0.0:
                            step[d] = min(step[d] * 1.15, 0.5 * spans[d])
            else:
                # occasional "opposite" try around best for rugged landscapes
                if random.random() < 0.12 and time.time() < deadline:
                    oc = opposite_point(cand)
                    foc = eval_f(oc)
                    if foc < bestf:
                        bestf = foc
                        bestx = oc
                        success += 1

            # 1+1 step size control: if failing a lot, shrink
            if attempts % 10 == 0:
                rate = success / float(attempts)
                if rate < 0.12:
                    for d in range(dim):
                        if spans[d] > 0.0:
                            step[d] = max(step[d] * 0.65, 1e-14 * spans[d])
                elif rate > 0.30:
                    for d in range(dim):
                        if spans[d] > 0.0:
                            step[d] = min(step[d] * 1.10, 0.5 * spans[d])

        return bestx, bestf

    # -------------------------- initialization --------------------------
    random.seed()

    # pop sizes (time-bounded, reasonably small)
    NP_max = max(18, min(96, 14 + 5 * dim))
    NP_min = max(8, min(40, 7 + 2 * dim))

    bases = first_primes(dim)

    pop, fit = [], []
    best = float("inf")
    bestx = None
    k_hal = 1

    # init: halton + random + opposition (evaluate both if time allows)
    for i in range(NP_max):
        if time.time() >= deadline:
            return best
        if i < (NP_max * 2) // 3:
            x = halton_point(k_hal, bases)
            k_hal += 1
        else:
            x = rand_point()

        fx = eval_f(x)

        if time.time() < deadline:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            if fxo < fx:
                x, fx = xo, fxo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            bestx = x

    # -------------------------- SHADE/JADE-style DE core --------------------------
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    # diagonal gaussian sampler state (CMA-ES-like, but very light)
    # sigma per dimension as fraction of span
    sigma = [0.25 * spans[d] if spans[d] > 0.0 else 0.0 for d in range(dim)]
    sig_min = [1e-14 * spans[d] for d in range(dim)]
    sig_max = [0.50 * spans[d] for d in range(dim)]

    while time.time() < deadline:
        frac = time_frac()
        if frac >= 1.0:
            break

        # reduce population size linearly
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        if len(pop) > target_NP:
            idx = list(range(len(pop)))
            idx.sort(key=lambda i: fit[i])
            idx_keep = idx[:target_NP]
            pop = [pop[i] for i in idx_keep]
            fit = [fit[i] for i in idx_keep]
            if len(archive) > target_NP:
                archive = random.sample(archive, target_NP)

        NP = len(pop)
        if NP < 4:
            return best

        # dynamic pbest fraction
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))

        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        # decide evaluation intensity based on remaining time and eval cost
        now = time.time()
        rem = max(0.0, deadline - now)
        est = avg_eval if avg_eval is not None else 1e-3
        # keep some margin for refinement and overhead
        safe_rem = max(0.0, rem - 0.03 * max_time)
        eval_budget = int(safe_rem / max(est, 1e-6))
        # candidates per individual to pre-generate; evaluate only 1 (sometimes 2) best-scored
        n_trials = 2 if (dim <= 20) else 3
        # allow occasional second evaluation if we can afford it
        allow_second = (eval_budget > 3 * NP) and (random.random() < 0.35)

        S_F, S_CR, weights = [], [], []

        union = pop + archive

        # --- DE generation ---
        for i in range(NP):
            if time.time() >= deadline:
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

            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            xr2 = None
            for _ in range(25):
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                xr2 = cand
                break
            if xr2 is None:
                xr2 = union[random.randrange(len(union))]

            # generate a few trials, rank by cheap score, then evaluate top1 (and maybe top2)
            trial_list = []
            best_score = None

            for _t in range(n_trials):
                use_ctpbest = (random.random() < (0.78 - 0.25 * (1.0 - frac)))  # favors ctpbest most of time
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

                # cheap score: closeness to best + mild diversity term (avoid full collapse)
                # lower is better
                db = l2_dist2(u, bestx) if bestx is not None else 0.0
                dpar = l2_dist2(u, xi)
                score = db + 0.02 * dpar
                if best_score is None or score < best_score:
                    best_score = score
                trial_list.append((score, u))

            trial_list.sort(key=lambda z: z[0])
            u1 = trial_list[0][1]
            fu1 = eval_f(u1)

            chosen_u = u1
            chosen_fu = fu1

            if allow_second and time.time() < deadline and len(trial_list) > 1:
                # evaluate second best-scored if it might beat parent (late-stage hedge)
                u2 = trial_list[1][1]
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

        # --- diagonal gaussian sampling around best (very effective on smooth problems) ---
        # Trigger more often late, and when DE stagnates
        if bestx is not None and time.time() < deadline:
            do_gauss = (random.random() < (0.06 + 0.30 * frac)) or (stagn >= 6 and random.random() < 0.65)
            if do_gauss:
                # a few samples; count depends on time left
                m = 3 if dim <= 10 else (4 if dim <= 30 else 5)
                for _ in range(m):
                    if time.time() >= deadline:
                        break
                    cand = [0.0] * dim
                    for d in range(dim):
                        if spans[d] == 0.0:
                            cand[d] = lo[d]
                        else:
                            cand[d] = clamp(bestx[d] + sigma[d] * randn(), lo[d], hi[d])
                    fc = eval_f(cand)
                    if fc < best:
                        best = fc
                        bestx = cand
                        # success -> slightly expand (but bounded)
                        for d in range(dim):
                            if spans[d] > 0.0:
                                sigma[d] = min(sigma[d] * 1.10, sig_max[d])
                    else:
                        # failure -> gently shrink
                        for d in range(dim):
                            if spans[d] > 0.0:
                                sigma[d] = max(sigma[d] * 0.95, sig_min[d])

        # --- local refinement (polish near end / after improvements) ---
        if bestx is not None and time.time() < deadline:
            p_ref = 0.04 + 0.42 * frac
            if random.random() < p_ref:
                rx, rf = local_pattern_search(bestx, best, deadline)
                if rf < best:
                    best, bestx = rf, rx
                    # inject into population by replacing worst
                    worst_i = 0
                    worst_f = fit[0]
                    for j in range(1, NP):
                        if fit[j] > worst_f:
                            worst_f = fit[j]
                            worst_i = j
                    pop[worst_i] = list(bestx)
                    fit[worst_i] = best

        # --- stagnation & diversity injection based on spread ---
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (stagn >= 10 or (frac < 0.70 and random.random() < 0.03)) and time.time() < deadline:
            # compute a cheap spread proxy: mean dist to best for a few points
            # if too clustered and not improving, inject diversity
            sample_n = min(NP, max(6, NP // 2))
            idxs = random.sample(range(NP), sample_n)
            md = 0.0
            for j in idxs:
                md += l2_dist2(pop[j], bestx)
            md /= float(sample_n)
            # threshold scales with spans
            span_norm = 0.0
            for s in spans:
                span_norm += s * s
            # if clustered or stagnating hard -> refresh worst
            if md < 1e-6 * (span_norm + 1e-30) or stagn >= 12:
                replace_n = max(2, NP // 5)
                worst = list(range(NP))
                worst.sort(key=lambda i: fit[i], reverse=True)
                for t in range(replace_n):
                    if time.time() >= deadline:
                        return best
                    j = worst[t]
                    xnew = halton_point(k_hal, bases) if random.random() < 0.60 else rand_point()
                    k_hal += 1 if xnew is not None else 0
                    fnew = eval_f(xnew)
                    if time.time() < deadline and random.random() < 0.50:
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
