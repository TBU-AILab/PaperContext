import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (self-contained, no numpy).

    Improvements over the provided best (JADE-like DE):
      - L-SHADE-style DE (linear pop-size reduction + success-history adaptation).
      - Two mutation strategies mixed: current-to-pbest/1 (with archive) and rand/1.
      - "Cheap surrogate screening": generate multiple trials, evaluate only the most
        promising by a distance-based score (reduces wasted expensive evaluations).
      - Trust-region local refinement around the best with adaptive radius.
      - Robust restarts for stagnation + bound handling with midpoint repair.
      - Strict deadline checks.

    Returns:
        best fitness (float)
    """

    # -------------------------- helpers --------------------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(x))

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
            lo, hi = bounds[d]
            s = spans[d]
            if s == 0.0:
                x[d] = lo
            else:
                u = halton_value(k, bases[d])
                x[d] = lo + u * s
        return x

    def random_point():
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            s = spans[d]
            x[d] = lo if s == 0.0 else (lo + random.random() * s)
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                xo[d] = lo
            else:
                xo[d] = clamp(lo + hi - x[d], lo, hi)
        return xo

    def rand_normal(mu, sigma):
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def rand_cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def repair(u, x_parent):
        # midpoint repair then clamp; stable and fast
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                u[d] = lo
            else:
                if u[d] < lo:
                    u[d] = 0.5 * (lo + x_parent[d])
                elif u[d] > hi:
                    u[d] = 0.5 * (hi + x_parent[d])
                u[d] = clamp(u[d], lo, hi)
        return u

    def l2_dist2(a, b):
        s = 0.0
        for d in range(dim):
            t = a[d] - b[d]
            s += t * t
        return s

    def centroid(pop):
        c = [0.0] * dim
        inv = 1.0 / max(1, len(pop))
        for x in pop:
            for d in range(dim):
                c[d] += x[d]
        for d in range(dim):
            c[d] *= inv
        return c

    def local_trust_refine(x, fx, deadline, frac):
        # Trust-region-ish random coordinate perturbations around best.
        # Radius shrinks as time passes and when not improving.
        bestx = list(x)
        bestf = fx

        # base radius fraction: larger early, small late
        rad0 = (0.22 * (1.0 - 0.85 * frac) + 0.006)
        rad = [rad0 * s for s in spans]

        # attempts budget
        tries = max(20, min(140, 8 * dim))
        noimp = 0

        for _ in range(tries):
            if time.time() >= deadline:
                break

            # pick a few coordinates to change
            k = 1 if dim <= 5 else (2 if dim <= 25 else 3)
            coords = random.sample(range(dim), k)

            cand = list(bestx)
            for d in coords:
                if spans[d] == 0.0:
                    cand[d] = bounds[d][0]
                    continue
                lo, hi = bounds[d]
                # symmetric perturb; occasional larger jump
                r = rad[d]
                if r <= 0.0:
                    continue
                mult = 1.0 if random.random() < 0.85 else (0.3 + 2.0 * random.random())
                delta = (2.0 * random.random() - 1.0) * r * mult
                cand[d] = clamp(cand[d] + delta, lo, hi)

            f = eval_f(cand)
            if f < bestf:
                bestf = f
                bestx = cand
                noimp = 0
                # slight expand where we moved
                for d in coords:
                    if spans[d] > 0.0:
                        rad[d] = min(rad[d] * 1.15, 0.5 * spans[d])
            else:
                noimp += 1
                if noimp % 7 == 0:
                    # shrink region if stuck
                    for d in coords:
                        if spans[d] > 0.0:
                            rad[d] = max(rad[d] * 0.65, 1e-14 * spans[d])

        return bestx, bestf

    # -------------------------- main --------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    spans = []
    for lo, hi in bounds:
        if hi < lo:
            raise ValueError("Each bound must be (low, high) with high >= low")
        spans.append(hi - lo)

    if all(s == 0.0 for s in spans):
        x0 = [bounds[d][0] for d in range(dim)]
        return eval_f(x0)

    random.seed()

    # L-SHADE-ish population schedule
    NP_max = max(18, min(90, 12 + 5 * dim))
    NP_min = max(8, min(30, 6 + 2 * dim))

    bases = first_primes(dim)

    # ---- initialization: Halton + random + opposition (keep best of pair) ----
    pop, fit = [], []
    best = float("inf")
    bestx = None
    k_hal = 1

    for i in range(NP_max):
        if time.time() >= deadline:
            return best
        if i < (NP_max * 2) // 3:
            x = halton_point(k_hal, bases)
            k_hal += 1
        else:
            x = random_point()

        fx = eval_f(x)
        xo = opposite_point(x)
        fxo = eval_f(xo) if time.time() < deadline else float("inf")
        if fxo < fx:
            x, fx = xo, fxo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, bestx = fx, x

    # ---- success-history adaptation (simplified SHADE) ----
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    h_idx = 0
    archive = []

    last_best = best
    stagn = 0

    # precompute a stable "diversity reference" point sometimes
    cen = centroid(pop)

    while time.time() < deadline:
        elapsed = time.time() - t0
        frac = elapsed / max(1e-12, float(max_time))
        if frac >= 1.0:
            break

        # linear pop reduction
        target_NP = int(round(NP_max - (NP_max - NP_min) * frac))
        if target_NP < NP_min:
            target_NP = NP_min

        # shrink population by removing worst
        if len(pop) > target_NP:
            idx = list(range(len(pop)))
            idx.sort(key=lambda i: fit[i])  # ascending
            idx_keep = idx[:target_NP]
            pop = [pop[i] for i in idx_keep]
            fit = [fit[i] for i in idx_keep]
            # also shrink archive
            if len(archive) > target_NP:
                archive = random.sample(archive, target_NP)

        NP = len(pop)
        if NP < 4:
            return best

        # choose pbest fraction (more exploitation later)
        p = min(0.35, max(2.0 / NP, 0.10 + 0.22 * frac))
        pbest_count = max(2, int(round(p * NP)))

        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        S_F, S_CR, weights = [], [], []

        union = pop + archive
        cen = cen if (random.random() < 0.7) else centroid(pop)  # occasional refresh

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i], fit[i]

            # draw from memory
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample CR and F
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

            # pick pbest
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # pick r1 from pop != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # pick r2 from union, not same object as xi/xr1 when possible
            # (union is small; loop is fine)
            xr2 = None
            for _ in range(25):
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                xr2 = cand
                break
            if xr2 is None:
                xr2 = union[random.randrange(len(union))]

            # ---- candidate generation: make a few trials, pick 1 to evaluate ----
            # score uses: closeness to best and some diversity (avoid too close to centroid)
            n_trials = 2 if dim <= 20 else 3
            best_u = None
            best_score = None

            for _t in range(n_trials):
                # mix two strategies
                use_ctpbest = (random.random() < (0.75 - 0.35 * (1.0 - frac)))  # more ctpbest early-mid
                v = [0.0] * dim
                if use_ctpbest:
                    for d in range(dim):
                        v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
                else:
                    # rand/1 from three randoms (choose fresh r3,r4,r5)
                    a = random.randrange(NP)
                    b = random.randrange(NP)
                    c3 = random.randrange(NP)
                    while b == a:
                        b = random.randrange(NP)
                    while c3 == a or c3 == b:
                        c3 = random.randrange(NP)
                    xa, xb, xc = pop[a], pop[b], pop[c3]
                    for d in range(dim):
                        v[d] = xa[d] + Fi * (xb[d] - xc[d])

                # crossover
                jrand = random.randrange(dim)
                u = [0.0] * dim
                for d in range(dim):
                    if spans[d] == 0.0:
                        u[d] = bounds[d][0]
                    else:
                        if random.random() < CRi or d == jrand:
                            u[d] = v[d]
                        else:
                            u[d] = xi[d]
                u = repair(u, xi)

                # score: prefer closer to best, but not collapsing to centroid too hard
                # (lower is better)
                d_best = l2_dist2(u, bestx) if bestx is not None else 0.0
                d_cen = l2_dist2(u, cen)
                score = d_best - 0.05 * d_cen  # small push away from centroid
                if best_score is None or score < best_score:
                    best_score = score
                    best_u = u

            u = best_u
            fu = eval_f(u)

            if fu <= fi:
                # archive old
                if len(archive) < NP:
                    archive.append(xi)
                else:
                    archive[random.randrange(NP)] = xi

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, bestx = fu, u

                df = abs(fi - fu)
                w = df if df > 0.0 else 1.0
                S_F.append(Fi)
                S_CR.append(CRi)
                weights.append(w)

        # update memories
        if S_F:
            wsum = sum(weights)
            if wsum <= 0.0:
                wsum = 1.0

            # Lehmer mean for F; weighted mean for CR
            num, den, cr = 0.0, 0.0, 0.0
            for Fi, CRi, w in zip(S_F, S_CR, weights):
                num += w * Fi * Fi
                den += w * Fi
                cr += w * CRi
            newF = (num / den) if den != 0.0 else M_F[h_idx]
            newCR = cr / wsum

            # keep in reasonable range
            if newF < 0.05: newF = 0.05
            if newF > 0.95: newF = 0.95
            if newCR < 0.0: newCR = 0.0
            if newCR > 1.0: newCR = 1.0

            M_F[h_idx] = newF
            M_CR[h_idx] = newCR
            h_idx = (h_idx + 1) % H

        # local refinement near end or occasionally when improvement occurs
        if bestx is not None and time.time() < deadline:
            p_ref = 0.05 + 0.40 * frac
            if random.random() < p_ref:
                rx, rf = local_trust_refine(bestx, best, deadline, frac)
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

        # stagnation / partial restart
        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if stagn >= 9 and time.time() < deadline:
            # restart a few worst individuals
            replace_n = max(2, NP // 5)
            worst = list(range(NP))
            worst.sort(key=lambda i: fit[i], reverse=True)
            for t in range(replace_n):
                if time.time() >= deadline:
                    return best
                j = worst[t]
                if random.random() < 0.55:
                    xnew = halton_point(k_hal, bases); k_hal += 1
                else:
                    xnew = random_point()
                fnew = eval_f(xnew)
                xo = opposite_point(xnew)
                fxo = eval_f(xo) if time.time() < deadline else float("inf")
                if fxo < fnew:
                    xnew, fnew = xo, fxo
                pop[j] = xnew
                fit[j] = fnew
                if fnew < best:
                    best, bestx = fnew, xnew
            archive = []
            stagn = 0

    return best
