import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only) targeting strong anytime performance.

    Main changes vs previous:
      - Switch to "L-SHADE-like" DE with current-to-pbest mutation + archive (more robust)
      - Use quasi-Latin hypercube init + opposition points (better initial coverage)
      - Adaptive (per-individual) F/CR sampling with success-memory update
      - Cheap periodic local refinement using multi-scale coordinate search around best
      - Stagnation-triggered diversified refresh (keeps elites + re-diversifies)

    Returns:
        best (float): best fitness found within max_time seconds
    """

    # ---------------------------
    # Helpers
    # ---------------------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        try:
            y = func(x)
        except TypeError:
            y = func(*x)
        try:
            y = float(y)
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == -float("inf"):
            return float("inf")
        return y

    if dim <= 0:
        return eval_f([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_ok = [span[i] > 0.0 for i in range(dim)]

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] if not span_ok[i] else random.uniform(lo[i], hi[i])
        return x

    def opposite_vec(x):
        ox = x[:]
        for i in range(dim):
            if span_ok[i]:
                ox[i] = (lo[i] + hi[i]) - x[i]
                if ox[i] < lo[i]: ox[i] = lo[i]
                elif ox[i] > hi[i]: ox[i] = hi[i]
            else:
                ox[i] = lo[i]
        return ox

    # Quasi-Latin hypercube (no numpy): stratified per-dimension, random permutation
    def qlhs(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    x[d] = lo[d]
                else:
                    u = (perms[d][i] + random.random()) / n
                    x[d] = lo[d] + u * span[d]
            pop.append(x)
        return pop

    # Small local coordinate refinement (multi-scale), evaluation-light
    def local_refine(x, fx, budget):
        # Start with steps relative to span; shrink quickly
        if budget <= 0:
            return x, fx, 0
        steps = []
        for d in range(dim):
            if span_ok[d]:
                steps.append(0.06 * span[d])
            else:
                steps.append(0.0)

        bestx = x[:]
        bestf = fx
        used = 0

        # A few rounds; each round tries all dims in random order
        for _ in range(3):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for d in order:
                if used >= budget:
                    return bestx, bestf, used
                s = steps[d]
                if s <= 0.0:
                    continue

                base = bestx[d]
                # try both directions
                for sgn in (-1.0, 1.0):
                    if used >= budget:
                        return bestx, bestf, used
                    cand = bestx[:]
                    cand[d] = clamp(base + sgn * s, lo[d], hi[d])
                    if cand[d] == base:
                        continue
                    fc = eval_f(cand)
                    used += 1
                    if fc < bestf:
                        bestx, bestf = cand, fc
                        improved = True
                        base = bestx[d]

            # adjust steps
            if improved:
                for d in range(dim):
                    steps[d] = min(steps[d] * 1.15, 0.25 * span[d] if span_ok[d] else 0.0)
            else:
                for d in range(dim):
                    steps[d] *= 0.30

        return bestx, bestf, used

    # Heavy-tail step for escapes
    def cauchy(scale):
        u = random.random()
        t = math.tan(math.pi * (u - 0.5))
        if t > 30.0: t = 30.0
        if t < -30.0: t = -30.0
        return scale * t

    # ---------------------------
    # Time bookkeeping
    # ---------------------------
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------------------
    # Population size (anytime-friendly)
    # ---------------------------
    # Slightly larger than before for better DE dynamics, still capped.
    NP = max(18, min(90, 12 * dim))

    # ---------------------------
    # Initialization: QLHS + opposition, keep best NP
    # ---------------------------
    cand = qlhs(NP)
    cand += [opposite_vec(x) for x in cand]

    pop = []
    fit = []
    for x in cand:
        if time.time() >= deadline:
            break
        pop.append(x)
        fit.append(eval_f(x))

    # If we couldn't evaluate anything
    if not fit:
        return float("inf")

    # Fill if needed
    while len(pop) < NP and time.time() < deadline:
        x = rand_vec()
        pop.append(x)
        fit.append(eval_f(x))

    # Trim to NP best
    if len(pop) > NP:
        idx = sorted(range(len(pop)), key=lambda i: fit[i])[:NP]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    best_i = min(range(len(pop)), key=lambda i: fit[i])
    bestx = pop[best_i][:]
    best = fit[best_i]

    # ---------------------------
    # L-SHADE-like adaptive DE components
    # ---------------------------
    H = 6  # memory size
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_pos = 0

    # external archive for diversity
    archive = []
    arch_max = NP

    # for stagnation handling
    stall = 0
    last_best = best
    max_stall = max(10, 3 + dim)

    gen = 0
    while time.time() < deadline:
        gen += 1
        tl = deadline - time.time()
        near_end = tl <= 0.20 * max_time

        # p-best fraction
        p = 0.18 if not near_end else 0.10
        pnum = max(2, int(p * NP))

        # Precompute ranking
        order = sorted(range(NP), key=lambda i: fit[i])

        SF = []
        SCR = []
        dF = []  # fitness improvements (weights)

        new_pop = [None] * NP
        new_fit = [None] * NP

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            # choose memory index
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # sample CR ~ N(mu, 0.1), clipped
            CR = mu_cr + 0.1 * (random.random() + random.random() + random.random() - 1.5)  # approx N
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # sample F ~ Cauchy(mu_f, 0.1) until in (0,1]
            F = -1.0
            for _ in range(6):
                F = mu_f + cauchy(0.1)
                if 0.0 < F <= 1.0:
                    break
            if not (0.0 < F <= 1.0):
                F = 0.5

            # pick pbest
            pbest = order[random.randrange(pnum)]

            # r1 from population, != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            # r2 from pop+archive, != i,r1
            total_pool = NP + len(archive)
            if total_pool <= 2:
                r2 = (i + 1) % NP
                x_r2 = pop[r2]
            else:
                while True:
                    r2 = random.randrange(total_pool)
                    if r2 < NP:
                        if r2 != i and r2 != r1:
                            x_r2 = pop[r2]
                            break
                    else:
                        aidx = r2 - NP
                        x_r2 = archive[aidx]
                        # archive entries are never "i" or "r1"
                        break

            x_r1 = pop[r1]
            x_pb = pop[pbest]

            # current-to-pbest/1 mutation
            v = [0.0] * dim
            for d in range(dim):
                if not span_ok[d]:
                    v[d] = lo[d]
                else:
                    v[d] = xi[d] + F * (x_pb[d] - xi[d]) + F * (x_r1[d] - x_r2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if (random.random() < CR) or (d == jrand):
                    u[d] = v[d]

            # bounds: reflection + clamp
            for d in range(dim):
                if not span_ok[d]:
                    u[d] = lo[d]
                    continue
                if u[d] < lo[d]:
                    u[d] = lo[d] + (lo[d] - u[d])
                    if u[d] > hi[d]:
                        u[d] = lo[d] + random.random() * span[d]
                elif u[d] > hi[d]:
                    u[d] = hi[d] - (u[d] - hi[d])
                    if u[d] < lo[d]:
                        u[d] = lo[d] + random.random() * span[d]
                if u[d] < lo[d]: u[d] = lo[d]
                elif u[d] > hi[d]: u[d] = hi[d]

            fu = eval_f(u)

            if fu <= fi:
                new_pop[i] = u
                new_fit[i] = fu

                # archive the replaced parent if it was strictly worse
                if fu < fi and len(archive) < arch_max:
                    archive.append(xi[:])
                elif fu < fi and arch_max > 0:
                    # random replacement in archive
                    archive[random.randrange(len(archive))] = xi[:]

                if fu < best:
                    best = fu
                    bestx = u[:]

                # record successes for memory update
                if fu < fi:
                    SF.append(F)
                    SCR.append(CR)
                    dF.append(fi - fu)
            else:
                new_pop[i] = xi
                new_fit[i] = fi

        # commit generation if we computed it
        if new_fit[0] is not None:
            pop = new_pop
            fit = new_fit

        # update adaptive memories (weighted Lehmer mean for F)
        if dF:
            wsum = sum(dF)
            if wsum <= 0.0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            # MCR: weighted mean
            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            # MF: weighted Lehmer mean (sum w*F^2 / sum w*F)
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            mf = (num / den) if den > 1e-12 else MF[mem_pos]

            MCR[mem_pos] = mcr
            MF[mem_pos] = mf
            mem_pos = (mem_pos + 1) % H

        # periodic local refine + small best-kick
        if time.time() < deadline:
            if near_end or (gen % 9 == 0):
                budget = 10 + 2 * dim  # small and predictable
                bx, bf, _ = local_refine(bestx, best, budget)
                if bf < best:
                    best, bestx = bf, bx

            if (gen % 7 == 0) and dim > 0 and not near_end:
                # a couple of kicked candidates around best
                for _ in range(2):
                    if time.time() >= deadline:
                        break
                    candx = bestx[:]
                    sc = 0.02
                    for d in range(dim):
                        if span_ok[d]:
                            candx[d] = clamp(candx[d] + cauchy(sc * span[d]), lo[d], hi[d])
                    fc = eval_f(candx)
                    if fc < best:
                        best, bestx = fc, candx[:]

        # stagnation detection & diversified refresh
        if best < last_best:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= max_stall and time.time() < deadline:
            stall = 0
            # keep elites, refresh others (some near-best, some random)
            order = sorted(range(NP), key=lambda i: fit[i])
            keep = max(4, NP // 6)
            elites = order[:keep]

            for idx in order[keep:]:
                if time.time() >= deadline:
                    break
                if random.random() < 0.65:
                    x = bestx[:]
                    scale = 0.18
                    for d in range(dim):
                        if span_ok[d]:
                            x[d] = clamp(x[d] + random.uniform(-1.0, 1.0) * scale * span[d], lo[d], hi[d])
                    pop[idx] = x
                else:
                    pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])

            # shrink/refresh archive a bit to avoid stale bias
            if len(archive) > arch_max:
                random.shuffle(archive)
                archive = archive[:arch_max]
            elif len(archive) > 0 and random.random() < 0.3:
                # randomly drop some
                k = max(1, len(archive) // 4)
                for _ in range(k):
                    if archive:
                        archive.pop(random.randrange(len(archive)))

            best_i = min(range(NP), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                bestx = pop[best_i][:]

    return best
