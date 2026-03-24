import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (standard library only).

    Improved algorithm vs the provided ones:
      - Multi-start: keeps a global best, but also runs short "intensification" phases.
      - Two-population idea in one: a main DE population + an aggressive small "elite"
        search around best using heavy-tailed + gaussian steps.
      - SHADE/JADE-like DE/current-to-pbest/1 with archive and success-history memories.
      - Budget-aware: adapts population size to dimension and time, and avoids heavy sorts.

    Returns: best fitness (float).
    """
    t0 = time.time()
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def time_up():
        return (time.time() - t0) >= max_time

    def reflect(x):
        # reflection mapping into [lo,hi]
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            w = hi - lo
            v = y[i] - lo
            m = v % (2.0 * w)
            if m <= w:
                y[i] = lo + m
            else:
                y[i] = lo + (2.0 * w - m)
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # --- set NP with dim but keep moderate for time-bounded runs ---
    # In practice, too-large NP is bad when func is costly.
    NP = 12 + 3 * dim
    if NP < 18:
        NP = 18
    if NP > 70:
        NP = 70

    # --- init population (use a bit of opposition-like sampling) ---
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    for i in range(NP):
        x = rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]
        if time_up():
            return best

    # opposition trials for a few random individuals
    opp_trials = max(2, NP // 6)
    for _ in range(opp_trials):
        if time_up():
            return best
        i = random.randrange(NP)
        xo = [lows[d] + highs[d] - pop[i][d] for d in range(dim)]
        xo = reflect(xo)
        fo = eval_f(xo)
        if fo < fit[i]:
            pop[i], fit[i] = xo, fo
            if fo < best:
                best, best_x = fo, xo[:]

    # --- Archive for JADE/SHADE style ---
    archive = []
    arch_max = NP

    # --- SHADE memories (H entries) ---
    H = 8
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    p = 0.20  # p-best fraction

    # --- stagnation / restart control ---
    last_improve_t = time.time()
    patience = max(0.12 * max_time, 0.8)

    # --- Elite local explorer settings (cheap but effective) ---
    # This helps a lot when DE gets close but needs refinement.
    next_elite = time.time() + max(0.12 * max_time, 0.25)
    elite_batch = max(10, 3 * dim)
    # per-dim step scales (relative to span), will shrink if no improvements
    elite_scale = 0.08

    # helper: get indices of top p fraction without sorting everything expensively too often
    # We'll do a full sort once per generation (NP <= 70 so ok).
    def top_p_indices():
        order = sorted(range(NP), key=lambda i: fit[i])
        pnum = int(math.ceil(p * NP))
        if pnum < 2:
            pnum = 2
        return order, pnum

    def elite_explore():
        nonlocal best, best_x, last_improve_t, elite_scale
        if best_x is None:
            return
        base = best_x[:]
        base_f = best

        # if we're stuck, shrink scale a bit; if improving, can keep or slightly grow
        improved = False

        for _ in range(elite_batch):
            if time_up():
                break

            # mixture: heavy-tailed single-coordinate + gaussian all-dim
            if random.random() < 0.55:
                y = base[:]
                j = random.randrange(dim)
                # cauchy-ish kick on one coordinate
                y[j] += elite_scale * spans[j] * math.tan(math.pi * (random.random() - 0.5))
            else:
                y = base[:]
                sig = max(1e-15, 0.5 * elite_scale)
                for d in range(dim):
                    y[d] += random.gauss(0.0, sig * spans[d])

            y = reflect(y)
            fy = eval_f(y)

            if fy < best:
                best, best_x = fy, y[:]
                base, base_f = best_x[:], best
                last_improve_t = time.time()
                improved = True

        # adapt elite_scale mildly
        if improved:
            elite_scale = min(0.20, elite_scale * 1.05)
        else:
            elite_scale = max(1e-6, elite_scale * 0.85)

    # --- main loop ---
    while not time_up():
        # periodic elite exploration (intensification)
        if time.time() >= next_elite and not time_up():
            elite_explore()
            # become more frequent later
            rem = max_time - (time.time() - t0)
            next_elite = time.time() + max(0.10, 0.06 * rem)

        # stagnation: partial restart of worst individuals
        if (time.time() - last_improve_t) >= patience and not time_up():
            # replace worst ~ 1/3
            k = max(3, NP // 3)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                if time_up():
                    break
                if random.random() < 0.55:
                    x = rand_point()
                else:
                    # around best with heavier tails
                    x = best_x[:]
                    for d in range(dim):
                        x[d] += 0.10 * spans[d] * math.tan(math.pi * (random.random() - 0.5))
                    x = reflect(x)
                fx = eval_f(x)
                pop[idx], fit[idx] = x, fx
                if fx < best:
                    best, best_x = fx, x[:]
                    last_improve_t = time.time()

            # trim archive to refresh dynamics
            if len(archive) > 0:
                # keep only last half
                archive = archive[len(archive) // 2 :]
            # reset elite scale a bit larger to escape
            elite_scale = min(0.15, max(elite_scale, 0.08))
            last_improve_t = time.time()

        order, pnum = top_p_indices()

        # successes for SHADE update
        S_F, S_CR, S_w = [], [], []

        for i in range(NP):
            if time_up():
                break

            xi = pop[i]
            fi = fit[i]

            # choose memory index r
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample CR ~ N(muCR, 0.1), clipped
            CR = random.gauss(muCR, 0.1)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # sample F ~ Cauchy(muF, 0.1), re-sample if <=0, clip to <=1
            F = cauchy(muF, 0.1)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = cauchy(muF, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose pbest from top pnum
            pbest_idx = order[random.randrange(pnum)]
            xpbest = pop[pbest_idx]

            # r1 from pop (not i, not pbest)
            r1 = i
            while r1 == i or r1 == pbest_idx:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # r2 from pop U archive (prefer archive sometimes if available)
            use_arch = (len(archive) > 0 and random.random() < (len(archive) / float(len(archive) + NP)))
            if use_arch:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == pbest_idx or r2 == r1:
                    r2 = random.randrange(NP)
                xr2 = pop[r2]

            # current-to-pbest/1
            v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            u = reflect(u)
            fu = eval_f(u)

            if fu <= fi:
                # success: push parent into archive
                archive.append(xi[:])
                if len(archive) > arch_max:
                    # random drop
                    j = random.randrange(len(archive))
                    archive[j] = archive[-1]
                    archive.pop()

                pop[i], fit[i] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = time.time()

                # record for memory update (weight by improvement)
                w = fi - fu
                if w <= 0.0:
                    w = 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(w)

        # update memories (SHADE-style)
        if S_F:
            wsum = sum(S_w)
            if wsum <= 0.0:
                wsum = float(len(S_w))

            # weighted mean CR
            meanCR = 0.0
            for cr, w in zip(S_CR, S_w):
                meanCR += cr * (w / wsum)

            # weighted Lehmer mean F
            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_w):
                wf = (w / wsum)
                num += wf * f * f
                den += wf * f
            meanF = (num / den) if den > 0.0 else M_F[k_mem]

            M_CR[k_mem] = meanCR
            M_F[k_mem] = meanF
            k_mem = (k_mem + 1) % H

    return best
