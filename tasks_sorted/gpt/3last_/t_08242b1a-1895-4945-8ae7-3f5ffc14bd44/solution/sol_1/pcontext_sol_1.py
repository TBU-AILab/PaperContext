import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid:
      1) Differential Evolution "current-to-pbest/1" (JADE-like) with an archive
      2) Success-history style adaptation of F and CR
      3) Bound handling via reflection + clipping
      4) Occasional local polish: stochastic coordinate/pattern search
      5) Restarts when stagnating

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ----------------------- helpers -----------------------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    span_max = max(spans) if dim > 0 else 1.0
    if span_max <= 0:
        # Degenerate bounds; evaluate the single point
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_into_bounds(x):
        # reflection (with safety) then clip
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect a few times if far outside
            for _ in range(3):
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
                else:
                    break
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def eval_vec(x):
        return float(func(reflect_into_bounds(x)))

    def mean(xs):
        return sum(xs) / float(len(xs)) if xs else 0.0

    def median(xs):
        if not xs:
            return 0.0
        ys = sorted(xs)
        m = len(ys) // 2
        return ys[m] if len(ys) % 2 == 1 else 0.5 * (ys[m - 1] + ys[m])

    # Local polish: stochastic coordinate search with adaptive step
    def local_polish(x, fx, time_slice):
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        xb = x[:]
        fb = fx

        # start step relative to bounds
        step = [0.08 * s if s > 0 else 1.0 for s in spans]
        # randomized coordinate order each sweep
        no_improve_sweeps = 0

        while time.time() < t_end:
            order = list(range(dim))
            random.shuffle(order)
            improved = False

            for i in order:
                if time.time() >= t_end:
                    break
                if step[i] <= 1e-14 * (spans[i] if spans[i] > 0 else 1.0):
                    continue

                # try multiple probes along this coordinate
                base = xb[i]
                best_i = base
                best_f = fb

                # probe set (small -> large)
                for mult in (1.0, 0.5, 1.5, 2.0):
                    delta = step[i] * mult

                    x1 = xb[:]
                    x1[i] = base + delta
                    f1 = eval_vec(x1)
                    if f1 < best_f:
                        best_f = f1
                        best_i = x1[i]

                    x2 = xb[:]
                    x2[i] = base - delta
                    f2 = eval_vec(x2)
                    if f2 < best_f:
                        best_f = f2
                        best_i = x2[i]

                if best_f < fb:
                    xb[i] = best_i
                    fb = best_f
                    improved = True
                else:
                    step[i] *= 0.7

            if improved:
                no_improve_sweeps = 0
                # cautiously expand steps a bit to keep moving
                for i in range(dim):
                    step[i] *= 1.05
            else:
                no_improve_sweeps += 1
                for i in range(dim):
                    step[i] *= 0.85
                if no_improve_sweeps >= 3:
                    break

        return xb, fb

    # ----------------------- algorithm params -----------------------
    # Population size: moderate, scaled with dim
    pop_size = max(12, min(80, 12 * dim))
    # p-best fraction for current-to-pbest mutation
    p = 0.15

    # Success-history memories for F and CR (small for speed)
    H = 6
    MF = [0.6] * H
    MCR = [0.85] * H
    hist_idx = 0

    # DE control params
    c = 0.1          # learning rate for memories
    archive = []     # external archive (JADE)
    arch_max = pop_size

    best = float("inf")
    best_x = None

    # stagnation / restart
    last_improve_time = time.time()
    stall_seconds = max(0.15 * float(max_time), 0.5)  # restart if no improvement for this long
    min_progress = 1e-12

    # ----------------------- main loop with restarts -----------------------
    while time.time() < deadline:
        # initialize population (plus a few biased samples around best if available)
        pop = []
        if best_x is not None:
            for _ in range(min(pop_size // 4, 10)):
                # small gaussian perturbation around best
                x = best_x[:]
                for d in range(dim):
                    sd = 0.15 * spans[d]
                    if sd > 0:
                        x[d] = x[d] + random.gauss(0.0, sd)
                pop.append(reflect_into_bounds(x))
        while len(pop) < pop_size:
            pop.append(rand_vec())

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = eval_vec(x)
            fit.append(fx)
            if fx + min_progress < best:
                best = fx
                best_x = x[:]
                last_improve_time = time.time()

        # evolution loop (until restart condition or time)
        while time.time() < deadline:
            # build p-best set indices
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            pcount = max(2, int(math.ceil(p * pop_size)))
            pbest_set = idx_sorted[:pcount]

            S_F = []
            S_CR = []
            delta_f = []

            new_pop = pop[:]
            new_fit = fit[:]

            # For each target vector
            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi = pop[i]
                fi = fit[i]

                # sample memory index
                r = random.randrange(H)
                muF = MF[r]
                muCR = MCR[r]

                # generate CR ~ N(muCR, 0.1) clipped
                CR = random.gauss(muCR, 0.1)
                if CR < 0.0:
                    CR = 0.0
                elif CR > 1.0:
                    CR = 1.0

                # generate F ~ Cauchy(muF, 0.1) positive and <=1
                # Cauchy: mu + gamma * tan(pi*(u-0.5))
                F = -1.0
                for _ in range(10):
                    u = random.random()
                    F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                    if F > 0.0:
                        break
                if F <= 0.0:
                    F = 0.5
                if F > 1.0:
                    F = 1.0

                # choose pbest
                pbest = pop[random.choice(pbest_set)]

                # choose r1 from population != i
                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                # choose r2 from pop U archive, != i, != r1
                use_arch = (archive and random.random() < 0.5)
                if use_arch:
                    pool_size = pop_size + len(archive)
                    # sample index in combined pool
                    while True:
                        k = random.randrange(pool_size)
                        if k < pop_size:
                            r2vec = pop[k]
                            if k != i and k != r1:
                                break
                        else:
                            r2vec = archive[k - pop_size]
                            # archive has no index conflict with i/r1; accept
                            break
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    r2vec = pop[r2]

                xr1 = pop[r1]
                xr2 = r2vec

                # mutation: current-to-pbest/1
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

                # crossover (binomial)
                jrand = random.randrange(dim)
                uvec = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

                uvec = reflect_into_bounds(uvec)
                fu = eval_vec(uvec)

                if fu <= fi:
                    # success: put parent into archive
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        # random removal
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = uvec
                    new_fit[i] = fu

                    # record success params for adaptation
                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    delta_f.append(df if df > 0.0 else 1e-16)

                    if fu + min_progress < best:
                        best = fu
                        best_x = uvec[:]
                        last_improve_time = time.time()

            pop, fit = new_pop, new_fit

            # update memories (Lehmer mean for F, arithmetic mean for CR, weighted by improvements)
            if S_F:
                wsum = sum(delta_f)
                if wsum <= 0.0:
                    wsum = float(len(delta_f))
                    weights = [1.0 / wsum] * len(delta_f)
                else:
                    weights = [df / wsum for df in delta_f]

                # Lehmer mean: sum(w*F^2)/sum(w*F)
                num = 0.0
                den = 0.0
                for w, f in zip(weights, S_F):
                    num += w * f * f
                    den += w * f
                F_new = num / den if den > 0 else mean(S_F)
                CR_new = sum(w * cr for w, cr in zip(weights, S_CR))

                MF[hist_idx] = (1.0 - c) * MF[hist_idx] + c * F_new
                MCR[hist_idx] = (1.0 - c) * MCR[hist_idx] + c * CR_new
                hist_idx = (hist_idx + 1) % H

            # occasional polish when we have a good incumbent
            if best_x is not None and random.random() < 0.08:
                remaining = deadline - time.time()
                if remaining > 0.02:
                    # small slice
                    slice_t = min(0.06, 0.15 * remaining)
                    bx, bf = local_polish(best_x, best, slice_t)
                    if bf + min_progress < best:
                        best, best_x = bf, bx
                        last_improve_time = time.time()

            # restart if stalled
            if time.time() - last_improve_time > stall_seconds:
                # before restarting, do one stronger polish if time allows
                remaining = deadline - time.time()
                if best_x is not None and remaining > 0.03:
                    bx, bf = local_polish(best_x, best, min(0.12, 0.25 * remaining))
                    if bf + min_progress < best:
                        best, best_x = bf, bx
                        last_improve_time = time.time()
                break

    return best
