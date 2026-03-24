import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (no external libs).

    Hybrid strategy:
      1) Multi-start: low-discrepancy-ish (stratified) init + jittered restarts
      2) DE (current-to-pbest/1 + archive) with success-history adaptation (JADE/SHADE-like)
      3) Opposition-based injection when progress stalls
      4) Periodic local search: randomized coordinate + small pattern moves
      5) Tight time management; returns best fitness found

    Notes:
      - func must accept a list of floats (len=dim) and return a float.
      - bounds is list of (lo, hi) for each dimension.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if any(s < 0 for s in spans):
        # swap invalid bounds
        for i in range(dim):
            if spans[i] < 0:
                lows[i], highs[i] = highs[i], lows[i]
                spans[i] = -spans[i]

    span_max = max(spans) if spans else 1.0
    if span_max <= 0.0:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # --------------------- utilities ---------------------
    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflection; few iterations then clip (fast + stable)
            for _ in range(4):
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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_of(x):
        # opposite point w.r.t. bounds: lo+hi-x
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def argmin(vals):
        bi, bv = 0, vals[0]
        for i in range(1, len(vals)):
            if vals[i] < bv:
                bi, bv = i, vals[i]
        return bi

    # Stratified init (reduces clustering vs pure random)
    def stratified_population(n, jitter=0.15):
        # For each dimension, use a random permutation of n strata
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pop = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    x[d] = lows[d]
                else:
                    # base in stratum + small jitter
                    u = (perms[d][k] + 0.5) / n
                    u += (random.random() - 0.5) * (jitter / n)
                    if u < 0.0: u = 0.0
                    if u > 1.0: u = 1.0
                    x[d] = lows[d] + u * spans[d]
            pop.append(x)
        return pop

    # Local search: coordinate probes + occasional 2D pattern move
    def local_search(x, fx, time_slice):
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        xb, fb = x[:], fx

        step = [0.06 * s if s > 0 else 1.0 for s in spans]
        min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

        no_imp = 0
        while time.time() < t_end:
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            # coordinate probing
            for i in order:
                if time.time() >= t_end:
                    break
                if step[i] <= min_step[i]:
                    continue

                base = xb[i]
                best_i = base
                best_f = fb

                # a few step multipliers
                for mult in (1.0, 0.5, 1.5):
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
                    step[i] *= 0.75

            # occasional small 2D pattern move (helps on rotated valleys)
            if time.time() < t_end and dim >= 2 and random.random() < 0.35:
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1
                if step[i] > min_step[i] and step[j] > min_step[j]:
                    di = step[i] * (1.0 if random.random() < 0.5 else -1.0)
                    dj = step[j] * (1.0 if random.random() < 0.5 else -1.0)
                    xt = xb[:]
                    xt[i] += di
                    xt[j] += dj
                    ft = eval_vec(xt)
                    if ft < fb:
                        xb, fb = reflect_into_bounds(xt), ft
                        improved = True

            if improved:
                no_imp = 0
                for d in range(dim):
                    step[d] *= 1.03
            else:
                no_imp += 1
                for d in range(dim):
                    step[d] *= 0.88
                if no_imp >= 3:
                    break

        return xb, fb

    # --------------------- main algorithm ---------------------
    # Population size: balanced; avoid exploding eval count in high dim
    pop_size = max(16, min(96, 10 * dim))
    p_best_frac = 0.12  # smaller p-best tends to be more exploitative

    # Success-history memory (SHADE-like)
    H = 8
    MF = [0.55] * H
    MCR = [0.85] * H
    hist_idx = 0
    c_learn = 0.12

    archive = []
    arch_max = pop_size

    best = float("inf")
    best_x = None

    # Stagnation & restarts
    last_improve = time.time()
    min_progress = 1e-12
    stall_seconds = max(0.12 * float(max_time), 0.35)

    # A little evaluation budget reserved for end polishing
    end_polish_reserved = 0.08 * float(max_time)

    def try_update_best(x, fx):
        nonlocal best, best_x, last_improve
        if fx + min_progress < best:
            best = fx
            best_x = x[:]
            last_improve = time.time()

    # multi-start loop
    while time.time() < deadline:
        # initialization: stratified + a few around incumbent
        pop = []
        if best_x is not None:
            k = min(pop_size // 4, 12)
            for _ in range(k):
                x = best_x[:]
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, 0.18 * spans[d])
                pop.append(reflect_into_bounds(x))

        # fill remaining via stratified
        rem = pop_size - len(pop)
        pop.extend(stratified_population(rem, jitter=0.25))

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = eval_vec(x)
            fit.append(fx)
            try_update_best(x, fx)

        # evolution
        while time.time() < deadline:
            # p-best set
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            pcount = max(2, int(math.ceil(p_best_frac * pop_size)))
            pbest_idx = idx_sorted[:pcount]

            S_F, S_CR, dF = [], [], []

            new_pop = pop[:]
            new_fit = fit[:]

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]

                # sample from memory
                r = random.randrange(H)
                muF, muCR = MF[r], MCR[r]

                # CR ~ N(muCR, 0.1)
                CR = random.gauss(muCR, 0.1)
                if CR < 0.0: CR = 0.0
                if CR > 1.0: CR = 1.0

                # F ~ Cauchy(muF, 0.1), resample until positive, cap at 1
                F = -1.0
                for _ in range(12):
                    u = random.random()
                    F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                    if F > 0.0:
                        break
                if F <= 0.0:
                    F = 0.45
                if F > 1.0:
                    F = 1.0

                # choose pbest
                pb = pop[random.choice(pbest_idx)]

                # choose r1 != i
                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)

                # choose r2 from pop U archive, try to diversify
                use_arch = (archive and random.random() < 0.5)
                if use_arch:
                    pool = pop + archive
                    # ensure not same object; accept archive always
                    r2v = pool[random.randrange(len(pool))]
                    # if accidentally same as xi and from pop, resample a bit
                    for _ in range(6):
                        if r2v is xi or r2v is pop[r1]:
                            r2v = pool[random.randrange(len(pool))]
                        else:
                            break
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    r2v = pop[r2]

                xr1 = pop[r1]
                xr2 = r2v

                # mutation: current-to-pbest/1
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + F * (pb[d] - xi[d]) + F * (xr1[d] - xr2[d])

                # binomial crossover
                jrand = random.randrange(dim)
                uvec = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

                uvec = reflect_into_bounds(uvec)
                fu = eval_vec(uvec)

                if fu <= fi:
                    # archive parent
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = uvec
                    new_fit[i] = fu

                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    dF.append(df if df > 0.0 else 1e-16)

                    try_update_best(uvec, fu)

            pop, fit = new_pop, new_fit

            # update memories
            if S_F:
                wsum = sum(dF)
                inv = 1.0 / wsum if wsum > 0 else 1.0 / len(dF)
                weights = [(df * inv) if wsum > 0 else inv for df in dF]

                # Lehmer mean for F
                num = 0.0
                den = 0.0
                for w, f in zip(weights, S_F):
                    num += w * f * f
                    den += w * f
                F_new = num / den if den > 0 else sum(S_F) / len(S_F)

                # weighted arithmetic mean for CR
                CR_new = 0.0
                for w, cr in zip(weights, S_CR):
                    CR_new += w * cr

                MF[hist_idx] = (1.0 - c_learn) * MF[hist_idx] + c_learn * F_new
                MCR[hist_idx] = (1.0 - c_learn) * MCR[hist_idx] + c_learn * CR_new
                hist_idx = (hist_idx + 1) % H

            # mild opposition-based injection when stalled (but not too often)
            if time.time() - last_improve > 0.6 * stall_seconds and best_x is not None:
                # replace a couple worst individuals with opposites of best + noise
                wcount = max(1, pop_size // 12)
                worst_idx = idx_sorted[-wcount:]
                for wi in worst_idx:
                    if time.time() >= deadline:
                        return best
                    x = opposite_of(best_x)
                    for d in range(dim):
                        if spans[d] > 0:
                            x[d] += random.gauss(0.0, 0.08 * spans[d])
                    x = reflect_into_bounds(x)
                    fx = eval_vec(x)
                    pop[wi] = x
                    fit[wi] = fx
                    try_update_best(x, fx)

            # periodic local search on incumbent (more aggressive near the end)
            remaining = deadline - time.time()
            if best_x is not None and remaining > 0.02:
                prob = 0.06 if remaining > end_polish_reserved else 0.18
                if random.random() < prob:
                    slice_t = min(0.08, 0.18 * remaining)
                    bx, bf = local_search(best_x, best, slice_t)
                    try_update_best(bx, bf)

            # restart if stalled
            if time.time() - last_improve > stall_seconds:
                # last-chance stronger local search before restart
                remaining = deadline - time.time()
                if best_x is not None and remaining > 0.03:
                    bx, bf = local_search(best_x, best, min(0.14, 0.28 * remaining))
                    try_update_best(bx, bf)
                break

    return best
