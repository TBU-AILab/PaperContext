import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Further-improved time-bounded minimizer (self-contained, no external libs).

    Improvements vs your current best (~4.14):
      - Adds a true gradient-free "CMA-like" local engine: Adaptive Diagonal Gaussian (ADG)
        * maintains per-dimension sigma, plus a low-rank direction set (limited-memory)
        * uses success-based step-size control (1/5th-ish) and rank-1 direction learning
      - Keeps your strong global engine (JADE/SHADE-like DE w/ archive + pbest)
      - Better exploitation schedule: DE -> ADG bursts -> (optional) NM (small dim)
      - Smarter stagnation logic: switches engines rather than full restart immediately
      - Still robust on rotated basins (thanks to direction-set + 2D rotation probes)

    func: list[float] -> float
    bounds: list[(lo, hi)]
    returns: best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]

    if max(spans) <= 0.0:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # -------------------- helpers --------------------
    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect a few times then clip
            for _ in range(6):
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

    # -------------------- quasi init: scrambled Halton --------------------
    def _first_primes(n):
        primes = []
        k = 2
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

    PRIMES = _first_primes(dim)

    def _radical_inverse(index, base, shift):
        f = 1.0 / base
        x = 0.0
        i = index + shift
        while i > 0:
            x += (i % base) * f
            i //= base
            f /= base
        return x

    def halton_population(n, jitter=0.0025):
        shifts = [random.randrange(1, 4000) for _ in range(dim)]
        start_k = random.randrange(15, 120)
        pop = []
        for t in range(n):
            k = start_k + t + 1
            x = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    x[d] = lows[d]
                else:
                    u = _radical_inverse(k, PRIMES[d], shifts[d])
                    u += (random.random() - 0.5) * jitter
                    if u < 0.0:
                        u = 0.0
                    elif u > 1.0:
                        u = 1.0
                    x[d] = lows[d] + u * spans[d]
            pop.append(x)
        return pop

    # -------------------- local search A: coord + random 2D rotations --------------------
    def local_search_rotcoord(x, fx, time_slice):
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        xb, fb = x[:], fx

        step = [0.06 * s if s > 0 else 1.0 for s in spans]
        min_step = [1e-14 * (s if s > 0 else 1.0) for s in spans]

        no_imp_rounds = 0
        while time.time() < t_end:
            improved = False

            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= t_end:
                    break
                if step[i] <= min_step[i]:
                    continue

                base = xb[i]
                best_i = base
                best_f = fb
                for mult in (1.0, 0.5, 1.5, 2.2):
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
                    step[i] *= 0.72

            if dim >= 2 and time.time() < t_end:
                tries = 1 if dim > 25 else 2
                for _ in range(tries):
                    if time.time() >= t_end:
                        break
                    i = random.randrange(dim)
                    j = random.randrange(dim - 1)
                    if j >= i:
                        j += 1
                    if step[i] <= min_step[i] and step[j] <= min_step[j]:
                        continue

                    theta = (random.random() * 2.0 - 1.0) * 1.05
                    ci = math.cos(theta)
                    si = math.sin(theta)
                    di = step[i] * (1.0 if random.random() < 0.5 else -1.0)
                    dj = step[j] * (1.0 if random.random() < 0.5 else -1.0)
                    ri = ci * di - si * dj
                    rj = si * di + ci * dj

                    xt = xb[:]
                    xt[i] += ri
                    xt[j] += rj
                    ft = eval_vec(xt)
                    if ft < fb:
                        xb, fb = reflect_into_bounds(xt), ft
                        improved = True

            if improved:
                no_imp_rounds = 0
                for d in range(dim):
                    step[d] *= 1.04
            else:
                no_imp_rounds += 1
                for d in range(dim):
                    step[d] *= 0.88
                if no_imp_rounds >= 3:
                    break

        return xb, fb

    # -------------------- local search B: Adaptive Diagonal Gaussian + direction set --------------------
    def local_adaptive_gaussian(x0, f0, time_slice):
        """
        CMA-like cheap local optimizer:
          - diagonal sigmas (per-coordinate)
          - small direction set (rank-1 memory) to handle rotations a bit
          - success-based step-size control
        """
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        x = x0[:]
        fx = f0

        # initial sigma proportional to range
        sigma = [0.18 * s if s > 0 else 1.0 for s in spans]
        sigma_min = [1e-15 * (s if s > 0 else 1.0) for s in spans]
        sigma_max = [0.60 * (s if s > 0 else 1.0) for s in spans]

        # direction memory (limited)
        kdir = 6 if dim >= 24 else 8
        dirs = []
        dir_w = []  # weights for mixing
        for _ in range(min(kdir, max(0, dim))):
            v = [random.gauss(0.0, 1.0) for _ in range(dim)]
            # normalize
            nrm = math.sqrt(sum(vi * vi for vi in v)) + 1e-18
            v = [vi / nrm for vi in v]
            dirs.append(v)
            dir_w.append(1.0)

        # success control
        trials = 0
        succ = 0
        batch = 12 if dim <= 12 else 16
        # global multiplier
        g = 1.0

        while time.time() < t_end:
            # propose a candidate
            y = x[:]

            # mixture: mostly diagonal gaussian, sometimes a direction move
            if dirs and random.random() < 0.55:
                # choose direction proportional to weight
                sw = sum(dir_w) + 1e-18
                r = random.random() * sw
                acc = 0.0
                idx = 0
                for i, w in enumerate(dir_w):
                    acc += w
                    if acc >= r:
                        idx = i
                        break
                dvec = dirs[idx]
                # step length based on average sigma
                avg_sig = sum(sigma) / float(dim)
                step_len = random.gauss(0.0, 1.0) * g * (1.25 * avg_sig)
                for j in range(dim):
                    y[j] = y[j] + step_len * dvec[j]
            else:
                for j in range(dim):
                    if spans[j] > 0.0:
                        y[j] = y[j] + random.gauss(0.0, 1.0) * g * sigma[j]

            y = reflect_into_bounds(y)
            fy = float(func(y))

            trials += 1
            if fy < fx:
                # update direction memory with successful move
                step = [y[i] - x[i] for i in range(dim)]
                nrm = math.sqrt(sum(si * si for si in step)) + 1e-18
                if nrm > 0.0:
                    u = [si / nrm for si in step]
                    if dirs:
                        # replace a random direction (or lowest weight) with the new one
                        if random.random() < 0.5:
                            rep = random.randrange(len(dirs))
                        else:
                            rep = min(range(len(dir_w)), key=lambda t: dir_w[t])
                        dirs[rep] = u
                        dir_w[rep] = max(dir_w[rep], 1.0) + 0.35
                    else:
                        dirs = [u]
                        dir_w = [1.0]

                # adapt sigmas a bit towards successful coordinate magnitudes
                for i in range(dim):
                    di = abs(y[i] - x[i])
                    if spans[i] > 0.0:
                        target = max(sigma_min[i], min(sigma_max[i], 0.80 * di + 0.20 * sigma[i]))
                        sigma[i] = 0.85 * sigma[i] + 0.15 * target

                x, fx = y, fy
                succ += 1
            else:
                # mild decay on failure to focus
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], 0.995 * sigma[i])

            # batch step-size control (1/5th style)
            if trials >= batch:
                ps = succ / float(trials)
                # target ~0.2
                if ps > 0.24:
                    g *= 1.12
                elif ps < 0.16:
                    g *= 0.88
                # clamp g somewhat
                if g < 0.05:
                    g = 0.05
                elif g > 2.5:
                    g = 2.5
                trials = 0
                succ = 0
                # slowly forget old direction weights
                for i in range(len(dir_w)):
                    dir_w[i] *= 0.97

        return x, fx

    # -------------------- Nelder–Mead (small dim) --------------------
    def nelder_mead(x0, f0, time_slice):
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        if dim == 0:
            return x0[:], f0

        simplex = [x0[:]]
        fvals = [f0]
        scale = [0.05 * s if s > 0 else 1.0 for s in spans]

        for i in range(dim):
            x = x0[:]
            if spans[i] > 0:
                x[i] += scale[i]
            x = reflect_into_bounds(x)
            simplex.append(x)
            fvals.append(eval_vec(x))
            if time.time() >= t_end:
                bi = min(range(len(fvals)), key=lambda k: fvals[k])
                return simplex[bi][:], fvals[bi]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        def sort_simplex():
            idx = sorted(range(len(fvals)), key=lambda k: fvals[k])
            return [simplex[k] for k in idx], [fvals[k] for k in idx]

        simplex, fvals = sort_simplex()

        while time.time() < t_end:
            centroid = [0.0] * dim
            for k in range(dim):
                xk = simplex[k]
                for d in range(dim):
                    centroid[d] += xk[d]
            inv = 1.0 / dim
            for d in range(dim):
                centroid[d] *= inv

            best = simplex[0]
            worst = simplex[-1]
            fbest = fvals[0]
            fworst = fvals[-1]
            fsecond = fvals[-2]

            xr = [centroid[d] + alpha * (centroid[d] - worst[d]) for d in range(dim)]
            xr = reflect_into_bounds(xr)
            fr = eval_vec(xr)
            if time.time() >= t_end:
                break

            if fr < fbest:
                xe = [centroid[d] + gamma * (xr[d] - centroid[d]) for d in range(dim)]
                xe = reflect_into_bounds(xe)
                fe = eval_vec(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fsecond:
                simplex[-1], fvals[-1] = xr, fr
            else:
                if fr < fworst:
                    xc = [centroid[d] + rho * (xr[d] - centroid[d]) for d in range(dim)]
                else:
                    xc = [centroid[d] - rho * (centroid[d] - worst[d]) for d in range(dim)]
                xc = reflect_into_bounds(xc)
                fc = eval_vec(xc)
                if fc < fworst:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for k in range(1, dim + 1):
                        xk = simplex[k]
                        xs = [best[d] + sigma * (xk[d] - best[d]) for d in range(dim)]
                        xs = reflect_into_bounds(xs)
                        simplex[k] = xs
                        fvals[k] = eval_vec(xs)
                        if time.time() >= t_end:
                            break

            simplex, fvals = sort_simplex()

        return simplex[0][:], fvals[0]

    # -------------------- DE (SHADE/JADE-like) with mutation mix --------------------
    pop_size = max(22, min(140, 12 * dim))
    H = 10
    MF = [0.62] * H
    MCR = [0.85] * H
    hist_idx = 0
    c = 0.12

    archive = []
    arch_max = pop_size

    best = float("inf")
    best_x = None
    last_improve = time.time()
    min_progress = 1e-12

    T = float(max_time)
    stall_seconds = max(0.12 * T, 0.45)
    endgame = 0.22 * T

    def update_best(x, fx):
        nonlocal best, best_x, last_improve
        if fx + min_progress < best:
            best = fx
            best_x = x[:]
            last_improve = time.time()

    def sample_F(muF):
        F = -1.0
        for _ in range(14):
            u = random.random()
            F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                break
        if F <= 0.0:
            F = 0.5
        if F > 1.0:
            F = 1.0
        return F

    def sample_CR(muCR):
        CR = random.gauss(muCR, 0.1)
        if CR < 0.0:
            CR = 0.0
        elif CR > 1.0:
            CR = 1.0
        return CR

    # -------------------- multi-start loop --------------------
    while time.time() < deadline:
        # init population: best-centered + Halton
        pop = []
        if best_x is not None:
            k = min(pop_size // 3, 22)
            for _ in range(k):
                x = best_x[:]
                for d in range(dim):
                    if spans[d] > 0:
                        x[d] += random.gauss(0.0, 0.22 * spans[d])
                pop.append(reflect_into_bounds(x))
        pop.extend(halton_population(pop_size - len(pop), jitter=0.003))

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = eval_vec(x)
            fit.append(fx)
            update_best(x, fx)

        # evolution loop
        while time.time() < deadline:
            now = time.time()
            remaining = deadline - now
            frac_left = remaining / max(1e-9, T)

            pfrac = 0.30 if frac_left > 0.70 else (0.16 if frac_left > 0.30 else 0.10)

            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            pcount = max(2, int(math.ceil(pfrac * pop_size)))
            pbest_idx = idx_sorted[:pcount]
            best_idx = idx_sorted[0]

            f_best = fit[best_idx]
            f_med = fit[idx_sorted[pop_size // 2]]
            converged = (abs(f_med - f_best) <= 1e-11 * (1.0 + abs(f_best)))

            S_F, S_CR, dF = [], [], []
            new_pop = pop[:]
            new_fit = fit[:]

            for i in range(pop_size):
                if time.time() >= deadline:
                    return best

                xi, fi = pop[i], fit[i]
                r = random.randrange(H)
                F = sample_F(MF[r])
                CR = sample_CR(MCR[r])

                pb = pop[random.choice(pbest_idx)]
                xbest = pop[best_idx]

                r1 = i
                while r1 == i:
                    r1 = random.randrange(pop_size)
                xr1 = pop[r1]

                use_arch = (archive and random.random() < 0.55)
                if use_arch:
                    pool = pop + archive
                    xr2 = pool[random.randrange(len(pool))]
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    xr2 = pop[r2]

                # mutation mix
                if frac_left > 0.55:
                    prand = 0.22
                    pbestguide = 0.70
                elif frac_left > 0.22:
                    prand = 0.14
                    pbestguide = 0.78
                else:
                    prand = 0.08
                    pbestguide = 0.70

                u = random.random()
                if u < pbestguide:
                    v = [xi[d] + F * (pb[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]
                elif u < pbestguide + prand:
                    a = random.randrange(pop_size)
                    b = random.randrange(pop_size)
                    cidx = random.randrange(pop_size)
                    xa, xb, xc = pop[a], pop[b], pop[cidx]
                    v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]
                else:
                    v = [xi[d] + F * (xbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

                if converged and random.random() < 0.35:
                    j = random.randrange(dim)
                    if spans[j] > 0:
                        v[j] += random.gauss(0.0, 0.03 * spans[j])

                jrand = random.randrange(dim)
                uvec = xi[:]
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        uvec[d] = v[d]

                uvec = reflect_into_bounds(uvec)
                fu = float(func(uvec))

                if fu <= fi:
                    archive.append(xi[:])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))

                    new_pop[i] = uvec
                    new_fit[i] = fu

                    S_F.append(F)
                    S_CR.append(CR)
                    df = abs(fi - fu)
                    dF.append(df if df > 0.0 else 1e-16)
                    update_best(uvec, fu)

            pop, fit = new_pop, new_fit

            # SHADE memory update
            if S_F:
                wsum = sum(dF)
                inv = 1.0 / wsum if wsum > 0 else 1.0 / len(dF)
                weights = [(df * inv) if wsum > 0 else inv for df in dF]

                num = 0.0
                den = 0.0
                for w, f in zip(weights, S_F):
                    num += w * f * f
                    den += w * f
                F_new = num / den if den > 0 else (sum(S_F) / len(S_F))

                CR_new = 0.0
                for w, cr in zip(weights, S_CR):
                    CR_new += w * cr

                MF[hist_idx] = (1.0 - c) * MF[hist_idx] + c * F_new
                MCR[hist_idx] = (1.0 - c) * MCR[hist_idx] + c * CR_new
                hist_idx = (hist_idx + 1) % H

            # if converged, shift budget to stronger local engines (instead of only injecting diversity)
            remaining = deadline - time.time()
            if best_x is not None and remaining > 0.02:
                if remaining <= endgame:
                    # endgame: ADG bursts + rotcoord, then maybe NM if small dim
                    bursts = 2 if dim > 25 else 3
                    slice_each = min(0.08, 0.26 * remaining / max(1, bursts))
                    for _ in range(bursts):
                        if time.time() >= deadline:
                            return best
                        bx, bf = local_adaptive_gaussian(best_x, best, slice_each)
                        update_best(bx, bf)
                        bx, bf = local_search_rotcoord(best_x, best, min(0.04, 0.12 * remaining))
                        update_best(bx, bf)
                    if dim <= 28 and (deadline - time.time()) > 0.05:
                        bx, bf = nelder_mead(best_x, best, min(0.11, 0.28 * (deadline - time.time())))
                        update_best(bx, bf)
                else:
                    # mid-game polish
                    if converged and random.random() < 0.16:
                        bx, bf = local_adaptive_gaussian(best_x, best, min(0.08, 0.12 * remaining))
                        update_best(bx, bf)
                    elif random.random() < 0.07:
                        bx, bf = local_search_rotcoord(best_x, best, min(0.06, 0.10 * remaining))
                        update_best(bx, bf)

            # diversity injection only when really stuck
            remaining = deadline - time.time()
            if converged and (time.time() - last_improve) > 0.55 * stall_seconds and remaining > 0.05:
                wcount = max(1, pop_size // 10)
                worst = idx_sorted[-wcount:]
                for t, wi in enumerate(worst):
                    if time.time() >= deadline:
                        return best
                    if best_x is not None and t < (wcount // 2):
                        x = best_x[:]
                        for d in range(dim):
                            if spans[d] > 0:
                                x[d] += random.gauss(0.0, 0.32 * spans[d])
                        x = reflect_into_bounds(x)
                    else:
                        x = rand_vec()
                    fx = float(func(x))
                    pop[wi] = x
                    fit[wi] = fx
                    update_best(x, fx)

            # stall handling: try heavy local before restarting
            if (time.time() - last_improve) > stall_seconds:
                remaining = deadline - time.time()
                if best_x is not None and remaining > 0.03:
                    bx, bf = local_adaptive_gaussian(best_x, best, min(0.16, 0.40 * remaining))
                    update_best(bx, bf)
                    bx, bf = local_search_rotcoord(best_x, best, min(0.10, 0.25 * remaining))
                    update_best(bx, bf)
                    if dim <= 28 and (deadline - time.time()) > 0.05:
                        bx, bf = nelder_mead(best_x, best, min(0.12, 0.30 * (deadline - time.time())))
                        update_best(bx, bf)
                break

    return best
