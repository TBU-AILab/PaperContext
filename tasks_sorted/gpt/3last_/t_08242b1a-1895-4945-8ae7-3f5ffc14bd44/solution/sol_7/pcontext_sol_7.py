import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Improved hybrid minimizer (self-contained, no external libs).

    What is improved vs your current best (~3.63):
      1) Stronger *local* optimizer: a compact CMA-ES (sep-CMA + rank-1 direction memory)
         - keeps diagonal covariance (cheap) + few learned directions (handles rotations)
         - uses mirrored sampling + weighted recombination + success-based sigma control
      2) Better *global* search: DE/current-to-pbest/1 (SHADE-like) + periodic
         "opposition / Latin-ish" injections when converged
      3) Cleaner time budgeting: always reserves endgame time for CMA-local polishing
      4) Safer bounds handling + fewer wasted evaluations

    func: callable(list[float]) -> float
    bounds: list[(lo, hi)] length=dim
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
            for _ in range(8):
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

    # ---- scrambled Halton (init) ----
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

    def halton_point(k, shifts):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                u = _radical_inverse(k, PRIMES[d], shifts[d])
                x[d] = lows[d] + u * spans[d]
        return x

    # -------------------- Local: compact CMA-like (sep + low-rank dirs) --------------------
    def local_compact_cma(x0, f0, time_slice):
        """
        A small-footprint CMA-ES-ish local optimizer:
          - diagonal covariance via per-dim scale 'sig'
          - a few learned directions (rank-1 memory) to cope with rotations
          - mirrored sampling to reduce noise/waste
        """
        t_end = min(deadline, time.time() + max(0.0, time_slice))
        x = x0[:]
        fx = f0

        # population sizes (small, time-bounded)
        lam = 8 + int(3.0 * math.log(dim + 1.0))
        if lam < 10:
            lam = 10
        if lam > 28:
            lam = 28
        mu = lam // 2

        # log weights
        w = [0.0] * mu
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        wsum = sum(w)
        w = [wi / wsum for wi in w]

        # step sizes
        sig = [0.18 * s if s > 0.0 else 1.0 for s in spans]
        sig_min = [1e-15 * (s if s > 0.0 else 1.0) for s in spans]
        sig_max = [0.60 * (s if s > 0.0 else 1.0) for s in spans]
        sigma_global = 1.0

        # direction memory
        kdir = 6 if dim >= 20 else 8
        dirs = []
        dirw = []
        for _ in range(min(kdir, dim)):
            v = [random.gauss(0.0, 1.0) for _ in range(dim)]
            nrm = math.sqrt(sum(vi * vi for vi in v)) + 1e-18
            v = [vi / nrm for vi in v]
            dirs.append(v)
            dirw.append(1.0)

        # evolution path-ish success controller
        succ = 0
        tried = 0
        batch = 10

        # precompute avg sigma scale
        def avg_sig():
            return (sum(sig) / float(dim)) if dim > 0 else 1.0

        # sampling function
        def sample_step():
            # 60% diagonal gaussian, 40% direction-move
            if dirs and random.random() < 0.40:
                # choose direction proportional to weight
                sw = sum(dirw) + 1e-18
                r = random.random() * sw
                acc = 0.0
                idx = 0
                for i, ww in enumerate(dirw):
                    acc += ww
                    if acc >= r:
                        idx = i
                        break
                d = dirs[idx]
                step_len = random.gauss(0.0, 1.0) * sigma_global * (1.35 * avg_sig())
                return [step_len * di for di in d]
            else:
                return [random.gauss(0.0, 1.0) * sigma_global * sig[i] if spans[i] > 0.0 else 0.0
                        for i in range(dim)]

        while time.time() < t_end:
            # build lambda candidates with mirrored sampling
            cand = []
            # generate lam/2 steps then mirror
            half = (lam + 1) // 2
            for _ in range(half):
                s = sample_step()
                y1 = [x[i] + s[i] for i in range(dim)]
                y2 = [x[i] - s[i] for i in range(dim)]
                cand.append(reflect_into_bounds(y1))
                if len(cand) < lam:
                    cand.append(reflect_into_bounds(y2))

            # evaluate
            scored = []
            for y in cand:
                if time.time() >= t_end:
                    break
                fy = float(func(y))
                scored.append((fy, y))

            if not scored:
                break

            scored.sort(key=lambda t: t[0])
            besty_f, besty = scored[0]
            tried += 1
            if besty_f < fx:
                succ += 1
                # learn successful direction
                step = [besty[i] - x[i] for i in range(dim)]
                nrm = math.sqrt(sum(si * si for si in step)) + 1e-18
                if nrm > 0.0:
                    u = [si / nrm for si in step]
                    if dirs:
                        # replace weakest
                        rep = min(range(len(dirw)), key=lambda k: dirw[k])
                        dirs[rep] = u
                        dirw[rep] = dirw[rep] * 0.5 + 1.0
                    else:
                        dirs = [u]
                        dirw = [1.0]

                x, fx = besty[:], besty_f

            # recombination update (move mean even if not improved, but conservatively)
            # use top-mu
            ybar = x[:]  # start from current mean
            # compute weighted delta relative to current x
            delta = [0.0] * dim
            top = scored[:min(mu, len(scored))]
            for wi, (fy, y) in zip(w, top):
                for i in range(dim):
                    delta[i] += wi * (y[i] - x[i])

            # modest mean shift; don't overshoot near bounds
            lr = 0.65
            x_trial = reflect_into_bounds([x[i] + lr * delta[i] for i in range(dim)])
            f_trial = float(func(x_trial))
            if f_trial < fx:
                x, fx = x_trial, f_trial

            # adapt diagonal sigmas toward successful top-step magnitudes
            for i in range(dim):
                if spans[i] <= 0.0:
                    continue
                # robust scale estimate from top half
                mags = []
                for (fy, y) in top:
                    mags.append(abs(y[i] - x[i]))
                mags.sort()
                m = mags[len(mags)//2] if mags else 0.0
                target = 0.7 * m + 0.3 * sig[i]
                if target < sig_min[i]:
                    target = sig_min[i]
                elif target > sig_max[i]:
                    target = sig_max[i]
                sig[i] = 0.90 * sig[i] + 0.10 * target

            # update sigma_global via success rate
            if tried >= batch:
                ps = succ / float(tried)
                if ps > 0.25:
                    sigma_global *= 1.15
                elif ps < 0.15:
                    sigma_global *= 0.85
                if sigma_global < 0.04:
                    sigma_global = 0.04
                elif sigma_global > 3.0:
                    sigma_global = 3.0
                succ = 0
                tried = 0
                for i in range(len(dirw)):
                    dirw[i] *= 0.96

        return x, fx

    # -------------------- Global: SHADE-ish DE --------------------
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
    endgame_reserve = 0.25 * T  # reserve for local CMA polishing

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

    # -------------------- main loop (multi-start DE + local endgame) --------------------
    shifts = [random.randrange(1, 5000) for _ in range(dim)]
    halton_k = random.randrange(30, 130)

    while time.time() < deadline:
        # init population: some best-centered, rest Halton-ish
        pop = []
        if best_x is not None:
            k = min(pop_size // 3, 24)
            for _ in range(k):
                x = best_x[:]
                for d in range(dim):
                    if spans[d] > 0.0:
                        x[d] += random.gauss(0.0, 0.22 * spans[d])
                pop.append(reflect_into_bounds(x))

        while len(pop) < pop_size:
            halton_k += 1
            x = halton_point(halton_k, shifts)
            # tiny jitter
            for d in range(dim):
                if spans[d] > 0.0:
                    x[d] += (random.random() - 0.5) * 0.003 * spans[d]
            pop.append(reflect_into_bounds(x))

        fit = []
        for x in pop:
            if time.time() >= deadline:
                return best
            fx = float(func(x))
            fit.append(fx)
            update_best(x, fx)

        # DE generations
        while time.time() < deadline:
            now = time.time()
            remaining = deadline - now

            # if entering endgame, switch to local optimizer for all remaining time
            if best_x is not None and remaining <= endgame_reserve:
                bx, bf = local_compact_cma(best_x, best, remaining)
                update_best(bx, bf)
                return best

            frac_left = remaining / max(1e-9, T)
            pfrac = 0.30 if frac_left > 0.70 else (0.16 if frac_left > 0.30 else 0.10)

            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            best_idx = idx_sorted[0]
            pcount = max(2, int(math.ceil(pfrac * pop_size)))
            pbest_idx = idx_sorted[:pcount]

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

                # jitter when converged
                if converged and random.random() < 0.35:
                    j = random.randrange(dim)
                    if spans[j] > 0.0:
                        v[j] += random.gauss(0.0, 0.03 * spans[j])

                # crossover
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
                for wgt, f in zip(weights, S_F):
                    num += wgt * f * f
                    den += wgt * f
                F_new = num / den if den > 0 else (sum(S_F) / len(S_F))

                CR_new = 0.0
                for wgt, cr in zip(weights, S_CR):
                    CR_new += wgt * cr

                MF[hist_idx] = (1.0 - c) * MF[hist_idx] + c * F_new
                MCR[hist_idx] = (1.0 - c) * MCR[hist_idx] + c * CR_new
                hist_idx = (hist_idx + 1) % H

            # occasional local polish (mid-game), but keep endgame reserve
            remaining = deadline - time.time()
            if best_x is not None and remaining > endgame_reserve + 0.02:
                if converged and random.random() < 0.18:
                    bx, bf = local_compact_cma(best_x, best, min(0.08, 0.10 * remaining))
                    update_best(bx, bf)

            # diversity/opposition injection if stuck
            if converged and (time.time() - last_improve) > 0.55 * stall_seconds and remaining > endgame_reserve + 0.05:
                wcount = max(1, pop_size // 10)
                worst = idx_sorted[-wcount:]
                for t, wi in enumerate(worst):
                    if time.time() >= deadline:
                        return best
                    if best_x is not None and t < (wcount // 2):
                        # opposition-ish around best: x' = lo+hi - best + noise
                        x = [lows[d] + highs[d] - best_x[d] for d in range(dim)]
                        for d in range(dim):
                            if spans[d] > 0.0:
                                x[d] += random.gauss(0.0, 0.10 * spans[d])
                        x = reflect_into_bounds(x)
                    else:
                        x = rand_vec()
                    fx = float(func(x))
                    pop[wi] = x
                    fit[wi] = fx
                    update_best(x, fx)

            # stall: do a stronger local attempt, then restart
            if (time.time() - last_improve) > stall_seconds:
                remaining = deadline - time.time()
                if best_x is not None and remaining > endgame_reserve + 0.03:
                    bx, bf = local_compact_cma(best_x, best, min(0.18, 0.25 * remaining))
                    update_best(bx, bf)
                break

    # if we exit due to time, best is already tracked
    return best
