import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided DE variant:
      1) Multi-strategy DE (Ensemble): rand/1, current-to-pbest/1 (JADE-like), best/1
         with per-individual parameter memories (SHADE-lite).
      2) Rank-based p-best selection for exploitation without premature collapse.
      3) External archive to increase diversity (classic JADE/SHADE feature).
      4) Budget-aware local search: randomized coordinate line-search + step adaptation,
         invoked adaptively on improvements and periodically.
      5) Low-overhead, robust boundary handling: reflection with periodic folding.
      6) Time-aware control: caps on population, local search, and sorting.

    Returns:
      best fitness (float) found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    var_idx = [i for i in range(dim) if spans[i] > 0.0]
    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # ---------------- utilities ----------------
    def now():
        return time.time()

    def eval_f(x):
        return float(func(x))

    def reflect_inplace(x):
        # Reflect/fold into bounds; handles big jumps robustly.
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            w = hi - lo
            xi = x[i]
            if xi < lo or xi > hi:
                # fold into [lo, lo+2w), then reflect if needed
                xi = lo + (xi - lo) % (2.0 * w)
                if xi > hi:
                    xi = hi - (xi - hi)
                x[i] = xi
        return x

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] > 0.0:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Halton for init coverage
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # Fast-ish sampling of distinct indices
    def pick_distinct(n, exclude_set):
        # returns list of n distinct indices from [0,NP), excluding exclude_set
        out = []
        while len(out) < n:
            r = random.randrange(NP)
            if r in exclude_set:
                continue
            exclude_set.add(r)
            out.append(r)
        return out

    # Cauchy-like sampling for F (JADE) without external libs:
    def cauchy(mu, gamma=0.1):
        # inverse CDF: mu + gamma * tan(pi*(u-0.5))
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def clamp01(x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # ---------------- local search ----------------
    def local_refine(x_best, f_best, budget, base_step_frac=0.06):
        """
        Randomized coordinate line-search with adaptive step.
        Very cheap: tries +/- step on a few dims, with step decay.
        """
        if budget <= 0:
            return f_best, x_best
        x = x_best[:]
        fx = f_best

        # prioritize wider dims but randomize
        dims = var_idx[:]
        dims.sort(key=lambda i: spans[i], reverse=True)
        # random shuffle tail for robustness
        if len(dims) > 8:
            tail = dims[4:]
            random.shuffle(tail)
            dims = dims[:4] + tail

        step = [0.0] * dim
        for j in var_idx:
            step[j] = base_step_frac * spans[j]

        used = 0
        stagn_rounds = 0
        while used < budget and now() < deadline:
            improved = False
            for j in dims:
                if used >= budget or now() >= deadline:
                    break
                sj = step[j]
                if sj <= 0.0:
                    continue

                xj = x[j]
                # try +, -
                best_cand = None
                best_fc = fx
                for sgn in (1.0, -1.0):
                    cand = x[:]
                    cand[j] = xj + sgn * sj
                    reflect_inplace(cand)
                    fc = eval_f(cand)
                    used += 1
                    if fc < best_fc:
                        best_fc = fc
                        best_cand = cand
                    if used >= budget or now() >= deadline:
                        break

                if best_cand is not None:
                    x, fx = best_cand, best_fc
                    improved = True
                    # slight step increase on success
                    step[j] *= 1.2
                else:
                    # shrink on failure
                    step[j] *= 0.6

            if improved:
                stagn_rounds = 0
            else:
                stagn_rounds += 1
                # global shrink if no progress
                for j in var_idx:
                    step[j] *= 0.7
                if stagn_rounds >= 2:
                    break

        return fx, x

    # ---------------- initialization ----------------
    # population size: balance exploration/exploitation for unknown func eval cost
    NP = int(max(22, min(160, 10 + 7 * dim)))
    if max_time <= 0.5:
        NP = min(NP, 36)
    if max_time <= 1.0:
        NP = min(NP, 60)

    pop = []
    fit = []
    x_best = None
    f_best = float("inf")

    # init mix: halton, random, opposition, and some around mid-box
    mid = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    for i in range(NP):
        if now() >= deadline:
            return f_best if x_best is not None else float(eval_f(rand_point()))

        r = i % 5
        if r == 0:
            x = halton_point()
        elif r == 1:
            x = rand_point()
        elif r == 2:
            x = halton_point()
            xo = opposite_point(x)
            reflect_inplace(x)
            reflect_inplace(xo)
            fx = eval_f(x)
            if now() >= deadline:
                return min(f_best, fx)
            fo = eval_f(xo)
            x, fx = (xo, fo) if fo < fx else (x, fx)
            pop.append(x)
            fit.append(fx)
            if fx < f_best:
                f_best, x_best = fx, x[:]
            continue
        elif r == 3:
            # gaussian around middle
            x = mid[:]
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.25 * spans[j])
        else:
            # gaussian around a random halton point
            x = halton_point()
            for j in var_idx:
                x[j] += random.gauss(0.0, 0.18 * spans[j])

        reflect_inplace(x)
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # archive for JADE-like mutation
    archive = []
    archive_max = NP  # typical

    # SHADE-lite memories for F and CR
    H = max(6, min(25, dim + 6))
    M_F = [0.5] * H
    M_CR = [0.8] * H
    mem_idx = 0

    # per-individual strategy probabilities (ensemble)
    # s0: current-to-pbest/1, s1: rand/1, s2: best/1
    p_s = [0.55, 0.30, 0.15]

    # stagnation control
    last_best = f_best
    stagn = 0
    gen = 0

    # ---------------- main loop ----------------
    while now() < deadline:
        gen += 1

        # rank individuals for p-best selection (partial sort via indices)
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda k: fit[k])
        best_idx = idx_sorted[0]
        if fit[best_idx] < f_best:
            f_best = fit[best_idx]
            x_best = pop[best_idx][:]

        # occasional local refinement
        if gen % 7 == 0 and now() < deadline:
            fb2, xb2 = local_refine(x_best, f_best, budget=2 * len(var_idx) + 10, base_step_frac=0.08)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]

        # p in p-best: choose from top p% (JADE); vary with time/stagnation
        # more exploitation later
        time_left = max(0.0, deadline - now())
        # heuristic schedule: p from 0.25 -> 0.08
        frac = time_left / max(1e-9, float(max_time))
        p_top = 0.08 + 0.17 * frac
        p_top = max(0.05, min(0.3, p_top))
        p_num = max(2, int(p_top * NP))

        # generation bookkeeping for memory update (Lehmer mean)
        S_F = []
        S_CR = []
        dF = []  # improvements

        # create trial pop
        for i in range(NP):
            if now() >= deadline:
                return f_best

            # sample F and CR from memory
            r_mem = random.randrange(H)
            Fi = cauchy(M_F[r_mem], gamma=0.12)
            # resample if invalid
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(M_F[r_mem], gamma=0.12)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.2
            if Fi > 1.0:
                Fi = 1.0

            CRi = random.gauss(M_CR[r_mem], 0.12)
            CRi = clamp01(CRi)

            # select strategy
            u = random.random()
            if u < p_s[0]:
                strat = 0
            elif u < p_s[0] + p_s[1]:
                strat = 1
            else:
                strat = 2

            xi = pop[i]

            # choose pbest
            pbest = pop[idx_sorted[random.randrange(p_num)]]

            # choose r1 from pop excluding i; choose r2 from pop+archive excluding i,r1
            excl = {i}
            r1 = pick_distinct(1, excl)[0]
            x_r1 = pop[r1]

            # for r2: union size
            use_arch = (len(archive) > 0) and (random.random() < 0.5)
            if use_arch:
                # pick from combined by probability
                if random.random() < (len(archive) / (len(archive) + NP)):
                    x_r2 = archive[random.randrange(len(archive))]
                else:
                    # pick from pop excluding i and r1
                    r2 = pick_distinct(1, excl)[0]
                    x_r2 = pop[r2]
            else:
                r2 = pick_distinct(1, excl)[0]
                x_r2 = pop[r2]

            # mutation
            v = xi[:]  # base
            if strat == 0:
                # current-to-pbest/1: v = x + F*(pbest-x) + F*(r1-r2)
                for j in var_idx:
                    v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (x_r1[j] - x_r2[j])
            elif strat == 1:
                # rand/1: v = r1 + F*(r2 - r3) with r2 from pop and r3 from pop/archive
                # pick r2_pop, r3_comb distinct from r1 and i
                excl2 = {i, r1}
                r2p = pick_distinct(1, excl2)[0]
                x2 = pop[r2p]
                # r3 from combined
                if len(archive) > 0 and random.random() < 0.5:
                    x3 = archive[random.randrange(len(archive))]
                else:
                    r3p = pick_distinct(1, excl2)[0]
                    x3 = pop[r3p]
                for j in var_idx:
                    v[j] = x_r1[j] + Fi * (x2[j] - x3[j])
            else:
                # best/1: v = best + F*(r1 - r2)
                xb = x_best
                for j in var_idx:
                    v[j] = xb[j] + Fi * (x_r1[j] - x_r2[j])

            # crossover (binomial)
            uvec = xi[:]
            jrand = var_idx[random.randrange(len(var_idx))]
            for j in var_idx:
                if random.random() < CRi or j == jrand:
                    uvec[j] = v[j]

            reflect_inplace(uvec)
            fu = eval_f(uvec)

            # selection with archive update
            if fu <= fit[i]:
                # add parent to archive (if strictly better or equal, still helps diversity)
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                old = fit[i]
                pop[i] = uvec
                fit[i] = fu

                if fu < f_best:
                    f_best = fu
                    x_best = uvec[:]

                # collect successful params for memory update
                imp = abs(old - fu)
                if imp <= 0.0:
                    imp = 1e-12
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(imp)

        # update memories (SHADE style: weighted Lehmer mean for F, weighted mean for CR)
        if dF:
            wsum = sum(dF)
            if wsum <= 0.0:
                wsum = 1e-12
            # weighted means
            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            numF = 0.0
            denF = 0.0
            numCR = 0.0
            for w, Fi, CRi in zip(dF, S_F, S_CR):
                ww = w / wsum
                numF += ww * (Fi * Fi)
                denF += ww * Fi
                numCR += ww * CRi
            if denF <= 1e-12:
                new_MF = M_F[mem_idx]
            else:
                new_MF = numF / denF
            new_MCR = numCR

            # keep reasonable ranges
            M_F[mem_idx] = min(1.0, max(0.05, new_MF))
            M_CR[mem_idx] = min(1.0, max(0.0, new_MCR))
            mem_idx = (mem_idx + 1) % H

        # adapt strategy probabilities based on stagnation (more exploration when stuck)
        if f_best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = f_best
            stagn = 0
            # slightly favor exploitation when improving
            p_s[0] = min(0.72, p_s[0] + 0.02)
            p_s[1] = max(0.18, p_s[1] - 0.015)
            p_s[2] = 1.0 - p_s[0] - p_s[1]
        else:
            stagn += 1
            if stagn >= 6:
                stagn = 0
                # inject exploration: broaden memories and reinit worst fraction
                p_s[0] = max(0.40, p_s[0] - 0.07)
                p_s[1] = min(0.45, p_s[1] + 0.06)
                p_s[2] = 1.0 - p_s[0] - p_s[1]

                # reinitialize worst ~25%
                idx_sorted = list(range(NP))
                idx_sorted.sort(key=lambda k: fit[k], reverse=True)
                k_re = max(2, NP // 4)
                for t in range(k_re):
                    if now() >= deadline:
                        return f_best
                    k = idx_sorted[t]
                    # mixture: around best, halton, random
                    rr = random.random()
                    if rr < 0.55 and x_best is not None:
                        x = x_best[:]
                        for j in var_idx:
                            x[j] += random.gauss(0.0, 0.22 * spans[j])
                    elif rr < 0.80:
                        x = halton_point()
                    else:
                        x = rand_point()
                    reflect_inplace(x)
                    fx = eval_f(x)
                    pop[k] = x
                    fit[k] = fx
                    if fx < f_best:
                        f_best = fx
                        x_best = x[:]

                # quick local search after shake-up (small)
                if now() < deadline and x_best is not None:
                    fb2, xb2 = local_refine(x_best, f_best, budget=len(var_idx) + 6, base_step_frac=0.10)
                    if fb2 < f_best:
                        f_best, x_best = fb2, xb2[:]

    return f_best
