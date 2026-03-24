import random, time, math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libraries).

    Main upgrades vs the provided variants:
      - Strong global search: L-SHADE/JADE style DE (current-to-pbest/1 + archive + success-history F/CR)
      - Population reduction (LPSR): gradually reduces pop size to focus search
      - Stronger exploitation: fast diagonal-CMA-like sampling around best + lightweight coordinate/quadratic steps
      - Better time usage: dynamic block scheduling, frequent time checks, low overhead
      - Robust restarts: triggered by stagnation, with near-best + global reseeding

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ----------------- helpers -----------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect(v, lo, hi):
        # reflection keeps steps meaningful; handles mild out-of-bounds well
        if v < lo:
            v = lo + (lo - v)
            if v > hi:
                v = lo
        elif v > hi:
            v = hi - (v - hi)
            if v < lo:
                v = hi
        return v

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # approx N(0,1) (cheap, no random.gauss)
    def gauss01():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    # very small cache (quantized) to avoid repeats in local search / CMA samples
    cache = {}
    CACHE_MAX = 25000
    def eval_f(x):
        if dim <= 14:
            key = []
            for j in range(dim):
                lo, hi = bounds[j]
                s = hi - lo
                if s <= 0.0:
                    q = 0
                else:
                    q = int(8192.0 * (x[j] - lo) / s + 0.5)
                    if q < 0: q = 0
                    if q > 8192: q = 8192
                key.append(q)
            key = tuple(key)
            v = cache.get(key)
            if v is not None:
                return v
            fx = float(func(x))
            if len(cache) < CACHE_MAX:
                cache[key] = fx
            return fx
        return float(func(x))

    # ----------------- edge cases -----------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    spans = []
    for j in range(dim):
        lo, hi = bounds[j]
        s = hi - lo
        if not (s > 0.0):
            x = [0.5 * (b[0] + b[1]) for b in bounds]
            return eval_f(x)
        spans.append(s)

    # ----------------- initialization -----------------
    # initial population (some stratified-ish seeding)
    pop0 = max(24, min(110, 12 * dim))
    pop_min = max(10, min(40, 4 * dim + 6))
    pop_size = pop0

    pop = []
    fit = []

    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if now() >= deadline - eps_time:
            return best
        x = [0.0] * dim
        if i < pop_size // 2:
            for j in range(dim):
                lo, hi = bounds[j]
                u = (i + random.random()) / max(1, pop_size // 2)
                u += 0.13 * (random.random() - 0.5)
                u -= math.floor(u)  # wrap to [0,1)
                x[j] = lo + (hi - lo) * u
        else:
            x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    # archive for DE diversity
    archive = []
    archive_max = pop_size

    # SHADE memories
    H = 6
    M_F = [0.55] * H
    M_CR = [0.75] * H
    mem_k = 0

    # pbest fraction
    pmin = 2.0 / pop_size
    pmax = 0.2

    def pick_top_pool(pfrac):
        k = max(2, int(pop_size * pfrac))
        chosen, used = [], set()
        for _ in range(k):
            bi, bf = None, float("inf")
            for idx in range(pop_size):
                if idx in used:
                    continue
                f = fit[idx]
                if f < bf:
                    bf, bi = f, idx
            used.add(bi)
            chosen.append(bi)
        return chosen

    # ----------------- local exploitation: diagonal-CMA-ish + coord/quadratic -----------------
    # Diagonal variances in normalized units; mean at best
    mean = best_x[:] if best_x is not None else pop[0][:]
    C = [1.0] * dim
    sigma = 0.20

    # local step scale for coordinate moves
    ls_sigma = 0.10
    ls_succ = 0
    ls_tr = 0

    def local_improve(bx, bf, tries):
        nonlocal best, best_x, ls_sigma, ls_succ, ls_tr
        xbest = bx[:]
        fbest = bf

        for _ in range(tries):
            if now() >= deadline - eps_time:
                break
            ls_tr += 1

            x = xbest[:]
            if random.random() < 0.70:
                j = random.randrange(dim)
                step = gauss01() * spans[j] * ls_sigma
                x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])
            else:
                for j in range(dim):
                    step = gauss01() * spans[j] * (ls_sigma * 0.35)
                    x[j] = reflect(x[j] + step, bounds[j][0], bounds[j][1])

            fx = eval_f(x)
            if fx < fbest:
                xbest, fbest = x, fx
                ls_succ += 1
                if fx < best:
                    best, best_x = fx, x[:]

            # occasional 1D quadratic interpolation around current best on a random coordinate
            if now() >= deadline - eps_time:
                break
            if random.random() < 0.10:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                a = xbest[j]
                delta = spans[j] * max(1e-7, ls_sigma * 0.45)
                x1 = xbest[:]; x1[j] = reflect(a - delta, lo, hi)
                x2 = xbest[:]; x2[j] = reflect(a + delta, lo, hi)
                f1 = eval_f(x1)
                if now() >= deadline - eps_time:
                    break
                f2 = eval_f(x2)

                denom = (f1 - 2.0 * fbest + f2)
                if abs(denom) > 1e-18:
                    t = 0.5 * (f1 - f2) / denom
                    if t < -1.5: t = -1.5
                    if t >  1.5: t =  1.5
                    xq = xbest[:]
                    xq[j] = reflect(a + t * delta, lo, hi)
                    fq = eval_f(xq)
                    if fq < fbest:
                        xbest, fbest = xq, fq
                        if fq < best:
                            best, best_x = fq, xq[:]

        # adapt ls_sigma with a small 1/5 success rule
        if ls_tr >= 30:
            rate = ls_succ / float(ls_tr)
            if rate > 0.22:
                ls_sigma *= 1.12
            else:
                ls_sigma *= 0.88
            if ls_sigma < 1e-8: ls_sigma = 1e-8
            if ls_sigma > 0.35: ls_sigma = 0.35
            ls_succ = 0
            ls_tr = 0

        return xbest, fbest

    def cma_diag_step(iters, lam):
        nonlocal mean, C, sigma, best, best_x
        # sample lam candidates, update mean and C (diagonal) from best half
        mu = max(2, lam // 2)
        # log weights
        w = [math.log(mu + 0.5) - math.log(i) for i in range(1, mu + 1)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        c_up = 0.18

        for _ in range(iters):
            if now() >= deadline - eps_time:
                return
            cand = []
            for _k in range(lam):
                if now() >= deadline - eps_time:
                    return
                x = [0.0] * dim
                for j in range(dim):
                    z = gauss01() * math.sqrt(C[j])
                    xj = mean[j] + (sigma * spans[j]) * z
                    x[j] = reflect(xj, bounds[j][0], bounds[j][1])
                fx = eval_f(x)
                cand.append((fx, x))
            cand.sort(key=lambda t: t[0])

            if cand[0][0] < best:
                best = cand[0][0]
                best_x = cand[0][1][:]

            old_mean = mean[:]
            new_mean = mean[:]
            for j in range(dim):
                s = 0.0
                for i in range(mu):
                    s += w[i] * cand[i][1][j]
                new_mean[j] = s

            for j in range(dim):
                denom = sigma * spans[j]
                if denom < 1e-18:
                    denom = 1e-18
                v = 0.0
                m0 = old_mean[j]
                for i in range(mu):
                    d = (cand[i][1][j] - m0) / denom
                    v += w[i] * (d * d)
                C[j] = max(1e-12, (1.0 - c_up) * C[j] + c_up * v)

            mean = new_mean

            # sigma adaptation (simple)
            if cand[mu - 1][0] <= cand[0][0] + 1e-12:
                sigma *= 1.02
            else:
                sigma *= 0.90
            if sigma < 1e-12: sigma = 1e-12
            if sigma > 0.7: sigma = 0.7

    # ----------------- main loop -----------------
    last_best = best
    stagnant = 0
    gen = 0

    while now() < deadline - eps_time:
        gen += 1

        # linear population size reduction (LPSR-ish)
        if pop_size > pop_min:
            # reduce slowly; do not reduce too aggressively in early stage
            target = int(pop0 - (pop0 - pop_min) * min(1.0, (now() - t0) / max(1e-9, max_time)))
            if target < pop_size:
                # remove worst individuals
                rm = pop_size - target
                for _ in range(rm):
                    wi, wf = None, -float("inf")
                    for idx in range(pop_size):
                        f = fit[idx]
                        if f > wf:
                            wf, wi = f, idx
                    pop[wi] = pop[-1]; fit[wi] = fit[-1]
                    pop.pop(); fit.pop()
                    pop_size -= 1
                archive_max = pop_size
                if len(archive) > archive_max:
                    archive = archive[:archive_max]
                pmin = 2.0 / pop_size

        # DE generation
        p = random.uniform(pmin, pmax)
        pbest_pool = pick_top_pool(p)

        S_F, S_CR, dF = [], [], []

        for i in range(pop_size):
            if now() >= deadline - eps_time:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            CRi = muCR + 0.1 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 6:
                Fi = muF + 0.1 * math.tan(math.pi * (random.random() - 0.5))
                tries += 1
            if Fi <= 0.0: Fi = 0.08
            if Fi > 1.0: Fi = 1.0

            pbest = pop[random.choice(pbest_pool)]

            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            if archive and random.random() < 0.5:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = r1
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                xr2 = pop[r2]

            xr1 = pop[r1]

            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = reflect(vj, bounds[j][0], bounds[j][1])

            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                u[j] = v[j] if (random.random() < CRi or j == jrand) else xi[j]

            fu = eval_f(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_max:
                    k = random.randrange(len(archive))
                    archive[k] = archive[-1]
                    archive.pop()

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = u[:]
                    mean = best_x[:]  # sync CMA mean fast

                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fi - fu))

        # update memories
        if S_F:
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            num = 0.0
            den = 0.0
            for ww, ff in zip(weights, S_F):
                num += ww * ff * ff
                den += ww * ff
            new_MF = (num / den) if den > 1e-12 else M_F[mem_k]

            new_MCR = 0.0
            for ww, cc in zip(weights, S_CR):
                new_MCR += ww * cc

            M_F[mem_k] = clip(new_MF, 0.05, 1.0)
            M_CR[mem_k] = clip(new_MCR, 0.0, 1.0)
            mem_k = (mem_k + 1) % H

        # exploitation blocks (time-adaptive)
        if best_x is not None and now() < deadline - eps_time:
            # small local improvement almost every generation
            budget = 2 + dim // 5
            best_x, best = local_improve(best_x, best, budget)

            # occasional CMA-ish step (strong in smooth basins)
            if random.random() < 0.35:
                lam = max(10, min(40, 6 + 2 * int(math.log(dim + 1.0) * 6)))
                iters = 1
                cma_diag_step(iters, lam)

        # stagnation handling / restart injection
        if best < last_best - 1e-12:
            last_best = best
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= 8 and now() < deadline - eps_time:
            # replace worst 25% with half near-best, half random; reset archive
            k = max(2, pop_size // 4)
            worst = []
            used = set()
            for _ in range(k):
                wi, wf = None, -float("inf")
                for idx in range(pop_size):
                    if idx in used:
                        continue
                    f = fit[idx]
                    if f > wf:
                        wf, wi = f, idx
                used.add(wi)
                worst.append(wi)

            for t, idx in enumerate(worst):
                if now() >= deadline - eps_time:
                    return best
                if best_x is not None and t < k // 2:
                    x = best_x[:]
                    rad = 0.22
                    for j in range(dim):
                        x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad,
                                    bounds[j][0], bounds[j][1])
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    mean = best_x[:]

            archive = []
            stagnant = 0
            sigma = min(0.35, sigma * 1.2)
            ls_sigma = min(0.22, ls_sigma * 1.15)

    return best
