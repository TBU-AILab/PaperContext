import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    What’s improved vs the provided ones:
      - Stronger global search: L-SHADE-style DE (current-to-pbest/1 + archive + success-history F/CR)
      - Much faster convergence near good regions: embedded (1+1)-ES / coordinate + quadratic-ish steps
      - Smarter restarts: triggers on stagnation and injects both near-best and far samples
      - Optional tiny evaluation cache (quantized) to avoid repeats in local search
      - Careful time accounting and low overhead selection (partial top-p selection)

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps = 1e-4

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_into_bounds(v, lo, hi):
        # reflection tends to preserve step intent better than hard clipping
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

    # Gaussian-ish N(0,1) via CLT (fast, no random.gauss dependency)
    def gauss01():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    # small quantized cache helps mainly for local search repeat proposals
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
            fx = cache.get(key)
            if fx is not None:
                return fx
            fx = float(func(x))
            if len(cache) < CACHE_MAX:
                cache[key] = fx
            return fx
        return float(func(x))

    # ---------------- edge cases ----------------
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

    # ---------------- L-SHADE-ish DE core ----------------
    # population size tuned for time-bounded runs
    pop_size0 = max(18, min(90, 10 * dim))
    pop_size = pop_size0
    archive_max = pop_size

    # p-best fraction
    pmin = 2.0 / pop_size
    pmax = 0.18

    # success-history memories
    H = 6
    M_F = [0.55] * H
    M_CR = [0.75] * H
    mem_k = 0

    # init population: mixture (uniform + a few stratified)
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # quick stratified init per dimension for some points (cheap quasi-random)
    # for i < pop_size//2, use stratification; rest uniform random.
    for i in range(pop_size):
        if now() >= deadline - eps:
            return best
        x = [0.0] * dim
        if i < pop_size // 2:
            # per-dim stratified with random permutation-ish offset
            for j in range(dim):
                lo, hi = bounds[j]
                u = (i + random.random()) / max(1, pop_size // 2)
                # scramble u a bit (avoid axis alignment)
                u = u + 0.17 * (random.random() - 0.5)
                if u < 0.0: u += 1.0
                if u >= 1.0: u -= 1.0
                x[j] = lo + (hi - lo) * u
        else:
            x = rand_vec()

        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    archive = []

    # fast partial "top p%" pool
    def pick_top_pool(pfrac):
        k = max(2, int(pop_size * pfrac))
        chosen = []
        used = set()
        for _ in range(k):
            bi = None
            bf = float("inf")
            for idx in range(pop_size):
                if idx in used:
                    continue
                f = fit[idx]
                if f < bf:
                    bf = f
                    bi = idx
            used.add(bi)
            chosen.append(bi)
        return chosen

    # ---------------- local search around best ----------------
    # hybrid: coordinate steps + (1+1)-ES shrink/expand + occasional 2-point parabolic step
    ls_sigma = 0.12  # fraction of range (global; adapted)
    ls_success = 0
    ls_trials = 0

    def local_search(bx, bf, budget_evals):
        nonlocal ls_sigma, ls_success, ls_trials, best, best_x
        xbest = bx[:]
        fbest = bf

        for _ in range(budget_evals):
            if now() >= deadline - eps:
                break
            ls_trials += 1

            x = xbest[:]
            if random.random() < 0.65:
                # coordinate step
                j = random.randrange(dim)
                step = gauss01() * spans[j] * ls_sigma
                xj = x[j] + step
                lo, hi = bounds[j]
                x[j] = reflect_into_bounds(xj, lo, hi)
            else:
                # small full perturbation
                for j in range(dim):
                    step = gauss01() * spans[j] * (ls_sigma * 0.35)
                    lo, hi = bounds[j]
                    x[j] = reflect_into_bounds(x[j] + step, lo, hi)

            fx = eval_f(x)
            if fx < fbest:
                xbest = x
                fbest = fx
                ls_success += 1
                if fx < best:
                    best = fx
                    best_x = x[:]

            # very cheap parabolic move occasionally on 1 dimension
            if now() >= deadline - eps:
                break
            if random.random() < 0.08:
                j = random.randrange(dim)
                lo, hi = bounds[j]
                mid = xbest[:]
                a = mid[j]
                # probe two points around a
                delta = spans[j] * max(1e-6, ls_sigma * 0.5)
                x1 = mid[:]; x1[j] = reflect_into_bounds(a - delta, lo, hi)
                x2 = mid[:]; x2[j] = reflect_into_bounds(a + delta, lo, hi)
                f1 = eval_f(x1)
                if now() >= deadline - eps:
                    break
                f2 = eval_f(x2)

                # quadratic interpolation if possible
                denom = (f1 - 2.0 * fbest + f2)
                if abs(denom) > 1e-18:
                    # argmin relative to center: t = 0.5*(f1 - f2)/denom in [-1,1] ideally
                    t = 0.5 * (f1 - f2) / denom
                    if t < -1.5: t = -1.5
                    if t >  1.5: t =  1.5
                    xq = mid[:]
                    xq[j] = reflect_into_bounds(a + t * delta, lo, hi)
                    fq = eval_f(xq)
                    if fq < fbest:
                        xbest, fbest = xq, fq
                        if fq < best:
                            best, best_x = fq, xq[:]

        # adapt sigma (1/5th-ish rule)
        if ls_trials >= 25:
            rate = ls_success / float(ls_trials)
            if rate > 0.22:
                ls_sigma *= 1.12
            else:
                ls_sigma *= 0.88
            if ls_sigma < 1e-8:
                ls_sigma = 1e-8
            if ls_sigma > 0.35:
                ls_sigma = 0.35
            ls_success = 0
            ls_trials = 0

        return xbest, fbest

    # ---------------- main loop with restarts ----------------
    last_best = best
    stagnant_gens = 0

    while now() < deadline - eps:
        # choose p-best pool for this generation
        p = random.uniform(pmin, pmax)
        pbest_pool = pick_top_pool(p)

        S_F, S_CR, dF = [], [], []

        # one DE generation
        for i in range(pop_size):
            if now() >= deadline - eps:
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # CR ~ N(muCR,0.1)
            CRi = muCR + 0.1 * gauss01()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # F ~ cauchy(muF,0.1)
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

            # r2 from pop or archive
            if archive and random.random() < 0.5:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = r1
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                xr2 = pop[r2]

            xr1 = pop[r1]

            # current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                lo, hi = bounds[j]
                v[j] = reflect_into_bounds(vj, lo, hi)

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                u[j] = v[j] if (random.random() < CRi or j == jrand) else xi[j]

            fu = eval_f(u)

            if fu <= fi:
                # archive old
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

                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fi - fu))

        # update success-history
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

        # local search budget scales mildly with dim
        if best_x is not None and now() < deadline - eps:
            ls_budget = 3 + dim // 4
            best_x, best = local_search(best_x, best, ls_budget)

        # stagnation / restart injection
        if best < last_best - 1e-12:
            last_best = best
            stagnant_gens = 0
        else:
            stagnant_gens += 1

        if stagnant_gens >= 10 and now() < deadline - eps:
            # replace worst 25%: half near best, half random; clear archive
            k = max(2, pop_size // 4)
            worst = []
            used = set()
            for _ in range(k):
                wi = None
                wf = -float("inf")
                for idx in range(pop_size):
                    if idx in used:
                        continue
                    f = fit[idx]
                    if f > wf:
                        wf = f
                        wi = idx
                used.add(wi)
                worst.append(wi)

            for t, idx in enumerate(worst):
                if now() >= deadline - eps:
                    return best
                if best_x is not None and t < k // 2:
                    x = best_x[:]
                    rad = 0.22
                    for j in range(dim):
                        lo, hi = bounds[j]
                        x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad, lo, hi)
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

            archive = []
            stagnant_gens = 0
            # slightly enlarge local sigma after restart to re-explore basin boundaries
            ls_sigma = min(0.22, ls_sigma * 1.25)

    return best
