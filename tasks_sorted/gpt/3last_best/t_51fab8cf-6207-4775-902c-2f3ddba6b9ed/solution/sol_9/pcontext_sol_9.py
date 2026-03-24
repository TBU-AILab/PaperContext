import random
import time
import math

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    # reflection tends to work better than clip for local steps
    def reflect(v, lo, hi):
        if hi <= lo:
            return lo
        # reflect until in range
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
        return v

    def eval_f(x):
        return float(func(x))

    # ---------- scrambled Halton for seeding / restarts ----------
    def _vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v

    def _is_prime(p):
        if p < 2:
            return False
        if p % 2 == 0:
            return p == 2
        r = int(math.isqrt(p))
        q = 3
        while q <= r:
            if p % q == 0:
                return False
            q += 2
        return True

    primes = []
    p = 2
    while len(primes) < dim:
        if _is_prime(p):
            primes.append(p)
        p += 1

    scramble = [random.random() for _ in range(dim)]

    def halton_point(k):
        kk = k + 1
        x = [0.0] * dim
        for i in range(dim):
            u = _vdc(kk, primes[i]) + scramble[i]
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- init population (bigger + keep diversity) ----------
    best = float("inf")
    best_x = None

    pop_size = max(26, min(120, 14 * dim))
    init_n = max(pop_size, 22 * dim)

    pop, fit = [], []

    # add center point (often helps)
    if time.time() < deadline:
        xc = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        fc = eval_f(xc)
        pop.append(xc)
        fit.append(fc)
        best, best_x = fc, xc[:]

    k = 0
    while len(pop) < init_n and time.time() < deadline:
        x = halton_point(k) if (k < int(0.85 * init_n)) else rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        k += 1

    if not pop:
        return best

    # keep best pop_size
    if len(pop) > pop_size:
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fit[i])
        idx = idx[:pop_size]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    pop_size = len(pop)

    def argsort_fit(farr):
        idx = list(range(len(farr)))
        idx.sort(key=lambda i: farr[i])
        return idx

    # ---------- archive + SHADE memories ----------
    archive = []
    arch_max = pop_size

    H = max(8, min(40, 3 * dim))  # memory size
    M_F = [0.6] * H
    M_CR = [0.7] * H
    mem_idx = 0

    # ---------- elite stats for injections ----------
    def elite_stats(order, elite_k):
        m = elite_k
        mu = [0.0] * dim
        for t in range(m):
            x = pop[order[t]]
            for d in range(dim):
                mu[d] += x[d]
        invm = 1.0 / m
        for d in range(dim):
            mu[d] *= invm

        var = [0.0] * dim
        for t in range(m):
            x = pop[order[t]]
            for d in range(dim):
                dd = x[d] - mu[d]
                var[d] += dd * dd
        for d in range(dim):
            var[d] *= invm

        sigma = [math.sqrt(v) for v in var]
        for d in range(dim):
            s = sigma[d] if sigma[d] > 0.0 else 0.25 * spans[d]
            sigma[d] = max(1e-12 * spans[d], min(0.85 * spans[d], s))
        return mu, sigma

    # ---------- local refine (coordinate + sparse ES), reflection bounds ----------
    def local_refine(x0, f0, time_budget, sigma_hint):
        endt = min(deadline, time.time() + time_budget)
        x = x0[:]
        f = f0

        step_min = [1e-12 * spans[d] for d in range(dim)]
        step = [max(step_min[d], min(0.30 * spans[d], 0.80 * sigma_hint[d])) for d in range(dim)]
        sig = [max(step_min[d], 0.45 * step[d]) for d in range(dim)]

        lam = 6 if dim <= 10 else 8 if dim <= 24 else 10
        stall = 0
        coords_all = list(range(dim)) if dim <= 22 else None

        while time.time() < endt:
            improved = False

            coords = coords_all
            if coords is None:
                m = max(7, dim // 3)
                coords = random.sample(range(dim), m)
            random.shuffle(coords)

            # coordinate probes
            for d in coords:
                if time.time() >= endt:
                    break
                sd = step[d]
                if sd <= 2.0 * step_min[d]:
                    continue

                xd = x[d]

                y = x[:]
                y[d] = reflect(xd + sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved = True
                    continue

                y = x[:]
                y[d] = reflect(xd - sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved = True
                    continue

                step[d] = max(step_min[d], step[d] * 0.70)
                sig[d] = max(step_min[d], sig[d] * 0.78)

            # sparse ES burst
            if time.time() < endt:
                best_y = None
                best_fy = f
                m = dim if dim <= 14 else max(3, dim // 3)

                for _ in range(lam):
                    if time.time() >= endt:
                        break
                    y = x[:]
                    idxs = random.sample(range(dim), m) if m < dim else range(dim)
                    for d in idxs:
                        y[d] = reflect(y[d] + random.gauss(0.0, 1.0) * sig[d], lows[d], highs[d])
                    fy = eval_f(y)
                    if fy < best_fy:
                        best_fy, best_y = fy, y

                if best_y is not None and best_fy < f:
                    x, f = best_y, best_fy
                    improved = True
                    for d in range(dim):
                        step[d] = min(0.45 * spans[d], step[d] * 1.10)
                        sig[d] = min(0.45 * spans[d], sig[d] * 1.10)

            if improved:
                stall = 0
            else:
                stall += 1
                if stall >= 4:
                    for d in range(dim):
                        step[d] = min(0.25 * spans[d], step[d] * 1.18)
                        sig[d] = min(0.25 * spans[d], sig[d] * 1.18)
                    stall = 0

            if max(step) <= max(step_min) * 120.0:
                break

        return x, f

    # ---------- main: SHADE-ish current-to-pbest/1 + archive + injections + pop reduction ----------
    order = argsort_fit(fit)
    pbest_rate = 0.12  # starts exploitative, can widen under stagnation
    no_improve = 0
    it = 0

    pop0 = pop_size
    min_pop = max(14, min(52, 7 * dim))
    refine_every = max(12, 4 * dim)

    while time.time() < deadline:
        it += 1
        frac_time = (time.time() - t0) / (max_time if max_time > 0 else 1.0)
        if frac_time < 0.0:
            frac_time = 0.0
        if frac_time > 1.0:
            frac_time = 1.0

        # population reduction
        target_pop = int(round(pop0 - (pop0 - min_pop) * frac_time))
        if target_pop < pop_size and (it % 6 == 0):
            order = argsort_fit(fit)
            keep = order[:target_pop]
            pop = [pop[j] for j in keep]
            fit = [fit[j] for j in keep]
            pop_size = target_pop
            arch_max = pop_size
            while len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))
            order = argsort_fit(fit)

        # local refinement of best (more frequent, short slices)
        if best_x is not None and (it % refine_every == 0):
            remaining = deadline - time.time()
            if remaining > 0:
                order = argsort_fit(fit)
                elite_k = max(6, min(pop_size, 2 * dim))
                mu, sigma = elite_stats(order, elite_k)
                budget = min((0.06 + 0.18 * frac_time) * max_time, 0.22 * remaining)
                bx, bf = local_refine(best_x, best, budget, sigma)
                if bf < best:
                    best, best_x = bf, bx[:]
                    wi = max(range(pop_size), key=lambda j: fit[j])
                    archive.append(pop[wi])
                    if len(archive) > arch_max:
                        archive.pop(random.randrange(len(archive)))
                    pop[wi] = bx[:]
                    fit[wi] = bf
                    no_improve = 0
                    order = argsort_fit(fit)

        if time.time() >= deadline:
            break

        # refresh ranking sometimes
        if it % max(3, pop_size // 2) == 0:
            order = argsort_fit(fit)

        # elite stats for injections
        elite_k = max(6, min(pop_size, 2 * dim))
        mu, sigma = elite_stats(order, elite_k)

        # pick target
        i = random.randrange(pop_size)
        xi = pop[i]
        fi = fit[i]

        # choose pbest from top p%
        pcount = max(2, int(pbest_rate * pop_size))
        pbest_i = order[random.randrange(pcount)]
        xpbest = pop[pbest_i]

        # SHADE memory index
        r = random.randrange(H)
        muF = M_F[r]
        muCR = M_CR[r]

        # sample Fi ~ Cauchy(muF, 0.1)
        Fi = 0.5
        for _ in range(12):
            u = random.random()
            cand = muF + 0.10 * math.tan(math.pi * (u - 0.5))
            if cand > 0.0:
                Fi = cand
                break
        if Fi > 1.0:
            Fi = 1.0
        if Fi < 1e-6:
            Fi = 1e-6

        # sample CRi ~ Normal(muCR, 0.1) with late tightening
        cr_sd = 0.10 * (1.0 - 0.55 * frac_time)
        CRi = muCR + random.gauss(0.0, cr_sd)
        if CRi < 0.0:
            CRi = 0.0
        elif CRi > 1.0:
            CRi = 1.0

        # choose r1, r2
        idxs = list(range(pop_size))
        idxs.remove(i)
        r1 = random.choice(idxs)
        idxs.remove(r1)
        r2 = random.choice(idxs)

        xr1 = pop[r1]
        xr2 = archive[random.randrange(len(archive))] if (archive and random.random() < 0.55) else pop[r2]

        # mutation current-to-pbest/1 with bounce-back
        v = [0.0] * dim
        for d in range(dim):
            vd = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            if vd < lows[d]:
                vd = lows[d] + random.random() * (xi[d] - lows[d])
            elif vd > highs[d]:
                vd = highs[d] - random.random() * (highs[d] - xi[d])
            v[d] = vd

        # crossover
        jrand = random.randrange(dim)
        uvec = xi[:]
        for d in range(dim):
            if d == jrand or random.random() < CRi:
                uvec[d] = v[d]

        # occasional Gaussian injection (mean/best) for robustness
        rr = random.random()
        if rr < (0.05 + 0.05 * frac_time):
            # elite-mean injection
            for d in range(dim):
                uvec[d] = clip(mu[d] + random.gauss(0.0, 1.0) * (0.90 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif rr < (0.07 + 0.06 * frac_time) and best_x is not None:
            # best-centered injection
            for d in range(dim):
                uvec[d] = clip(best_x[d] + random.gauss(0.0, 1.0) * (0.55 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif rr < 0.095:
            # rare global jump
            uvec = halton_point(random.randrange(1, 6000000))

        fu = eval_f(uvec)

        if fu <= fi:
            # selection + archive
            archive.append(xi)
            if len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))
            pop[i] = uvec
            fit[i] = fu

            # successful parameter memory update (weighted by improvement)
            df = abs(fi - fu)
            w = df if df > 1e-12 else 1e-12

            # Lehmer mean for F (approx with weighted update)
            # keep simple & stable without external libs
            newF = Fi
            newCR = CRi

            # move memory slot towards successful params
            M_F[mem_idx] = 0.85 * M_F[mem_idx] + 0.15 * newF
            M_CR[mem_idx] = 0.85 * M_CR[mem_idx] + 0.15 * newCR
            mem_idx = (mem_idx + 1) % H

            if fu < best:
                best, best_x = fu, uvec[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # stagnation: widen pbest and partial restart with better global coverage
        if no_improve > (45 + 6 * dim):
            no_improve = 0
            pbest_rate = min(0.75, pbest_rate * 1.45)

            replace = max(1, pop_size // 4)
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)[:replace]
            for widx in worst:
                if time.time() >= deadline:
                    break
                xnew = halton_point(random.randrange(1, 9000000)) if random.random() < 0.80 else rand_point()
                fnew = eval_f(xnew)
                pop[widx] = xnew
                fit[widx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # also "reheat" memories a bit
            for h in range(H):
                M_F[h] = min(0.95, max(0.2, M_F[h] + 0.05))
                M_CR[h] = min(0.95, max(0.05, M_CR[h] + 0.05))

        # anneal pbest_rate back slowly
        pbest_rate = max(0.10, pbest_rate * 0.9992)

    return best
