import random
import time
import math

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def eval_f(x):
        return float(func(x))

    # ---------- scrambled Halton (seed + restarts) ----------
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

    # ---------- init population ----------
    best = float("inf")
    best_x = None

    # Slightly larger + more robust than previous best (still time-safe)
    pop_size0 = max(24, min(120, 12 * dim))
    init_n = max(pop_size0, 20 * dim)

    pop, fit = [], []
    k = 0
    while k < init_n and time.time() < deadline:
        x = halton_point(k) if (k < int(0.85 * init_n)) else rand_point()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        k += 1

    if not pop:
        return best

    # truncate to pop_size0 best
    if len(pop) > pop_size0:
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fit[i])
        idx = idx[:pop_size0]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    pop_size = len(pop)

    # ---------- helpers ----------
    def argsort_fit():
        idx = list(range(pop_size))
        idx.sort(key=lambda i: fit[i])
        return idx

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
        # keep sigmas non-degenerate + capped
        for d in range(dim):
            s = sigma[d] if sigma[d] > 0.0 else 0.30 * spans[d]
            sigma[d] = max(1e-12 * spans[d], min(0.85 * spans[d], s))
        return mu, sigma

    # ---------- bound-handling reflection (often better than clip/bounce for local steps) ----------
    def reflect_into_bounds(v, lo, hi):
        if lo >= hi:
            return lo
        # reflect until within bounds (rarely loops more than once)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            elif v > hi:
                v = hi - (v - hi)
        return v

    # ---------- local refine: short BOBYQA-like substitute (pattern + ES + diagonal quadratic probe) ----------
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

        # occasional diagonal-quadratic move estimator
        quad_every = 5
        pass_id = 0

        while time.time() < endt:
            pass_id += 1
            improved = False

            # (A) coordinate probes (pattern search)
            coords = coords_all
            if coords is None:
                m = max(7, dim // 3)
                coords = random.sample(range(dim), m)
            random.shuffle(coords)

            for d in coords:
                if time.time() >= endt:
                    break
                sd = step[d]
                if sd <= 2.0 * step_min[d]:
                    continue

                xd = x[d]

                y = x[:]
                y[d] = reflect_into_bounds(xd + sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved = True
                    continue

                y = x[:]
                y[d] = reflect_into_bounds(xd - sd, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved = True
                    continue

                # shrink if no improvement on this coordinate
                step[d] = max(step_min[d], step[d] * 0.70)
                sig[d] = max(step_min[d], sig[d] * 0.78)

            # (B) ES burst (captures interactions)
            if time.time() < endt:
                best_y = None
                best_fy = f

                if dim <= 14:
                    m = dim
                else:
                    m = max(3, dim // 3)

                for _ in range(lam):
                    if time.time() >= endt:
                        break
                    y = x[:]
                    idxs = random.sample(range(dim), m) if m < dim else range(dim)
                    for d in idxs:
                        y[d] = reflect_into_bounds(y[d] + random.gauss(0.0, 1.0) * sig[d], lows[d], highs[d])
                    fy = eval_f(y)
                    if fy < best_fy:
                        best_fy, best_y = fy, y

                if best_y is not None and best_fy < f:
                    x, f = best_y, best_fy
                    improved = True
                    for d in range(dim):
                        step[d] = min(0.45 * spans[d], step[d] * 1.10)
                        sig[d] = min(0.45 * spans[d], sig[d] * 1.10)

            # (C) diagonal quadratic probe (cheap 1D along random direction)
            if time.time() < endt and (pass_id % quad_every == 0):
                # random sparse direction
                ydir = [0.0] * dim
                m = dim if dim <= 16 else max(3, dim // 4)
                idxs = random.sample(range(dim), m) if m < dim else range(dim)
                for d in idxs:
                    # scale direction by current step
                    ydir[d] = random.choice([-1.0, 1.0]) * step[d]

                # evaluate at alpha in {-1, +1}
                y1 = x[:]
                y2 = x[:]
                for d in idxs:
                    y1[d] = reflect_into_bounds(y1[d] + ydir[d], lows[d], highs[d])
                    y2[d] = reflect_into_bounds(y2[d] - ydir[d], lows[d], highs[d])
                f1 = eval_f(y1)
                f2 = eval_f(y2)

                # take best among x,y1,y2 (avoid fitting if no signal)
                if f1 < f:
                    x, f = y1, f1
                    improved = True
                if f2 < f:
                    x, f = y2, f2
                    improved = True

            if improved:
                stall = 0
            else:
                stall += 1
                if stall >= 4:
                    # mild reheat to escape tiny-step deadlock
                    for d in range(dim):
                        step[d] = min(0.25 * spans[d], step[d] * 1.18)
                        sig[d] = min(0.25 * spans[d], sig[d] * 1.18)
                    stall = 0

            if max(step) <= max(step_min) * 120.0:
                break

        return x, f

    # ---------- main: L-SHADE-ish DE/current-to-pbest + archive + occasional CMA-lite sampling ----------
    archive = []
    arch_max = pop_size

    mu_F = 0.62
    mu_CR = 0.70
    pbest_rate = 0.18

    it = 0
    no_improve = 0

    order = argsort_fit()
    rank_refresh = max(3, pop_size // 3)

    pop0 = pop_size
    min_pop = max(14, min(52, 7 * dim))
    refine_every = max(14, 4 * dim)

    # small "CMA-lite" sampler: correlated gaussian via elite covariance diagonal + rank-1 direction
    # (no heavy matrix ops; keeps it library-free)
    def cma_lite_sample(mu, sigma, best_x, strength):
        y = [0.0] * dim
        # rank-1 direction towards best
        if best_x is not None and random.random() < 0.55:
            for d in range(dim):
                y[d] = (best_x[d] - mu[d])
        # normalize direction roughly (avoid zero)
        norm = 0.0
        for d in range(dim):
            norm += y[d] * y[d]
        norm = math.sqrt(norm) if norm > 0.0 else 1.0

        x = [0.0] * dim
        for d in range(dim):
            z = random.gauss(0.0, 1.0)
            # mixture: diagonal + rank-1 (scaled)
            step = sigma[d] * (0.85 * z + 0.35 * strength * (y[d] / norm) * random.gauss(0.0, 1.0))
            x[d] = clip(mu[d] + step, lows[d], highs[d])
        return x

    while time.time() < deadline:
        it += 1

        if it % rank_refresh == 0:
            order = argsort_fit()

        frac_time = (time.time() - t0) / max(1e-12, max_time)

        # population reduction over time
        target_pop = int(round(pop0 - (pop0 - min_pop) * min(1.0, frac_time)))
        if target_pop < pop_size and it % 5 == 0:
            order = argsort_fit()
            keep = order[:target_pop]
            pop = [pop[j] for j in keep]
            fit = [fit[j] for j in keep]
            pop_size = target_pop
            arch_max = pop_size
            while len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))
            order = argsort_fit()

        # elite stats
        elite_k = max(6, min(pop_size, 2 * dim))
        if it % rank_refresh == 0:
            order = argsort_fit()
        mu, sigma = elite_stats(order, elite_k)

        # periodic local refinement of best (more frequent, time-sliced)
        if best_x is not None and (it % refine_every == 0):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            budget = min((0.07 + 0.16 * frac_time) * max_time, 0.22 * remaining)
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
                order = argsort_fit()

        if time.time() >= deadline:
            break

        # occasionally use CMA-lite sample instead of DE offspring (helps on rotated valleys)
        if random.random() < (0.06 + 0.08 * frac_time):
            cand = cma_lite_sample(mu, sigma, best_x, strength=1.0 + 1.5 * frac_time)
            fc = eval_f(cand)
            if fc < best:
                best, best_x = fc, cand[:]
            # insert by replacing worst
            wi = max(range(pop_size), key=lambda j: fit[j])
            if fc <= fit[wi]:
                archive.append(pop[wi])
                if len(archive) > arch_max:
                    archive.pop(random.randrange(len(archive)))
                pop[wi] = cand
                fit[wi] = fc
                order = argsort_fit()
            continue

        # DE: one-offspring per loop (time-safe)
        i = random.randrange(pop_size)
        xi = pop[i]
        fi = fit[i]

        # choose pbest
        pcount = max(2, int(pbest_rate * pop_size))
        pbest_i = order[random.randrange(pcount)]
        xpbest = pop[pbest_i]

        # sample Fi (cauchy-ish) and CRi (normal-ish) with late tightening
        Fi = 0.5
        scale = 0.10 * (1.0 - 0.65 * min(1.0, frac_time))
        for _ in range(12):
            u = random.random()
            cand = mu_F + scale * math.tan(math.pi * (u - 0.5))
            if cand > 0.0:
                Fi = cand
                break
        if Fi > 1.0:
            Fi = 1.0
        if Fi < 1e-6:
            Fi = 1e-6

        CRi = mu_CR + random.gauss(0.0, 0.10 * (1.0 - 0.55 * min(1.0, frac_time)))
        if CRi < 0.0:
            CRi = 0.0
        elif CRi > 1.0:
            CRi = 1.0

        # choose r1, r2 distinct
        idxs = list(range(pop_size))
        idxs.remove(i)
        r1 = random.choice(idxs)
        idxs.remove(r1)
        r2 = random.choice(idxs)

        xr1 = pop[r1]
        xr2 = archive[random.randrange(len(archive))] if (archive and random.random() < 0.55) else pop[r2]

        # mutation: current-to-pbest/1 with bounce-back
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

        # occasional elite-mean injection (diagonal gaussian)
        r = random.random()
        if r < 0.05:
            for d in range(dim):
                uvec[d] = clip(mu[d] + random.gauss(0.0, 1.0) * (0.95 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif r < 0.085 and best_x is not None:
            for d in range(dim):
                uvec[d] = clip(best_x[d] + random.gauss(0.0, 1.0) * (0.55 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif r < 0.10:
            uvec = halton_point(random.randrange(1, 5000000))

        fu = eval_f(uvec)

        if fu <= fi:
            archive.append(xi)
            if len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))

            pop[i] = uvec
            fit[i] = fu

            mu_F = 0.92 * mu_F + 0.08 * Fi
            mu_CR = 0.92 * mu_CR + 0.08 * CRi

            if fu < best:
                best, best_x = fu, uvec[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # stagnation: partial restart + widen pbest + parameter kick
        if no_improve > (45 + 6 * dim):
            no_improve = 0
            pbest_rate = min(0.75, pbest_rate * 1.45)

            replace = max(1, pop_size // 4)
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)[:replace]
            for w in worst:
                if time.time() >= deadline:
                    break
                xnew = halton_point(random.randrange(1, 7000000)) if random.random() < 0.80 else rand_point()
                fnew = eval_f(xnew)
                pop[w] = xnew
                fit[w] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            mu_F = min(0.95, mu_F + 0.08)
            mu_CR = min(0.95, mu_CR + 0.08)

        # anneal pbest_rate back slowly
        pbest_rate = max(0.12, pbest_rate * 0.9990)

    return best
