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
        return lo if v < lo else (hi if v > hi else v)

    def eval_f(x):
        return float(func(x))

    # ---------- scrambled Halton init ----------
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
        r = int(math.sqrt(p))
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

    # ---------- seed population ----------
    best = float("inf")
    best_x = None

    pop_size = max(22, min(96, 12 * dim))
    init_n = max(pop_size, 16 * dim)

    pop, fit = [], []
    k = 0
    while k < init_n and time.time() < deadline:
        x = halton_point(k) if (k < int(0.85 * init_n)) else rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        pop.append(x)
        fit.append(fx)
        k += 1

    if not pop:
        return best

    if len(pop) > pop_size:
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fit[i])
        idx = idx[:pop_size]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    # ---------- utilities ----------
    def elite_stats(elite_idx):
        m = len(elite_idx)
        mu = [0.0] * dim
        for j in elite_idx:
            x = pop[j]
            for d in range(dim):
                mu[d] += x[d]
        invm = 1.0 / m
        for d in range(dim):
            mu[d] *= invm

        var = [0.0] * dim
        for j in elite_idx:
            x = pop[j]
            for d in range(dim):
                dd = x[d] - mu[d]
                var[d] += dd * dd
        for d in range(dim):
            var[d] *= invm

        sigma = [math.sqrt(v) for v in var]
        for d in range(dim):
            s = sigma[d] if sigma[d] > 0.0 else 0.30 * spans[d]
            sigma[d] = max(1e-12 * spans[d], min(0.75 * spans[d], s))
        return mu, sigma

    # ---------- stronger local refine: (1+λ)-ES + occasional coordinate probe ----------
    def local_refine(x0, f0, time_budget):
        endt = min(deadline, time.time() + time_budget)
        x = x0[:]
        f = f0

        sig = [0.14 * spans[d] for d in range(dim)]
        sig_min = [1e-12 * spans[d] for d in range(dim)]
        sig_max = [0.9 * spans[d] for d in range(dim)]

        lam = 6 if dim <= 10 else 8 if dim <= 25 else 10
        succ_batches = 0
        batches = 0

        while time.time() < endt:
            best_y = None
            best_fy = float("inf")

            # sparse mutation in higher dims
            if dim <= 14:
                m = dim
            else:
                m = max(3, dim // 3)

            for _ in range(lam):
                y = x[:]
                idxs = random.sample(range(dim), m) if m < dim else range(dim)
                for d in idxs:
                    y[d] = clip(y[d] + random.gauss(0.0, 1.0) * sig[d], lows[d], highs[d])
                fy = eval_f(y)
                if fy < best_fy:
                    best_fy, best_y = fy, y
                if time.time() >= endt:
                    break

            if best_y is None:
                break

            improved = False
            if best_fy < f:
                x, f = best_y, best_fy
                improved = True

            # occasional cheap coordinate probe near current x (helps on separable-ish problems)
            if time.time() < endt and random.random() < 0.20:
                d = random.randrange(dim)
                step = (0.35 * sig[d]) * (1.0 if random.random() < 0.5 else -1.0)
                y = x[:]
                y[d] = clip(y[d] + step, lows[d], highs[d])
                fy = eval_f(y)
                if fy < f:
                    x, f = y, fy
                    improved = True

            batches += 1
            if improved:
                succ_batches += 1

            # 1/5 success rule on batch level
            if batches >= 8:
                rate = succ_batches / float(batches)
                mult = 1.23 if rate > 0.2 else 0.84
                for d in range(dim):
                    sig[d] = max(sig_min[d], min(sig_max[d], sig[d] * mult))
                succ_batches = 0
                batches = 0

            if max(sig) <= max(sig_min) * 80.0:
                break

        return x, f

    # ---------- main: JADE-like DE/current-to-pbest + archive + dual injection ----------
    mu_F = 0.62
    mu_CR = 0.72
    pbest_rate = 0.18

    archive = []
    arch_max = pop_size

    it = 0
    no_improve = 0

    # reduce sorting cost: refresh ranking only every few steps
    rank_refresh = max(3, pop_size // 3)
    order = list(range(pop_size))
    order.sort(key=lambda j: fit[j])

    # cadence for local refine
    refine_every = max(18, 5 * dim)

    while time.time() < deadline:
        it += 1

        if it % rank_refresh == 0:
            order.sort(key=lambda j: fit[j])

        # periodic local refinement of best with a bit more budget near the end
        if best_x is not None and (it % refine_every == 0):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            # spend more refine time later (when remaining is small)
            frac = 0.10 + 0.15 * (1.0 - min(1.0, remaining / max(1e-9, max_time)))
            bx, bf = local_refine(best_x, best, time_budget=min(frac * max_time, 0.30 * remaining))
            if bf < best:
                best, best_x = bf, bx[:]
                wi = max(range(pop_size), key=lambda j: fit[j])
                archive.append(pop[wi])
                if len(archive) > arch_max:
                    archive.pop(random.randrange(len(archive)))
                pop[wi] = bx[:]
                fit[wi] = bf
                no_improve = 0

        if time.time() >= deadline:
            break

        # elite stats for injections
        elite_k = max(5, min(pop_size, 2 * dim))
        elite_idx = order[:elite_k]
        mu, sigma = elite_stats(elite_idx)

        # pick target index
        i = random.randrange(pop_size)
        xi = pop[i]
        fi = fit[i]

        # choose pbest from top p%
        pcount = max(2, int(pbest_rate * pop_size))
        pbest_i = order[random.randrange(pcount)]
        xpbest = pop[pbest_i]

        # sample Fi ~ cauchy(mu_F, 0.1), CRi ~ normal(mu_CR, 0.1)
        Fi = None
        for _ in range(12):
            u = random.random()
            Fi = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
            if Fi is not None and Fi > 0.0:
                break
        if Fi is None or Fi <= 0.0:
            Fi = 0.5
        if Fi > 1.0:
            Fi = 1.0

        CRi = mu_CR + random.gauss(0.0, 0.1)
        if CRi < 0.0:
            CRi = 0.0
        elif CRi > 1.0:
            CRi = 1.0

        # pick r1, r2
        idxs = list(range(pop_size))
        idxs.remove(i)
        r1 = random.choice(idxs)
        idxs.remove(r1)
        r2 = random.choice(idxs)

        xr1 = pop[r1]
        xr2 = archive[random.randrange(len(archive))] if (archive and random.random() < 0.40) else pop[r2]

        # mutation: current-to-pbest/1
        v = [0.0] * dim
        for d in range(dim):
            vd = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # bounce-back bound handling (often better than hard clip)
            if vd < lows[d]:
                vd = lows[d] + random.random() * (xi[d] - lows[d])
            elif vd > highs[d]:
                vd = highs[d] - random.random() * (highs[d] - xi[d])
            v[d] = vd

        # binomial crossover
        jrand = random.randrange(dim)
        uvec = xi[:]
        for d in range(dim):
            if d == jrand or random.random() < CRi:
                uvec[d] = v[d]

        # dual injection: (a) elite-mean gaussian, (b) best-centered gaussian
        r = random.random()
        if r < 0.05:
            for d in range(dim):
                uvec[d] = clip(mu[d] + random.gauss(0.0, 1.0) * (0.75 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])
        elif r < 0.09 and best_x is not None:
            for d in range(dim):
                uvec[d] = clip(best_x[d] + random.gauss(0.0, 1.0) * (0.45 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])

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

        # stagnation control: diversify + widen pbest, plus a cheap halton injection
        if no_improve > (70 + 7 * dim):
            no_improve = 0
            pbest_rate = min(0.55, pbest_rate * 1.35)

            replace = max(1, pop_size // 5)
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)[:replace]
            for _idx in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.55:
                    xnew = halton_point(random.randrange(1, 2000000))
                else:
                    xnew = rand_point()
                fnew = eval_f(xnew)
                pop[_idx] = xnew
                fit[_idx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # slightly reheat parameters
            mu_F = min(0.85, mu_F + 0.05)
            mu_CR = min(0.9, mu_CR + 0.05)

        pbest_rate = max(0.12, pbest_rate * 0.9992)

    return best
