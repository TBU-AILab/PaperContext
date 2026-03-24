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

    # ---------- low-discrepancy init (scrambled Halton) ----------
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
        x = [0.0] * dim
        kk = k + 1
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

    pop_size = max(18, min(80, 10 * dim))
    pop = []
    fit = []

    init_n = max(pop_size, 12 * dim)
    k = 0
    while k < init_n and time.time() < deadline:
        # mostly Halton, some pure random
        x = halton_point(k) if (k < int(0.8 * init_n)) else rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        pop.append(x)
        fit.append(fx)
        k += 1

    if not pop:
        return best

    # keep best pop_size points to start (truncate if init_n > pop_size)
    if len(pop) > pop_size:
        idx = list(range(len(pop)))
        idx.sort(key=lambda i: fit[i])
        idx = idx[:pop_size]
        pop = [pop[i] for i in idx]
        fit = [fit[i] for i in idx]

    # ---------- helpers for adaptive sampling ----------
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
            # avoid collapse + cap
            s = sigma[d] if sigma[d] > 0.0 else 0.25 * spans[d]
            sigma[d] = max(1e-12 * spans[d], min(0.7 * spans[d], s))
        return mu, sigma

    # ---------- (1+λ)-ES around best with 1/5 rule ----------
    def local_es(x0, f0, time_budget):
        endt = min(deadline, time.time() + time_budget)
        x = x0[:]
        f = f0

        sig = [0.15 * spans[d] for d in range(dim)]
        sig_min = [1e-12 * spans[d] for d in range(dim)]
        sig_max = [0.9 * spans[d] for d in range(dim)]

        # batch size (lambda)
        lam = 4 if dim <= 8 else 6 if dim <= 20 else 8
        succ = 0
        trials = 0

        while time.time() < endt:
            # generate λ candidates and take best
            best_y = None
            best_fy = float("inf")

            # mutate subset size (sparse in high-d)
            if dim <= 12:
                m = dim
            else:
                m = max(2, dim // 3)

            for _ in range(lam):
                y = x[:]
                idxs = random.sample(range(dim), m) if m < dim else range(dim)
                for d in idxs:
                    y[d] = clip(y[d] + random.gauss(0.0, 1.0) * sig[d], lows[d], highs[d])
                fy = eval_f(y)
                if fy < best_fy:
                    best_fy, best_y = fy, y
                trials += 1
                if time.time() >= endt:
                    break

            if best_y is None:
                break

            if best_fy < f:
                x, f = best_y, best_fy
                succ += 1

            # 1/5 success rule adaptation every few batches
            if trials >= 24:
                rate = succ / float(max(1, trials // lam))
                mult = 1.22 if rate > 0.2 else 0.82
                for d in range(dim):
                    sig[d] = max(sig_min[d], min(sig_max[d], sig[d] * mult))
                succ = 0
                trials = 0

            # stop if steps are tiny
            if max(sig) <= max(sig_min) * 50.0:
                break

        return x, f

    # ---------- main optimizer: JADE-like DE/current-to-pbest + gaussian injection ----------
    # parameters
    mu_F = 0.6
    mu_CR = 0.7
    pbest_rate = 0.2  # top p% as pbest candidates
    archive = []      # store replaced individuals for diversity (like JADE)
    arch_max = pop_size

    it = 0
    no_improve = 0
    refine_every = max(25, 6 * dim)

    while time.time() < deadline:
        it += 1

        # periodic local refinement of current best
        if best_x is not None and (it % refine_every == 0):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            bx, bf = local_es(best_x, best, time_budget=min(0.10 * max_time, 0.25 * remaining))
            if bf < best:
                best, best_x = bf, bx[:]
                # inject into population replacing worst
                wi = max(range(len(pop)), key=lambda j: fit[j])
                archive.append(pop[wi])
                if len(archive) > arch_max:
                    archive.pop(random.randrange(len(archive)))
                pop[wi] = bx[:]
                fit[wi] = bf
                no_improve = 0

        # elite stats for occasional gaussian proposals
        elite_k = max(4, min(pop_size, 2 * dim))
        elite_idx = sorted(range(pop_size), key=lambda j: fit[j])[:elite_k]
        mu, sigma = elite_stats(elite_idx)

        # DE generation (one target per loop to be time-safe)
        i = random.randrange(pop_size)
        xi = pop[i]
        fi = fit[i]

        # choose pbest from top p%
        pcount = max(2, int(pbest_rate * pop_size))
        pbest_i = sorted(range(pop_size), key=lambda j: fit[j])[random.randrange(pcount)]
        xpbest = pop[pbest_i]

        # sample Fi from Cauchy around mu_F, CRi from Normal around mu_CR
        Fi = None
        for _ in range(10):
            # cauchy: mu + gamma*tan(pi*(u-0.5))
            u = random.random()
            Fi = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
            if Fi > 0:
                break
        if Fi is None or Fi <= 0:
            Fi = 0.5
        if Fi > 1.0:
            Fi = 1.0

        CRi = mu_CR + random.gauss(0.0, 0.1)
        if CRi < 0.0:
            CRi = 0.0
        elif CRi > 1.0:
            CRi = 1.0

        # pick r1, r2 distinct from i, and from each other
        idxs = list(range(pop_size))
        idxs.remove(i)
        r1 = random.choice(idxs)
        idxs.remove(r1)
        r2 = random.choice(idxs)

        xr1 = pop[r1]
        # r2 may come from pop or archive (JADE)
        if archive and random.random() < 0.35:
            xr2 = archive[random.randrange(len(archive))]
        else:
            xr2 = pop[r2]

        # mutation: current-to-pbest/1
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])
            # keep in bounds (clip)
            if v[d] < lows[d]:
                v[d] = lows[d] + random.random() * (xi[d] - lows[d])
            elif v[d] > highs[d]:
                v[d] = highs[d] - random.random() * (highs[d] - xi[d])

        # crossover
        jrand = random.randrange(dim)
        uvec = xi[:]
        for d in range(dim):
            if d == jrand or random.random() < CRi:
                uvec[d] = v[d]

        # occasional gaussian injection centered at elite mean (helps if DE stagnates)
        if random.random() < 0.06:
            for d in range(dim):
                uvec[d] = clip(mu[d] + random.gauss(0.0, 1.0) * (0.65 * sigma[d] + 1e-12 * spans[d]),
                               lows[d], highs[d])

        fu = eval_f(uvec)

        if fu <= fi:
            # selection
            archive.append(xi)
            if len(archive) > arch_max:
                archive.pop(random.randrange(len(archive)))
            pop[i] = uvec
            fit[i] = fu

            # update JADE means
            mu_F = 0.9 * mu_F + 0.1 * Fi
            mu_CR = 0.9 * mu_CR + 0.1 * CRi

            if fu < best:
                best, best_x = fu, uvec[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # stagnation control: increase pbest_rate + diversify
        if no_improve > (80 + 8 * dim):
            no_improve = 0
            pbest_rate = min(0.5, pbest_rate * 1.25)
            # partial refresh: replace a fraction with random/halton
            replace = max(1, pop_size // 5)
            worst = sorted(range(pop_size), key=lambda j: fit[j], reverse=True)[:replace]
            for idx in worst:
                if time.time() >= deadline:
                    break
                xnew = halton_point(random.randrange(1, 1000000)) if random.random() < 0.6 else rand_point()
                fnew = eval_f(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

        # slowly anneal pbest_rate back
        pbest_rate = max(0.12, pbest_rate * 0.999)

    return best
