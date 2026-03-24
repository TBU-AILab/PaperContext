import random
import time
import math

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + max_time

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

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---- Low-discrepancy (Halton) to seed diverse good points quickly ----
    def _vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v

    # first primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    if dim > len(primes):
        # extend primes naively
        p = primes[-1] + 2
        while len(primes) < dim:
            is_p = True
            r = int(math.sqrt(p))
            for q in range(3, r + 1, 2):
                if p % q == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(p)
            p += 2

    def halton_vec(idx, scramble):
        x = [0.0] * dim
        for i in range(dim):
            h = _vdc(idx + 1, primes[i])
            # Cranley-Patterson rotation scramble in [0,1)
            u = h + scramble[i]
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---- Seed best and an elite pool ----
    best = float("inf")
    best_x = None

    elite_k = max(8, 3 * dim)  # a bit larger than before
    elites = []  # (f, x)

    # scramble once (fixed during run)
    scramble = [random.random() for _ in range(dim)]

    # initial budget: halton + a few pure random
    init_n = max(64, 16 * dim)
    # keep init within time
    idx = 0
    while idx < init_n and time.time() < deadline:
        if idx < int(0.8 * init_n):
            x = halton_vec(idx, scramble)
        else:
            x = rand_uniform()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        elites.append((fx, x))
        idx += 1

    if not elites:
        return best

    elites.sort(key=lambda t: t[0])
    elites = elites[:elite_k]

    # ---- Utilities for search distributions ----
    def compute_mu_sigma(elist):
        k = len(elist)
        mu = [0.0] * dim
        for f, x in elist:
            for i in range(dim):
                mu[i] += x[i]
        invk = 1.0 / k
        for i in range(dim):
            mu[i] *= invk

        var = [0.0] * dim
        for f, x in elist:
            for i in range(dim):
                d = x[i] - mu[i]
                var[i] += d * d
        for i in range(dim):
            var[i] = var[i] * invk

        sigma = [math.sqrt(v) for v in var]
        # floors/ceilings
        for i in range(dim):
            s = sigma[i]
            if not (s > 0.0):
                s = 0.25 * spans[i]
            sigma[i] = max(1e-10 * spans[i], min(0.6 * spans[i], s))
        return mu, sigma

    mu, sigma = compute_mu_sigma(elites)

    # ---- Local improvement: (1+1)-ES with 1/5 success rule (fast + robust) ----
    def local_es(x0, f0, budget_seconds):
        endt = min(deadline, time.time() + budget_seconds)
        x = x0[:]
        f = f0

        # start with moderate radius, adapted online
        sig = [0.12 * spans[i] for i in range(dim)]
        sig_min = [1e-12 * spans[i] for i in range(dim)]
        sig_max = [0.8 * spans[i] for i in range(dim)]

        succ = 0
        tries = 0

        while time.time() < endt:
            # generate one candidate
            y = x[:]
            # mutate a subset sometimes (sparse mutation helps in higher dims)
            if dim <= 10:
                coords = range(dim)
            else:
                m = max(1, dim // 3)
                coords = random.sample(range(dim), m)

            for i in coords:
                step = random.gauss(0.0, 1.0) * sig[i]
                y[i] = clip(y[i] + step, lows[i], highs[i])

            fy = eval_f(y)
            tries += 1
            if fy < f:
                x, f = y, fy
                succ += 1

            # adapt every few steps (1/5 success rule)
            if tries >= 12:
                rate = succ / float(tries)
                # if success rate > 0.2 increase step, else decrease
                if rate > 0.2:
                    mult = 1.25
                else:
                    mult = 0.82
                for i in range(dim):
                    sig[i] = max(sig_min[i], min(sig_max[i], sig[i] * mult))
                succ = 0
                tries = 0

            # stop if steps are tiny
            if max(sig) <= max(sig_min) * 50.0:
                break

        return x, f

    # ---- Main loop: DE/rand-to-best/1/bin + periodic ES refine + restarts ----
    # Population from elites + extra randoms
    pop_size = max(12, min(60, 8 * dim))
    pop = [x for _, x in elites]
    while len(pop) < pop_size and time.time() < deadline:
        pop.append(rand_uniform())

    # evaluate pop (cache fitness separately)
    fit = []
    for x in pop:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # DE params (self-adapted mildly)
    F = 0.7
    CR = 0.9
    no_improve = 0

    # refinement cadence
    refine_every = max(20, 6 * dim)
    it = 0

    while time.time() < deadline:
        it += 1

        # periodic strong local refinement of incumbent
        if best_x is not None and (it % refine_every == 0):
            # small slice of remaining time, capped
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            budget = min(0.12 * max_time, 0.25 * remaining)
            bx, bf = local_es(best_x, best, budget)
            if bf < best:
                best, best_x = bf, bx
                # inject improved point into population (replace worst)
                wi = max(range(len(pop)), key=lambda j: fit[j])
                pop[wi] = bx[:]
                fit[wi] = bf
                no_improve = 0

        if time.time() >= deadline:
            break

        # pick target
        i = random.randrange(pop_size)

        # choose a,b,c distinct and not i
        idxs = list(range(pop_size))
        idxs.remove(i)
        a, b, c = random.sample(idxs, 3)

        xa, xb, xc = pop[a], pop[b], pop[c]
        xi = pop[i]
        xbest = best_x if best_x is not None else pop[min(range(pop_size), key=lambda j: fit[j])]

        # rand-to-best/1 mutation: v = xa + F*(xbest-xa) + F*(xb-xc)
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xa[d] + F * (xbest[d] - xa[d]) + F * (xb[d] - xc[d])
            v[d] = clip(v[d], lows[d], highs[d])

        # binomial crossover
        jrand = random.randrange(dim)
        u = xi[:]
        for d in range(dim):
            if d == jrand or random.random() < CR:
                u[d] = v[d]

        fu = eval_f(u)
        if fu <= fit[i]:
            pop[i] = u
            fit[i] = fu
            if fu < best:
                best, best_x = fu, u[:]
                no_improve = 0
            else:
                no_improve += 0
        else:
            no_improve += 1

        # mild parameter jitter + restart if stagnating
        if it % (pop_size * 3) == 0:
            # adapt F, CR slightly
            F = min(0.95, max(0.35, F + random.uniform(-0.08, 0.08)))
            CR = min(0.98, max(0.10, CR + random.uniform(-0.10, 0.10)))

        if no_improve > (60 + 10 * dim):
            no_improve = 0
            # partial restart: keep best, reinit a chunk using halton around mu/sigma + random
            mu, sigma = compute_mu_sigma(sorted(zip(fit, pop), key=lambda t: t[0])[:max(6, dim)])
            for _ in range(max(1, pop_size // 3)):
                if time.time() >= deadline:
                    break
                j = random.randrange(pop_size)
                if pop[j] == best_x:
                    continue
                if random.random() < 0.5:
                    # gaussian around best with sigma from elites
                    y = [0.0] * dim
                    for d in range(dim):
                        step = random.gauss(0.0, 1.0) * (0.6 * sigma[d] + 1e-12 * spans[d])
                        y[d] = clip(best_x[d] + step, lows[d], highs[d])
                else:
                    y = rand_uniform()
                fy = eval_f(y)
                pop[j] = y
                fit[j] = fy
                if fy < best:
                    best, best_x = fy, y[:]

    return best
