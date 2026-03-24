import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Hybrid algorithm:
      1) Low-discrepancy initialization (Halton) + opposition points
      2) Differential Evolution with:
           - JADE-style "current-to-pbest/1" mutation
           - external archive (for diversity)
           - adaptive parameters (mu_F, mu_CR) via success history
      3) Lightweight caching of evaluations
      4) Periodic local refinement on the incumbent best:
           - coordinate/pattern search with adaptive step sizes
      5) Stagnation detection + partial re-seeding near best and random immigrants

    Returns:
      best fitness (float) found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ----------------------- helpers -----------------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Cache: rounding trades a little precision for more cache hits
    cache = {}
    def evaluate(x):
        key = tuple(round(v, 12) for v in x)
        val = cache.get(key)
        if val is None:
            val = float(func(list(x)))
            cache[key] = val
        return val

    # --- Halton sequence (no libs) ---
    def _first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def _van_der_corput(k, base):
        v = 0.0
        denom = 1.0
        while k > 0:
            k, rem = divmod(k, base)
            denom *= base
            v += rem / denom
        return v

    primes = _first_primes(max(1, dim))

    def halton_point(index):  # index >= 1
        return [_van_der_corput(index, primes[j]) for j in range(dim)]

    def from_unit(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ----------------------- local search -----------------------
    def local_search(x0, f0, time_limit):
        x = x0[:]
        fx = f0

        # start steps relative to span
        steps = [0.12 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = 1e-14

        # Track a simple "pattern" direction
        last_x = x[:]

        no_improve = 0
        order = list(range(dim))

        while time.time() < time_limit:
            improved_any = False
            random.shuffle(order)

            for i in order:
                if time.time() >= time_limit:
                    break
                si = steps[i]
                if si <= min_step:
                    continue

                best_local_f = fx
                best_local_x = None

                # Try +/- step
                xp = x[:]
                xp[i] += si
                clip_inplace(xp)
                fp = evaluate(xp)
                if fp < best_local_f:
                    best_local_f, best_local_x = fp, xp

                xm = x[:]
                xm[i] -= si
                clip_inplace(xm)
                fm = evaluate(xm)
                if fm < best_local_f:
                    best_local_f, best_local_x = fm, xm

                if best_local_x is not None:
                    # accept improvement
                    last_x = x
                    x = best_local_x
                    fx = best_local_f
                    improved_any = True

                    # pattern move (a small extrapolation in the same coordinate direction)
                    if time.time() < time_limit:
                        di = x[i] - last_x[i]
                        if di != 0.0:
                            xt = x[:]
                            xt[i] += 0.6 * di
                            clip_inplace(xt)
                            ft = evaluate(xt)
                            if ft < fx:
                                last_x = x
                                x, fx = xt, ft

            if improved_any:
                no_improve = 0
                # slightly expand steps where possible to speed up basin descent
                for i in range(dim):
                    steps[i] *= 1.05
                    # cap to avoid exploding
                    if steps[i] > 0.25 * spans[i]:
                        steps[i] = 0.25 * spans[i]
            else:
                no_improve += 1
                shrink = 0.55 if no_improve < 3 else 0.35
                for i in range(dim):
                    steps[i] *= shrink
                if max(steps) <= min_step:
                    break

        return x, fx

    # ----------------------- init population -----------------------
    # Keep pop moderate: JADE benefits from some width, but time-bounded
    pop_size = max(14, min(60, 10 * dim + 10))

    pop = []
    # Halton + opposition for half population
    halton_pairs = min(pop_size // 2, max(4, pop_size // 2))
    for k in range(1, halton_pairs + 1):
        x = from_unit(halton_point(k))
        pop.append(x)
        if len(pop) < pop_size:
            pop.append(opposition(x))

    while len(pop) < pop_size:
        pop.append(rand_vec())

    fit = [evaluate(ind) for ind in pop]
    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # ----------------------- JADE-like DE -----------------------
    # External archive for diversity (stores replaced solutions)
    archive = []
    archive_max = pop_size

    # Adaptive parameter means
    mu_F = 0.55
    mu_CR = 0.6
    c = 0.1  # learning rate for mu updates

    # p-best fraction: choose pbest from top p*NP
    p = 0.15  # common JADE value

    # stagnation control
    last_best = best
    stagnation_gens = 0
    gen = 0

    # small helper for sampling from union(pop, archive)
    def pick_from_union(exclude_idx, exclude_set):
        # exclude_set contains ids to avoid (we pass indices, but we only have lists)
        # We'll just attempt a few times; fallback to random.
        union = pop + archive
        # Ensure at least pop has enough; archive may be empty.
        for _ in range(10):
            cand = random.choice(union)
            # exclude_set stores "object id" to avoid duplicates in mutation
            if id(cand) not in exclude_set:
                return cand
        return random.choice(union)

    def cauchy(mu, gamma=0.1):
        # simple Cauchy(mu, gamma) via inverse CDF
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    while True:
        if time.time() >= deadline:
            return best

        gen += 1

        # sort indices by fitness (for pbest selection)
        ranked = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p * pop_size)))

        S_F = []
        S_CR = []

        # Each generation produce one trial per individual
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]

            # Sample CR ~ N(mu_CR, 0.1) clipped to [0,1]
            # Sample F  ~ Cauchy(mu_F, 0.1) resample until >0, then clip to 1
            CR = mu_CR + 0.1 * (random.random() + random.random() + random.random() - 1.5) * 2.0  # approx normal-ish
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            F = cauchy(mu_F, 0.1)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = cauchy(mu_F, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # Choose pbest from top pcount
            pbest_idx = ranked[random.randrange(pcount)]
            xpbest = pop[pbest_idx]

            # Choose r1 from pop, r2 from pop+archive, distinct
            # Use object identity to avoid same vector in mutation
            exclude = {id(xi), id(xpbest)}

            # r1 from population (not i, not pbest ideally)
            for _ in range(12):
                r1_idx = random.randrange(pop_size)
                xr1 = pop[r1_idx]
                if r1_idx != i and id(xr1) not in exclude:
                    break
            else:
                xr1 = pop[(i + 1) % pop_size]
            exclude.add(id(xr1))

            # r2 from union (pop + archive)
            xr2 = pick_from_union(i, exclude)

            # current-to-pbest/1 with archive:
            # v = xi + F*(xpbest - xi) + F*(xr1 - xr2)
            v = [xi[j] + F * (xpbest[j] - xi[j]) + F * (xr1[j] - xr2[j]) for j in range(dim)]

            # Binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]

            clip_inplace(u)
            fu = evaluate(u)

            if fu <= fit[i]:
                # archive stores the replaced target
                archive.append(xi)
                if len(archive) > archive_max:
                    # random removal to keep it simple and fast
                    archive.pop(random.randrange(len(archive)))

                pop[i] = u
                fit[i] = fu

                S_F.append(F)
                S_CR.append(CR)

                if fu < best:
                    best = fu
                    best_x = u[:]

        # Update mu_F, mu_CR using successful parameters
        if S_F:
            # Lehmer mean for F: sum(F^2)/sum(F)
            sf1 = sum(S_F)
            if sf1 > 0.0:
                lehmer = sum(f * f for f in S_F) / sf1
                mu_F = (1.0 - c) * mu_F + c * lehmer

            mu_CR = (1.0 - c) * mu_CR + c * (sum(S_CR) / len(S_CR))

        # Stagnation detection
        if best < last_best - 1e-12:
            last_best = best
            stagnation_gens = 0
        else:
            stagnation_gens += 1

        # Periodic local improvement (more frequent later, but always budgeted)
        if gen % max(6, 4 + dim // 2) == 0 and time.time() < deadline:
            remaining = deadline - time.time()
            # small but useful budget
            budget = min(0.18 * float(max_time), 0.25 * remaining)
            if budget > 0.0:
                xls, fls = local_search(best_x, best, time.time() + budget)
                if fls < best:
                    best = fls
                    best_x = xls[:]
                    last_best = best
                    stagnation_gens = 0

        # Diversity injection if stuck
        if stagnation_gens >= 8 and time.time() < deadline:
            # Replace worst ~40%: some near-best perturbations, some random
            order = sorted(range(pop_size), key=lambda k: fit[k])
            keep = max(4, int(0.6 * pop_size))

            for idx in order[keep:]:
                if time.time() >= deadline:
                    return best

                if random.random() < 0.7:
                    # near-best perturbation using triangular noise
                    xnew = best_x[:]
                    # scale increases a bit when stuck to jump basins
                    base = 0.03 + 0.10 * min(1.0, stagnation_gens / 20.0)
                    for j in range(dim):
                        if spans[j] <= 0.0:
                            continue
                        noise = (random.random() - random.random())  # [-1,1] triangular
                        xnew[j] += noise * base * spans[j]
                    clip_inplace(xnew)
                else:
                    xnew = rand_vec()

                fnew = evaluate(xnew)
                pop[idx] = xnew
                fit[idx] = fnew
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]
                    last_best = best

            # also trim archive to avoid stale dominance
            if len(archive) > archive_max:
                archive = archive[-archive_max:]

            stagnation_gens = 0
