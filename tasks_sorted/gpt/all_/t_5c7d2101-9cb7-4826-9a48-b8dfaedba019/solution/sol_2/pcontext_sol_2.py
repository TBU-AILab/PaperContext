import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Strategy (fast + robust):
      1) Low-discrepancy Sobol-like sampler replacement: scrambled Halton for seeding
      2) Differential Evolution with:
           - current-to-pbest/1 mutation (JADE-style) for better balance
           - success-history adaptation of F and CR (SHADE-lite)
           - external archive to increase diversity
      3) Embedded local search around best using coordinate + gaussian-ish steps
      4) Stagnation detector triggers partial re-seeding

    Returns:
        best (float): best fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def eval_f(x):
        return float(func(x))

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]

    # ---------------- scrambled Halton ----------------
    def primes_first(n):
        ps = []
        x = 2
        while len(ps) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(x)
            x += 1
        return ps

    def vdc_scrambled(index, base, scramble):
        # radical inverse with digit scrambling (perm = (d + scramble) % base)
        v = 0.0
        denom = 1.0
        n = index
        while n > 0:
            n, d = divmod(n, base)
            d = (d + scramble) % base
            denom *= base
            v += d / denom
        return v

    primes = primes_first(dim)
    scrambles = [random.randrange(1, p) for p in primes]
    halton_i = 1

    def halton_vec():
        nonlocal halton_i
        idx = halton_i
        halton_i += 1
        x = [0.0] * dim
        for j in range(dim):
            u = vdc_scrambled(idx, primes[j], scrambles[j])
            lo, hi = bounds[j]
            x[j] = lo + (hi - lo) * u
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Gaussian-ish using CLT (sum uniforms)
    def gaussish():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random()) - 3.0

    # ---------------- SHADE-lite DE ----------------
    # Pop size: moderate but time-bounded
    pop_size = max(16, min(80, 10 * dim))
    archive_max = pop_size

    # p-best fraction (current-to-pbest): choose from top p%
    pmin = 2.0 / pop_size
    pmax = 0.2

    # Historical memories for F and CR (small)
    H = 6
    M_F = [0.6] * H
    M_CR = [0.9] * H
    mem_k = 0

    # Initialize population (mix Halton + random)
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if now() >= deadline:
            return best
        x = halton_vec() if i < (pop_size * 3) // 4 else rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = x[:]

    # Archive for diversity (stores replaced solutions)
    archive = []

    # Stagnation tracking
    last_improve_time = now()
    last_best = best
    no_improve_gens = 0

    # Local search scale (shrinks slowly)
    local_scale = 0.12
    local_shrink = 0.992

    # Utility: get indices sorted by fitness (avoid full sort often)
    def top_indices(p_frac):
        # returns list of indices of approximately top p_frac fraction by fitness
        k = max(2, int(pop_size * p_frac))
        # partial selection by repeated scan (k small)
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

    # Main loop
    gen = 0
    while True:
        if now() >= deadline:
            return best

        # Update best quickly
        if best < last_best - 1e-15:
            last_best = best
            last_improve_time = now()
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # Choose p for this generation (mildly randomized)
        p = random.uniform(pmin, pmax)
        pbest_pool = top_indices(p)

        S_F = []
        S_CR = []
        dF = []  # fitness improvements for weighting memory update

        # One generation
        for i in range(pop_size):
            if now() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick memory index r
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample CR ~ Normal(muCR, 0.1) clipped to [0,1]
            CRi = muCR + 0.1 * gaussish()
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            # sample F using Cauchy-like: muF + 0.1*tan(pi*(u-0.5))
            # (no heavy math: use tan from math)
            u = random.random()
            Fi = muF + 0.1 * math.tan(math.pi * (u - 0.5))
            # Resample if invalid
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 6:
                u = random.random()
                Fi = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if Fi <= 0.0:
                Fi = 0.1
            if Fi > 1.0:
                Fi = 1.0

            # choose pbest from top
            pbest = pop[random.choice(pbest_pool)]

            # choose r1 != i from pop
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from union(pop, archive), distinct from i and r1
            use_archive = (len(archive) > 0) and (random.random() < 0.5)
            if use_archive:
                # pick from archive with fallback to pop
                r2_vec = archive[random.randrange(len(archive))]
            else:
                r2 = r1
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                r2_vec = pop[r2]

            xr1 = pop[r1]

            # current-to-pbest/1:
            # v = xi + F*(pbest - xi) + F*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - r2_vec[j])
                lo, hi = bounds[j]
                # reflection bounds handling
                if vj < lo:
                    vj = lo + (lo - vj)
                    if vj > hi:
                        vj = lo
                elif vj > hi:
                    vj = hi - (vj - hi)
                    if vj < lo:
                        vj = hi
                v[j] = vj

            # binomial crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]

            fu = eval_f(uvec)

            # selection
            if fu <= fi:
                # add replaced solution to archive
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                pop[i] = uvec
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = uvec[:]

                # store successful parameters
                S_F.append(Fi)
                S_CR.append(CRi)
                dF.append(max(0.0, fi - fu))

        # Update memories (weighted Lehmer mean for F, weighted mean for CR)
        if len(S_F) > 0:
            # weights proportional to improvement
            wsum = sum(dF)
            if wsum <= 0.0:
                weights = [1.0 / len(dF)] * len(dF)
            else:
                weights = [di / wsum for di in dF]

            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * (f * f)
                den += w * f
            new_MF = (num / den) if den > 1e-12 else M_F[mem_k]

            # Mean for CR
            new_MCR = 0.0
            for w, cr in zip(weights, S_CR):
                new_MCR += w * cr

            M_F[mem_k] = clip(new_MF, 0.05, 1.0)
            M_CR[mem_k] = clip(new_MCR, 0.0, 1.0)
            mem_k = (mem_k + 1) % H

        # -------- local search around best (cheap) --------
        if best_x is not None:
            # coordinate + gaussian-ish perturbations
            tries = 4 + dim // 4
            for _ in range(tries):
                if now() >= deadline:
                    return best
                x = best_x[:]

                if random.random() < 0.5:
                    # coordinate move (good for separable-ish landscapes)
                    j = random.randrange(dim)
                    step = gaussish() * spans[j] * local_scale
                    x[j] = clip(x[j] + step, bounds[j][0], bounds[j][1])
                else:
                    # full small perturbation
                    for j in range(dim):
                        step = gaussish() * spans[j] * (local_scale * 0.35)
                        x[j] = clip(x[j] + step, bounds[j][0], bounds[j][1])

                fx = eval_f(x)
                if fx < best:
                    best = fx
                    best_x = x[:]

            local_scale *= local_shrink
            if local_scale < 1e-6:
                local_scale = 1e-6

        # -------- stagnation: partial reseed --------
        gen += 1
        if no_improve_gens >= 12 and (now() - last_improve_time) > 0.25:
            # replace worst ~25% with new points (half near best, half global)
            k = max(2, pop_size // 4)
            # find worst indices
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
                if now() >= deadline:
                    return best
                if best_x is not None and t < k // 2:
                    # near-best reseed
                    x = best_x[:]
                    rad = 0.25
                    for j in range(dim):
                        x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * spans[j] * rad,
                                    bounds[j][0], bounds[j][1])
                else:
                    x = halton_vec()

                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

            no_improve_gens = 0
            last_improve_time = now()

        # time check
        if now() >= deadline:
            return best
