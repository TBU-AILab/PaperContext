import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improvements over the provided hybrid DE:
      - Keeps an explicit evaluated archive + elite pool (for reuse and diversity).
      - Uses a stronger DE variant: "DE/current-to-pbest/1" with an external
        archive (JADE-style) so that difference vectors use past solutions too.
      - Success-history adaptation for F and CR (JADE) + mild jitter.
      - Better boundary handling (reflection + clamp) vs pure clamp (reduces
        boundary sticking).
      - Adds a cheap local optimizer: multi-start SPSA-like gradient sign steps
        around the best + coordinate parabolic refinement.
      - Asynchronous/stochastic update order, periodic mini-restarts that keep elites.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float(func([]))

    # ---------------------------- helpers ----------------------------

    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def reflect_into_bounds(v):
        # Reflect per coordinate to reduce "stuck on boundary" issues.
        out = v[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                out[i] = lo
                continue
            x = out[i]
            span = hi - lo
            # reflect repeatedly if far out
            while x < lo or x > hi:
                if x < lo:
                    x = lo + (lo - x)
                elif x > hi:
                    x = hi - (x - hi)
            # numeric safety
            if x < lo: x = lo
            if x > hi: x = hi
            out[i] = x
        return out

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    avgw = sum(widths) / dim
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    # ---- quick primes + scrambled Halton for seeding ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(dim)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def vdc_scrambled(n, base):
        v = 0.0
        denom = 1.0
        perm = digit_perm[base]
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += perm[rem] / denom
        return v

    def halton_point(index):
        x = []
        for i in range(dim):
            u = vdc_scrambled(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    def opposition(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    # ---------------------- local refinements ----------------------

    def quadratic_refine_1d(x, i, delta):
        if delta <= 0.0:
            return None
        lo, hi = bounds[i]
        x0 = x[:]
        f0 = eval_f(x0)

        xm = x[:]
        xp = x[:]
        xm[i] = clamp(xm[i] - delta, lo, hi)
        xp[i] = clamp(xp[i] + delta, lo, hi)
        fm = eval_f(xm)
        fp = eval_f(xp)

        best_f, best_x = f0, x0
        if fm < best_f:
            best_f, best_x = fm, xm
        if fp < best_f:
            best_f, best_x = fp, xp

        denom = (fm - 2.0 * f0 + fp)
        if denom != 0.0:
            t_star = 0.5 * (fm - fp) / denom  # in [-1,1] ideally
            if -1.0 <= t_star <= 1.0:
                xv = x[:]
                xv[i] = clamp(xv[i] + t_star * delta, lo, hi)
                fv = eval_f(xv)
                if fv < best_f:
                    best_f, best_x = fv, xv
        return best_f, best_x

    def spsa_refine(x, f_x, budget, base_step):
        # Very cheap approximate gradient using two evaluations per iteration.
        # Works well as an "anytime" local improver on smooth-ish objectives.
        best_loc_f = f_x
        best_loc_x = x[:]
        a = base_step
        c = 0.05 * base_step + 1e-12
        for k in range(1, budget + 1):
            if time.time() >= deadline:
                break
            ck = c / (k ** 0.101)
            ak = a / (k ** 0.602)

            delta = [1 if random.random() < 0.5 else -1 for _ in range(dim)]
            xp = best_loc_x[:]
            xm = best_loc_x[:]
            for i in range(dim):
                xp[i] += ck * delta[i]
                xm[i] -= ck * delta[i]
            xp = reflect_into_bounds(xp)
            xm = reflect_into_bounds(xm)
            fp = eval_f(xp)
            fm = eval_f(xm)
            # gradient estimate component ~ (fp-fm)/(2*ck*delta_i)
            # descent step: x - ak * g
            xn = best_loc_x[:]
            diff = (fp - fm) / (2.0 * ck + 1e-12)
            for i in range(dim):
                g_i = diff * delta[i]
                xn[i] = xn[i] - ak * g_i
            xn = reflect_into_bounds(xn)
            fn = eval_f(xn)
            if fn < best_loc_f:
                best_loc_f = fn
                best_loc_x = xn
        return best_loc_f, best_loc_x

    # ---------------------- initialization ----------------------

    # Population size: moderate, time-friendly
    NP = max(20, min(90, 8 * dim + 10))

    # Build a strong candidate set then take best NP
    seed_n = max(NP, 12 * dim + 30)
    candidates = [centers[:]]

    # LHS-like
    lhs_n = max(8, seed_n // 3)
    strata = []
    for i in range(dim):
        idx = list(range(lhs_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(lhs_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            u = (strata[i][k] + random.random()) / lhs_n
            x.append(lo + u * (hi - lo))
        candidates.append(x)

    # Halton
    halton_n = max(8, seed_n // 3)
    offset = random.randint(1, 9000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(offset + k))

    # Random + opposition of some
    for _ in range(max(12, seed_n - len(candidates))):
        candidates.append(rand_uniform_vec())
    for x in candidates[:max(10, len(candidates) // 6)]:
        candidates.append(opposition(x))

    # Evaluate candidates, keep best NP
    scored = []
    best = float("inf")
    best_x = centers[:]

    # evaluated archive: store a limited pool of past solutions for JADE
    # as list of vectors (no fitness needed for diff vectors)
    archive = []

    for x in candidates:
        if time.time() >= deadline:
            return best
        x = reflect_into_bounds(x)
        fx = eval_f(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]
    scored.sort(key=lambda t: t[0])
    pop = [scored[i][1][:] for i in range(min(NP, len(scored)))]
    while len(pop) < NP and time.time() < deadline:
        x = rand_uniform_vec()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        pop.append(x)

    fit = [eval_f(x) for x in pop]
    for i in range(NP):
        if fit[i] < best:
            best, best_x = fit[i], pop[i][:]

    # ---------------------- JADE-style adaptation ----------------------

    mu_F = 0.55
    mu_CR = 0.60
    c_adapt = 0.10  # learning rate

    def sample_F():
        # Cauchy around mu_F, truncated to (0,1]
        while True:
            # tan(pi*(u-0.5)) is standard Cauchy
            u = random.random()
            F = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                if F > 1.0:
                    F = 1.0
                return F

    def sample_CR():
        # Normal around mu_CR, truncated to [0,1]
        CR = random.gauss(mu_CR, 0.1)
        if CR < 0.0: CR = 0.0
        if CR > 1.0: CR = 1.0
        return CR

    # local step sizes
    step = [0.15 * (w if w > 0 else 1.0) for w in widths]
    step_min = [1e-14 * ((w if w > 0 else 1.0) + 1.0) for w in widths]
    step_max = [0.8 * (w if w > 0 else (avgw if avgw > 0 else 1.0)) for w in widths]

    stagn = 0
    last_best = best
    gen = 0
    archive_max = max(NP, 2 * NP)

    # ---------------------- main loop ----------------------

    while time.time() < deadline:
        gen += 1
        if best < last_best:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # rank population for pbest selection
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = 0.18
        pcount = max(2, int(p * NP))

        # success lists for JADE update
        succ_F = []
        succ_CR = []

        # shuffled index order (helps)
        idxs = list(range(NP))
        random.shuffle(idxs)

        for i in idxs:
            if time.time() >= deadline:
                return best

            F = sample_F()
            CR = sample_CR()

            # pbest from top p%
            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest_idx]

            # choose r1 from population (not i), r2 from union(pop, archive)
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            union_size = NP + len(archive)
            r2u = None
            while True:
                r2u = random.randrange(union_size)
                # avoid selecting the same vector as r1 or i (approx check by index)
                if r2u != i and r2u != r1:
                    break

            x_r1 = pop[r1]
            x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

            # mutation: current-to-pbest/1
            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = x_i[d] + F * (x_p[d] - x_i[d]) + F * (x_r1[d] - x_r2[d])

            mutant = reflect_into_bounds(mutant)

            # bin crossover
            trial = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]

            trial = reflect_into_bounds(trial)
            f_trial = eval_f(trial)

            if f_trial <= fit[i]:
                # add replaced parent to archive
                archive.append(x_i[:])
                if len(archive) > archive_max:
                    # random prune
                    del archive[random.randrange(len(archive))]

                pop[i] = trial
                fit[i] = f_trial

                succ_F.append(F)
                succ_CR.append(CR)

                if f_trial < best:
                    best, best_x = f_trial, trial[:]
            # else keep parent

        # JADE parameter update
        if succ_F:
            # Lehmer mean for F; arithmetic mean for CR
            num = 0.0
            den = 0.0
            for f in succ_F:
                num += f * f
                den += f
            lehmer_F = num / (den + 1e-12)
            mean_CR = sum(succ_CR) / len(succ_CR)
            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * lehmer_F
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * mean_CR

        # -------- local refinement (best-focused) --------
        if time.time() < deadline and (gen % 4 == 0 or stagn > 18):
            xref = best_x[:]
            fref = best

            # SPSA refine: small budget, scaled to search space
            base_step = 0.15 * (avgw if avgw > 0 else 1.0)
            spsa_budget = 2 + max(2, dim // 6)
            f_loc, x_loc = spsa_refine(xref, fref, spsa_budget, base_step)
            if f_loc < best:
                best, best_x = f_loc, x_loc[:]
                xref, fref = best_x[:], best

            # Coordinate parabolic on a few coords
            coords = list(range(dim))
            random.shuffle(coords)
            coord_budget = max(2, min(dim, 10))
            for t in range(coord_budget):
                if time.time() >= deadline:
                    return best
                j = coords[t]
                dlt = step[j]
                if dlt < step_min[j]:
                    continue
                res = quadratic_refine_1d(xref, j, dlt)
                if res is None:
                    continue
                fnew, xnew = res
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    xref, fref = best_x[:], best
                    step[j] = min(step_max[j], step[j] * 1.22)
                else:
                    step[j] = max(step_min[j], step[j] * 0.75)

        # -------- mini restart / diversification --------
        if stagn >= 70 and time.time() < deadline:
            stagn = 0
            # keep top elites, refresh others around best + random/halton
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep = max(4, NP // 4)
            elites = [pop[order[k]][:] for k in range(keep)]
            elites_fit = [fit[order[k]] for k in range(keep)]

            # rebuild remaining
            pop = elites[:]
            fit = elites_fit[:]
            # slightly widen mutation means to encourage exploration
            mu_F = min(0.9, max(0.2, mu_F * 1.10))
            mu_CR = min(0.95, max(0.05, mu_CR * 0.95))

            refill = NP - keep
            offset = random.randint(1, 12000)
            for k in range(refill):
                if time.time() >= deadline:
                    return best
                r = random.random()
                if r < 0.40:
                    x = rand_uniform_vec()
                elif r < 0.75:
                    x = halton_point(offset + k + 1)
                else:
                    # around best with diagonal noise
                    x = best_x[:]
                    for d in range(dim):
                        sd = 0.20 * (widths[d] if widths[d] > 0 else 1.0)
                        x[d] += random.gauss(0.0, sd)
                x = reflect_into_bounds(x)
                fx = eval_f(x)
                pop.append(x)
                fit.append(fx)
                if fx < best:
                    best, best_x = fx, x[:]

            # reset some local steps after restart
            step = [max(step[d], 0.18 * (widths[d] if widths[d] > 0 else 1.0)) for d in range(dim)]
            # also shrink archive a bit to stay relevant
            if len(archive) > archive_max // 2:
                archive = archive[-(archive_max // 2):]

    return best
