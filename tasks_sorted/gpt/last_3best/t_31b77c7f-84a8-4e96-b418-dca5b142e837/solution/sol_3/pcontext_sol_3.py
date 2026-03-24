import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Hybrid algorithm tuned for strong anytime performance:
      1) Strong initial design: LHS-like + scrambled Halton + center/opposition.
      2) Population-based Differential Evolution with *adaptive* parameters
         (jDE: self-adaptive F and CR per individual).
      3) Periodic local refinement on the current best using:
           - coordinate-wise quadratic (parabolic) step
           - small stochastic hillclimb (subset perturbations)
      4) Stagnation handling with partial reinitialization while keeping elites.

    Returns:
      best fitness (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ------------------------- helpers -------------------------

    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def clamp_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    avgw = sum(widths) / max(1, dim)
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposition(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    # ---- scrambled Halton (very lightweight scrambling) ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
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

    primes = first_primes(max(1, dim))
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def van_der_corput_scrambled(n, base):
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
            u = van_der_corput_scrambled(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------------------- local refine ----------------------

    def quadratic_refine_1d(x, i, delta):
        """Try x[i]-d, x[i], x[i]+d + parabolic vertex in [-d,d]."""
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

        best_f = f0
        best_x = x0
        if fm < best_f:
            best_f, best_x = fm, xm
        if fp < best_f:
            best_f, best_x = fp, xp

        denom = (fm - 2.0 * f0 + fp)
        if denom != 0.0:
            t_star = 0.5 * (fm - fp) / denom  # in units of "delta"
            if -1.0 <= t_star <= 1.0:
                xv = x[:]
                xv[i] = clamp(xv[i] + t_star * delta, lo, hi)
                fv = eval_f(xv)
                if fv < best_f:
                    best_f, best_x = fv, xv

        return best_f, best_x

    # ---------------------- initialization ----------------------

    # Population size (DE): common heuristic 10*dim, capped for time.
    NP = max(18, min(80, 10 * dim if dim > 0 else 18))

    # Seeding budget
    seed_base = max(NP, 10 * dim + 20)

    init_points = []
    init_points.append(centers[:])

    # LHS-like
    lhs_n = seed_base // 2
    if lhs_n < 4:
        lhs_n = 4
    strata = []
    for i in range(dim):
        idx = list(range(lhs_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(lhs_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            a = strata[i][k]
            u = (a + random.random()) / lhs_n
            x.append(lo + u * (hi - lo))
        init_points.append(x)

    # Halton
    halton_n = seed_base // 2
    offset = random.randint(1, 5000)
    for k in range(1, halton_n + 1):
        init_points.append(halton_point(offset + k))

    # Some randoms
    for _ in range(max(8, dim * 2)):
        init_points.append(rand_uniform_vec())

    # Some opposition points (subset)
    opp_subset = init_points[:max(8, len(init_points) // 5)]
    for x in opp_subset:
        init_points.append(opposition(x))

    # Evaluate and build initial population by taking best NP points seen so far
    # (this tends to dominate purely random initial populations).
    scored = []
    best = float("inf")
    best_x = centers[:]
    # guard: dim==0 edge case
    if dim == 0:
        # Only one candidate: empty vector
        return eval_f([])

    for x in init_points:
        if time.time() >= deadline:
            return best
        x = clamp_vec(x)
        fx = eval_f(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])

    # Fill population; if not enough, add random.
    pop = []
    while len(pop) < NP and scored:
        pop.append(scored.pop(0)[1])

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

    # jDE self-adaptive parameters per individual
    # F in [0.1, 0.9], CR in [0, 1]
    Fi = [random.uniform(0.4, 0.9) for _ in range(NP)]
    CRi = [random.uniform(0.2, 0.95) for _ in range(NP)]
    tau1 = 0.1
    tau2 = 0.1

    # Local step sizes (for refine)
    step = [0.12 * (w if w > 0 else 1.0) for w in widths]
    step_min = [1e-12 * ((w if w > 0 else 1.0) + 1.0) for w in widths]
    step_max = [0.75 * (w if w > 0 else (avgw if avgw > 0 else 1.0)) for w in widths]

    stagn = 0
    last_best = best
    gen = 0

    # ------------------------- main loop -------------------------

    while time.time() < deadline:
        gen += 1

        if best < last_best:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # One DE generation (DE/current-to-pbest/1 + bin crossover)
        # Choose pbest from top p fraction
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        p = 0.2
        pcount = max(2, int(p * NP))

        for i in range(NP):
            if time.time() >= deadline:
                return best

            # jDE adaptation
            F = Fi[i]
            CR = CRi[i]
            if random.random() < tau1:
                F = random.uniform(0.1, 0.9)
            if random.random() < tau2:
                CR = random.random()

            # select pbest
            pbest_idx = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest_idx]

            # pick r1, r2 distinct from i and each other
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)

            x_r1 = pop[r1]
            x_r2 = pop[r2]

            # mutation: current-to-pbest + rand diff
            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = x_i[d] + F * (x_p[d] - x_i[d]) + F * (x_r1[d] - x_r2[d])
            mutant = clamp_vec(mutant)

            # crossover
            trial = x_i[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = mutant[d]
            trial = clamp_vec(trial)

            f_trial = eval_f(trial)

            # selection
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                Fi[i] = F
                CRi[i] = CR

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
            # else keep parent, keep old Fi/CRi

        # -------- periodic local refinement of best --------
        # Keep it cheap: only occasionally and only a few coordinates
        if time.time() < deadline and (gen % 3 == 0 or stagn > 20):
            xref = best_x[:]

            # stochastic subset hillclimb proposals
            tries = max(6, dim)
            for _ in range(tries):
                if time.time() >= deadline:
                    return best
                x = xref[:]
                k = max(1, dim // 4)
                for __ in range(k):
                    j = random.randrange(dim)
                    s = step[j]
                    x[j] = clamp(x[j] + random.gauss(0.0, s), bounds[j][0], bounds[j][1])
                fx = eval_f(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    xref = x[:]

            # coordinate parabolic on a few random coords
            coords = list(range(dim))
            random.shuffle(coords)
            coord_budget = max(1, min(dim, 8))
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
                    xref = xnew[:]
                    step[j] = min(step_max[j], step[j] * 1.25)
                else:
                    step[j] = max(step_min[j], step[j] * 0.72)

        # -------- stagnation handling: partial refresh --------
        if stagn >= 80 and time.time() < deadline:
            stagn = 0

            # keep top survivors
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep = max(3, NP // 3)

            survivors = [pop[order[k]][:] for k in range(keep)]
            survivors_fit = [fit[order[k]] for k in range(keep)]
            survivors_F = [Fi[order[k]] for k in range(keep)]
            survivors_CR = [CRi[order[k]] for k in range(keep)]

            pop = survivors
            fit = survivors_fit
            Fi = survivors_F
            CRi = survivors_CR

            # refill with diverse points (Halton+random+best-mix)
            refill = NP - keep
            offset = random.randint(1, 8000)
            for k in range(refill):
                if time.time() >= deadline:
                    return best
                if random.random() < 0.45:
                    x = rand_uniform_vec()
                else:
                    x = halton_point(offset + k + 1)

                # directed mix with best sometimes
                if random.random() < 0.55:
                    a = random.random()
                    x = [a * best_x[d] + (1.0 - a) * x[d] for d in range(dim)]

                x = clamp_vec(x)
                fx = eval_f(x)
                pop.append(x)
                fit.append(fx)
                Fi.append(random.uniform(0.4, 0.9))
                CRi.append(random.uniform(0.2, 0.95))
                if fx < best:
                    best, best_x = fx, x[:]

            # re-expand local steps a bit after refresh
            step = [max(step[d], 0.15 * (widths[d] if widths[d] > 0 else 1.0)) for d in range(dim)]

    return best
