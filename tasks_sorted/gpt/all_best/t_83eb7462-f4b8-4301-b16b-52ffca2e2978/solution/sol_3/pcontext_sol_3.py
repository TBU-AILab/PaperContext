import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libs).

    Improvements over plain DE:
      - JADE-style DE/current-to-pbest/1 with external archive
      - Success-history adaptation of F and CR (SHADE-like)
      - Very small, budgeted local search around the incumbent (stochastic pattern search)
      - Stagnation-triggered partial restart (elite preserve + best-centered reseeding)
      - Fast time checks; robust safe_eval

    Returns
    -------
    best : float
        Best (minimum) fitness found within time budget.
    """

    # ------------------------- helpers -------------------------
    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            lo, hi = bounds[i]
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def bounce(vj, lo, hi):
        # reflection-like boundary handling; tends to preserve search dynamics
        if vj < lo:
            vj = lo + (lo - vj) * 0.5
            if vj > hi:
                vj = lo
        elif vj > hi:
            vj = hi - (vj - hi) * 0.5
            if vj < lo:
                vj = hi
        return vj

    def cauchy(mu, gamma):
        u = random.random() - 0.5
        return mu + gamma * math.tan(math.pi * u)

    def time_frac(now, start, deadline):
        den = max(1e-12, deadline - start)
        t = (now - start) / den
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    widths = []
    for i in range(dim):
        lo, hi = bounds[i]
        w = hi - lo
        widths.append(w if w > 0 else 1.0)

    # ------------------------- time -------------------------
    start = time.perf_counter()
    deadline = start + float(max_time)

    # ------------------------- initialization -------------------------
    # modestly sized but robust; caps to avoid slowdowns
    pop_size = max(14, min(70, 18 + 3 * dim))

    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best_val = fit[best_i]

    # external archive (JADE)
    archive = []
    archive_max = pop_size

    # p-best fraction
    p_best_frac = 0.2

    # SHADE memory for parameter adaptation (success-history)
    H = 8
    MF = [0.6] * H
    MCR = [0.7] * H
    mem_idx = 0

    # Stagnation / restart
    last_best = best_val
    no_improve_gens = 0
    restart_after = max(28, 10 * dim)

    # Local search trust region (relative); adapted by success
    tr = 0.35

    # ------------------------- local search -------------------------
    def local_search(x0, f0, budget, scale):
        # stochastic pattern search with tiny budget
        x = x0[:]
        f = f0
        steps = [max(1e-12, scale * w) for w in widths]

        for _ in range(budget):
            if time.perf_counter() >= deadline:
                break

            trial = x[:]
            if dim == 1 or random.random() < 0.75:
                j = random.randrange(dim)
                s = steps[j] * (1.0 if random.random() < 0.85 else 2.5)
                trial[j] += s if random.random() < 0.5 else -s
            else:
                for j in range(dim):
                    trial[j] += random.gauss(0.0, steps[j])

            clip_inplace(trial)
            ft = safe_eval(trial)

            if ft < f:
                x, f = trial, ft
                # slightly expand around improvements
                if random.random() < 0.35:
                    steps = [s * 1.18 for s in steps]
            else:
                # contract a bit if not improving
                if random.random() < 0.45:
                    steps = [s * 0.88 for s in steps]

        return x, f

    # ------------------------- main loop -------------------------
    gen = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return best_val

        gen += 1
        t = time_frac(now, start, deadline)

        order = sorted(range(pop_size), key=lambda i: fit[i])
        pcount = max(2, int(math.ceil(p_best_frac * pop_size)))

        succ_F = []
        succ_CR = []
        succ_dF = []

        # DE generation
        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return best_val

            # pick memory slot
            r = random.randrange(H)
            muF = MF[r]
            muCR = MCR[r]

            # sample CR ~ N(muCR, 0.1), clamp
            CRi = random.gauss(muCR, 0.1)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # sample F ~ Cauchy(muF, 0.1), resample if <=0; clamp to 1
            Fi = cauchy(muF, 0.1)
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(muF, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.05
            if Fi > 1.0:
                Fi = 1.0

            # slight anneal of Fi late for fine tuning
            Fi = max(0.05, min(1.0, Fi * (1.0 - 0.20 * t)))

            # choose pbest
            pbest = order[random.randrange(pcount)]

            # choose r1 != i,pbest
            while True:
                r1 = random.randrange(pop_size)
                if r1 != i and r1 != pbest:
                    break

            # choose r2 from pop U archive
            pool = pop_size + len(archive)
            if pool == 0:
                x_r2 = pop[random.randrange(pop_size)]
            else:
                while True:
                    k = random.randrange(pool)
                    if k < pop_size:
                        if k != i and k != pbest and k != r1:
                            x_r2 = pop[k]
                            break
                    else:
                        x_r2 = archive[k - pop_size]
                        break

            xi = pop[i]
            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = x_r2

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                lo, hi = bounds[j]
                v[j] = bounce(vj, lo, hi)

            # binomial crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]

            ft = safe_eval(trial)

            # selection + archive update
            if ft <= fit[i]:
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                df = fit[i] - ft
                pop[i] = trial
                fit[i] = ft

                succ_F.append(Fi)
                succ_CR.append(CRi)
                succ_dF.append(df if df > 0.0 else 1e-12)

                if ft < best_val:
                    best_val = ft
                    best_x = trial[:]

        # update memories (SHADE-style)
        if succ_F:
            wsum = sum(succ_dF)
            if wsum <= 0.0:
                wsum = float(len(succ_dF))

            # weighted Lehmer mean for F and weighted arithmetic mean for CR
            numF = 0.0
            denF = 0.0
            meanCR = 0.0
            for Fi, CRi, dfi in zip(succ_F, succ_CR, succ_dF):
                w = dfi / wsum
                numF += w * Fi * Fi
                denF += w * Fi
                meanCR += w * CRi

            if denF > 1e-12:
                newMF = numF / denF
                MF[mem_idx] = max(0.05, min(0.95, newMF))
            MCR[mem_idx] = max(0.0, min(1.0, meanCR))
            mem_idx = (mem_idx + 1) % H

        # light local search more frequently later, still cheap
        if time.perf_counter() < deadline and (gen % (3 if t < 0.65 else 2) == 0):
            base = 0.20 * (1.0 - t) + 0.012 * t
            ls_budget = max(8, min(55, 3 * dim + 10))
            bx, bv = local_search(best_x, best_val, ls_budget, max(1e-8, tr * base))
            if bv < best_val:
                best_x, best_val = bx, bv
                tr = min(2.5, tr * 1.18)
            else:
                tr = max(0.18, tr * 0.90)

        # stagnation detection
        if best_val < last_best - 1e-12:
            last_best = best_val
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # partial restart (elite preserve + mixed reseed)
        if no_improve_gens >= restart_after and time.perf_counter() < deadline:
            no_improve_gens = 0

            order = sorted(range(pop_size), key=lambda i: fit[i])
            elite = max(2, pop_size // 5)

            new_pop = [pop[idx][:] for idx in order[:elite]]
            new_fit = [fit[idx] for idx in order[:elite]]

            center_scale = 0.55 * (1.0 - t) + 0.06 * t
            while len(new_pop) < pop_size and time.perf_counter() < deadline:
                if random.random() < 0.78:
                    x = []
                    for d in range(dim):
                        lo, hi = bounds[d]
                        s = center_scale * widths[d]
                        v = random.gauss(best_x[d], s)
                        if v < lo:
                            v = lo
                        elif v > hi:
                            v = hi
                        x.append(v)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                new_pop.append(x)
                new_fit.append(fx)
                if fx < best_val:
                    best_val = fx
                    best_x = x[:]

            pop, fit = new_pop, new_fit
            archive = []
            # keep learned memories but reset trust region a bit
            tr = 0.35
