import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-budgeted box-constrained minimizer (no external libs).

    What’s improved vs your best (#3 DE + local refine):
      1) Uses a stronger DE variant: JADE-style "current-to-pbest/1" with:
         - adaptive F and CR learned online from successful trials
         - external archive for (r1-r2) difference vectors (better diversity)
      2) Better bound handling: reflection + final clamp (stable near edges).
      3) More effective local search: short Powell-like random-subspace + coordinate
         trust-region with success-based step adaptation.
      4) Time-aware scheduling: spends more on exploration early, intensifies late.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]
    avg_span = (sum(span_safe) / float(dim)) if dim > 0 else 1.0

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def reflect_then_clamp_inplace(x):
        # reflect into [lo, hi] with period 2w, then clamp for numeric safety
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                x[i] = (a + y) if (y <= w) else (b - (y - w))
        return clamp_inplace(x)

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(random.random(), 1e-300)
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare = z1
        _has_spare = True
        return z0

    def l2norm(v):
        s = 0.0
        for i in range(dim):
            s += v[i] * v[i]
        return math.sqrt(s)

    # ---------------- best bookkeeping ----------------
    best = float("inf")
    best_x = None

    def try_update(x):
        nonlocal best, best_x
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    if now() >= deadline:
        return best

    # ---------------- seeding ----------------
    # Center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    reflect_then_clamp_inplace(x0)
    try_update(x0)

    # A few corners (cheap diversity)
    corner_bits = min(dim, 6)  # up to 64 corners, but cap count
    max_corners = min(14, 1 << corner_bits)
    for mask in range(max_corners):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if i < corner_bits:
                x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
            else:
                x[i] = 0.5 * (lo[i] + hi[i])
        try_update(x)

    # Random seeds
    seed_count = 12 + 5 * dim
    for _ in range(seed_count):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------------- local refinement (trust-region + random subspace) ----------------
    def local_refine(x_start, f_start, time_limit):
        x = x_start[:]
        fx = f_start

        # per-dimension trust region
        step = [0.10 * s for s in span_safe]
        min_step = [1e-14 * s for s in span_safe]
        max_step = [0.50 * s for s in span_safe]

        sweeps = 2  # keep cheap (called periodically)
        for _ in range(sweeps):
            if now() >= time_limit:
                break

            improved = False

            # coordinate moves (random order)
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if now() >= time_limit:
                    return x, fx
                sj = step[j]
                if sj <= min_step[j]:
                    continue
                base = x[j]

                # try + and -
                y = x[:]
                y[j] = base + sj
                reflect_then_clamp_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

                y = x[:]
                y[j] = base - sj
                reflect_then_clamp_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    continue

            # random-subspace directional probes (handles rotated valleys)
            if now() < time_limit and dim > 0:
                k = max(1, int(math.sqrt(dim)))
                idxs = random.sample(range(dim), k) if dim >= k else list(range(dim))
                d = [0.0] * dim
                for j in idxs:
                    d[j] = randn()
                dn = l2norm(d)
                if dn > 0.0:
                    inv = 1.0 / dn
                    for j in idxs:
                        d[j] *= inv

                    base_step = 0.0
                    for j in idxs:
                        base_step += step[j]
                    base_step = base_step / float(len(idxs)) if idxs else 0.0
                    base_step = max(base_step, 1e-16 * avg_span)

                    # small 1D search (few probes)
                    for alpha in (-base_step, base_step, 2.0 * base_step, -2.0 * base_step):
                        if now() >= time_limit:
                            break
                        y = x[:]
                        for j in idxs:
                            y[j] += alpha * d[j]
                        reflect_then_clamp_inplace(y)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved = True

            # adapt trust region
            if improved:
                for j in range(dim):
                    step[j] = min(max_step[j], step[j] * 1.20)
            else:
                for j in range(dim):
                    step[j] *= 0.5

        return x, fx

    # ---------------- JADE-like Differential Evolution ----------------
    # population size: moderate, time-limited
    NP = max(16, min(70, 10 + 2 * dim))
    if dim >= 25:
        NP = max(20, min(90, 12 + int(3.0 * math.sqrt(dim)) * 4))

    # Initialize population (inject best + perturbations)
    pop = []
    fit = []

    if best_x is not None:
        pop.append(best_x[:])
        fit.append(best)

        # a few perturbed best points
        perturb = min(5, NP - 1)
        for _ in range(perturb):
            if now() >= deadline:
                return best
            x = best_x[:]
            for j in range(dim):
                x[j] += (2.0 * random.random() - 1.0) * 0.07 * span_safe[j]
            reflect_then_clamp_inplace(x)
            fx = try_update(x)
            pop.append(x)
            fit.append(fx)

    while len(pop) < NP:
        if now() >= deadline:
            return best
        x = rand_point()
        fx = try_update(x)
        pop.append(x)
        fit.append(fx)

    # external archive (JADE); stores replaced parent vectors
    archive = []
    arch_max = NP  # typical JADE choice

    # JADE adaptation parameters
    mu_F = 0.55
    mu_CR = 0.85
    c_adapt = 0.10  # learning rate

    # p-best fraction (top p of population)
    p_min = 0.05
    p_max = 0.30

    # local refine scheduling (more frequent late)
    last_local = now()
    base_local_period = max(0.20, min(1.0, 0.12 * float(max_time)))

    # stagnation-based injections
    last_best = best
    no_improve = 0

    def sorted_indices():
        idx = list(range(NP))
        idx.sort(key=lambda i: fit[i])
        return idx

    # utility: choose r from union pop+archive excluding a set
    def pick_from_union(exclude_set):
        # union size = NP + len(archive)
        total = NP + len(archive)
        if total <= 0:
            return None, None
        # try a few times to avoid expensive bookkeeping
        for _ in range(30):
            r = random.randrange(total)
            if r < NP:
                if r in exclude_set:
                    continue
                return pop[r], r
            else:
                ar = r - NP
                # archive has no "indices" in pop; safe unless object identity equals
                return archive[ar], None
        # fallback: just random pop not in exclude
        for r in range(NP):
            if r not in exclude_set:
                return pop[r], r
        return pop[0], 0

    # main loop
    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, (deadline - t0))  # 0..1
        # p increases slightly over time: more exploitation late
        p = p_min + (p_max - p_min) * (0.35 + 0.65 * frac)
        pcount = max(2, int(math.ceil(p * NP)))

        # precompute ranking
        rank = sorted_indices()
        pbest_pool = rank[:pcount]

        # success memories for adapting mu_F, mu_CR
        SF = []
        SCR = []
        dF = []  # improvements for weighted mean

        for i in range(NP):
            if now() >= deadline:
                return best

            xi = pop[i]

            # Sample Fi from a truncated normal around mu_F (JADE uses Cauchy; normal is fine and stable)
            Fi = mu_F + 0.15 * randn()
            if Fi <= 0.05:
                Fi = 0.05
            elif Fi > 0.95:
                Fi = 0.95

            # Sample CRi similarly
            CRi = mu_CR + 0.10 * randn()
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # pick pbest
            pb = pbest_pool[random.randrange(len(pbest_pool))]
            xpbest = pop[pb]

            # choose r1 != i and r2 from union (pop+archive) distinct-ish
            exclude = {i, pb}
            xr1, r1_idx = pick_from_union(exclude)
            if r1_idx is not None:
                exclude.add(r1_idx)
            xr2, _ = pick_from_union(exclude)

            # mutation: current-to-pbest/1
            # v = xi + F*(xpbest - xi) + F*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            reflect_then_clamp_inplace(v)

            # binomial crossover
            jrand = random.randrange(dim) if dim > 0 else 0
            u = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]

            reflect_then_clamp_inplace(u)
            fu = evaluate(u)

            # selection
            if fu <= fit[i]:
                # archive stores replaced parent (JADE)
                archive.append(xi[:])
                if len(archive) > arch_max:
                    # random removal keeps it simple
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                # improvement size for weighting adaptation
                imp = max(0.0, fit[i] - fu)
                if imp > 0.0:
                    SF.append(Fi)
                    SCR.append(CRi)
                    dF.append(imp)

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = u[:]

        # adapt mu_F, mu_CR from successes
        if SF:
            # weighted Lehmer mean for F (common in DE variants)
            wsum = sum(dF) if dF else float(len(SF))
            if wsum <= 0.0:
                wsum = float(len(SF))
                weights = [1.0 / wsum] * len(SF)
            else:
                weights = [imp / wsum for imp in dF]

            num = 0.0
            den = 0.0
            for w, fval in zip(weights, SF):
                num += w * (fval * fval)
                den += w * fval
            F_lehmer = (num / den) if den > 0.0 else mu_F

            CR_mean = 0.0
            for w, crval in zip(weights, SCR):
                CR_mean += w * crval

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * CR_mean

            # keep in range
            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.95, max(0.05, mu_CR))

        # stagnation management
        if best < last_best - 1e-15 * (1.0 + abs(last_best)):
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > (8 + dim // 2):
            no_improve = 0
            # inject diversity: replace a few worst with random or perturbed best
            rank = sorted_indices()
            inject = max(1, NP // 7)
            for k in range(inject):
                if now() >= deadline:
                    return best
                idx = rank[-1 - k]  # worsts
                if best_x is not None and random.random() < 0.65:
                    x = best_x[:]
                    for j in range(dim):
                        x[j] += 0.18 * span_safe[j] * randn()
                    reflect_then_clamp_inplace(x)
                else:
                    x = rand_point()
                fx = evaluate(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

        # periodic local refinement (more frequent near end)
        if best_x is not None:
            local_period = base_local_period * (1.15 - 0.70 * frac)  # smaller late
            local_period = max(0.10, min(1.2, local_period))
            if (now() - last_local) >= local_period and now() < deadline:
                last_local = now()
                remain = deadline - now()
                # spend a small budget; slightly more late
                budget = min(0.06 * float(max_time) + 0.04 * float(max_time) * frac,
                             0.28 * remain,
                             0.75)
                tl = now() + max(0.02, budget)
                xr, fr = local_refine(best_x, best, tl)
                if fr < best:
                    best, best_x = fr, xr[:]
                    # insert refined best by replacing current worst
                    worst = max(range(NP), key=lambda ii: fit[ii])
                    pop[worst] = xr[:]
                    fit[worst] = fr

    return best
