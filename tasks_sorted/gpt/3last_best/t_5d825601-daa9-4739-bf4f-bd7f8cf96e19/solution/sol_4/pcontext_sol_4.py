import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (stdlib-only).

    What’s improved vs the provided best (L-SHADE-ish DE + local search):
      1) Multi-start "batch seeding" + opposition seeding (better initial coverage).
      2) Two-mutation ensemble with bandit-style adaptation:
           - current-to-pbest/1 (fast convergence)
           - rand/1 (diversity / escape)
         The algorithm learns which one works better on the current problem online.
      3) "Basin hopping" around current best using heavy-tailed steps (Cauchy-like),
         plus deterministic coordinate/pattern refinement (cheap local optimizer).
      4) Diversity-aware stagnation handling: partial restart + around-best sprays.
      5) More robust bound handling via reflection (repeatedly) to avoid sticking.

    Returns best fitness found within max_time seconds.
    """

    t0 = time.time()
    end_time = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ----------------- utilities -----------------
    def timed_out():
        return time.time() >= end_time

    def safe_eval(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_bounds_inplace(x):
        # Robust reflection: keep reflecting until in bounds (handles big jumps)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            xi = x[i]
            # reflect repeatedly if needed
            while xi < lo or xi > hi:
                if xi < lo:
                    xi = lo + (lo - xi)
                if xi > hi:
                    xi = hi - (xi - hi)
            x[i] = xi
        return x

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def cauchy_0_1(scale=1.0):
        # Cauchy(0,scale) via tan(pi*(u-0.5))
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # SHADE parameter sampling
    def sample_F(mu):
        # Cauchy around mu, ensure in (0,1]
        for _ in range(12):
            f = mu + 0.1 * cauchy_0_1(1.0)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        # fallback
        if mu <= 0.0:
            return 0.5
        return 1.0 if mu > 1.0 else mu

    def sample_CR(mu):
        cr = mu + 0.1 * random.gauss(0.0, 1.0)
        if cr < 0.0:
            return 0.0
        if cr > 1.0:
            return 1.0
        return cr

    # ----------------- initialization -----------------
    # population size: moderate, but a bit more generous early (better global search)
    NP_init = max(28, 14 * dim)
    NP_init = min(NP_init, 160)
    NP_min = max(10, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(10, NP_init // 2)

    # SHADE memories
    H = 12
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    p_best_rate = 0.12
    p_best_rate = max(0.05, min(0.30, p_best_rate))

    # archive
    archive = []
    archive_cap = NP_init

    # Multi-start seeding: random + opposition + a few "sprays"
    pop = []
    while len(pop) < NP_init:
        x = rand_vec()
        pop.append(x)
        if len(pop) < NP_init:
            pop.append(reflect_bounds_inplace(opposition(x)))
    pop = pop[:NP_init]

    fit = [float("inf")] * NP_init

    best = float("inf")
    best_x = None

    # evaluate initial pop
    for i in range(NP_init):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # ----------------- local improvement primitives -----------------
    ls_sigma = 0.12
    ls_sigma_min = 1e-7
    ls_sigma_max = 0.45

    def coord_pattern_refine(xb, fb, frac):
        """Cheap deterministic-ish local improvement around best."""
        nonlocal ls_sigma
        if xb is None:
            return xb, fb

        x = xb[:]
        f = fb

        # more exploitation later
        coord_tries = min(dim, 5 + int(8 * frac))
        pattern_tries = 1 + int(3 * frac)

        # coordinate moves
        idxs = list(range(dim))
        random.shuffle(idxs)
        for t in range(coord_tries):
            if timed_out():
                break
            d = idxs[t % dim]
            step = ls_sigma * spans[d]
            if step <= 0.0:
                continue

            improved = False
            for s in (1.0, -1.0):
                xt = x[:]
                xt[d] += s * step
                reflect_bounds_inplace(xt)
                ft = safe_eval(xt)
                if ft < f:
                    x, f = xt, ft
                    improved = True
                    break
            if improved:
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.90)

        # sparse pattern moves (few dims at a time)
        for _ in range(pattern_tries):
            if timed_out():
                break
            xt = x[:]
            prob = 0.10 + 0.20 * (1.0 - frac)
            for d in range(dim):
                if random.random() < prob:
                    xt[d] += (random.random() * 2.0 - 1.0) * ls_sigma * spans[d]
            reflect_bounds_inplace(xt)
            ft = safe_eval(xt)
            if ft < f:
                x, f = xt, ft
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.93)

        return x, f

    def basin_hop(best_x, best_f, frac):
        """
        Heavy-tailed 'kick' around best to escape local minima, then refine.
        Keeps kicks rare early, more focused late.
        """
        nonlocal ls_sigma
        if best_x is None:
            return best_x, best_f

        # time-dependent intensity: smaller later
        kick_scale = (0.25 * (1.0 - frac) + 0.03)
        # attempt a few hops; keep tiny budget
        hops = 1 if frac < 0.5 else 2

        x_best = best_x[:]
        f_best = best_f

        for _ in range(hops):
            if timed_out():
                break
            xt = x_best[:]
            # cauchy kicks in a few random dims
            k = 1 + int(0.15 * dim)
            for __ in range(k):
                d = random.randrange(dim)
                xt[d] += cauchy_0_1(1.0) * kick_scale * spans[d]
            reflect_bounds_inplace(xt)
            ft = safe_eval(xt)
            if ft < f_best:
                x_best, f_best = xt, ft
                x_best, f_best = coord_pattern_refine(x_best, f_best, frac)
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.92)
            else:
                ls_sigma = min(ls_sigma_max, ls_sigma * 1.04)

        return x_best, f_best

    # ----------------- DE main loop with strategy bandit -----------------
    # Two strategies: 0=current-to-pbest/1, 1=rand/1
    # Maintain weights; update by recent improvements (simple bandit)
    strat_w = [1.0, 1.0]
    strat_decay = 0.90

    no_improve_gens = 0
    stagnation_limit = max(10, 3 * dim)

    gen = 0
    while not timed_out():
        gen += 1
        now = time.time()
        frac = (now - t0) / (max_time + 1e-12)
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0

        # linear population size reduction
        NP_target = int(round(NP_init - frac * (NP_init - NP_min)))
        if NP_target < NP_min:
            NP_target = NP_min

        NP = len(pop)
        if NP_target < NP:
            keep = sorted(range(NP), key=lambda i: fit[i])[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = NP_target
            archive_cap = max(NP, NP_init // 2)
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

        # ranking
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_best_rate * NP)))

        # SHADE success histories
        S_F, S_CR, S_df = [], [], []
        improved_gen = False

        # strategy success (for bandit)
        strat_gain = [0.0, 0.0]

        for i in range(NP):
            if timed_out():
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # choose strategy by weights
            sw0, sw1 = strat_w
            if random.random() < (sw0 / (sw0 + sw1 + 1e-12)):
                strat = 0
            else:
                strat = 1

            # pick r1, r2, r3
            # helper to pick from pop excluding a set
            def pick_pop(excl):
                while True:
                    j = random.randrange(NP)
                    if j not in excl:
                        return j

            if strat == 0:
                # current-to-pbest/1
                pbest = pop[idx_sorted[random.randrange(p_count)]]
                r1 = pick_pop({i})
                # r2 from pop or archive
                use_arch = (archive and random.random() < 0.5)
                if use_arch:
                    combined_n = NP + len(archive)
                    while True:
                        j = random.randrange(combined_n)
                        if j == i or j == r1:
                            continue
                        xr2 = pop[j] if j < NP else archive[j - NP]
                        break
                else:
                    r2 = pick_pop({i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]

                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            else:
                # rand/1 with archive option on r3 (diversity)
                r1 = pick_pop({i})
                r2 = pick_pop({i, r1})
                x1 = pop[r1]
                x2 = pop[r2]
                use_arch = (archive and random.random() < 0.35)
                if use_arch:
                    x3 = archive[random.randrange(len(archive))]
                else:
                    r3 = pick_pop({i, r1, r2})
                    x3 = pop[r3]

                v = [0.0] * dim
                for d in range(dim):
                    v[d] = x1[d] + F * (x2[d] - x3[d])

            reflect_bounds_inplace(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CR) else xi[d]

            # small best-injection late
            if best_x is not None and random.random() < (0.008 + 0.05 * frac):
                d = random.randrange(dim)
                u[d] = 0.80 * u[d] + 0.20 * best_x[d]
                reflect_bounds_inplace(u)

            fu = safe_eval(u)

            if fu <= fi:
                # archive parent
                archive.append(xi[:])
                if len(archive) > archive_cap:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = u
                fit[i] = fu

                df = fi - fu
                if df > 0.0:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(df)
                    strat_gain[strat] += df

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_gen = True

        # update SHADE memories
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                weights = [1.0 / float(len(S_df))] * len(S_df)
            else:
                inv = 1.0 / wsum
                weights = [df * inv for df in S_df]

            meanCR = 0.0
            for w, cr in zip(weights, S_CR):
                meanCR += w * cr

            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            meanF = (num / den) if den != 0.0 else M_F[mem_idx]

            M_CR[mem_idx] = min(1.0, max(0.0, meanCR))
            M_F[mem_idx] = min(1.0, max(0.05, meanF))
            mem_idx = (mem_idx + 1) % H

        # update strategy weights (bandit)
        # decay, then add normalized gains
        strat_w[0] = strat_w[0] * strat_decay + (strat_gain[0] + 1e-12)
        strat_w[1] = strat_w[1] * strat_decay + (strat_gain[1] + 1e-12)
        # prevent collapse
        minw = 1e-6
        if strat_w[0] < minw: strat_w[0] = minw
        if strat_w[1] < minw: strat_w[1] = minw

        # local improvement schedule
        if best_x is not None and not timed_out():
            # refine more often late, or after improvements
            if improved_gen or random.random() < (0.06 + 0.40 * frac):
                bx, bf = coord_pattern_refine(best_x, best, frac)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

            # occasional basin hop (escape)
            if random.random() < (0.02 + 0.10 * (1.0 - frac)):
                bx, bf = basin_hop(best_x, best, frac)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # stagnation handling
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagnation_limit and not timed_out():
            no_improve_gens = 0
            NP = len(pop)
            k = max(2, NP // 5)  # refresh ~20%
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for wi in worst:
                if timed_out():
                    return best
                if best_x is not None and random.random() < 0.70:
                    x = best_x[:]
                    # spray around best (shrinks over time)
                    rad = (0.22 * (1.0 - frac) + 0.04)
                    for d in range(dim):
                        if random.random() < 0.35:
                            x[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
                    reflect_bounds_inplace(x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, x[:]

            # thin archive
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

    return best
