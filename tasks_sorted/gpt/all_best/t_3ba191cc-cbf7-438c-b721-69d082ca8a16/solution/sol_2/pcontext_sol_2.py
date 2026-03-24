import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Hybrid:
      - Differential Evolution (DE/current-to-pbest/1) with JADE-like p-best selection
      - External archive (as in JADE) to increase diversity
      - Success-history adaptation of F and CR (lightweight SHADE-style)
      - Opposition-based init + occasional re-sampling of worst
      - Budget-aware local search around the incumbent best (coordinate + random direction)
      - Strict time checks

    Returns
    -------
    best : float
        Best objective value found within time budget.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- bounds helpers
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def time_up():
        return time.time() >= deadline

    def clip_inplace(x):
        for j in range(dim):
            if x[j] < lows[j]:
                x[j] = lows[j]
            elif x[j] > highs[j]:
                x[j] = highs[j]
        return x

    def rand_vec():
        return [lows[j] + random.random() * spans[j] for j in range(dim)]

    def opposite(x):
        return [lows[j] + highs[j] - x[j] for j in range(dim)]

    # ---- sampling utilities
    def randn_approx():
        # sum of 12 uniforms - 6 ~ N(0,1) (CLT), cheap and no external libs
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_like():
        # heavy-tailed; clip later
        u = random.random()
        # avoid tan blowups
        u = 1e-12 if u <= 1e-12 else (1.0 - 1e-12 if u >= 1.0 - 1e-12 else u)
        return math.tan(math.pi * (u - 0.5))

    # ---- population size
    # Keep moderate; too large wastes evals, too small stagnates.
    pop_size = max(18, min(80, 8 * dim + 10))

    # ---- init: opposition-based, select best pop_size from 2*pop_size
    pop = [rand_vec() for _ in range(pop_size)]
    opp = [clip_inplace(opposite(x[:])) for x in pop]
    candidates = pop + opp

    fits = []
    best = float("inf")
    best_vec = None

    for x in candidates:
        if time_up():
            return best
        fx = func(x)
        fits.append(fx)
        if fx < best:
            best = fx
            best_vec = x[:]

    idx_sorted = sorted(range(len(candidates)), key=lambda i: fits[i])
    pop = [candidates[i][:] for i in idx_sorted[:pop_size]]
    fit_pop = [fits[i] for i in idx_sorted[:pop_size]]

    # ---- archive (JADE): store replaced solutions
    archive = []
    archive_max = pop_size

    # ---- success-history parameters (SHADE-lite)
    H = 8  # memory size
    M_CR = [0.8] * H
    M_F = [0.6] * H
    mem_idx = 0

    # ---- p-best fraction
    p_min = 0.08
    p_max = 0.25

    # ---- stagnation controls
    no_improve_gens = 0
    restart_patience = max(25, 6 * dim)
    restart_frac = 0.30
    elite_count = max(2, pop_size // 10)

    # ---- local search controls
    local_every = 6  # generations
    base_local_trials = 8

    gen = 0
    while not time_up():
        gen += 1
        improved_gen = False

        # sort indices by fitness once per generation for p-best selection
        order = sorted(range(pop_size), key=lambda i: fit_pop[i])

        # time-progress for scheduling
        elapsed = time.time() - t0
        progress = elapsed / max(1e-12, float(max_time))
        if progress > 1.0:
            progress = 1.0

        # p in [p_min, p_max], larger early for exploration
        p = p_max - (p_max - p_min) * progress
        pbest_count = max(2, int(math.ceil(p * pop_size)))

        # generation success lists for adapting memory
        S_CR = []
        S_F = []
        S_df = []  # improvements

        # evolve each individual
        for i in range(pop_size):
            if time_up():
                return best

            xi = pop[i]

            # sample CR from N(M_CR[k], 0.1), clamp to [0,1]
            k = random.randrange(H)
            CR = M_CR[k] + 0.1 * randn_approx()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # sample F from Cauchy(M_F[k], 0.1), re-sample if <=0, clamp <=1
            Fi = 0.0
            for _ in range(10):
                Fi = M_F[k] + 0.1 * cauchy_like()
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # choose p-best from top pbest_count
            pbest = pop[order[random.randrange(pbest_count)]]

            # choose r1 from population, r2 from population U archive, all distinct
            # build index pool for r1
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # r2 can come from archive if non-empty
            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                # ensure not equal to xi / r1 by value is hard; use index-based distinctness on pop, archive is separate
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                xr2 = pop[r2]

            xr1 = pop[r1]

            # mutation: current-to-pbest/1 with archive option
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

                # boundary handling: if out, bounce back towards xi (keeps feasibility without sticking)
                if v[j] < lows[j]:
                    v[j] = lows[j] + random.random() * (xi[j] - lows[j])
                elif v[j] > highs[j]:
                    v[j] = highs[j] - random.random() * (highs[j] - xi[j])

            # binomial crossover
            j_rand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # evaluate
            f_trial = func(trial)
            if f_trial <= fit_pop[i]:
                # archive the replaced vector
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # random removal
                    archive[random.randrange(len(archive))] = archive[-1]
                    archive.pop()

                # selection
                old = fit_pop[i]
                pop[i] = trial
                fit_pop[i] = f_trial

                # record success for adaptation
                df = max(0.0, old - f_trial)
                S_CR.append(CR)
                S_F.append(Fi)
                S_df.append(df if df > 0.0 else 1e-12)

                if f_trial < best:
                    best = f_trial
                    best_vec = trial[:]
                    improved_gen = True

        # adapt memories if we had successes
        if S_F:
            # weighted means by improvement (as in JADE/SHADE)
            w_sum = sum(S_df)
            if w_sum <= 0.0:
                w_sum = float(len(S_df))

            # Lehmer mean for F (favors larger successful F)
            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_df):
                ww = w / w_sum
                num += ww * f * f
                den += ww * f
            new_MF = num / max(1e-12, den)

            # weighted arithmetic mean for CR
            new_MCR = 0.0
            for cr, w in zip(S_CR, S_df):
                new_MCR += (w / w_sum) * cr

            # smooth update to avoid volatility
            M_F[mem_idx] = 0.7 * M_F[mem_idx] + 0.3 * new_MF
            M_CR[mem_idx] = 0.7 * M_CR[mem_idx] + 0.3 * new_MCR
            mem_idx = (mem_idx + 1) % H

        # stagnation bookkeeping
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # local refinement around best (more towards end or if stagnating)
        if best_vec is not None and (gen % local_every == 0 or no_improve_gens >= 6 or progress >= 0.7):
            if time_up():
                return best

            # step shrinks with progress; slightly grows with stagnation
            stagn_boost = 1.0 + min(2.0, no_improve_gens / max(1.0, restart_patience))
            step = 0.12 * (1.0 - progress) * stagn_boost
            if step > 1e-12:
                trials = base_local_trials + (dim // 4)
                x0 = best_vec[:]
                f0 = best

                for _ in range(trials):
                    if time_up():
                        return best

                    cand = x0[:]

                    if random.random() < 0.65:
                        # coordinate-wise: tweak 1-3 coordinates
                        kcoords = 1 if dim == 1 else (1 + (1 if random.random() < 0.7 else min(2, dim - 1)))
                        for __ in range(kcoords):
                            j = random.randrange(dim)
                            u = (random.random() + random.random()) - 1.0  # ~[-1,1]
                            cand[j] += u * step * spans[j]
                    else:
                        # random direction
                        for j in range(dim):
                            cand[j] += randn_approx() * (0.35 * step) * spans[j]

                    clip_inplace(cand)
                    fc = func(cand)
                    if fc < f0:
                        x0, f0 = cand, fc
                        if fc < best:
                            best, best_vec = fc, cand[:]
                            improved_gen = True
                            no_improve_gens = 0

        # partial restart if stuck: keep elites, reinit worst
        if no_improve_gens >= restart_patience:
            if time_up():
                return best

            elite_idxs = sorted(range(pop_size), key=lambda k: fit_pop[k])[:elite_count]
            elite_set = set(elite_idxs)

            k_restart = max(1, int(pop_size * restart_frac))
            worst = [k for k in sorted(range(pop_size), key=lambda k: fit_pop[k], reverse=True) if k not in elite_set]
            worst = worst[:k_restart]

            # reinit around best half the time, otherwise random
            for k in worst:
                if time_up():
                    return best

                if best_vec is not None and random.random() < 0.55:
                    x = best_vec[:]
                    # broadened perturbation
                    for j in range(dim):
                        x[j] += randn_approx() * 0.25 * spans[j]
                    clip_inplace(x)
                else:
                    x = rand_vec()

                fx = func(x)
                pop[k] = x
                fit_pop[k] = fx
                if fx < best:
                    best = fx
                    best_vec = x[:]
                    improved_gen = True

            # also trim archive to avoid staleness after restart
            if len(archive) > archive_max:
                archive = archive[:archive_max]

            no_improve_gens = 0

    return best
