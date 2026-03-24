import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Further-improved time-bounded derivative-free minimization.

    Main upgrades vs previous best:
      - L-SHADE / JADE-style DE/current-to-pbest/1 + archive
      - Linear population size reduction (more exploration early, more exploitation late)
      - "Centered" (quasi-Latin) initialization + opposition augmentation
      - Two-stage local refinement:
          (a) adaptive coordinate search with per-dimension step sizes (success-based)
          (b) lightweight Nelder-Mead-like simplex around best (occasionally, low eval count)
      - Stronger stagnation recovery: diversify around best + random + opposition
      - Careful time checks and no external libraries

    Returns
    -------
    best : float
        Best objective value found within time budget.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s > 0 else 0.0 for s in spans]

    def time_up():
        return time.time() >= deadline

    def clip_inplace(x):
        for j in range(dim):
            lj = lows[j]
            hj = highs[j]
            if x[j] < lj:
                x[j] = lj
            elif x[j] > hj:
                x[j] = hj
        return x

    def rand_vec():
        return [lows[j] + random.random() * spans[j] for j in range(dim)]

    def opposite(x):
        return [lows[j] + highs[j] - x[j] for j in range(dim)]

    # approx N(0,1)
    def randn_approx():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_like():
        u = random.random()
        if u < 1e-12:
            u = 1e-12
        elif u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------- Initialization: quasi-Latin "centered bins" + opposition ----------
    # Larger initial pop, later reduced
    pop_max = max(24, min(120, 10 * dim + 20))
    pop_min = max(10, min(40, 4 * dim + 8))

    # Build "centered bin" samples: for each dimension, permute bins and take mid+small jitter
    pop = []
    for i in range(pop_max):
        x = [0.0] * dim
        pop.append(x)

    # per-dimension permutation of bins
    for j in range(dim):
        perm = list(range(pop_max))
        random.shuffle(perm)
        for i in range(pop_max):
            # bin center in [0,1]
            u = (perm[i] + 0.5) / float(pop_max)
            # small jitter within bin
            u += (random.random() - 0.5) / float(pop_max)
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            pop[i][j] = lows[j] + u * spans[j]

    # Add opposition points and pick best pop_max from 2*pop_max
    candidates = pop + [clip_inplace(opposite(x[:])) for x in pop]

    best = float("inf")
    best_vec = None
    fits = []
    for x in candidates:
        if time_up():
            return best
        fx = func(x)
        fits.append(fx)
        if fx < best:
            best = fx
            best_vec = x[:]

    order = sorted(range(len(candidates)), key=lambda i: fits[i])
    pop = [candidates[i][:] for i in order[:pop_max]]
    fit_pop = [fits[i] for i in order[:pop_max]]

    # ---------- L-SHADE-ish adaptation memory ----------
    H = 10
    M_CR = [0.85] * H
    M_F = [0.55] * H
    mem_idx = 0

    archive = []
    archive_max = pop_max

    # p-best fraction schedule
    p_min, p_max_frac = 0.06, 0.25

    # Stagnation handling
    no_improve_gens = 0
    restart_patience = max(20, 5 * dim)

    # Local search: adaptive coordinate steps
    coord_steps = [0.12 * spans[j] for j in range(dim)]
    coord_min = [1e-12 * spans[j] + 1e-15 for j in range(dim)]
    coord_max = [0.50 * spans[j] for j in range(dim)]

    # ---------- helpers for population reduction ----------
    def target_pop_size(progress):
        # Linear reduction from pop_max -> pop_min
        n = int(round(pop_max - (pop_max - pop_min) * progress))
        if n < pop_min:
            n = pop_min
        if n > pop_max:
            n = pop_max
        return n

    def shrink_population(new_size):
        nonlocal pop, fit_pop, archive_max, archive
        if len(pop) <= new_size:
            return
        idx = sorted(range(len(pop)), key=lambda i: fit_pop[i])[:new_size]
        pop = [pop[i] for i in idx]
        fit_pop = [fit_pop[i] for i in idx]
        archive_max = new_size
        if len(archive) > archive_max:
            # keep random subset
            random.shuffle(archive)
            archive = archive[:archive_max]

    # ---------- Simplex local search around best (tiny NM-like) ----------
    def simplex_refine(x_best, f_best, max_evals):
        # Build simplex with dim+1 points: x_best + e_j
        # Very lightweight (few iterations), works well near optimum sometimes.
        if dim == 1:
            # 1D: just try +/- step
            x0 = x_best[:]
            f0 = f_best
            step = max(coord_steps[0], 1e-12)
            for _ in range(max_evals):
                if time_up():
                    return x0, f0
                cand = x0[:]
                cand[0] = cand[0] + (step if random.random() < 0.5 else -step)
                clip_inplace(cand)
                fc = func(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    step *= 1.2
                else:
                    step *= 0.7
            return x0, f0

        # size of initial simplex steps scaled down late
        base_scale = 0.06
        simplex = [x_best[:]]
        fvals = [f_best]
        # construct orthogonal perturbations
        for j in range(dim):
            x = x_best[:]
            step = base_scale * spans[j]
            if step <= 0:
                step = 1e-12
            x[j] += step
            if x[j] > highs[j]:
                x[j] = x_best[j] - step
            clip_inplace(x)
            if time_up():
                return x_best, f_best
            fx = func(x)
            simplex.append(x)
            fvals.append(fx)
            if fx < f_best:
                x_best, f_best = x[:], fx

        evals_used = len(simplex) - 1
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while evals_used < max_evals and not time_up():
            # order
            idx = sorted(range(len(simplex)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            if fvals[0] < f_best:
                x_best, f_best = simplex[0][:], fvals[0]

            # centroid of best dim points (exclude worst)
            centroid = [0.0] * dim
            for i in range(dim):
                si = simplex[i]
                for j in range(dim):
                    centroid[j] += si[j]
            invd = 1.0 / float(dim)
            for j in range(dim):
                centroid[j] *= invd

            worst = simplex[-1]
            # reflect
            xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(dim)]
            clip_inplace(xr)
            fr = func(xr); evals_used += 1
            if fr < fvals[0] and evals_used < max_evals and not time_up():
                # expand
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(dim)]
                clip_inplace(xe)
                fe = func(xe); evals_used += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contract
                xc = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(dim)]
                clip_inplace(xc)
                fc = func(xc); evals_used += 1
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink towards best
                    bestp = simplex[0]
                    for i in range(1, len(simplex)):
                        if evals_used >= max_evals or time_up():
                            break
                        xi = simplex[i]
                        xs = [bestp[j] + sigma * (xi[j] - bestp[j]) for j in range(dim)]
                        clip_inplace(xs)
                        fs = func(xs); evals_used += 1
                        simplex[i], fvals[i] = xs, fs

        # return best in simplex
        bi = min(range(len(simplex)), key=lambda i: fvals[i])
        return simplex[bi][:], fvals[bi]

    # ---------- Main loop ----------
    gen = 0
    while not time_up():
        gen += 1

        elapsed = time.time() - t0
        progress = elapsed / max(1e-12, float(max_time))
        if progress > 1.0:
            progress = 1.0

        # Population reduction
        newN = target_pop_size(progress)
        if newN < len(pop):
            shrink_population(newN)

        N = len(pop)
        archive_max = N
        if len(archive) > archive_max:
            random.shuffle(archive)
            archive = archive[:archive_max]

        # p-best schedule
        pfrac = p_max_frac - (p_max_frac - p_min) * progress
        pbest_count = max(2, int(math.ceil(pfrac * N)))

        # ranking for p-best
        order = sorted(range(N), key=lambda i: fit_pop[i])

        improved_gen = False
        S_CR, S_F, S_df = [], [], []

        for i in range(N):
            if time_up():
                return best

            xi = pop[i]
            fi_old = fit_pop[i]

            k = random.randrange(H)
            # CR ~ N(M_CR[k], 0.1)
            CR = M_CR[k] + 0.1 * randn_approx()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # F ~ Cauchy(M_F[k], 0.1), ensure >0
            Fi = -1.0
            for _ in range(12):
                Fi = M_F[k] + 0.1 * cauchy_like()
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # choose p-best from top pbest_count
            pbest = pop[order[random.randrange(pbest_count)]]

            # choose r1 from pop != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(N)

            # choose r2 from pop U archive, try to avoid selecting same as xi by reference
            use_archive = (archive and random.random() < 0.35)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(N)
                xr2 = pop[r2]
            xr1 = pop[r1]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                val = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                # bounce-back handling (keeps diversity and feasibility)
                if val < lows[j]:
                    val = lows[j] + random.random() * (xi[j] - lows[j])
                elif val > highs[j]:
                    val = highs[j] - random.random() * (highs[j] - xi[j])
                v[j] = val

            # crossover
            j_rand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            f_trial = func(trial)
            if f_trial <= fi_old:
                # archive replaced
                archive.append(xi[:])
                if len(archive) > archive_max:
                    # remove random
                    r = random.randrange(len(archive))
                    archive[r] = archive[-1]
                    archive.pop()

                pop[i] = trial
                fit_pop[i] = f_trial

                df = fi_old - f_trial
                if df <= 0.0:
                    df = 1e-12
                S_CR.append(CR)
                S_F.append(Fi)
                S_df.append(df)

                if f_trial < best:
                    best = f_trial
                    best_vec = trial[:]
                    improved_gen = True

        # adapt memory
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = float(len(S_df))

            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_df):
                ww = w / wsum
                num += ww * f * f
                den += ww * f
            new_MF = num / max(1e-12, den)

            # weighted mean for CR
            new_MCR = 0.0
            for cr, w in zip(S_CR, S_df):
                new_MCR += (w / wsum) * cr

            # update slot
            M_F[mem_idx] = 0.6 * M_F[mem_idx] + 0.4 * new_MF
            M_CR[mem_idx] = 0.6 * M_CR[mem_idx] + 0.4 * new_MCR
            mem_idx = (mem_idx + 1) % H

        # stagnation bookkeeping
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # ---------- Local refinement (coordinate search) ----------
        # Do more later and when stagnating
        if best_vec is not None:
            do_coord = (gen % 5 == 0) or (progress > 0.6) or (no_improve_gens >= 5)
            if do_coord and not time_up():
                x0 = best_vec[:]
                f0 = best

                # number of local trials
                trials = 6 + (dim // 3)
                # scale steps down over time, up slightly on stagnation
                time_scale = max(0.02, 1.0 - progress)
                stag_scale = 1.0 + min(1.5, no_improve_gens / max(1.0, restart_patience))
                for _ in range(trials):
                    if time_up():
                        return best

                    # pick a coordinate and direction
                    j = random.randrange(dim)
                    step = coord_steps[j] * time_scale * stag_scale
                    if step < coord_min[j]:
                        step = coord_min[j]
                    if step > coord_max[j]:
                        step = coord_max[j]

                    cand = x0[:]
                    if random.random() < 0.5:
                        cand[j] += step
                    else:
                        cand[j] -= step
                    clip_inplace(cand)
                    fc = func(cand)

                    if fc < f0:
                        x0, f0 = cand, fc
                        # success: slightly increase step for this coordinate
                        coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.15)
                    else:
                        # failure: decrease step for this coordinate
                        coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.85)

                    if fc < best:
                        best = fc
                        best_vec = cand[:]
                        no_improve_gens = 0

        # ---------- Occasional simplex refinement near the end ----------
        if best_vec is not None and (progress > 0.75 or (no_improve_gens >= 8 and progress > 0.4)):
            # limit evaluation spend
            if not time_up():
                max_evals = 4 + 2 * dim
                xb, fb = simplex_refine(best_vec, best, max_evals=max_evals)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # ---------- Stagnation recovery / partial restart ----------
        if no_improve_gens >= restart_patience and not time_up():
            # keep elites; diversify others: mixture of around-best, random, opposition
            N = len(pop)
            elite_count = max(2, N // 10)
            elite_idx = sorted(range(N), key=lambda i: fit_pop[i])[:elite_count]
            elite_set = set(elite_idx)

            # restart fraction increases with stagnation severity
            frac = 0.25 + 0.20 * min(1.0, no_improve_gens / (restart_patience * 2.0))
            k_restart = max(1, int(frac * N))

            # pick worst non-elites
            worst = [i for i in sorted(range(N), key=lambda i: fit_pop[i], reverse=True) if i not in elite_set]
            worst = worst[:k_restart]

            for idx in worst:
                if time_up():
                    return best

                r = random.random()
                if best_vec is not None and r < 0.55:
                    # diversify around best with heavy-tailed noise
                    x = best_vec[:]
                    # scale noise by spans and stagnation
                    amp = 0.18 * (1.0 + min(1.5, no_improve_gens / float(restart_patience)))
                    for j in range(dim):
                        x[j] += (0.6 * randn_approx() + 0.4 * cauchy_like()) * amp * spans[j]
                    clip_inplace(x)
                elif r < 0.75:
                    # opposition of a random elite or random point
                    base = pop[random.choice(elite_idx)][:] if elite_idx else rand_vec()
                    x = clip_inplace(opposite(base))
                else:
                    x = rand_vec()

                fx = func(x)
                pop[idx] = x
                fit_pop[idx] = fx
                if fx < best:
                    best = fx
                    best_vec = x[:]

            # reset stagnation counter and slightly reset memories to encourage movement
            no_improve_gens = 0
            # gentle memory reset (not full)
            for h in range(H):
                M_F[h] = 0.5 * M_F[h] + 0.5 * (0.45 + 0.35 * random.random())
                M_CR[h] = 0.5 * M_CR[h] + 0.5 * (0.65 + 0.30 * random.random())

    return best
