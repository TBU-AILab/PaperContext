import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no external libs).

    Improvements over the provided JADE/SHADE-lite:
      - Better initialization: centered Latin-hypercube + opposition, then elitist downselect
      - L-SHADE-like core: DE/current-to-pbest/1 + archive + success-history adaptation
      - Linear population size reduction (explore early, exploit late)
      - Robust bound handling (bounce-back) and occasional re-evaluation-free diversity injections
      - Stronger local refinement: (1) adaptive coordinate pattern search (per-dimension steps)
                                 (2) small simplex (very lightweight Nelder-Mead-like) near the end
      - Stagnation recovery: keep elites, re-seed others via around-best heavy-tail / random / opposition

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

    # ---------------- utils ----------------
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

    def randn_approx():
        # ~N(0,1) via CLT, cheap
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy_like():
        # heavy tail (avoid tan blow-ups)
        u = random.random()
        if u < 1e-12:
            u = 1e-12
        elif u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------------- initialization (centered LHS + opposition) ----------------
    pop_max = max(24, min(120, 10 * dim + 20))
    pop_min = max(10, min(50, 4 * dim + 10))

    # centered Latin hypercube-ish (per-dimension permuted bins, mid + small jitter)
    pop = [[0.0] * dim for _ in range(pop_max)]
    for j in range(dim):
        perm = list(range(pop_max))
        random.shuffle(perm)
        inv = 1.0 / float(pop_max)
        for i in range(pop_max):
            u = (perm[i] + 0.5) * inv
            u += (random.random() - 0.5) * inv
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            pop[i][j] = lows[j] + u * spans[j]

    # add opposition points, then downselect best pop_max among 2*pop_max
    candidates = pop + [clip_inplace(opposite(x[:])) for x in pop]
    cand_fit = []
    best = float("inf")
    best_vec = None
    for x in candidates:
        if time_up():
            return best
        fx = func(x)
        cand_fit.append(fx)
        if fx < best:
            best = fx
            best_vec = x[:]

    order = sorted(range(len(candidates)), key=lambda i: cand_fit[i])
    pop = [candidates[i][:] for i in order[:pop_max]]
    fit_pop = [cand_fit[i] for i in order[:pop_max]]

    # ---------------- L-SHADE-ish memory + archive ----------------
    H = 10
    M_CR = [0.85] * H
    M_F = [0.55] * H
    mem_idx = 0

    archive = []
    archive_max = len(pop)

    # pbest schedule
    p_min, p_max_frac = 0.06, 0.25

    # population size reduction
    def target_pop_size(progress):
        n = int(round(pop_max - (pop_max - pop_min) * progress))
        if n < pop_min:
            n = pop_min
        if n > pop_max:
            n = pop_max
        return n

    def shrink_population(new_size):
        nonlocal pop, fit_pop, archive, archive_max
        if len(pop) <= new_size:
            return
        idx = sorted(range(len(pop)), key=lambda i: fit_pop[i])[:new_size]
        pop = [pop[i] for i in idx]
        fit_pop = [fit_pop[i] for i in idx]
        archive_max = new_size
        if len(archive) > archive_max:
            random.shuffle(archive)
            archive = archive[:archive_max]

    # ---------------- local search: adaptive coordinate pattern search ----------------
    coord_steps = [0.12 * spans[j] if spans[j] > 0 else 1e-9 for j in range(dim)]
    coord_min = [1e-12 * (spans[j] if spans[j] > 0 else 1.0) + 1e-15 for j in range(dim)]
    coord_max = [0.50 * spans[j] if spans[j] > 0 else 1.0 for j in range(dim)]

    def coord_refine(x_best, f_best, progress, tries):
        # per-try: pick a dim, try +/- step, adapt step on success/failure
        x0 = x_best[:]
        f0 = f_best
        time_scale = max(0.02, 1.0 - progress)
        for _ in range(tries):
            if time_up():
                return x0, f0
            j = random.randrange(dim)
            step = coord_steps[j] * time_scale
            if step < coord_min[j]:
                step = coord_min[j]
            if step > coord_max[j]:
                step = coord_max[j]

            # try one direction (random), if fails occasionally try opposite
            for attempt in (0, 1):
                if time_up():
                    return x0, f0
                cand = x0[:]
                direction = 1.0 if (random.random() < 0.5) else -1.0
                if attempt == 1:
                    direction = -direction
                cand[j] += direction * step
                clip_inplace(cand)
                fc = func(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.18)
                    break
                else:
                    coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.86)

            if f0 < f_best:
                f_best = f0
        return x0, f0

    # ---------------- tiny simplex refine (very lightweight Nelder-Mead-ish) ----------------
    def simplex_refine(x_best, f_best, max_evals):
        if dim == 1:
            x0, f0 = x_best[:], f_best
            step = max(coord_steps[0], 1e-12)
            for _ in range(max_evals):
                if time_up():
                    return x0, f0
                cand = x0[:]
                cand[0] += (step if random.random() < 0.5 else -step)
                clip_inplace(cand)
                fc = func(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    step *= 1.2
                else:
                    step *= 0.7
            return x0, f0

        base_scale = 0.06
        simplex = [x_best[:]]
        fvals = [f_best]

        # create dim additional points
        for j in range(dim):
            if time_up():
                return x_best, f_best
            x = x_best[:]
            step = base_scale * (spans[j] if spans[j] > 0 else 1.0)
            if step <= 0.0:
                step = 1e-12
            x[j] += step
            if x[j] > highs[j]:
                x[j] = x_best[j] - step
            clip_inplace(x)
            fx = func(x)
            simplex.append(x)
            fvals.append(fx)

        evals_used = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while evals_used < max_evals and not time_up():
            idx = sorted(range(len(simplex)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            if fvals[0] < f_best:
                x_best, f_best = simplex[0][:], fvals[0]

            # centroid excluding worst
            centroid = [0.0] * dim
            for i in range(dim):
                si = simplex[i]
                for j in range(dim):
                    centroid[j] += si[j]
            invd = 1.0 / float(dim)
            for j in range(dim):
                centroid[j] *= invd

            worst = simplex[-1]

            xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(dim)]
            clip_inplace(xr)
            fr = func(xr); evals_used += 1

            if fr < fvals[0] and evals_used < max_evals and not time_up():
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
                xc = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(dim)]
                clip_inplace(xc)
                fc = func(xc); evals_used += 1
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    bestp = simplex[0]
                    for i in range(1, len(simplex)):
                        if evals_used >= max_evals or time_up():
                            break
                        xi = simplex[i]
                        xs = [bestp[j] + sigma * (xi[j] - bestp[j]) for j in range(dim)]
                        clip_inplace(xs)
                        fs = func(xs); evals_used += 1
                        simplex[i], fvals[i] = xs, fs

        bi = min(range(len(simplex)), key=lambda i: fvals[i])
        return simplex[bi][:], fvals[bi]

    # ---------------- main loop ----------------
    gen = 0
    no_improve_gens = 0
    restart_patience = max(20, 5 * dim)

    while not time_up():
        gen += 1

        elapsed = time.time() - t0
        progress = elapsed / max(1e-12, float(max_time))
        if progress > 1.0:
            progress = 1.0

        # reduce population
        newN = target_pop_size(progress)
        if newN < len(pop):
            shrink_population(newN)

        N = len(pop)
        archive_max = N
        if len(archive) > archive_max:
            random.shuffle(archive)
            archive = archive[:archive_max]

        # p-best
        pfrac = p_max_frac - (p_max_frac - p_min) * progress
        pbest_count = max(2, int(math.ceil(pfrac * N)))
        order = sorted(range(N), key=lambda i: fit_pop[i])

        improved_gen = False
        S_CR, S_F, S_df = [], [], []

        for i in range(N):
            if time_up():
                return best

            xi = pop[i]
            fi_old = fit_pop[i]

            k = random.randrange(H)

            CR = M_CR[k] + 0.1 * randn_approx()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            Fi = -1.0
            for _ in range(12):
                Fi = M_F[k] + 0.1 * cauchy_like()
                if Fi > 0.0:
                    break
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            pbest = pop[order[random.randrange(pbest_count)]]

            r1 = i
            while r1 == i:
                r1 = random.randrange(N)

            use_archive = (archive and random.random() < 0.35)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(N)
                xr2 = pop[r2]

            xr1 = pop[r1]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                val = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

                # bounce-back bounds handling
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
                archive.append(xi[:])
                if len(archive) > archive_max:
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

            M_F[mem_idx] = 0.6 * M_F[mem_idx] + 0.4 * new_MF
            M_CR[mem_idx] = 0.6 * M_CR[mem_idx] + 0.4 * new_MCR
            mem_idx = (mem_idx + 1) % H

        # stagnation
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # local refinement: coordinate search (more later / if stagnating)
        if best_vec is not None and not time_up():
            do_coord = (gen % 5 == 0) or (progress > 0.60) or (no_improve_gens >= 5)
            if do_coord:
                tries = 6 + (dim // 3)
                xb, fb = coord_refine(best_vec, best, progress, tries)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # near-end simplex refinement
        if best_vec is not None and not time_up():
            if progress > 0.78 or (progress > 0.45 and no_improve_gens >= 8):
                max_evals = 4 + 2 * dim
                xb, fb = simplex_refine(best_vec, best, max_evals)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # stagnation recovery: keep elites, reseed some worst
        if no_improve_gens >= restart_patience and not time_up():
            N = len(pop)
            elite_count = max(2, N // 10)
            elite_idx = sorted(range(N), key=lambda i: fit_pop[i])[:elite_count]
            elite_set = set(elite_idx)

            frac = 0.25 + 0.20 * min(1.0, no_improve_gens / (restart_patience * 2.0))
            k_restart = max(1, int(frac * N))

            worst = [i for i in sorted(range(N), key=lambda i: fit_pop[i], reverse=True) if i not in elite_set]
            worst = worst[:k_restart]

            for idx in worst:
                if time_up():
                    return best

                r = random.random()
                if best_vec is not None and r < 0.55:
                    # around best with heavy-tail noise
                    x = best_vec[:]
                    amp = 0.18 * (1.0 + min(1.5, no_improve_gens / float(restart_patience)))
                    for j in range(dim):
                        x[j] += (0.6 * randn_approx() + 0.4 * cauchy_like()) * amp * (spans[j] if spans[j] > 0 else 1.0)
                    clip_inplace(x)
                elif r < 0.75:
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

            # mild memory randomization to encourage movement
            for h in range(H):
                M_F[h] = 0.5 * M_F[h] + 0.5 * (0.45 + 0.35 * random.random())
                M_CR[h] = 0.5 * M_CR[h] + 0.5 * (0.65 + 0.30 * random.random())

            no_improve_gens = 0

    return best
