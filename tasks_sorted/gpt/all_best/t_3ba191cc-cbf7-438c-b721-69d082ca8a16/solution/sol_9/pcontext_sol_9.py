import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Compared to your best (L-SHADE-ish current-to-pbest/1 + archive + coord + SPSA),
    this version focuses on (a) spending evaluations where they matter, and
    (b) making local exploitation more reliable under short budgets.

    Key improvements:
      1) Evaluation caching (rounded key) to avoid wasting calls on duplicates (common with reflections/local steps).
      2) Dual-mode DE mutation:
           - current-to-pbest/1 (as before) for fast convergence
           - rand-to-pbest/1 (triggered on stagnation / late) to re-inject diversity without full restart
      3) Better bound handling: reflection with "random re-entry" if still problematic.
      4) Stronger local search:
           - deterministic coordinate pattern search (± step) with per-dimension adaptation (kept)
           - *random subspace quadratic line-search* (1D parabola fit) on a few dims near best (3 evals each)
             which often beats SPSA when objective is smooth-ish/noisy but not too chaotic.
           - SPSA kept but made cheaper and only used when dim is large or very stuck.
      5) Restart strategy refined:
           - small frequent "micro-reseeds" of a few worst individuals (cheap, prevents collapse)
           - full partial restart only when truly stuck

    Returns
    -------
    best : float
        Best objective value found within the time budget.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------------- time / cache ----------------
    def time_up():
        return time.time() >= deadline

    # Cache: rounded vector -> fitness
    # Rounding reduces key size and increases hit rate; scaled by span to be roughly relative.
    cache = {}
    # A slightly coarse rounding is usually best under tight budgets.
    # Using relative bins: ~1e-10 of span is too fine; ~1e-6..1e-8 is ok.
    rel_round = 1e-8

    def key_of(x):
        # scale-aware rounding; avoid division by zero via spans already fixed
        return tuple(int(round((x[j] - lows[j]) / spans[j] / rel_round)) for j in range(dim))

    def eval_f(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = func(x)
        cache[k] = fx
        return fx

    # ---------------- random utils ----------------
    def rand_vec():
        return [lows[j] + random.random() * spans[j] for j in range(dim)]

    def opposite(x):
        return [lows[j] + highs[j] - x[j] for j in range(dim)]

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

    # ---------------- bounds handling ----------------
    def reflect_reenter_inplace(x, parent=None):
        # reflect; if still out (can happen after big jumps), re-enter randomly (optionally around parent)
        for j in range(dim):
            lo, hi = lows[j], highs[j]
            if x[j] < lo:
                x[j] = lo + (lo - x[j])
                if x[j] > hi:
                    if parent is None:
                        x[j] = lo + random.random() * (hi - lo)
                    else:
                        # random point between bound and parent coordinate
                        pj = parent[j]
                        if pj < lo: pj = lo
                        if pj > hi: pj = hi
                        x[j] = lo + random.random() * (pj - lo) if pj >= lo else lo
            elif x[j] > hi:
                x[j] = hi - (x[j] - hi)
                if x[j] < lo:
                    if parent is None:
                        x[j] = lo + random.random() * (hi - lo)
                    else:
                        pj = parent[j]
                        if pj < lo: pj = lo
                        if pj > hi: pj = hi
                        x[j] = pj + random.random() * (hi - pj) if pj <= hi else hi

            # final clamp
            if x[j] < lo: x[j] = lo
            elif x[j] > hi: x[j] = hi
        return x

    # ---------------- Initialization: centered LHS + opposition + elitist ----------------
    pop_max = max(24, min(140, 10 * dim + 30))
    pop_min = max(10, min(60, 4 * dim + 12))

    base = [[0.0] * dim for _ in range(pop_max)]
    for j in range(dim):
        perm = list(range(pop_max))
        random.shuffle(perm)
        inv = 1.0 / float(pop_max)
        for i in range(pop_max):
            u = (perm[i] + 0.5) * inv
            u += (random.random() - 0.5) * inv  # jitter
            if u < 0.0: u = 0.0
            elif u > 1.0: u = 1.0
            base[i][j] = lows[j] + u * spans[j]

    candidates = base + [reflect_reenter_inplace(opposite(x[:])) for x in base]

    best = float("inf")
    best_vec = None
    cand_fit = []
    for x in candidates:
        if time_up():
            return best
        fx = eval_f(x)
        cand_fit.append(fx)
        if fx < best:
            best = fx
            best_vec = x[:]

    order = sorted(range(len(candidates)), key=lambda i: cand_fit[i])
    pop = [candidates[i][:] for i in order[:pop_max]]
    fit_pop = [cand_fit[i] for i in order[:pop_max]]

    # ---------------- L-SHADE memory + archive + pop reduction ----------------
    H = 10
    M_CR = [0.85] * H
    M_F = [0.55] * H
    mem_idx = 0

    archive = []
    archive_max = len(pop)

    p_min, p_max_frac = 0.06, 0.25

    def target_pop_size(progress):
        n = int(round(pop_max - (pop_max - pop_min) * progress))
        if n < pop_min: n = pop_min
        if n > pop_max: n = pop_max
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

    # ---------------- Local search: coord steps + quadratic line-search + SPSA ----------------
    coord_steps = [0.12 * spans[j] for j in range(dim)]
    coord_min = [1e-12 * spans[j] + 1e-15 for j in range(dim)]
    coord_max = [0.50 * spans[j] for j in range(dim)]

    def coord_refine(x_best, f_best, progress, tries):
        x0 = x_best[:]
        f0 = f_best
        time_scale = max(0.02, 1.0 - progress)
        for _ in range(tries):
            if time_up():
                return x0, f0
            j = random.randrange(dim)
            step = coord_steps[j] * time_scale
            if step < coord_min[j]: step = coord_min[j]
            if step > coord_max[j]: step = coord_max[j]

            improved = False
            for direction in (1.0, -1.0):
                if time_up():
                    return x0, f0
                cand = x0[:]
                cand[j] += direction * step
                reflect_reenter_inplace(cand, parent=x0)
                fc = eval_f(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.20)
                    improved = True
                    break
            if not improved:
                coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.85)
        return x0, f0

    def quad_line_search_dims(x_best, f_best, progress, dims_to_try):
        """
        For each chosen dimension j, evaluate at -h, 0, +h and fit a parabola.
        If curvature is positive, jump to argmin. Uses 2-3 evals/dim (3 including center if not cached).
        """
        x = x_best[:]
        f = f_best
        # h shrinks with progress; also relates to coord_steps
        shrink = max(0.02, 1.0 - progress)
        for j in dims_to_try:
            if time_up():
                return x, f

            h = coord_steps[j] * 0.75 * shrink
            if h < coord_min[j]:
                h = coord_min[j]
            if h > coord_max[j]:
                h = coord_max[j]

            # center value (often cached)
            f0 = eval_f(x)

            xm = x[:]; xm[j] -= h
            xp = x[:]; xp[j] += h
            reflect_reenter_inplace(xm, parent=x)
            reflect_reenter_inplace(xp, parent=x)

            fm = eval_f(xm)
            if time_up():
                return x, f
            fp = eval_f(xp)

            # Parabola through (-h,fm), (0,f0), (+h,fp):
            # a = (fp + fm - 2f0) / (2h^2), b = (fp - fm) / (2h)
            denom = 2.0 * h * h
            a = (fp + fm - 2.0 * f0) / (denom + 1e-300)
            b = (fp - fm) / (2.0 * h + 1e-300)

            if a > 1e-18:
                t = -b / (2.0 * a)
                # limit step to reasonable range
                if t > 2.0 * h: t = 2.0 * h
                elif t < -2.0 * h: t = -2.0 * h

                cand = x[:]
                cand[j] += t
                reflect_reenter_inplace(cand, parent=x)
                fc = eval_f(cand)
                if fc < f:
                    x, f = cand, fc
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.10)
                else:
                    coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.92)
            else:
                # no positive curvature; just accept best among {xm, x, xp} if better
                if fm < f:
                    x, f = xm, fm
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.08)
                elif fp < f:
                    x, f = xp, fp
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.08)
                else:
                    coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.92)

        return x, f

    def spsa_refine(x_best, f_best, progress, steps):
        x = x_best[:]
        f = f_best
        a = 0.16 * (1.0 - progress) + 0.015
        c = 0.07 * (1.0 - progress) + 0.008
        for k in range(steps):
            if time_up():
                return x, f
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            ck = c / math.sqrt(1.0 + k)

            x_plus = [x[j] + ck * delta[j] * spans[j] for j in range(dim)]
            x_minus = [x[j] - ck * delta[j] * spans[j] for j in range(dim)]
            reflect_reenter_inplace(x_plus, parent=x)
            reflect_reenter_inplace(x_minus, parent=x)

            f_plus = eval_f(x_plus)
            if time_up():
                return x, f
            f_minus = eval_f(x_minus)

            ak = a / (1.0 + 0.20 * k)
            cand = x[:]
            for j in range(dim):
                g = (f_plus - f_minus) / (2.0 * ck * spans[j] * delta[j] + 1e-18)
                cand[j] -= ak * g
            reflect_reenter_inplace(cand, parent=x)
            fc = eval_f(cand)
            if fc < f:
                x, f = cand, fc
        return x, f

    # ---------------- Main loop ----------------
    gen = 0
    no_improve_gens = 0
    restart_patience = max(18, 4 * dim)

    # micro-reseed controls
    micro_every = 6               # generations
    micro_count_min = 1
    micro_count_max = 3

    while not time_up():
        gen += 1

        elapsed = time.time() - t0
        progress = elapsed / max(1e-12, float(max_time))
        if progress > 1.0:
            progress = 1.0

        # population size reduction
        newN = target_pop_size(progress)
        if newN < len(pop):
            shrink_population(newN)

        N = len(pop)
        archive_max = N
        if len(archive) > archive_max:
            random.shuffle(archive)
            archive = archive[:archive_max]

        pfrac = p_max_frac - (p_max_frac - p_min) * progress
        pbest_count = max(2, int(math.ceil(pfrac * N)))
        order = sorted(range(N), key=lambda i: fit_pop[i])

        improved_gen = False
        S_CR, S_F, S_df = [], [], []

        # choose mutation mode bias
        # more diversity late or when stagnating
        use_rand_to_pbest = (progress > 0.70) or (no_improve_gens >= 6)

        for i in range(N):
            if time_up():
                return best

            xi = pop[i]
            fi_old = fit_pop[i]
            k = random.randrange(H)

            CR = M_CR[k] + 0.1 * randn_approx()
            if CR < 0.0: CR = 0.0
            elif CR > 1.0: CR = 1.0

            Fi = -1.0
            for _ in range(12):
                Fi = M_F[k] + 0.1 * cauchy_like()
                if Fi > 0.0:
                    break
            if Fi <= 0.0: Fi = 0.5
            if Fi > 1.0: Fi = 1.0

            pbest = pop[order[random.randrange(pbest_count)]]

            # pick r1
            r1 = i
            while r1 == i:
                r1 = random.randrange(N)

            # pick r2 (pop or archive)
            use_archive = (archive and random.random() < 0.40)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(N)
                xr2 = pop[r2]
            xr1 = pop[r1]

            # mutation
            if use_rand_to_pbest and random.random() < 0.50:
                # rand-to-pbest/1: x_r + F*(pbest - x_r) + F*(xr1 - xr2)
                r0 = random.randrange(N)
                xr0 = pop[r0]
                v = [xr0[j] + Fi * (pbest[j] - xr0[j]) + Fi * (xr1[j] - xr2[j]) for j in range(dim)]
                reflect_reenter_inplace(v, parent=xr0)
                base_for_cr = xr0
            else:
                # current-to-pbest/1
                v = [xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j]) for j in range(dim)]
                reflect_reenter_inplace(v, parent=xi)
                base_for_cr = xi

            # crossover
            j_rand = random.randrange(dim)
            trial = base_for_cr[:]  # start from base (helps rand-to-pbest mode)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            f_trial = eval_f(trial)
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

        # memory adaptation
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                wsum = float(len(S_df))

            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, S_df):
                ww = w / wsum
                num += ww * fval * fval
                den += ww * fval
            new_MF = num / max(1e-12, den)

            new_MCR = 0.0
            for cr, w in zip(S_CR, S_df):
                new_MCR += (w / wsum) * cr

            # slightly faster adaptation late
            mix = 0.42 if progress > 0.6 else 0.35
            M_F[mem_idx] = (1.0 - mix) * M_F[mem_idx] + mix * new_MF
            M_CR[mem_idx] = (1.0 - mix) * M_CR[mem_idx] + mix * new_MCR
            mem_idx = (mem_idx + 1) % H

        # stagnation bookkeeping
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # micro-reseed: periodically reinitialize a couple of worst (cheap anti-collapse)
        if (gen % micro_every == 0) and not time_up() and N >= 8:
            k_micro = micro_count_min + int(random.random() * (micro_count_max - micro_count_min + 1))
            worst = sorted(range(N), key=lambda i: fit_pop[i], reverse=True)[:k_micro]
            for idx in worst:
                if time_up():
                    return best
                if best_vec is not None and random.random() < 0.65:
                    x = best_vec[:]
                    amp = (0.10 + 0.20 * random.random()) * (1.0 - 0.6 * progress)
                    for j in range(dim):
                        x[j] += (0.7 * randn_approx() + 0.3 * cauchy_like()) * amp * spans[j]
                    reflect_reenter_inplace(x, parent=best_vec)
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[idx] = x
                fit_pop[idx] = fx
                if fx < best:
                    best = fx
                    best_vec = x[:]
                    no_improve_gens = 0

        # local refinement schedule
        if best_vec is not None and not time_up():
            if (gen % 4 == 0) or (progress > 0.55) or (no_improve_gens >= 4):
                tries = 6 + (dim // 3)
                xb, fb = coord_refine(best_vec, best, progress, tries)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # quadratic line-search on a few random dimensions (very effective late-stage)
        if best_vec is not None and not time_up():
            if (progress > 0.62) or (no_improve_gens >= 5):
                kdim = 2 if dim < 10 else (3 if dim < 30 else 4)
                dims_to_try = [random.randrange(dim) for _ in range(kdim)]
                xb, fb = quad_line_search_dims(best_vec, best, progress, dims_to_try)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # SPSA only when helpful: higher-dim or very stuck
        if best_vec is not None and not time_up():
            if (dim >= 20 and progress > 0.65) or (no_improve_gens >= 8):
                steps = 1 if dim < 25 else (2 if dim < 60 else 3)
                xb, fb = spsa_refine(best_vec, best, progress, steps)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # stronger stagnation recovery
        if no_improve_gens >= restart_patience and not time_up():
            N = len(pop)
            elite_count = max(2, N // 10)
            elite_idx = sorted(range(N), key=lambda i: fit_pop[i])[:elite_count]
            elite_set = set(elite_idx)

            frac = 0.32 + 0.28 * min(1.0, no_improve_gens / (restart_patience * 2.0))
            k_restart = max(1, int(frac * N))
            worst = [i for i in sorted(range(N), key=lambda i: fit_pop[i], reverse=True) if i not in elite_set]
            worst = worst[:k_restart]

            for idx in worst:
                if time_up():
                    return best

                r = random.random()
                if best_vec is not None and r < 0.62:
                    x = best_vec[:]
                    amp = 0.25 * (1.0 + min(2.0, no_improve_gens / float(restart_patience)))
                    for j in range(dim):
                        x[j] += (0.50 * randn_approx() + 0.50 * cauchy_like()) * amp * spans[j]
                    reflect_reenter_inplace(x, parent=best_vec)
                elif r < 0.84:
                    base = pop[random.choice(elite_idx)][:] if elite_idx else rand_vec()
                    x = opposite(base)
                    reflect_reenter_inplace(x, parent=base)
                else:
                    x = rand_vec()

                fx = eval_f(x)
                pop[idx] = x
                fit_pop[idx] = fx
                if fx < best:
                    best = fx
                    best_vec = x[:]

            # memory shake to encourage movement
            for h in range(H):
                M_F[h] = 0.5 * M_F[h] + 0.5 * (0.35 + 0.55 * random.random())
                M_CR[h] = 0.5 * M_CR[h] + 0.5 * (0.45 + 0.50 * random.random())

            no_improve_gens = 0

    return best
