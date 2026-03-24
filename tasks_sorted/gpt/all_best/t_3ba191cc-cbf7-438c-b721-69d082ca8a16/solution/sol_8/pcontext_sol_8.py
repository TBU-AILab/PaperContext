import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over your best (JADE/SHADE-lite DE + local search):
      - Better initialization: centered-LHS + opposition + elitist downselect
      - L-SHADE core:
          * DE/current-to-pbest/1 + archive
          * success-history adaptation of F/CR (weighted means)
          * linear population size reduction
      - Smarter local exploitation:
          * adaptive coordinate pattern search (per-dimension step control)
          * occasional SPSA-style gradient sign step around best (2 evals/step)
      - Stronger stagnation recovery: partial restarts (around-best heavy tail / opposition / random)
      - Robust bound handling: reflection + fallback random reset for pathological cases
      - Strict time checks

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

    # ---------------- utilities ----------------
    def time_up():
        return time.time() >= deadline

    def reflect_clip_inplace(x):
        # reflect at bounds; if still out due to numerical issues, clamp
        for j in range(dim):
            lo = lows[j]
            hi = highs[j]
            if x[j] < lo:
                x[j] = lo + (lo - x[j])
                if x[j] > hi:
                    x[j] = lo
            elif x[j] > hi:
                x[j] = hi - (x[j] - hi)
                if x[j] < lo:
                    x[j] = hi
            # final clamp
            if x[j] < lo:
                x[j] = lo
            elif x[j] > hi:
                x[j] = hi
        return x

    def rand_vec():
        return [lows[j] + random.random() * spans[j] for j in range(dim)]

    def opposite(x):
        return [lows[j] + highs[j] - x[j] for j in range(dim)]

    def randn_approx():
        # ~N(0,1) via CLT
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

    # ---------------- Initialization: centered LHS + opposition + elitist ----------------
    pop_max = max(24, min(140, 10 * dim + 30))
    pop_min = max(10, min(60, 4 * dim + 12))

    # centered-LHS-ish
    base = [[0.0] * dim for _ in range(pop_max)]
    for j in range(dim):
        perm = list(range(pop_max))
        random.shuffle(perm)
        inv = 1.0 / float(pop_max)
        for i in range(pop_max):
            u = (perm[i] + 0.5) * inv
            u += (random.random() - 0.5) * inv  # jitter
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            base[i][j] = lows[j] + u * spans[j]

    candidates = base + [reflect_clip_inplace(opposite(x[:])) for x in base]

    best = float("inf")
    best_vec = None
    cand_fit = []
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

    # ---------------- Local search helpers ----------------
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
            if step < coord_min[j]:
                step = coord_min[j]
            if step > coord_max[j]:
                step = coord_max[j]

            improved = False
            # try + and - deterministically
            for direction in (1.0, -1.0):
                if time_up():
                    return x0, f0
                cand = x0[:]
                cand[j] += direction * step
                reflect_clip_inplace(cand)
                fc = func(cand)
                if fc < f0:
                    x0, f0 = cand, fc
                    coord_steps[j] = min(coord_max[j], coord_steps[j] * 1.20)
                    improved = True
                    break
            if not improved:
                coord_steps[j] = max(coord_min[j], coord_steps[j] * 0.85)

        return x0, f0

    def spsa_refine(x_best, f_best, progress, steps):
        """
        SPSA-style local search near best (2 evaluations per step).
        Very budget-efficient in higher dimensions.
        """
        x = x_best[:]
        f = f_best

        # scale shrinks with time
        a = 0.18 * (1.0 - progress) + 0.02
        c = 0.08 * (1.0 - progress) + 0.01

        for k in range(steps):
            if time_up():
                return x, f

            # random +/-1 perturbation
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            ck = c / math.sqrt(1.0 + k)

            x_plus = [x[j] + ck * delta[j] * spans[j] for j in range(dim)]
            x_minus = [x[j] - ck * delta[j] * spans[j] for j in range(dim)]
            reflect_clip_inplace(x_plus)
            reflect_clip_inplace(x_minus)

            f_plus = func(x_plus)
            if time_up():
                return x, f
            f_minus = func(x_minus)

            # gradient estimate and update
            ghat = [(f_plus - f_minus) / (2.0 * ck * spans[j] * delta[j] + 1e-18) for j in range(dim)]
            ak = a / (1.0 + 0.15 * k)

            cand = [x[j] - ak * ghat[j] for j in range(dim)]
            reflect_clip_inplace(cand)
            fc = func(cand)

            if fc < f:
                x, f = cand, fc
            else:
                # small random "trust region" move
                cand2 = x[:]
                scale = 0.15 * (1.0 - progress) + 0.02
                for j in range(dim):
                    cand2[j] += randn_approx() * scale * spans[j]
                reflect_clip_inplace(cand2)
                fc2 = func(cand2)
                if fc2 < f:
                    x, f = cand2, fc2

        return x, f

    # ---------------- Main loop ----------------
    gen = 0
    no_improve_gens = 0
    restart_patience = max(18, 4 * dim)

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

            use_archive = (archive and random.random() < 0.40)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
                # ensure r1 isn't identical vector reference doesn't matter; archive separate
                xr1 = pop[r1]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(N)
                xr1 = pop[r1]
                xr2 = pop[r2]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                val = xi[j] + Fi * (pbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                v[j] = val
            reflect_clip_inplace(v)

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

        # memory adaptation
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

            M_F[mem_idx] = 0.60 * M_F[mem_idx] + 0.40 * new_MF
            M_CR[mem_idx] = 0.60 * M_CR[mem_idx] + 0.40 * new_MCR
            mem_idx = (mem_idx + 1) % H

        # stagnation bookkeeping
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # local refinement schedule
        if best_vec is not None and not time_up():
            # coordinate refine regularly
            if (gen % 4 == 0) or (progress > 0.55) or (no_improve_gens >= 4):
                tries = 7 + (dim // 3)
                xb, fb = coord_refine(best_vec, best, progress, tries)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        if best_vec is not None and not time_up():
            # SPSA: great in medium/high-dim; do more near end or when stuck
            if (progress > 0.68) or (no_improve_gens >= 7):
                # keep it light: 1-3 steps, 2 evals per step (+1 for accept) => small overhead
                spsa_steps = 1 if dim < 8 else (2 if dim < 25 else 3)
                xb, fb = spsa_refine(best_vec, best, progress, spsa_steps)
                if fb < best:
                    best, best_vec = fb, xb[:]
                    no_improve_gens = 0

        # stagnation recovery: keep elites, reseed some worst
        if no_improve_gens >= restart_patience and not time_up():
            N = len(pop)
            elite_count = max(2, N // 10)
            elite_idx = sorted(range(N), key=lambda i: fit_pop[i])[:elite_count]
            elite_set = set(elite_idx)

            frac = 0.30 + 0.25 * min(1.0, no_improve_gens / (restart_patience * 2.0))
            k_restart = max(1, int(frac * N))
            worst = [i for i in sorted(range(N), key=lambda i: fit_pop[i], reverse=True) if i not in elite_set]
            worst = worst[:k_restart]

            for idx in worst:
                if time_up():
                    return best

                r = random.random()
                if best_vec is not None and r < 0.60:
                    x = best_vec[:]
                    amp = 0.22 * (1.0 + min(2.0, no_improve_gens / float(restart_patience)))
                    for j in range(dim):
                        x[j] += (0.55 * randn_approx() + 0.45 * cauchy_like()) * amp * spans[j]
                    reflect_clip_inplace(x)
                elif r < 0.82:
                    base = pop[random.choice(elite_idx)][:] if elite_idx else rand_vec()
                    x = reflect_clip_inplace(opposite(base))
                else:
                    x = rand_vec()

                fx = func(x)
                pop[idx] = x
                fit_pop[idx] = fx
                if fx < best:
                    best = fx
                    best_vec = x[:]

            # encourage movement
            for h in range(H):
                M_F[h] = 0.5 * M_F[h] + 0.5 * (0.40 + 0.45 * random.random())
                M_CR[h] = 0.5 * M_CR[h] + 0.5 * (0.55 + 0.40 * random.random())

            no_improve_gens = 0

    return best
