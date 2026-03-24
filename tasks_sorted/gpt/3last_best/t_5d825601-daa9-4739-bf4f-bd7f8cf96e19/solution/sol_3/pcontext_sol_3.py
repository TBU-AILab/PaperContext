import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Upgrade over the previous JADE-DE:
      - L-SHADE style memories (M_F, M_CR) instead of single mu_F/mu_CR
      - Linear population size reduction (explore early, exploit late)
      - External archive (JADE/L-SHADE)
      - "EIG" crossover (rotate in random 2D subspaces) occasionally helps on ill-conditioned problems
      - Stronger, time-aware local search around the best (SPSA-like + coordinate)
      - Stagnation rescue: re-seed part of the worst and inject around best

    Stdlib-only, time-bounded. Returns best fitness found.
    """

    t0 = time.time()
    end_time = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

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
        # reflection (keeps diversity better than clip)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            xi = x[i]
            if xi < lo:
                xi = lo + (lo - xi)
                if xi > hi:
                    xi = lo
            elif xi > hi:
                xi = hi - (xi - hi)
                if xi < lo:
                    xi = hi
            x[i] = xi
        return x

    # --- L-SHADE-ish sizes (kept moderate for unknown evaluation costs) ---
    NP_init = max(24, 12 * dim)
    NP_init = min(NP_init, 140)
    NP_min = max(8, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(8, NP_init // 2)

    # SHADE memories
    H = 10
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    # JADE/L-SHADE settings
    p_best_rate = 0.12
    p_best_rate = max(0.05, min(0.30, p_best_rate))

    # archive
    archive = []
    archive_cap = NP_init

    # init pop
    pop = [rand_vec() for _ in range(NP_init)]
    fit = [float("inf")] * NP_init

    best = float("inf")
    best_x = None

    for i in range(NP_init):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # ---- sampling F and CR (SHADE standard) ----
    def sample_F(mu):
        # Cauchy around mu, retry to ensure >0
        for _ in range(10):
            u = random.random()
            f = mu + 0.1 * math.tan(math.pi * (u - 0.5))
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

    # --- EIG crossover: rotate in random 2D planes (cheap version) ---
    # Applies to mutant v and parent x to create trial u in a rotated subspace.
    def eig_crossover(x, v, cr):
        # start with binomial, then rotate a few random pairs
        u = v[:]  # already mostly mutant; we will mix with x
        jrand = random.randrange(dim)
        for d in range(dim):
            if d != jrand and random.random() >= cr:
                u[d] = x[d]

        # rotate in k random 2D subspaces: helps when variables are correlated
        k = 1 if dim < 8 else 2
        for _ in range(k):
            a = random.randrange(dim)
            b = random.randrange(dim)
            if a == b:
                continue
            # random angle
            theta = (random.random() * 2.0 - 1.0) * (math.pi / 3.0)
            cth = math.cos(theta)
            sth = math.sin(theta)

            # rotate the *difference* (u - x) in plane (a,b)
            da = u[a] - x[a]
            db = u[b] - x[b]
            ra = cth * da - sth * db
            rb = sth * da + cth * db
            u[a] = x[a] + ra
            u[b] = x[b] + rb

        return reflect_bounds_inplace(u)

    # --- stronger local search around best (time-aware budget) ---
    ls_sigma = 0.10
    ls_sigma_min = 1e-6
    ls_sigma_max = 0.35

    def local_search(xb, fb, frac):
        # frac = progress 0..1, increase exploitation late
        nonlocal ls_sigma
        if xb is None:
            return xb, fb

        x = xb[:]
        f = fb

        # budgeted tries (few early, more late)
        coord_tries = min(dim, 4 + int(6 * frac))
        rand_tries = 1 + int(2 * frac)

        # SPSA-like 2-eval step
        if not timed_out():
            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]
            c = ls_sigma
            a = 0.7 * ls_sigma
            x_plus = x[:]
            x_minus = x[:]
            for d in range(dim):
                step = c * spans[d]
                x_plus[d] += delta[d] * step
                x_minus[d] -= delta[d] * step
            reflect_bounds_inplace(x_plus)
            reflect_bounds_inplace(x_minus)

            f_plus = safe_eval(x_plus)
            if timed_out():
                return x, f
            f_minus = safe_eval(x_minus)

            if f_plus < float("inf") and f_minus < float("inf"):
                g = (f_plus - f_minus)
                x_try = x[:]
                # if g>0 we want to move opposite delta, else along delta
                sgn = 1.0 if g > 0.0 else -1.0
                for d in range(dim):
                    x_try[d] -= a * spans[d] * sgn * delta[d]
                reflect_bounds_inplace(x_try)
                f_try = safe_eval(x_try)
                if f_try < f:
                    x, f = x_try, f_try
                    ls_sigma = max(ls_sigma_min, ls_sigma * 0.85)
                else:
                    ls_sigma = min(ls_sigma_max, ls_sigma * 1.08)

        # Coordinate probes (random order)
        idxs = list(range(dim))
        random.shuffle(idxs)
        for t in range(coord_tries):
            if timed_out():
                break
            d = idxs[t % dim]
            step = ls_sigma * spans[d]
            if step <= 0.0:
                continue
            for s in (1.0, -1.0):
                xt = x[:]
                xt[d] += s * step
                reflect_bounds_inplace(xt)
                ft = safe_eval(xt)
                if ft < f:
                    x, f = xt, ft
                    ls_sigma = max(ls_sigma_min, ls_sigma * 0.90)
                    break

        # A couple sparse random pattern moves (good when close)
        for _ in range(rand_tries):
            if timed_out():
                break
            xt = x[:]
            prob = 0.15 + 0.20 * (1.0 - frac)  # a bit denser earlier
            for d in range(dim):
                if random.random() < prob:
                    xt[d] += (random.random() * 2.0 - 1.0) * ls_sigma * spans[d]
            reflect_bounds_inplace(xt)
            ft = safe_eval(xt)
            if ft < f:
                x, f = xt, ft
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.92)

        return x, f

    # stagnation
    no_improve_gens = 0
    stagnation_limit = max(10, 3 * dim)

    # main loop
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
            # keep best NP_target
            keep = sorted(range(NP), key=lambda i: fit[i])[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = NP_target
            archive_cap = max(NP, NP_init // 2)
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

        # pbest pool
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_best_rate * NP)))

        S_F, S_CR, S_df = [], [], []
        improved_gen = False

        # generation
        for i in range(NP):
            if timed_out():
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            pbest = pop[idx_sorted[random.randrange(p_count)]]

            # choose r1 != i
            while True:
                r1 = random.randrange(NP)
                if r1 != i:
                    break

            # choose r2 from pop U archive, != i,r1
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
                while True:
                    r2 = random.randrange(NP)
                    if r2 != i and r2 != r1:
                        xr2 = pop[r2]
                        break

            xr1 = pop[r1]

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (xr1[d] - xr2[d])
            reflect_bounds_inplace(v)

            # crossover: mostly binomial, sometimes EIG
            if dim >= 2 and random.random() < 0.18:
                u = eig_crossover(xi, v, CR)
            else:
                jrand = random.randrange(dim)
                u = [0.0] * dim
                for d in range(dim):
                    u[d] = v[d] if (d == jrand or random.random() < CR) else xi[d]

            # occasional "best-directed" injection late in the run
            if best_x is not None and random.random() < (0.01 + 0.04 * frac):
                d = random.randrange(dim)
                u[d] = 0.85 * u[d] + 0.15 * best_x[d]
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

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_gen = True

        # update memories
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

        # local search schedule: more likely late, or on improvement
        if best_x is not None and not timed_out():
            p_ls = 0.05 + 0.35 * frac
            if improved_gen or random.random() < p_ls:
                bx, bf = local_search(best_x, best, frac)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # stagnation rescue
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagnation_limit and not timed_out():
            no_improve_gens = 0
            NP = len(pop)
            k = max(2, NP // 6)

            # 1) re-seed worst
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for wi in worst:
                if timed_out():
                    return best
                # mix: either pure random or around best
                if best_x is not None and random.random() < 0.6:
                    x = best_x[:]
                    # gaussian-ish via sum of uniforms (no numpy)
                    for d in range(dim):
                        # smaller radius later
                        rad = (0.20 * (1.0 - frac) + 0.03) * spans[d]
                        g = (random.random() + random.random() + random.random() - 1.5)  # ~N(0,0.25)
                        x[d] += g * rad
                    reflect_bounds_inplace(x)
                else:
                    x = rand_vec()
                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

            # 2) thin archive
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

    return best
