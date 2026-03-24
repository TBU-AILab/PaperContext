import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs).

    Hybrid:
      1) JADE-like Differential Evolution: current-to-pbest/1 + archive
         - success-history adaptation of F and CR (Lehmer mean for F)
         - pbest fraction shrinks over time (more exploitation later)
      2) Heavy-tailed "shotgun" sampling around best (Cauchy-like steps)
      3) Cheap coordinate/pattern local search with adaptive step size
      4) Stagnation-triggered partial restart of worst individuals
      5) Reflection bounds handling (usually better than clipping)

    Returns:
      best fitness (float) found within max_time seconds.
    """

    # ---------------- helpers ----------------
    def clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_into_bounds(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly until inside bounds
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            elif x > hi:
                x = hi - (x - hi)
        # safety
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return x

    def ensure_bounds_reflect(vec):
        out = vec[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = reflect_into_bounds(out[i], lo, hi)
        return out

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(vec):
        try:
            v = func(vec)
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                v = float(v)
                if v != v or v == float("inf") or v == float("-inf"):
                    return float("inf")
                return v
            return float("inf")
        except Exception:
            return float("inf")

    def l2_sq(a, b):
        s = 0.0
        for i in range(dim):
            d = a[i] - b[i]
            s += d * d
        return s

    def sample_cauchy(mu, gamma):
        # mu + gamma * tan(pi*(u-0.5))
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    # ---------------- setup ----------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # spans for step scaling
    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        if s <= 0.0:
            s = 1.0
        span.append(s)

    # time-adaptive population size (smaller when dim is big)
    pop_size = 8 + 3 * dim
    pop_size = int(clip(pop_size, 14, 64))

    # init pop
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop)

    best = float("inf")
    best_x = None
    for i in range(pop_size):
        if fit[i] < best:
            best = fit[i]
            best_x = pop[i][:]

    # archive for JADE
    archive = []
    archive_max = pop_size

    # JADE success-history means
    mu_F = 0.6
    mu_CR = 0.6
    c_adapt = 0.1

    # local search state
    local_step_rel = 0.12  # relative to span
    local_step_min = 1e-6
    last_local = time.time()
    local_interval = 0.10  # seconds between local calls

    # stagnation state
    last_best = best
    no_improve_rounds = 0

    # ---------------- main loop ----------------
    while time.time() < deadline:
        now = time.time()
        if now >= deadline:
            break
        time_frac = (now - t0) / max(1e-12, float(max_time))
        time_frac = clip(time_frac, 0.0, 1.0)

        # pbest fraction shrinks over time (more exploit later)
        p_max, p_min = 0.30, 0.05
        p = p_max - (p_max - p_min) * time_frac
        p_num = max(2, int(math.ceil(p * pop_size)))

        # sort for pbest selection
        idx_sorted = sorted(range(pop_size), key=lambda k: fit[k])

        # success lists for parameter adaptation
        S_F, S_CR, S_df = [], [], []

        # --- one generation of JADE-like DE ---
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # pick pbest
            pbest_idx = idx_sorted[random.randrange(p_num)]
            xpbest = pop[pbest_idx]

            # sample CR from N(mu_CR, 0.1)
            CRi = clip(random.gauss(mu_CR, 0.1), 0.0, 1.0)

            # sample F from cauchy(mu_F, 0.1), retry a bit for positive
            Fi = None
            for _ in range(8):
                cand = sample_cauchy(mu_F, 0.1)
                if cand > 0.0:
                    Fi = cand
                    break
            if Fi is None:
                Fi = mu_F
            Fi = clip(Fi, 0.05, 1.0)

            # choose r1 from pop not i
            r1 = random.randrange(pop_size)
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from union(pop, archive) not (i or r1) when in pop part
            union = pop + archive
            uN = len(union)
            r2 = random.randrange(uN) if uN > 0 else random.randrange(pop_size)

            tries = 0
            while tries < 25 and r2 < pop_size and (r2 == i or r2 == r1):
                r2 = random.randrange(uN)
                tries += 1

            xr1 = pop[r1]
            xr2 = union[r2] if uN > 0 else pop[random.randrange(pop_size)]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            # binomial crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]

            uvec = ensure_bounds_reflect(uvec)
            fu = safe_eval(uvec)

            if fu <= fi:
                # archive replaced
                archive.append(xi[:])
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = uvec
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = uvec[:]

                df = fi - fu
                if df < 0.0:
                    df = 0.0
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(df)

        # --- adapt mu_F / mu_CR ---
        if S_F:
            sw = sum(S_df)
            if sw <= 0.0:
                weights = [1.0 / len(S_df)] * len(S_df)
            else:
                weights = [d / sw for d in S_df]

            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            F_lehmer = (num / den) if den > 0.0 else mu_F

            # weighted mean for CR
            CR_mean = 0.0
            for w, cr in zip(weights, S_CR):
                CR_mean += w * cr

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * F_lehmer
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * CR_mean
            mu_F = clip(mu_F, 0.05, 0.95)
            mu_CR = clip(mu_CR, 0.0, 1.0)

        # --- stagnation tracking ---
        if best < last_best - 1e-12:
            last_best = best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        # --- heavy-tailed "shotgun" around best (cheap diversification/exploitation) ---
        # small budget per outer loop
        if best_x is not None:
            # more shots later when DE has focused
            shots = 1 + int(2 * time_frac)
            for _ in range(shots):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                # sparse perturbation with heavy tails
                p_change = min(0.35, 3.0 / max(3.0, dim))
                # scale decays with time, but keep a floor
                scale = 0.25 * (1.0 - time_frac) + 0.03
                for j in range(dim):
                    if random.random() < p_change:
                        # Cauchy-like step
                        step = sample_cauchy(0.0, scale) * span[j]
                        x[j] += step
                x = ensure_bounds_reflect(x)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best
                    no_improve_rounds = 0

        # --- local refinement around best (pattern/coordinate), periodic ---
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now
            x0 = best_x[:]
            f0 = best

            # pick a few coordinates (cheap)
            coords = list(range(dim))
            random.shuffle(coords)
            m = min(dim, 8)
            coords = coords[:m]

            step_rel = max(local_step_rel, local_step_min)
            for j in coords:
                if time.time() >= deadline:
                    return best
                step = step_rel * span[j]
                if step <= 0.0:
                    continue

                # try +/- step, then a smaller follow-up if improved
                improved = False
                for sgn in (1.0, -1.0):
                    xt = x0[:]
                    xt[j] += sgn * step
                    xt = ensure_bounds_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft
                        improved = True

                if improved and time.time() < deadline:
                    step2 = 0.35 * step
                    for sgn in (1.0, -1.0):
                        xt = x0[:]
                        xt[j] += sgn * step2
                        xt = ensure_bounds_reflect(xt)
                        ft = safe_eval(xt)
                        if ft < f0:
                            x0, f0 = xt, ft

            if f0 < best:
                best = f0
                best_x = x0[:]
                last_best = best
                no_improve_rounds = 0
                # if local works, reduce step a bit (fine-tune)
                local_step_rel = max(local_step_rel * 0.92, local_step_min)
                local_interval = clip(local_interval * 0.92, 0.05, 0.30)
            else:
                # if local doesn't help, slowly increase step & do it less often
                local_step_rel = clip(local_step_rel * 1.05, local_step_min, 0.25)
                local_interval = clip(local_interval * 1.05, 0.05, 0.35)

        # --- partial restart on stagnation ---
        if no_improve_rounds >= 8:
            no_improve_rounds = 0
            # replace worst ~20% (keep best)
            k = max(1, pop_size // 5)
            worst_idx = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]

            for wi in worst_idx:
                if time.time() >= deadline:
                    return best

                # avoid overwriting a vector identical to current best (cheap check)
                if best_x is not None and fit[wi] == best and l2_sq(pop[wi], best_x) < 1e-24:
                    continue

                if best_x is not None and random.random() < 0.65:
                    # re-seed near best with moderate radius
                    x = best_x[:]
                    rad = 0.20
                    for j in range(dim):
                        x[j] += random.uniform(-rad, rad) * span[j]
                    x = ensure_bounds_reflect(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best

            # clear/trim archive (refresh diversity)
            if len(archive) > archive_max // 2:
                # random downsample
                random.shuffle(archive)
                archive = archive[: archive_max // 2]

    return best
