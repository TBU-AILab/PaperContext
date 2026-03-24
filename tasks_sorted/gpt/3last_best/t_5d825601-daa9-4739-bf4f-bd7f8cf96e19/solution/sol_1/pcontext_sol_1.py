import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization:
    Differential Evolution (DE) + JADE-style parameter adaptation + current-to-pbest
    + external archive + lightweight local search (coordinate/pattern steps).

    Self-contained: uses only Python stdlib.
    Returns: best (minimum) fitness found within time.
    """
    t0 = time.time()

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0.0 else 0.0 for s in spans]

    def timed_out():
        return (time.time() - t0) >= max_time

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # Handle NaN/inf/exception by assigning a huge penalty.
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # --- population sizing (kept modest for speed) ---
    pop_size = max(16, 8 * dim)
    pop_size = min(pop_size, 90)

    # JADE knobs
    p_best_rate = 0.15  # choose p-best from top p%
    p_best_rate = max(0.05, min(0.3, p_best_rate))
    c = 0.10           # learning rate for mu_F, mu_CR
    mu_F = 0.55
    mu_CR = 0.85

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [0.0] * pop_size

    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # external archive for JADE (stores replaced parents)
    archive = []
    max_archive = pop_size

    # --- local search around best (very lightweight) ---
    # step size in normalized units; decreases on success/failure pattern
    ls_sigma = 0.12  # start fairly exploratory
    ls_min_sigma = 1e-4
    ls_max_tries = max(4, 2 * dim)

    def local_search(best_x, best_f, sigma):
        if best_x is None:
            return best_x, best_f, sigma

        # Convert sigma (normalized) to per-dimension absolute steps
        # Use coordinate + a couple random pattern directions.
        improved = False
        x0 = best_x[:]
        f0 = best_f

        # Coordinate search
        idx_order = list(range(dim))
        random.shuffle(idx_order)

        tries = 0
        for d in idx_order:
            if timed_out() or tries >= ls_max_tries:
                break
            tries += 1

            step = sigma * spans[d]
            if step <= 0.0:
                continue

            # Try +step, then -step
            for sgn in (1.0, -1.0):
                if timed_out():
                    break
                x = x0[:]
                x[d] = x[d] + sgn * step
                clip_inplace(x)
                fx = safe_eval(x)
                if fx < f0:
                    x0, f0 = x, fx
                    improved = True
                    break  # move to next dimension with updated x0
            # continue to next dimension

        # A couple of random pattern moves
        for _ in range(2):
            if timed_out():
                break
            x = x0[:]
            # sparse perturbation
            for d in range(dim):
                if random.random() < 0.25:
                    x[d] += (random.random() * 2.0 - 1.0) * sigma * spans[d]
            clip_inplace(x)
            fx = safe_eval(x)
            if fx < f0:
                x0, f0 = x, fx
                improved = True

        # Adapt sigma
        if improved:
            sigma = max(ls_min_sigma, sigma * 0.85)
        else:
            sigma = min(0.35, sigma * 1.12)

        return x0, f0, sigma

    # restart logic
    no_improve_gens = 0
    restart_after = max(25, 8 * dim)

    # helper: sample from population U archive, excluding some indices
    def pick_index_excluding(n, exclude_set):
        # simple rejection sampling; n is small enough
        while True:
            j = random.randrange(n)
            if j not in exclude_set:
                return j

    # Main loop
    while not timed_out():
        # Sort indices by fitness for p-best selection
        idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_best_rate * pop_size)))

        # Success histories for adaptation
        S_F = []
        S_CR = []
        dF = []  # fitness improvements (parent - child)

        improved_gen = False

        for i in range(pop_size):
            if timed_out():
                return best

            xi = pop[i]
            fi = fit[i]

            # --- sample adaptive F and CR (JADE style) ---
            # CR ~ N(mu_CR, 0.1), clipped to [0,1]
            CR = mu_CR + 0.1 * random.gauss(0.0, 1.0)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # F from Cauchy-like: mu_F + 0.1 * tan(pi*(u-0.5)), resample if <=0
            # (no external libs)
            F = -1.0
            for _ in range(6):
                u = random.random()
                cand = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                if cand > 0.0:
                    F = cand
                    break
            if F <= 0.0:
                F = mu_F
            if F > 1.0:
                F = 1.0

            # choose p-best
            pbest_idx = idx_sorted[random.randrange(p_count)]
            xpbest = pop[pbest_idx]

            # choose r1 from population excluding i
            r1 = pick_index_excluding(pop_size, {i})

            # choose r2 from population+archive excluding i and r1
            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                # combined pool indices: [0..pop_size-1] are pop, [pop_size..] archive
                combined_n = pop_size + len(archive)
                while True:
                    r2c = random.randrange(combined_n)
                    if r2c == i or r2c == r1:
                        continue
                    if r2c < pop_size:
                        xr2 = pop[r2c]
                    else:
                        xr2 = archive[r2c - pop_size]
                    break
            else:
                r2 = pick_index_excluding(pop_size, {i, r1})
                xr2 = pop[r2]

            xr1 = pop[r1]

            # --- mutation: current-to-pbest/1 with archive ---
            vi = [0.0] * dim
            for d in range(dim):
                vi[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # boundary handling: reflect to keep diversity (better than hard clip)
            for d in range(dim):
                lo = lows[d]
                hi = highs[d]
                if vi[d] < lo:
                    vi[d] = lo + (lo - vi[d])
                    if vi[d] > hi:
                        vi[d] = lo
                elif vi[d] > hi:
                    vi[d] = hi - (vi[d] - hi)
                    if vi[d] < lo:
                        vi[d] = hi

            # --- binomial crossover ---
            jrand = random.randrange(dim)
            ui = [0.0] * dim
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    ui[d] = vi[d]
                else:
                    ui[d] = xi[d]

            # tiny probability of "opposition-like" move to increase exploration
            if random.random() < 0.02:
                for d in range(dim):
                    if random.random() < 0.15:
                        ui[d] = lows[d] + highs[d] - ui[d]
                clip_inplace(ui)

            fui = safe_eval(ui)

            # selection
            if fui <= fi:
                # add parent to archive (as in JADE)
                archive.append(xi[:])
                if len(archive) > max_archive:
                    # remove random element to keep bounded
                    archive.pop(random.randrange(len(archive)))

                pop[i] = ui
                fit[i] = fui

                if fi - fui > 0.0:
                    S_F.append(F)
                    S_CR.append(CR)
                    dF.append(fi - fui)

                if fui < best:
                    best = fui
                    best_x = ui[:]
                    improved_gen = True
            # else keep parent

        # --- update mu_F and mu_CR ---
        if S_F:
            # weighted Lehmer mean for F
            wsum = sum(dF)
            if wsum <= 0.0:
                wsum = float(len(dF))
                weights = [1.0 / wsum] * len(dF)
            else:
                weights = [df / wsum for df in dF]

            # mu_CR: weighted arithmetic mean
            new_mu_CR = 0.0
            for w, cr in zip(weights, S_CR):
                new_mu_CR += w * cr

            # mu_F: weighted Lehmer mean = sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for w, ff in zip(weights, S_F):
                num += w * ff * ff
                den += w * ff
            new_mu_F = (num / den) if den != 0.0 else mu_F

            mu_CR = (1.0 - c) * mu_CR + c * new_mu_CR
            mu_F = (1.0 - c) * mu_F + c * new_mu_F

            # keep sane
            mu_CR = min(0.99, max(0.05, mu_CR))
            mu_F = min(0.95, max(0.10, mu_F))

        # Local search occasionally (more often when close to timeout, to exploit)
        # Trigger: every few generations, and also if we improved recently.
        if not timed_out() and best_x is not None:
            if improved_gen or (random.random() < 0.25):
                bx, bf, ls_sigma = local_search(best_x, best, ls_sigma)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # Stagnation / partial restart: reinitialize worst 20%
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= restart_after and not timed_out():
            no_improve_gens = 0
            k = max(2, pop_size // 5)
            worst_idx = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)[:k]
            for wi in worst_idx:
                if timed_out():
                    return best
                pop[wi] = rand_vec()
                fit[wi] = safe_eval(pop[wi])
                if fit[wi] < best:
                    best = fit[wi]
                    best_x = pop[wi][:]

            # also diversify archive
            if archive and random.random() < 0.5:
                # keep only a few
                keep = min(len(archive), max(5, pop_size // 6))
                random.shuffle(archive)
                archive = archive[:keep]

    return best
