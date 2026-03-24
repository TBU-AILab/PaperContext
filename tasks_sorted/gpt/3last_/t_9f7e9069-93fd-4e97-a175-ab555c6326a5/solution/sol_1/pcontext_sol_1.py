import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded optimizer (no external libs):
    - Differential Evolution (current-to-best/1) with jittered parameters
    - Opposition-based initialization + occasional opposition refresh
    - Local refinement around best (coordinate + random directions) with step-size decay
    - Stagnation handling via partial re-initialization near best + some global randoms
    Returns best fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]

    # --- helpers ---
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                x[i] = lo[i] + random.random() * span[i]
        return x

    def opposite(x):
        # opposition point across the center of bounds
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = lo[i]
            else:
                y[i] = lo[i] + (hi[i] - x[i])
        return y

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_unit_vec():
        # random direction (approx unit) without numpy
        v = [random.gauss(0.0, 1.0) for _ in range(dim)]
        s = math.sqrt(sum(a*a for a in v))
        if s == 0.0:
            j = random.randrange(dim)
            v = [0.0]*dim
            v[j] = 1.0
            return v
        return [a/s for a in v]

    # --- budget checks ---
    def time_left():
        return deadline - time.time()

    # --- population size ---
    # A bit larger than before to improve global coverage, capped for speed.
    pop_size = max(16, min(80, 12 * dim))

    # --- Opposition-based initialization ---
    pop = []
    fit = []
    # Generate pop_size candidates + their opposites, keep the best pop_size
    candidates = []
    for _ in range(pop_size):
        x = rand_vec()
        candidates.append(x)
        candidates.append(opposite(x))

    # Evaluate candidates (stop early if time is almost over)
    cand_fit = []
    for x in candidates:
        if time.time() >= deadline:
            break
        cand_fit.append((safe_eval(x), x))
    cand_fit.sort(key=lambda t: t[0])
    for f, x in cand_fit[:pop_size]:
        pop.append(x[:])
        fit.append(f)

    if not pop:
        return float("inf")

    best_idx = min(range(len(fit)), key=lambda i: fit[i])
    best = fit[best_idx]
    best_x = pop[best_idx][:]

    # --- DE controls ---
    # Use current-to-best/1 (often faster convergence) with jitter.
    # u = x_i + F*(best - x_i) + F*(x_r1 - x_r2)
    F_base_min, F_base_max = 0.35, 0.95
    CR_min, CR_max = 0.2, 0.98

    # --- Local refinement controls ---
    # start step as fraction of span; decay on stagnation
    step = [0.15 * s for s in span]  # per-dimension
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # --- stagnation / restart controls ---
    no_improve = 0
    # generations without improvement triggers actions
    restart_after = max(20, 6 * dim)
    opp_refresh_after = max(10, 3 * dim)
    local_refine_every = 3  # attempt local refinement every few generations

    gen = 0
    while time.time() < deadline:
        gen += 1

        # Jitter parameters each generation
        F = F_base_min + (F_base_max - F_base_min) * random.random()
        CR = CR_min + (CR_max - CR_min) * random.random()

        improved_this_gen = False

        # --- DE generation ---
        for i in range(len(pop)):
            if time.time() >= deadline:
                return best

            # pick r1, r2 distinct from i
            idxs = list(range(len(pop)))
            idxs.remove(i)
            if len(idxs) < 2:
                continue
            r1, r2 = random.sample(idxs, 2)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # mutation: current-to-best/1
            v = [0.0] * dim
            # add small jitter to F per individual (improves exploration)
            Fj = max(0.0, min(1.2, F * (0.8 + 0.4 * random.random())))
            for d in range(dim):
                if fixed[d]:
                    v[d] = lo[d]
                else:
                    v[d] = xi[d] + Fj * (best_x[d] - xi[d]) + Fj * (xr1[d] - xr2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
            clip_inplace(u)

            fu = safe_eval(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_this_gen = True

        # --- occasional opposition refresh for diversity ---
        if (gen % opp_refresh_after) == 0 and time_left() > 0.01:
            # take a few worst and try their opposites
            order = sorted(range(len(pop)), key=lambda k: fit[k], reverse=True)
            m = max(1, len(pop)//6)
            for idx in order[:m]:
                if time.time() >= deadline:
                    return best
                ox = opposite(pop[idx])
                clip_inplace(ox)
                fox = safe_eval(ox)
                if fox < fit[idx]:
                    pop[idx] = ox
                    fit[idx] = fox
                    if fox < best:
                        best = fox
                        best_x = ox[:]
                        improved_this_gen = True

        # --- local refinement around best ---
        # lightweight hill-climb with decaying step (coordinate + random directions)
        if (gen % local_refine_every) == 0 and time_left() > 0.01:
            # Try a few coordinate tweaks
            trials = 0
            max_trials = 2 * dim + 6
            while trials < max_trials and time.time() < deadline:
                trials += 1
                x = best_x[:]

                if random.random() < 0.65:
                    # coordinate move
                    d = random.randrange(dim)
                    if fixed[d] or step[d] <= min_step[d]:
                        continue
                    delta = step[d] * (1.0 if random.random() < 0.5 else -1.0)
                    x[d] += delta
                else:
                    # random direction move
                    dirv = rand_unit_vec()
                    # scale by average step
                    avg_step = 0.0
                    cnt = 0
                    for d in range(dim):
                        if not fixed[d]:
                            avg_step += step[d]
                            cnt += 1
                    if cnt == 0:
                        break
                    avg_step /= cnt
                    scale = avg_step * (0.5 + random.random())
                    for d in range(dim):
                        if not fixed[d]:
                            x[d] += scale * dirv[d]

                clip_inplace(x)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    best_x = x
                    improved_this_gen = True
                    no_improve = 0
                else:
                    # mild step decay on failed local moves
                    # (keeps local search from overshooting)
                    for d in range(dim):
                        if not fixed[d] and step[d] > min_step[d]:
                            step[d] *= 0.995

        if improved_this_gen:
            no_improve = 0
        else:
            no_improve += 1

        # --- stagnation handling ---
        if no_improve >= restart_after and time_left() > 0.02:
            no_improve = 0

            # Decay steps more aggressively to enable finer local search later
            for d in range(dim):
                if not fixed[d] and step[d] > min_step[d]:
                    step[d] = max(min_step[d], step[d] * 0.6)

            # Keep elites, reinit others: half near best (Gaussian), half global random
            order = sorted(range(len(pop)), key=lambda k: fit[k])
            k = max(2, len(pop)//5)
            elites = [pop[idx][:] for idx in order[:k]]
            elites_fit = [fit[idx] for idx in order[:k]]

            new_pop = elites[:]
            new_fit = elites_fit[:]

            # near-best sampling scale based on current step
            for _ in range(len(pop) - k):
                if time.time() >= deadline:
                    break
                if random.random() < 0.6:
                    x = best_x[:]
                    for d in range(dim):
                        if fixed[d]:
                            x[d] = lo[d]
                        else:
                            sd = max(1e-15, step[d])
                            x[d] += random.gauss(0.0, sd)
                    clip_inplace(x)
                else:
                    x = rand_vec()
                new_pop.append(x)
                new_fit.append(safe_eval(x))

            pop, fit = new_pop, new_fit

            # refresh best
            best_idx = min(range(len(fit)), key=lambda i: fit[i])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = pop[best_idx][:]

    return best
