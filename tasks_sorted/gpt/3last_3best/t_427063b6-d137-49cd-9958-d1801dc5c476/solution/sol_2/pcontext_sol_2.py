import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization:
      - Differential Evolution (jDE: self-adaptive F and CR per individual)
      - "DE/current-to-best/1" mixing (fast convergence) with diversity-friendly "rand/1"
      - Reflection bound handling (better than clipping)
      - Periodic small-budget local pattern search around best
      - Stagnation-triggered partial restart of worst individuals

    No external libraries required.
    Returns: best (float) best fitness found within max_time seconds.
    """

    # ------------------------ helpers ------------------------
    def clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect(x, lo, hi):
        if lo == hi:
            return lo
        # reflect until in-range (robust for large excursions)
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        # numeric safety
        if x < lo: x = lo
        if x > hi: x = hi
        return x

    def ensure_bounds_reflect(vec):
        out = vec[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = reflect(out[i], lo, hi)
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

    def pick3_excluding(n, exclude):
        # returns 3 distinct indices in [0,n) excluding `exclude`
        a = random.randrange(n)
        while a == exclude:
            a = random.randrange(n)
        b = random.randrange(n)
        while b == exclude or b == a:
            b = random.randrange(n)
        c = random.randrange(n)
        while c == exclude or c == a or c == b:
            c = random.randrange(n)
        return a, b, c

    # ------------------------ setup ------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # dimension spans (for local search step sizes)
    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    for i in range(dim):
        if span[i] <= 0.0:
            span[i] = 1.0

    best = float("inf")
    best_x = None

    # population size: modest but effective
    pop_size = max(10, min(40, 12 + 3 * dim))

    # initialize population and self-adaptive params
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    # jDE parameters per individual
    F = [random.uniform(0.4, 0.9) for _ in range(pop_size)]
    CR = [random.uniform(0.2, 0.95) for _ in range(pop_size)]
    tau1, tau2 = 0.1, 0.1   # probabilities to resample F/CR
    Fl, Fu = 0.1, 0.95

    # best
    for i in range(pop_size):
        if fit[i] < best:
            best = fit[i]
            best_x = pop[i][:]

    # stagnation + restart controls
    last_best = best
    no_improve_gens = 0

    # local search scheduling
    last_local = time.time()
    local_interval = 0.12  # seconds
    local_base = 0.10      # relative step to span, decays with progress
    local_decay = 0.995

    gen = 0
    while time.time() < deadline:
        gen += 1

        # mix strategies: more exploration early, more exploitation later
        time_frac = (time.time() - t0) / max(1e-12, float(max_time))
        p_exploit = clip(0.35 + 0.55 * time_frac, 0.35, 0.90)  # increase over time

        improved = False

        # one "generation"
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # jDE adaptation (trial parameters)
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                Fi = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CRi = random.random()

            # choose mutation strategy
            if best_x is None or random.random() > p_exploit:
                # DE/rand/1
                a, b, c = pick3_excluding(pop_size, i)
                xa, xb, xc = pop[a], pop[b], pop[c]
                v = [xa[j] + Fi * (xb[j] - xc[j]) for j in range(dim)]
            else:
                # DE/current-to-best/1 (fast convergence) + random difference
                a, b, c = pick3_excluding(pop_size, i)
                xb, xc = pop[b], pop[c]
                xi = pop[i]
                v = [xi[j] + Fi * (best_x[j] - xi[j]) + 0.5 * Fi * (xb[j] - xc[j]) for j in range(dim)]

            # binomial crossover
            jrand = random.randrange(dim)
            trial = [0.0] * dim
            xi = pop[i]
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    trial[j] = v[j]
                else:
                    trial[j] = xi[j]

            trial = ensure_bounds_reflect(trial)
            f_trial = safe_eval(trial)

            # selection + keep successful Fi/CRi (jDE rule)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                F[i] = Fi
                CR[i] = CRi
                improved = True

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        # stagnation tracking
        if best < last_best - 1e-12:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # partial restart when stuck (replace some worst)
        if no_improve_gens >= 8:
            no_improve_gens = 0
            k = max(1, pop_size // 4)  # replace worst 25%
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if time.time() >= deadline:
                    return best

                # re-seed: half near best, half global random
                if best_x is not None and random.random() < 0.6:
                    x = best_x[:]
                    rad = 0.25
                    for j in range(dim):
                        x[j] = x[j] + random.uniform(-rad, rad) * span[j]
                    x = ensure_bounds_reflect(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                F[idx] = random.uniform(0.4, 0.9)
                CR[idx] = random.uniform(0.2, 0.95)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_best = best

        # periodic local search around best (small evaluation budget)
        now = time.time()
        if best_x is not None and (now - last_local) >= local_interval and now < deadline:
            last_local = now

            x0 = best_x[:]
            f0 = best

            # probe a small subset of coordinates to keep it cheap
            coords = list(range(dim))
            random.shuffle(coords)
            m = min(dim, 8)
            coords = coords[:m]

            step_rel = local_base
            local_base *= local_decay  # very gentle decay

            for j in coords:
                if time.time() >= deadline:
                    return best
                step = step_rel * span[j]
                if step <= 0.0:
                    continue

                # pattern move: try + then - then smaller
                for s in (1.0, -1.0):
                    xt = x0[:]
                    xt[j] = xt[j] + s * step
                    xt = ensure_bounds_reflect(xt)
                    ft = safe_eval(xt)
                    if ft < f0:
                        x0, f0 = xt, ft

                # small refinement step if improved
                if f0 < best:
                    step2 = 0.35 * step
                    for s in (1.0, -1.0):
                        xt = x0[:]
                        xt[j] = xt[j] + s * step2
                        xt = ensure_bounds_reflect(xt)
                        ft = safe_eval(xt)
                        if ft < f0:
                            x0, f0 = xt, ft

            # sparse random-direction nudge (helps escape shallow traps)
            if time.time() < deadline:
                xt = x0[:]
                prob = min(0.25, 2.0 / max(2.0, dim))
                for j in range(dim):
                    if random.random() < prob:
                        xt[j] = xt[j] + random.uniform(-0.03, 0.03) * span[j]
                xt = ensure_bounds_reflect(xt)
                ft = safe_eval(xt)
                if ft < f0:
                    x0, f0 = xt, ft

            if f0 < best:
                best = f0
                best_x = x0[:]
                last_best = best

            # adapt local frequency slightly (more often if still improving)
            local_interval = clip(local_interval * (0.92 if improved else 1.03), 0.05, 0.30)

    return best
