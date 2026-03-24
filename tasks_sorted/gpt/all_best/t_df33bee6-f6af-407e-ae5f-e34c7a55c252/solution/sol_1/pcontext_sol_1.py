import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs).

    Hybrid strategy:
      1) Global exploration: Differential Evolution "current-to-best/1" (fast convergence)
         with dithered F and time-adapted CR.
      2) Exploitation: lightweight (1+1)-ES around best with success-based sigma control.
      3) Local refinement: occasional coordinate pattern search with shrinking steps.
      4) Restarts: partial reinitialization via Latin-hypercube-like stratified sampling
         when stalled.

    Returns:
      best (float): minimum fitness found within max_time seconds.
    """

    # ----------------- helpers -----------------
    def clip_inplace(x):
        for i, (lo, hi) in enumerate(bounds):
            xi = x[i]
            if xi < lo:
                x[i] = lo
            elif xi > hi:
                x[i] = hi
        return x

    def span(i):
        lo, hi = bounds[i]
        s = hi - lo
        return s if s > 0 else 1.0

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if not math.isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Approx standard normal using sum of uniforms (CLT), no external libs
    def randn():
        # mean 0, var ~1 with 12 uniforms - 6
        return (random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() + random.random() - 6.0)

    # Stratified (Latin-ish) samples per dimension; good coverage quickly
    def stratified_population(n):
        # For each dimension, create n strata and permute them
        perms = []
        for d in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            perms.append(idx)
        pop = []
        for i in range(n):
            x = []
            for d in range(dim):
                lo, hi = bounds[d]
                # sample inside stratum
                u = (perms[d][i] + random.random()) / float(n)
                x.append(lo + u * (hi - lo))
            pop.append(x)
        return pop

    # ----------------- setup -----------------
    start = time.time()
    deadline = start + float(max_time)

    # population sizing
    pop_size = max(10, min(60, 12 + 3 * dim))

    # initial population: stratified for better coverage than pure random
    pop = stratified_population(pop_size)
    fit = [eval_f(x) for x in pop]

    best_i = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # DE parameters (time-adapted); F is dithered per trial
    CR_hi, CR_lo = 0.95, 0.60

    # Local step sizes for pattern search
    step = [0.20 * span(i) for i in range(dim)]
    min_step = [1e-14 * span(i) + 1e-18 for i in range(dim)]

    # (1+1)-ES sigma (relative to bounds)
    sigma = 0.12  # start fairly exploratory
    sigma_min = 1e-12
    sigma_max = 0.35

    # Stall / restart controls
    last_improve_t = start
    stall_seconds = max(0.20, 0.12 * max_time)

    # ----------------- main loop -----------------
    # We alternate: DE sweep -> ES burst -> occasional pattern search -> stall handling
    while True:
        now = time.time()
        if now >= deadline:
            return best

        # progress fraction
        if max_time > 0:
            t = min(1.0, (now - start) / max_time)
        else:
            t = 1.0

        CR = CR_hi - (CR_hi - CR_lo) * t

        # ---------- DE sweep: current-to-best/1 (fast exploitation) ----------
        # trial = xi + F*(best-xi) + F*(xr1-xr2)
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose r1,r2 distinct and != i
            # (avoid O(n) list creation)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # dithered F per individual (often improves robustness)
            # start higher, end lower
            F_base_hi, F_base_lo = 0.95, 0.35
            F_base = F_base_hi - (F_base_hi - F_base_lo) * t
            F = random.uniform(0.5 * F_base, 1.0 * F_base)

            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = xi[d] + F * (best_x[d] - xi[d]) + F * (xr1[d] - xr2[d])
            clip_inplace(mutant)

            # binomial crossover
            trial = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if random.random() < CR or d == jrand:
                    trial[d] = mutant[d]

            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]
                    last_improve_t = time.time()

        # ---------- (1+1)-ES burst around best with success-based sigma ----------
        # small, very fast local improvement; sigma shrinks if failing repeatedly
        successes = 0
        attempts = 0
        # allocate more local attempts late in time budget
        es_tries = 4 + int(10 * t)
        for _ in range(es_tries):
            if time.time() >= deadline:
                return best
            attempts += 1

            cand = best_x[:]
            # scale per-dimension by span
            for d in range(dim):
                cand[d] += randn() * (sigma * span(d))
            clip_inplace(cand)

            f_c = eval_f(cand)
            if f_c < best:
                best = f_c
                best_x = cand
                successes += 1
                last_improve_t = time.time()

        # 1/5-like success rule (coarse but effective)
        if attempts > 0:
            rate = successes / float(attempts)
            if rate > 0.22:
                sigma = min(sigma_max, sigma * 1.25)
            elif rate < 0.12:
                sigma = max(sigma_min, sigma * 0.70)

        # ---------- occasional coordinate pattern search ----------
        # do it more often late, but keep it cheap
        if (t > 0.25 and random.random() < (0.15 + 0.45 * t)) or (t <= 0.25 and random.random() < 0.10):
            improved = False
            x = best_x[:]
            for d in range(dim):
                if time.time() >= deadline:
                    return best
                sd = step[d]
                if sd <= min_step[d]:
                    continue

                # Try a small set of moves including half-step (helps on ridges)
                for delta in (sd, -sd, 0.5 * sd, -0.5 * sd):
                    cand = x[:]
                    cand[d] += delta
                    clip_inplace(cand)
                    f_c = eval_f(cand)
                    if f_c < best:
                        best = f_c
                        best_x = cand
                        x = cand
                        improved = True
                        last_improve_t = time.time()
                        break

            if not improved:
                step = [s * 0.55 for s in step]
            else:
                # mild expansion, capped
                for d in range(dim):
                    step[d] = min(step[d] * 1.10, 0.30 * span(d))

        # ---------- stall handling / restart ----------
        if time.time() - last_improve_t > stall_seconds:
            # keep elite set + rebuild remainder with stratified + near-best noise
            elite_k = max(2, pop_size // 6)
            idx_sorted = sorted(range(pop_size), key=lambda i: fit[i])
            elites = [pop[i][:] for i in idx_sorted[:elite_k]]
            elite_fit = [fit[i] for i in idx_sorted[:elite_k]]

            # new population
            new_pop = elites[:]
            new_fit = elite_fit[:]

            remaining = pop_size - elite_k
            # half: stratified global samples, half: near-best samples
            n_global = remaining // 2
            n_local = remaining - n_global

            # global stratified
            if n_global > 0:
                gp = stratified_population(n_global)
                for x in gp:
                    new_pop.append(x)
                    new_fit.append(eval_f(x))

            # local cloud around best
            local_sigma = max(0.06, sigma)  # ensure some spread
            for _ in range(n_local):
                x = best_x[:]
                for d in range(dim):
                    x[d] += randn() * (local_sigma * 0.18 * span(d))
                clip_inplace(x)
                new_pop.append(x)
                new_fit.append(eval_f(x))

            pop, fit = new_pop, new_fit

            best_i = min(range(pop_size), key=lambda i: fit[i])
            if fit[best_i] < best:
                best = fit[best_i]
                best_x = pop[best_i][:]
            last_improve_t = time.time()

            # reset some local search radii to escape plateaus
            step = [max(0.10 * span(d), step[d]) for d in range(dim)]
            sigma = max(0.10, sigma)
