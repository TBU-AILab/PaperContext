import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimization (no external libs).

    Algorithm: DE (Differential Evolution) + local coordinate/pattern refinement.
    - Fast global search with a small population, adaptive mutation/crossover.
    - Maintains diversity; periodically injects random individuals.
    - When a new best is found, performs a short local pattern/coordinate search.
    - Time-aware: all loops check the deadline.

    Returns:
        best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [bounds[i][0] for i in range(dim)]
    highs = [bounds[i][1] for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # guard against degenerate spans
    spans = [s if s != 0 else 1.0 for s in spans]

    def now():
        return time.time()

    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # --- small, cheap Latin-hypercube-like init to seed the population ---
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + random.random()) / n
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # --- local refinement: short coordinate/pattern search around a given point ---
    def local_refine(x, fx, budget_evals):
        # step sizes: start at 5% of span, shrink quickly
        step = [0.05 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] for i in range(dim)]
        shrink = 0.5
        grow = 1.2

        evals = 0
        x_best = x[:]
        f_best = fx

        while evals < budget_evals and now() < deadline:
            improved = False
            # coordinate search
            for d in range(dim):
                if evals >= budget_evals or now() >= deadline:
                    break
                sd = step[d]
                if sd <= min_step[d]:
                    continue

                # +sd
                xp = x_best[:]
                xp[d] += sd
                clip(xp)
                fp = eval_f(xp); evals += 1
                if fp < f_best:
                    x_best, f_best = xp, fp
                    improved = True
                    continue

                if evals >= budget_evals or now() >= deadline:
                    break

                # -sd
                xm = x_best[:]
                xm[d] -= sd
                clip(xm)
                fm = eval_f(xm); evals += 1
                if fm < f_best:
                    x_best, f_best = xm, fm
                    improved = True
                    continue

            if improved:
                # modestly expand steps to move faster
                for d in range(dim):
                    step[d] = min(0.25 * spans[d], step[d] * grow)
            else:
                # shrink steps to refine
                tiny = True
                for d in range(dim):
                    step[d] = max(min_step[d], step[d] * shrink)
                    if step[d] > 10.0 * min_step[d]:
                        tiny = False
                if tiny:
                    break

        return x_best, f_best

    # --- DE parameters (chosen to be robust under many objective types) ---
    # population size: small but scales with dim
    pop_size = max(12, min(60, 8 * dim))
    # time-aware: if max_time is tiny, keep pop small
    if max_time <= 0.2:
        pop_size = max(10, min(pop_size, 16))

    # Adaptive control parameters
    F_min, F_max = 0.45, 0.95    # mutation factor range
    CR_min, CR_max = 0.10, 0.95  # crossover range

    # Initialize population
    pop = []
    fit = []

    # Seed with LHS points + randoms
    n_lhs = min(pop_size, max(4, int(math.sqrt(pop_size) + 3)))
    for x in lhs_points(n_lhs):
        if now() >= deadline:
            return float("inf") if not fit else min(fit)
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    while len(pop) < pop_size and now() < deadline:
        x = rand_vec()
        fx = eval_f(x)
        pop.append(x); fit.append(fx)

    # Best tracking
    best_i = min(range(len(fit)), key=lambda i: fit[i])
    best_x = pop[best_i][:]
    best = fit[best_i]

    # If we have essentially no time, return what we have
    if now() >= deadline:
        return best

    # Success memories for mild self-adaptation
    # start from reasonable defaults
    Fm = 0.7
    CRm = 0.6

    # Main DE loop
    gen = 0
    inject_period = max(3, int(0.5 * math.sqrt(pop_size)))
    refine_cooldown = 0

    while now() < deadline:
        gen += 1

        # time-graded aggressiveness:
        # early: explore more (higher F, CR), later: exploit (lower F, mid CR)
        remaining = max(0.0, deadline - now())
        frac = remaining / max(1e-9, max_time)
        # frac ~ 1 early, ~0 late
        explore = frac

        # Iterate individuals
        for i in range(pop_size):
            if now() >= deadline:
                return best

            # Choose r1, r2, r3 distinct and != i
            # Do it fast without heavy sampling
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = i
            while r3 == i or r3 == r2 or r3 == r1:
                r3 = random.randrange(pop_size)

            # Current-to-best/1 mutation for faster convergence:
            # v = x_i + F*(best - x_i) + F*(x_r1 - x_r2)
            # with occasional rand/1 to preserve diversity
            use_rand1 = (random.random() < 0.20 + 0.25 * explore)  # more rand early
            # sample F, CR around memories
            # triangular-ish perturbation without libs
            def jitter(a):
                return (random.random() - random.random()) * a

            F = Fm + jitter(0.25)
            if F < F_min: F = F_min
            if F > F_max: F = F_max

            CR = CRm + jitter(0.35)
            if CR < CR_min: CR = CR_min
            if CR > CR_max: CR = CR_max

            xi = pop[i]
            if use_rand1:
                base = pop[r3]
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = base[d] + F * (pop[r1][d] - pop[r2][d])
            else:
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = xi[d] + F * (best_x[d] - xi[d]) + F * (pop[r1][d] - pop[r2][d])

            clip(v)

            # Binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]  # trial
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            # Evaluate
            fu = eval_f(u)

            # Selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu

                # Update best
                if fu < best:
                    best = fu
                    best_x = u[:]

                    # Local refinement when we find a new best, but throttle it
                    # so it doesn't dominate time.
                    if refine_cooldown <= 0 and now() < deadline:
                        # budget depends on dim and remaining time
                        # keep it small and safe
                        budget = max(6, min(60, 4 * dim))
                        # late-stage: allow slightly more refinement
                        if frac < 0.35:
                            budget = min(120, max(budget, 8 * dim))
                        x2, f2 = local_refine(best_x, best, budget)
                        if f2 < best:
                            best = f2
                            best_x = x2[:]
                        refine_cooldown = 2  # wait a couple gens before refining again

                # Mild self-adaptation: if trial succeeded, move memories toward used params
                # (simple exponential smoothing)
                Fm = 0.90 * Fm + 0.10 * F
                CRm = 0.90 * CRm + 0.10 * CR

        refine_cooldown -= 1

        # Periodic diversity injection (replace a few worst with random)
        if gen % inject_period == 0 and now() < deadline:
            # replace k worst
            k = max(1, pop_size // 10)
            # get indices of worst k (partial selection)
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if now() >= deadline:
                    break
                x = rand_vec()
                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

    return best
