import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization, self-contained (stdlib only).

    Hybrid algorithm:
      1) Low-discrepancy (Halton) + random + opposition initialization
      2) Small elite archive
      3) Adaptive Differential Evolution (DE/rand/1/bin) on the population
      4) Fast coordinate pattern-search "polish" on the current best (opportunistic)
      5) Occasional re-seeding to avoid stagnation

    Returns
    -------
    best : float
        Best (minimum) objective value found within the time budget.
    """

    deadline = time.perf_counter() + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def clamp_inplace(x):
        for i in range(dim):
            lo = lows[i]; hi = highs[i]
            v = x[i]
            if v < lo:
                x[i] = lo
            elif v > hi:
                x[i] = hi

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # Halton sequence for quasi-random init (no external libs)
    # Small prime base list (extendable)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173]

    def radical_inverse(n, base):
        # n >= 1 recommended
        inv_base = 1.0 / base
        f = inv_base
        r = 0.0
        while n > 0:
            n, mod = divmod(n, base)
            r += mod * f
            f *= inv_base
        return r

    def halton_point(index):
        # index should start at 1
        x = [0.0] * dim
        for i in range(dim):
            base = primes[i % len(primes)]
            u = radical_inverse(index, base)
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Fast-ish approx N(0,1) (sum of uniforms), avoids log/sin/cos
    def randn():
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    # ---------- parameters ----------
    # Population size: DE likes ~8..20*dim; keep modest for speed
    pop_size = max(16, min(80, 10 * dim))
    elite_size = max(4, min(12, 2 + dim // 2))

    # DE control parameters (self-adapting around these)
    F_mean = 0.6
    CR_mean = 0.9

    # Pattern-search step (normalized)
    polish_step0 = 0.15
    polish_step_min = 1e-6

    # Reseed settings
    stall_limit = 60 + 10 * dim  # evaluations without global improvement -> reseed some
    reseed_frac = 0.25

    # ---------- elite management ----------
    elites = []  # list of (fx, x)

    best = float("inf")
    best_x = None

    def consider(x, fx):
        nonlocal best, best_x, elites
        if fx < best:
            best = fx
            best_x = list(x)

        # maintain elites (sorted by fx)
        if len(elites) < elite_size or fx < elites[-1][0]:
            elites.append((fx, list(x)))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_size:
                elites.pop()

    # ---------- initialization ----------
    # Mix: Halton + random, each with opposition
    pop = []
    pop_fx = []

    halton_n = pop_size  # one Halton per pop member
    for k in range(1, halton_n + 1):
        if time.perf_counter() >= deadline:
            return best

        if random.random() < 0.65:
            x = halton_point(k)
        else:
            x = rand_point()

        fx = eval_f(x)
        consider(x, fx)
        pop.append(x); pop_fx.append(fx)

        # opposition candidate
        xo = opposite_point(x)
        clamp_inplace(xo)
        fxo = eval_f(xo)
        consider(xo, fxo)

        # if opposition is better, replace into population
        if fxo < fx:
            pop[-1] = xo
            pop_fx[-1] = fxo

    if best_x is None:
        return best

    # ---------- local polish: coordinate pattern search ----------
    def polish_best(time_budget_fraction=0.10):
        nonlocal best, best_x
        if best_x is None:
            return
        end_polish = min(deadline, time.perf_counter() + (deadline - time.perf_counter()) * time_budget_fraction)

        x = list(best_x)
        fx = best
        step = polish_step0

        # random coordinate order helps
        coords = list(range(dim))

        while time.perf_counter() < end_polish and step > polish_step_min:
            improved = False
            random.shuffle(coords)
            for i in coords:
                if time.perf_counter() >= end_polish:
                    break
                if spans[i] == 0.0:
                    continue

                delta = step * safe_spans[i]

                # try +delta
                xp = list(x); xp[i] += delta
                clamp_inplace(xp)
                f1 = eval_f(xp)
                if f1 < fx:
                    x, fx = xp, f1
                    consider(x, fx)
                    improved = True
                    continue

                # try -delta
                xm = list(x); xm[i] -= delta
                clamp_inplace(xm)
                f2 = eval_f(xm)
                if f2 < fx:
                    x, fx = xm, f2
                    consider(x, fx)
                    improved = True

            if not improved:
                step *= 0.5

        # update global best (already done via consider), but keep best_x aligned
        if fx < best:
            best = fx
            best_x = list(x)

    # quick initial polish
    polish_best(time_budget_fraction=0.05)

    # ---------- main loop: adaptive DE + occasional polish ----------
    last_improve_eval = 0
    eval_count = 0

    # Track best to detect stall
    best_seen = best

    while True:
        if time.perf_counter() >= deadline:
            return best

        # Occasionally polish (short bursts) if we are making progress or near the end
        if eval_count > 0 and (eval_count % (40 + 5 * dim) == 0):
            polish_best(time_budget_fraction=0.03)

        # One DE "generation" (but time-bounded, so we may exit anytime)
        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return best

            # Sample adaptive parameters around means (jDE-ish lite)
            # Keep in reasonable ranges without external distributions
            Fi = max(0.1, min(1.0, F_mean + 0.20 * randn()))
            CRi = max(0.0, min(1.0, CR_mean + 0.15 * randn()))

            # choose r1,r2,r3 distinct and != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = r1
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = r2
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(pop_size)

            x1 = pop[r1]; x2 = pop[r2]; x3 = pop[r3]
            target = pop[i]

            # mutant
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x1[d] + Fi * (x2[d] - x3[d])

            # binomial crossover
            u = list(target)
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]

            # repair: clamp (fast)
            clamp_inplace(u)

            fu = eval_f(u)
            eval_count += 1

            # selection
            if fu <= pop_fx[i]:
                pop[i] = u
                pop_fx[i] = fu
                consider(u, fu)

                # move means slightly toward successful params
                # (very lightweight adaptation)
                F_mean = 0.90 * F_mean + 0.10 * Fi
                CR_mean = 0.90 * CR_mean + 0.10 * CRi

            # improvement tracking for reseed
            if best < best_seen:
                best_seen = best
                last_improve_eval = eval_count

            # reseed on stall (inject diversity)
            if (eval_count - last_improve_eval) > stall_limit:
                last_improve_eval = eval_count
                # reseed worst fraction with random/halton around best + noise
                # sort indices by fitness descending
                idxs = list(range(pop_size))
                idxs.sort(key=lambda k: pop_fx[k], reverse=True)
                m = max(1, int(reseed_frac * pop_size))
                for t in range(m):
                    if time.perf_counter() >= deadline:
                        return best
                    j = idxs[t]

                    if best_x is not None and random.random() < 0.7:
                        # sample around best with moderate noise
                        xnew = list(best_x)
                        for d in range(dim):
                            if spans[d] == 0.0:
                                continue
                            xnew[d] += (0.20 * safe_spans[d]) * randn()
                        clamp_inplace(xnew)
                    else:
                        # mix halton and random
                        if random.random() < 0.5:
                            xnew = halton_point(1 + random.randrange(1, 10000))
                        else:
                            xnew = rand_point()

                    fnew = eval_f(xnew)
                    eval_count += 1
                    pop[j] = xnew
                    pop_fx[j] = fnew
                    consider(xnew, fnew)

                # after reseed, do a tiny polish to exploit new best
                polish_best(time_budget_fraction=0.02)
                break  # restart generation loop to re-check time etc.
