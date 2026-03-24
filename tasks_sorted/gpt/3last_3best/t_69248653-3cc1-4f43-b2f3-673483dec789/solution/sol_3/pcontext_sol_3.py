import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (stdlib-only) improved vs prior portfolio approach.

    Key upgrades:
    - Better global exploration: scrambled Halton + opposition points + periodic global injections.
    - Stronger local search: Adaptive Differential Evolution (JADE-like) on a small population,
      plus a fast coordinate/pattern-search "polish" around the current best.
    - Stagnation handling: partial reinitialization, step-size control, and boundary-safe repair.
    - Robust objective evaluation: guards against exceptions/NaN/inf.

    Returns: best (minimum) objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [s if s > 0 else 1.0 for s in span]

    # ---------- utilities ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def opposite_point(x):
        # opposition-based: reflect across center
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            if v is None or isinstance(v, complex):
                return float("inf")
            v = float(v)
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ---------- Halton (scrambled) ----------
    def _primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for p in range(2, int(n ** 0.5) + 1):
            if sieve[p]:
                start = p * p
                step = p
                sieve[start:n + 1:step] = [False] * (((n - start) // step) + 1)
        return [i for i, ok in enumerate(sieve) if ok]

    def _first_n_primes(n):
        if n <= 0:
            return []
        # upper bound heuristic, grow if needed
        ub = max(50, int(n * (math.log(max(3, n)) + math.log(math.log(max(3, n))) + 3)))
        primes = _primes_upto(ub)
        while len(primes) < n:
            ub = int(ub * 1.7) + 10
            primes = _primes_upto(ub)
        return primes[:n]

    primes = _first_n_primes(dim)
    scramble = [random.random() for _ in range(dim)]

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_value(k, primes[i])
            u = (u + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # ---------- DE helpers ----------
    def ensure_in_bounds(trial, base):
        # "bounce-back" repair towards base if out of bounds, otherwise clip.
        # (improves over hard clip on some problems)
        for j in range(dim):
            if trial[j] < lo[j] or trial[j] > hi[j]:
                # pull back between base and bound-random
                r = random.random()
                if trial[j] < lo[j]:
                    trial[j] = lo[j] + r * (base[j] - lo[j])
                else:
                    trial[j] = hi[j] - r * (hi[j] - base[j])
        return clip_inplace(trial)

    def pick3_excluding(n, excl):
        # pick 3 distinct indices != excl
        a = excl
        while a == excl:
            a = random.randrange(n)
        b = a
        while b == excl or b == a:
            b = random.randrange(n)
        c = b
        while c == excl or c == a or c == b:
            c = random.randrange(n)
        return a, b, c

    # ---------- coordinate/pattern polish ----------
    def polish(best_x, best_f, max_evals):
        # Deterministic-ish coordinate probing with shrinking step.
        # Uses remaining time indirectly via eval budget.
        x = list(best_x)
        fx = best_f
        step = [0.12 * scale[i] for i in range(dim)]
        min_step = [1e-12 * scale[i] for i in range(dim)]
        evals = 0

        # try several passes; stop when steps tiny or budget used
        while evals < max_evals:
            improved = False
            # random order to avoid bias
            order = list(range(dim))
            random.shuffle(order)

            for j in order:
                if evals >= max_evals:
                    break
                sj = step[j]
                if sj <= min_step[j]:
                    continue

                # try +/- sj
                x0 = x[j]
                for sgn in (-1.0, 1.0):
                    if evals >= max_evals:
                        break
                    xn = list(x)
                    xn[j] = x0 + sgn * sj
                    clip_inplace(xn)
                    fn = safe_eval(xn)
                    evals += 1
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        break

                # small greedy acceleration along same coord
                if improved and evals < max_evals:
                    xn = list(x)
                    xn[j] = x[j] + (x[j] - x0)  # one more step in same direction
                    clip_inplace(xn)
                    fn = safe_eval(xn)
                    evals += 1
                    if fn < fx:
                        x, fx = xn, fn

            if not improved:
                # shrink steps
                for j in range(dim):
                    step[j] *= 0.55
                # stop if all tiny
                if max(step) <= max(min_step):
                    break

        return x, fx, evals

    # ---------- initialization ----------
    # Population size: small for speed but grows a bit with dimension
    NP = max(12, min(60, 10 + 4 * dim))

    pop = []
    fit = []

    # seed with Halton + random + opposition (cheap improvements on some landscapes)
    k = 1
    while len(pop) < NP and time.time() < deadline:
        if len(pop) % 3 == 0:
            x = rand_point()
        else:
            x = halton_point(k)
            k += 1

        fx = safe_eval(x)
        xo = opposite_point(x)
        fo = safe_eval(xo)

        if fo < fx:
            x, fx = xo, fo

        pop.append(list(x))
        fit.append(fx)

    if not pop:
        return float("inf")

    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    best_x = list(pop[best_idx])
    best = fit[best_idx]

    # JADE-like parameter adaptation (lightweight)
    mu_F = 0.5
    mu_CR = 0.5

    # p-best fraction for current-to-pbest mutation
    p_min = 2.0 / max(2, NP)
    p_frac = min(0.2, max(p_min, 0.08))

    # stagnation / injections
    no_best_improve = 0
    inj_period = max(40, 15 * dim)       # iterations between possible global injections
    hard_stagn = max(150, 50 * dim)

    it = 0
    while time.time() < deadline:
        it += 1

        # occasional polishing around best (cheap, improves final quality)
        if it % max(35, 10 * dim) == 0:
            # spend a small eval budget, not time-based
            budget = max(12, 3 * dim)
            px, pf, used = polish(best_x, best, budget)
            if pf < best:
                best = pf
                best_x = px
                no_best_improve = 0

        # global injection if stuck or periodically
        if (it % inj_period == 0) or (no_best_improve > hard_stagn):
            # replace a few worst individuals with new Halton/random around best
            n_replace = max(1, NP // 6)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:n_replace]
            for w in worst:
                if time.time() >= deadline:
                    break
                if random.random() < 0.6:
                    x = halton_point(k)
                    k += 1
                else:
                    # sample around best with moderate radius
                    x = [best_x[j] + random.gauss(0.0, 0.35 * scale[j]) for j in range(dim)]
                    clip_inplace(x)
                fx = safe_eval(x)
                # opposition try
                if random.random() < 0.3:
                    xo = opposite_point(x)
                    fo = safe_eval(xo)
                    if fo < fx:
                        x, fx = xo, fo
                pop[w] = list(x)
                fit[w] = fx

            best_idx = min(range(NP), key=lambda i: fit[i])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = list(pop[best_idx])
                no_best_improve = 0
            else:
                # if we were hard-stagnating, reset counter to avoid repeated nukes
                if no_best_improve > hard_stagn:
                    no_best_improve = hard_stagn // 2

        # prepare ranking for p-best
        order = sorted(range(NP), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_frac * NP)))

        # per-generation successful params
        s_F = []
        s_CR = []

        improved_this_gen = False

        for i in range(NP):
            if time.time() >= deadline:
                break

            xi = pop[i]

            # sample CR, F (clipped)
            CR = random.gauss(mu_CR, 0.1)
            if CR < 0.0: CR = 0.0
            if CR > 1.0: CR = 1.0

            # F from Cauchy around mu_F
            # basic cauchy: mu + gamma*tan(pi*(u-0.5))
            u = random.random()
            u = min(1.0 - 1e-12, max(1e-12, u))
            F = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
            # resample if non-positive
            tries = 0
            while F <= 0.0 and tries < 5:
                u = random.random()
                u = min(1.0 - 1e-12, max(1e-12, u))
                F = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose pbest and r1,r2
            pbest = order[random.randrange(p_count)]
            r1, r2, r3 = pick3_excluding(NP, i)  # 3 picks; we'll use two plus maybe third
            # ensure r1/r2 not pbest etc doesn't matter much; keep simple

            xp = pop[pbest]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # current-to-pbest/1:
            # v = x + F*(xp - x) + F*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + F * (xp[j] - xi[j]) + F * (xr1[j] - xr2[j])

            ensure_in_bounds(v, xi)

            # binomial crossover
            jrand = random.randrange(dim)
            uvec = [0.0] * dim
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    uvec[j] = v[j]
                else:
                    uvec[j] = xi[j]

            ensure_in_bounds(uvec, xi)
            fu = safe_eval(uvec)

            if fu <= fit[i]:
                pop[i] = uvec
                fit[i] = fu
                s_F.append(F)
                s_CR.append(CR)
                if fu < best:
                    best = fu
                    best_x = list(uvec)
                    improved_this_gen = True

        if improved_this_gen:
            no_best_improve = 0
        else:
            no_best_improve += 1

        # update mu_F, mu_CR using successful parameters (JADE style, simplified)
        if s_F:
            # Lehmer mean for F
            num = sum(f * f for f in s_F)
            den = sum(s_F)
            if den > 0:
                mu_F = 0.9 * mu_F + 0.1 * (num / den)

            mu_CR = 0.9 * mu_CR + 0.1 * (sum(s_CR) / len(s_CR))

        # small decay of exploration late in time (keeps it from thrashing)
        frac = (time.time() - t0) / max(1e-9, float(max_time))
        if frac > 0.7:
            p_frac = max(p_min, p_frac * 0.999)

    return best
