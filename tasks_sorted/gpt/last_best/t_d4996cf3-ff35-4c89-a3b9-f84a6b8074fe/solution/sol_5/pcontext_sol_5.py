import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Key upgrades vs the previous hybrids:
      1) Stronger global start: scrambled Halton + selective opposition + sparse random.
      2) Main engine: Differential Evolution (DE/rand-to-best/1/bin) with
         - adaptive F/CR (jDE-style per-individual)
         - a small "current-to-best" probability to accelerate convergence
         - boundary handling by reflection (better than hard clip for DE)
      3) Local refinement: lightweight coordinate + pattern search with step decay.
      4) Stagnation handling: partial reinitialization + sigma/step reset.
      5) Careful time checks: returns best fitness within max_time.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    var_idx = [i for i in range(dim) if spans[i] > 0.0]
    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    def now():
        return time.time()

    def eval_f(x):
        return float(func(x))

    # ---- boundary handling: reflect into [lo, hi] (works well for DE steps) ----
    def reflect_inplace(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            xi = x[i]
            if xi < lo or xi > hi:
                w = hi - lo
                if w <= 0.0:
                    x[i] = lo
                    continue
                # reflect repeatedly (handles big jumps)
                xi = lo + abs((xi - lo) % (2.0 * w))
                if xi > hi:
                    xi = hi - (xi - hi)
                x[i] = xi
        return x

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] > 0.0:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return xo

    # ---- scrambled Halton for global coverage ----
    def first_primes(n):
        primes = []
        p = 2
        while len(primes) < n:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def van_der_corput(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---- cheap local search: coordinate/pattern with decay ----
    def local_refine(x0, f0, eval_budget):
        x = x0[:]
        fx = f0
        order = var_idx[:]
        order.sort(key=lambda i: spans[i], reverse=True)

        # start step relative to box
        step = [0.0] * dim
        for i in var_idx:
            step[i] = 0.03 * spans[i]

        used = 0
        # a few rounds with shrinking steps
        while used < eval_budget and now() < deadline:
            improved = False
            for i in order:
                si = step[i]
                if si <= 0.0:
                    continue

                # try +/- step
                best_local = fx
                best_x = None
                xi = x[i]
                for d in (-1.0, 1.0):
                    cand = x[:]
                    cand[i] = xi + d * si
                    reflect_inplace(cand)
                    fc = eval_f(cand)
                    used += 1
                    if fc < best_local:
                        best_local = fc
                        best_x = cand
                    if used >= eval_budget or now() >= deadline:
                        break

                if best_x is not None:
                    x, fx = best_x, best_local
                    improved = True

                if used >= eval_budget or now() >= deadline:
                    break

            if not improved:
                # shrink
                alive = False
                for i in var_idx:
                    step[i] *= 0.5
                    if step[i] > 1e-14 * (spans[i] + 1.0):
                        alive = True
                if not alive:
                    break

        return fx, x

    # -------------------- Initialization --------------------
    # population size: DE benefits from ~10*dim but time-limited -> cap
    NP = int(max(18, min(140, 12 + 6 * dim)))
    if max_time <= 0.3:
        NP = min(NP, 28)
    if max_time <= 1.0:
        NP = min(NP, 50)

    pop = []
    fit = []

    x_best = None
    f_best = float("inf")

    # mix halton/random/opposition during init
    for i in range(NP):
        if now() >= deadline:
            return f_best if x_best is not None else float(eval_f(rand_point()))

        if i % 3 == 0:
            x = halton_point()
        elif i % 3 == 1:
            x = rand_point()
        else:
            x = halton_point()
            xo = opposite_point(x)
            # pick better of x and opposition
            reflect_inplace(x)
            reflect_inplace(xo)
            fx = eval_f(x)
            if now() >= deadline:
                return fx if fx < f_best else f_best
            fo = eval_f(xo)
            if fo < fx:
                x = xo
                fx = fo
            pop.append(x)
            fit.append(fx)
            if fx < f_best:
                f_best, x_best = fx, x[:]
            continue

        reflect_inplace(x)
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # per-individual parameters (jDE)
    F = [0.5 + 0.3 * random.random() for _ in range(NP)]   # mutation factor
    CR = [0.6 + 0.35 * random.random() for _ in range(NP)] # crossover rate

    # stagnation controls
    last_best = f_best
    stagn_gens = 0
    gen = 0

    # -------------------- Main DE loop --------------------
    while now() < deadline:
        gen += 1
        improved_this_gen = False

        # occasional local refine on incumbent best
        if gen % 6 == 0 and now() < deadline:
            fb2, xb2 = local_refine(x_best, f_best, eval_budget=2 * len(var_idx) + 8)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                improved_this_gen = True

        # produce next generation
        for i in range(NP):
            if now() >= deadline:
                return f_best

            # jDE parameter self-adaptation (small probabilities)
            if random.random() < 0.12:
                F[i] = 0.15 + 0.85 * random.random()
            if random.random() < 0.12:
                CR[i] = random.random()

            Fi = F[i]
            CRi = CR[i]

            # choose r1,r2,r3 distinct and != i
            # also use "rand-to-best" component sometimes
            idxs = list(range(NP))
            # cheap way without allocating a lot: sample until distinct
            def pick_excluding(excl_set):
                while True:
                    r = random.randrange(NP)
                    if r not in excl_set:
                        return r

            excl = {i}
            r1 = pick_excluding(excl); excl.add(r1)
            r2 = pick_excluding(excl); excl.add(r2)
            r3 = pick_excluding(excl); excl.add(r3)

            xi = pop[i]
            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]

            # mutation: mostly rand/1, sometimes current-to-best/1-ish
            v = xi[:]  # will be overwritten for var dims
            use_to_best = (random.random() < 0.35)
            if use_to_best and x_best is not None:
                # v = x1 + F*(x_best - x1) + F*(x2 - x3)
                for j in var_idx:
                    v[j] = x1[j] + Fi * (x_best[j] - x1[j]) + Fi * (x2[j] - x3[j])
            else:
                # v = x1 + F*(x2 - x3)
                for j in var_idx:
                    v[j] = x1[j] + Fi * (x2[j] - x3[j])

            # binomial crossover
            u = xi[:]
            jrand = var_idx[random.randrange(len(var_idx))]
            for j in var_idx:
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]

            reflect_inplace(u)
            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                improved_this_gen = True
                if fu < f_best:
                    f_best = fu
                    x_best = u[:]

        # stagnation detection / rescue
        if f_best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = f_best
            stagn_gens = 0
        else:
            stagn_gens += 1

        # if stuck: partial reinit + slightly more exploratory params
        if stagn_gens >= 8 and now() < deadline:
            stagn_gens = 0
            # reinitialize worst 30%
            idx = list(range(NP))
            idx.sort(key=lambda k: fit[k], reverse=True)
            k_re = max(2, NP // 3)

            for t in range(k_re):
                if now() >= deadline:
                    return f_best
                k = idx[t]
                # around best with noise + some pure global
                if random.random() < 0.65 and x_best is not None:
                    x = x_best[:]
                    for j in var_idx:
                        x[j] += random.gauss(0.0, 0.18 * spans[j])
                else:
                    x = halton_point() if random.random() < 0.6 else rand_point()
                reflect_inplace(x)
                fx = eval_f(x)
                pop[k] = x
                fit[k] = fx
                F[k] = 0.45 + 0.4 * random.random()
                CR[k] = 0.4 + 0.55 * random.random()
                if fx < f_best:
                    f_best, x_best = fx, x[:]

    return f_best
