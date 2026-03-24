import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs):
    - DE/current-to-best/1/bin with per-individual jDE-style self-adaptation (Fi, CRi)
    - Low-discrepancy (Halton) + opposition-based initialization
    - Lightweight memoization
    - Periodic multi-start local search:
        * randomized coordinate search + small pattern steps
        * occasional 2D subspace refinement
    - Diversity injection when stagnating

    Returns: best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0 else 0.0 for s in spans]

    # ---------------- helpers ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    # small rounding to enable cache hits while not collapsing search too much
    cache = {}
    def evaluate(x):
        key = tuple(round(v, 10) for v in x)
        v = cache.get(key)
        if v is None:
            v = float(func(list(x)))
            cache[key] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --- Halton sequence for better initial coverage (no libs) ---
    def _first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def _van_der_corput(k, base):
        # radical inverse
        v, denom = 0.0, 1.0
        while k > 0:
            k, rem = divmod(k, base)
            denom *= base
            v += rem / denom
        return v

    primes = _first_primes(max(1, dim))

    def halton_point(index):
        # index should start at 1
        return [_van_der_corput(index, primes[j]) for j in range(dim)]

    def from_unit(u):
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def opposition(x):
        # opposite point across the center of bounds
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------------- local search ----------------
    def local_search(x0, f0, time_limit):
        """
        Derivative-free local search:
        - randomized coordinate steps with decaying step sizes
        - occasional 2D subspace tweak
        """
        x = x0[:]
        fx = f0

        # start with moderately sized steps; shrink adaptively
        steps = [0.15 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
        min_step = 1e-12

        no_improve_rounds = 0
        order = list(range(dim))

        while time.time() < time_limit:
            random.shuffle(order)
            improved = False

            # coordinate moves
            for i in order:
                if time.time() >= time_limit:
                    break
                si = steps[i]
                if si <= min_step:
                    continue

                # try both directions; also a small "pattern" overshoot if improved
                best_local = (fx, None)

                # +si
                xp = x[:]
                xp[i] += si
                clip_inplace(xp)
                fp = evaluate(xp)
                if fp < best_local[0]:
                    best_local = (fp, xp)

                # -si
                xm = x[:]
                xm[i] -= si
                clip_inplace(xm)
                fm = evaluate(xm)
                if fm < best_local[0]:
                    best_local = (fm, xm)

                if best_local[1] is not None:
                    x = best_local[1]
                    fx = best_local[0]
                    improved = True

                    # pattern step in same direction (small)
                    if time.time() < time_limit:
                        # infer direction by comparing coordinate
                        direction = 1.0 if x[i] > x0[i] else -1.0
                        xt = x[:]
                        xt[i] += direction * 0.5 * si
                        clip_inplace(xt)
                        ft = evaluate(xt)
                        if ft < fx:
                            x, fx = xt, ft

            # occasional 2D random subspace adjustment
            if time.time() < time_limit and dim >= 2 and random.random() < 0.25:
                i = random.randrange(dim)
                j = random.randrange(dim - 1)
                if j >= i:
                    j += 1
                si = steps[i]
                sj = steps[j]
                if si > min_step or sj > min_step:
                    # random small combination
                    xt = x[:]
                    xt[i] += (random.random() * 2 - 1) * si
                    xt[j] += (random.random() * 2 - 1) * sj
                    clip_inplace(xt)
                    ft = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True

            if improved:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                # shrink steps when stuck
                shrink = 0.6 if no_improve_rounds < 3 else 0.35
                for k in range(dim):
                    steps[k] *= shrink
                if max(steps) <= min_step:
                    break

            x0 = x[:]  # update reference for pattern direction guess

        return x, fx

    # ---------------- population init ----------------
    # population size: scale with dimension but keep modest for time-bounded runs
    pop_size = max(12, min(50, 8 * dim + 8))

    pop = []
    fit = []

    # Use Halton points + their opposites for coverage, then fill randomly
    # Start index at 1 to avoid all zeros.
    halton_needed = min(pop_size, max(4, pop_size // 2))
    for k in range(1, halton_needed + 1):
        u = halton_point(k)
        x = from_unit(u)
        xo = opposition(x)
        pop.append(x)
        if len(pop) < pop_size:
            pop.append(xo)

    while len(pop) < pop_size:
        pop.append(rand_vec())

    # Evaluate and (optionally) keep the better of each pair (x, opposition(x))
    for idx in range(pop_size):
        f = evaluate(pop[idx])
        fit.append(f)

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # ------------- DE parameters: jDE self-adaptation -------------
    # Each individual has its own F and CR that evolve
    Fi = [0.5 + 0.3 * random.random() for _ in range(pop_size)]
    CRi = [0.2 + 0.6 * random.random() for _ in range(pop_size)]
    tau1 = 0.1
    tau2 = 0.1
    Fl, Fu = 0.1, 0.9

    # Stagnation tracking
    last_best = best
    stagnation = 0

    gen = 0
    while True:
        now = time.time()
        if now >= deadline:
            return best

        gen += 1

        # Choose strategy probability: sometimes use DE/rand/1 for diversity
        use_current_to_best = 0.8

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # jDE adaptation
            if random.random() < tau1:
                Fi[i] = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CRi[i] = random.random()

            F = Fi[i]
            CR = CRi[i]

            # pick distinct indices
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xi = pop[i]
            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation
            if random.random() < use_current_to_best:
                # current-to-best/1: v = xi + F*(best - xi) + F*(xb - xc)
                v = [xi[j] + F * (best_x[j] - xi[j]) + F * (xb[j] - xc[j]) for j in range(dim)]
            else:
                # rand/1: v = xa + F*(xb - xc)
                v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    u[j] = v[j]

            clip_inplace(u)
            fu = evaluate(u)

            # selection + "success-based" keep of parameters already handled by jDE
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # ---- stagnation / diversity injection ----
        if best < last_best - 1e-12:
            last_best = best
            stagnation = 0
        else:
            stagnation += 1

        # compute cheap diversity estimate occasionally
        if gen % max(5, 2 + dim // 3) == 0 and time.time() < deadline:
            # If stagnating, inject new candidates near best + random immigrants
            if stagnation >= 6:
                # keep top fraction, replace worst fraction
                order = sorted(range(pop_size), key=lambda k: fit[k])
                keep = max(3, pop_size // 3)

                # immigrants: some random, some gaussian-like around best (triangular)
                for idx in order[keep:]:
                    if time.time() >= deadline:
                        return best
                    if random.random() < 0.6:
                        # near-best perturbation (triangular approx)
                        xnew = best_x[:]
                        for j in range(dim):
                            if spans[j] == 0:
                                continue
                            # triangular noise in [-1,1]
                            noise = (random.random() - random.random())
                            # scale shrinks with stagnation but not too small
                            scale = (0.05 + 0.15 * (1.0 / (1.0 + stagnation))) * spans[j]
                            xnew[j] += noise * scale
                        clip_inplace(xnew)
                    else:
                        xnew = rand_vec()

                    fnew = evaluate(xnew)
                    pop[idx] = xnew
                    fit[idx] = fnew
                    Fi[idx] = Fl + random.random() * (Fu - Fl)
                    CRi[idx] = random.random()

                    if fnew < best:
                        best = fnew
                        best_x = xnew[:]

                stagnation = 0  # reset after injection

        # ---- periodic local search on best (budgeted) ----
        if gen % max(7, 6 + dim // 2) == 0 and time.time() < deadline:
            remaining = deadline - time.time()
            # spend a small but meaningful fraction on local search
            budget = min(0.20 * float(max_time), 0.30 * remaining)
            if budget > 0:
                xls, fls = local_search(best_x, best, time.time() + budget)
                if fls < best:
                    best = fls
                    best_x = xls[:]
                    last_best = best
                    stagnation = 0
