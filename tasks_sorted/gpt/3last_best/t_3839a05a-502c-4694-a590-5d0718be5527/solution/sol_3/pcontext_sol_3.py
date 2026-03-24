import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvement over provided best:
      - Keeps an elite archive but also maintains a *small evolving population*
        and runs a JADE/SHADE-like DE with:
          * current-to-pbest/1 mutation
          * adaptive F and CR (learned from successful moves)
          * external archive for diversity
      - Adds a deterministic-ish local refinement stage (short coordinate pattern search)
        when improvement is found, for fast polishing.
      - Uses boundary "bounce-back" (reflection) + clamp (safer than pure clamp).
      - Keeps a quantized cache (optional, low overhead) to reduce repeats.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return time.time() < deadline

    # --- bounds ---
    lo = [0.0] * dim
    hi = [0.0] * dim
    span = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
        span[i] = b - a

    def clamp_reflect(x):
        # reflect then clamp (handles large jumps better than clamp-only)
        for i in range(dim):
            a, b = lo[i], hi[i]
            if x[i] < a:
                if span[i] > 0:
                    x[i] = a + (a - x[i])  # reflect
                    if x[i] > b:
                        # if still out, clamp
                        x[i] = a
                else:
                    x[i] = a
            elif x[i] > b:
                if span[i] > 0:
                    x[i] = b - (x[i] - b)  # reflect
                    if x[i] < a:
                        x[i] = b
                else:
                    x[i] = b
            # final clamp
            if x[i] < a:
                x[i] = a
            elif x[i] > b:
                x[i] = b
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # --- quantized cache ---
    cache = {}
    cache_max = 9000

    def key_of(x):
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0 else 1.0
            # 1e-7 of range quantization
            k.append(int(((x[i] - lo[i]) / s) * 1e7 + 0.5))
        return tuple(k)

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        if len(cache) > cache_max:
            # prune random chunk
            kill = min(len(cache) // 6, 1500)
            for kk in random.sample(list(cache.keys()), k=kill):
                cache.pop(kk, None)
        return fx

    # --- random variates ---
    def randn():
        # approx N(0,1)
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy():
        u = random.random()
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # --- Halton init (low discrepancy) ---
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(max(1, dim))

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            i, rem = divmod(i, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(index, primes[i])
            x[i] = lo[i] + u * span[i]
        return x

    # --- elite archive (best few solutions) ---
    elite_k = max(6, min(24, 3 + dim // 2))
    elites = []  # list of (f, x)

    def add_elite(x, fx):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # --- local refinement (cheap pattern search around incumbent) ---
    def refine(best_x, best_f, budget_steps=18):
        # coordinate pokes + shrinking step
        if dim <= 0:
            return best_x, best_f
        base_step = [0.04 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        x = best_x[:]
        f = best_f
        step = base_step[:]
        it = 0
        while it < budget_steps and time_left():
            improved = False
            # random coordinate order (cheap)
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if not time_left():
                    break
                if span[i] <= 0:
                    continue
                si = step[i]
                if si <= 1e-16 * (span[i] if span[i] > 0 else 1.0):
                    continue
                # try +/- with small "momentum" when improving
                for sgn in (+1.0, -1.0):
                    xt = x[:]
                    xt[i] += sgn * si
                    clamp_reflect(xt)
                    ft = evaluate(xt)
                    if ft < f:
                        x, f = xt, ft
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # slight expand (but keep bounded)
                for i in range(dim):
                    step[i] = min(step[i] * 1.12, 0.25 * (span[i] if span[i] > 0 else 1.0))
            else:
                # shrink
                for i in range(dim):
                    step[i] *= 0.55
            it += 1
        return x, f

    # --- initialization: population for adaptive DE ---
    if dim <= 0:
        # degenerate
        return float(func([]))

    # population size: small but not tiny
    NP = max(10, min(40, 6 + 2 * int(math.sqrt(dim)) + dim // 3))

    pop = []
    fit = []

    # seed with midpoint
    mid = [lo[i] + 0.5 * span[i] for i in range(dim)]
    clamp_reflect(mid)
    best_x = mid[:]
    best = evaluate(best_x)
    add_elite(best_x, best)

    # fill population with Halton + opposition + random
    idx = 1
    while len(pop) < NP and time_left():
        x = halton_point(idx)
        fx = evaluate(x)
        pop.append(x)
        fit.append(fx)
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition
        xo = [lo[i] + hi[i] - x[i] for i in range(dim)]
        clamp_reflect(xo)
        fxo = evaluate(xo)
        if len(pop) < NP:
            pop.append(xo)
            fit.append(fxo)
        add_elite(xo, fxo)
        if fxo < best:
            best, best_x = fxo, xo[:]

        # occasional random
        if (idx & 3) == 0 and len(pop) < NP:
            xr = rand_point()
            fr = evaluate(xr)
            pop.append(xr)
            fit.append(fr)
            add_elite(xr, fr)
            if fr < best:
                best, best_x = fr, xr[:]
        idx += 1

    # if time was extremely short
    if not pop:
        return best

    # polish best once early
    if time_left():
        bx, bf = refine(best_x, best, budget_steps=10)
        if bf < best:
            best, best_x = bf, bx[:]
            add_elite(best_x, best)

    # --- Adaptive DE (JADE/SHADE-like) ---
    # memory of F and CR (rolling means of successful parameters)
    mu_F = 0.65
    mu_CR = 0.55

    p_best_rate = 0.2  # top p% are eligible as pbest
    archive = []       # external archive of replaced solutions
    arch_max = NP * 2

    # for stagnation and restarts
    no_improve = 0
    stagnate_after = 90 + 10 * dim

    def pick_indices(exclude_i, n=1):
        # pick n distinct indices != exclude_i
        res = []
        tries = 0
        while len(res) < n and tries < 50:
            j = random.randrange(NP)
            if j == exclude_i or j in res:
                tries += 1
                continue
            res.append(j)
        if len(res) < n:
            # fallback: allow repeats but avoid exclude_i
            while len(res) < n:
                j = random.randrange(NP)
                if j != exclude_i:
                    res.append(j)
        return res

    def sample_F():
        # Cauchy around mu_F (clipped to (0,1])
        for _ in range(8):
            val = mu_F + 0.1 * cauchy()
            if val > 0.0:
                return min(1.0, val)
        return min(1.0, max(0.01, mu_F))

    def sample_CR():
        # Normal around mu_CR (clipped to [0,1])
        val = mu_CR + 0.1 * randn()
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    # main loop
    while time_left():
        # build pbest candidate set size
        pN = max(2, int(math.ceil(p_best_rate * NP)))
        # get indices sorted by fitness (cheap: partial sort via args list)
        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        pbest_pool = order[:pN]

        succ_F = []
        succ_CR = []
        succ_df = []  # fitness improvements (for weighting)

        for i in range(NP):
            if not time_left():
                break

            xi = pop[i]
            fi = fit[i]

            # choose pbest
            pb = pop[random.choice(pbest_pool)]

            # choose r1 from population, r2 from pop+archive
            r1 = pick_indices(i, 1)[0]
            x1 = pop[r1]

            # create combined pool for r2
            use_arch = (archive and random.random() < 0.5)
            if use_arch:
                x2 = archive[random.randrange(len(archive))]
            else:
                r2 = pick_indices(i, 1)[0]
                x2 = pop[r2]

            F = sample_F()
            CR = sample_CR()

            # current-to-pbest/1: v = x + F*(pbest-x) + F*(x1-x2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (pb[d] - xi[d]) + F * (x1[d] - x2[d])

            # binomial crossover
            u = xi[:]  # trial
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            # occasional tiny isotropic noise to avoid collapse
            if random.random() < 0.08:
                # mutate ~sqrt(dim) dims
                m = max(1, int(math.sqrt(dim)))
                if m >= dim:
                    idxs = range(dim)
                else:
                    idxs = random.sample(range(dim), m)
                for d in idxs:
                    if span[d] > 0:
                        u[d] += (0.002 + 0.01 * random.random()) * span[d] * randn()

            clamp_reflect(u)
            fu = evaluate(u)

            # selection
            if fu <= fi:
                # add replaced to archive (diversity)
                archive.append(xi[:])
                if len(archive) > arch_max:
                    # random removal
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fit[i] = fu

                # record success for adapting mu_F/mu_CR
                succ_F.append(F)
                succ_CR.append(CR)
                succ_df.append(max(0.0, fi - fu))

                if fu < best:
                    best = fu
                    best_x = u[:]
                    add_elite(best_x, best)
                    no_improve = 0

                    # opportunistic polish when we find a new best
                    if time_left():
                        bx, bf = refine(best_x, best, budget_steps=8)
                        if bf < best:
                            best, best_x = bf, bx[:]
                            add_elite(best_x, best)
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # adapt mu_F, mu_CR using successful parameters
        if succ_F:
            # weighted Lehmer mean for F (common in SHADE); weights by improvement
            wsum = sum(succ_df) if any(succ_df) else float(len(succ_df))
            if wsum <= 0.0:
                w = [1.0 / len(succ_F)] * len(succ_F)
            else:
                w = [(df / wsum) for df in succ_df]

            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for k in range(len(succ_F)):
                fk = succ_F[k]
                wk = w[k]
                num += wk * fk * fk
                den += wk * fk
                cr_mean += wk * succ_CR[k]
            if den > 1e-12:
                lehmer_F = num / den
                mu_F = 0.85 * mu_F + 0.15 * lehmer_F
            mu_CR = 0.85 * mu_CR + 0.15 * cr_mean

            # keep in sensible ranges
            if mu_F < 0.05:
                mu_F = 0.05
            if mu_F > 0.95:
                mu_F = 0.95
            if mu_CR < 0.0:
                mu_CR = 0.0
            if mu_CR > 1.0:
                mu_CR = 1.0

        # restart / diversification on stagnation
        if no_improve > stagnate_after and time_left():
            no_improve = 0
            # reinitialize a fraction of worst individuals around elites + random
            order = list(range(NP))
            order.sort(key=lambda i: fit[i], reverse=True)
            k_re = max(2, NP // 3)

            for t in range(k_re):
                if not time_left():
                    break
                i = order[t]
                if elites and random.random() < 0.75:
                    _, xe = random.choice(elites)
                    xnew = xe[:]
                    # radius mixes local + global
                    rad = (0.10 + 0.40 * random.random())
                    for d in range(dim):
                        if span[d] > 0:
                            xnew[d] += rad * span[d] * randn()
                else:
                    xnew = rand_point()

                clamp_reflect(xnew)
                fnew = evaluate(xnew)
                pop[i] = xnew
                fit[i] = fnew
                add_elite(xnew, fnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # slightly reset adaptation to encourage exploration
            mu_F = 0.70
            mu_CR = 0.50
            archive.clear()

    return best
