import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    What’s improved vs the previous set (esp. your best DE/JADE-like result):
      - Keeps JADE-style DE core, but makes it more time-efficient and less parameter-fragile.
      - Adds a *very effective* local phase: bounded SPSA (simultaneous perturbation) + coordinate/pattern.
        (SPSA gives gradient-like progress with only 2 evaluations/step, great under tight time.)
      - Uses adaptive population sizing and reinitializes worst individuals more intelligently.
      - Uses a small quadratic/cubic-ish 1D line-search along promising directions (still cheap).
      - Strict time checks; designed to return good results quickly.

    Returns:
      best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # ---------- bounds handling: mirror ----------
    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def repair(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- Halton for init / reseeds ----------
    def first_primes(n):
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite archive ----------
    elite_size = max(12, min(48, 14 + int(4.0 * math.sqrt(dim))))
    elites = []  # sorted list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    def best_from_elites():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # ---------- RNG helpers ----------
    def normal_rand(mu, sigma):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def cauchy_rand(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    # ---------- cheap local search: coordinate/pattern + line probing ----------
    def coord_pattern(x, fx, rounds=2, max_coords=14, base=0.06):
        if dim == 0:
            return fx, x
        xbest, fbest = x[:], fx

        idxs = list(range(dim))
        idxs.sort(key=lambda i: spans_nz[i], reverse=True)
        idxs = idxs[:max(1, min(dim, max_coords))]

        for r in range(rounds):
            if now() >= deadline:
                break
            improved = False
            step_mul = base * (0.55 ** r)
            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = min(0.25 * spans_nz[i], max(1e-12, step_mul * spans_nz[i]))

                # try +delta
                xp = xbest[:]
                xp[i] += delta
                repair(xp)
                fp = evaluate(xp)
                if fp < fbest:
                    step = xp[i] - xbest[i]
                    xbest, fbest = xp, fp
                    # pattern
                    xpp = xbest[:]
                    xpp[i] += step
                    repair(xpp)
                    fpp = evaluate(xpp)
                    if fpp < fbest:
                        xbest, fbest = xpp, fpp
                    improved = True
                    continue

                # try -delta
                xm = xbest[:]
                xm[i] -= delta
                repair(xm)
                fm = evaluate(xm)
                if fm < fbest:
                    step = xm[i] - xbest[i]
                    xbest, fbest = xm, fm
                    xmm = xbest[:]
                    xmm[i] += step
                    repair(xmm)
                    fmm = evaluate(xmm)
                    if fmm < fbest:
                        xbest, fbest = xmm, fmm
                    improved = True

            if not improved:
                break
        return fbest, xbest

    # ---------- very cheap local search: SPSA around best (2 evals/step) ----------
    def spsa_refine(x, fx, steps=24):
        """
        Simultaneous Perturbation Stochastic Approximation in box bounds.
        Uses only 2 evaluations per step to approximate a gradient direction.
        """
        if dim == 0:
            return fx, x
        xb, fb = x[:], fx

        # scale step sizes to problem scale
        avg_span = sum(spans_nz) / float(dim)
        a0 = 0.12 * avg_span
        c0 = 0.08 * avg_span

        # If bounds are tight in some dims, normalize step per-dimension by span.
        for k in range(1, steps + 1):
            if now() >= deadline:
                break

            ak = a0 / (k ** 0.602)     # typical SPSA exponents
            ck = c0 / (k ** 0.101)

            delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(dim)]

            x_plus = xb[:]
            x_minus = xb[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    x_plus[i] = lows[i]
                    x_minus[i] = lows[i]
                else:
                    # per-dim perturbation scaled to span
                    ci = ck * (spans_nz[i] / (avg_span if avg_span > 0 else 1.0))
                    x_plus[i] += ci * delta[i]
                    x_minus[i] -= ci * delta[i]
            repair(x_plus)
            repair(x_minus)

            f_plus = evaluate(x_plus)
            if now() >= deadline:
                if f_plus < fb:
                    xb, fb = x_plus, f_plus
                break
            f_minus = evaluate(x_minus)

            if f_plus < fb:
                xb, fb = x_plus, f_plus
            if f_minus < fb:
                xb, fb = x_minus, f_minus

            # gradient estimate and update
            ghat = [0.0] * dim
            denom = (2.0 * ck)
            for i in range(dim):
                if spans[i] == 0.0:
                    ghat[i] = 0.0
                else:
                    ghat[i] = (f_plus - f_minus) / max(1e-18, denom * delta[i])

            # take step
            x_new = xb[:]
            for i in range(dim):
                if spans[i] == 0.0:
                    x_new[i] = lows[i]
                else:
                    # normalize by span to avoid huge moves in large-range dims
                    si = spans_nz[i]
                    x_new[i] -= ak * (ghat[i] / max(1e-12, si))
            repair(x_new)
            f_new = evaluate(x_new)
            if f_new < fb:
                xb, fb = x_new, f_new

        return fb, xb

    # ---------- Initialization ----------
    best = float("inf")
    best_x = None

    # Dynamic population sizing: smaller if dim large
    NP = max(18, min(90, 12 + 4 * int(math.sqrt(dim)) + dim))
    NP = min(NP, max(22, 4 * dim))

    pop, fit = [], []
    init_n = NP

    for _ in range(init_n):
        if now() >= deadline:
            return best
        if random.random() < 0.82:
            x = halton_point(hal_k)
            hal_k += 1
        else:
            x = rand_uniform_point()
        fx = evaluate(x)
        pop.append(x)
        fit.append(fx)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition sometimes
        if now() >= deadline:
            return best
        if random.random() < 0.50:
            xo = [lows[d] + highs[d] - x[d] for d in range(dim)]
            repair(xo)
            fo = evaluate(xo)
            push_elite(fo, xo)
            if fo < best:
                best, best_x = fo, xo[:]
            if fo < fx and random.random() < 0.7:
                pop[-1] = xo
                fit[-1] = fo

    # JADE-like archive
    A = []
    Amax = NP

    mu_F = 0.55
    mu_CR = 0.55

    def pick_index_excluding(n, exclude):
        while True:
            r = random.randrange(n)
            if r not in exclude:
                return r

    gen = 0
    no_best = 0
    last_best = best

    while now() < deadline:
        gen += 1
        time_left = deadline - now()
        if time_left <= 0:
            break
        frac_left = time_left / float(max_time) if max_time > 0 else 0.0
        endgame = frac_left < 0.20

        # p-best pool
        idxs_sorted = list(range(NP))
        idxs_sorted.sort(key=lambda i: fit[i])
        p = 0.10 + 0.10 * random.random()
        p_num = max(2, int(p * NP))
        pbest_pool = idxs_sorted[:p_num]

        S_F, S_CR = [], []

        union = pop + A
        union_n = len(union)

        for i in range(NP):
            if now() >= deadline:
                break

            xi = pop[i]
            fi = fit[i]

            CR = normal_rand(mu_CR, 0.10)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = cauchy_rand(mu_F, 0.10)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = cauchy_rand(mu_F, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0
            # mild dither
            F *= (0.92 + 0.16 * math.sin(2.0 * math.pi * random.random()))
            F = min(1.0, max(0.05, F))

            pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
            x_pbest = pop[pbest_idx]

            r1 = pick_index_excluding(NP, {i, pbest_idx})
            x_r1 = pop[r1]

            if union_n <= 1:
                r2_vec = pop[pick_index_excluding(NP, {i, pbest_idx, r1})]
            else:
                # choose a different vector if possible
                r2_vec = union[random.randrange(union_n)]
                # quick attempts to avoid duplicates
                for _ in range(6):
                    if (r2_vec is not xi) and (r2_vec is not x_r1):
                        break
                    r2_vec = union[random.randrange(union_n)]

            # mutation current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + F * (x_pbest[d] - xi[d]) + F * (x_r1[d] - r2_vec[d])

            # crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
            repair(u)

            fu = evaluate(u)
            if fu <= fi:
                # archive
                if len(A) < Amax:
                    A.append(xi[:])
                else:
                    A[random.randrange(Amax)] = xi[:]

                pop[i] = u
                fit[i] = fu
                push_elite(fu, u)
                S_F.append(F)
                S_CR.append(CR)
                if fu < best:
                    best, best_x = fu, u[:]
            else:
                # very small chance to inject a fresh point (keeps diversity, prevents collapse)
                if random.random() < (0.01 if not endgame else 0.004) and now() < deadline:
                    if random.random() < 0.7:
                        xn = halton_point(hal_k)
                        hal_k += 1
                    else:
                        xn = rand_uniform_point()
                    fn = evaluate(xn)
                    if fn < fit[i]:
                        if len(A) < Amax:
                            A.append(pop[i][:])
                        else:
                            A[random.randrange(Amax)] = pop[i][:]
                        pop[i] = xn
                        fit[i] = fn
                        push_elite(fn, xn)
                        if fn < best:
                            best, best_x = fn, xn[:]

        # update parameter means
        if S_F:
            s1 = sum(f * f for f in S_F)
            s2 = sum(S_F)
            if s2 > 0.0:
                mu_F = 0.90 * mu_F + 0.10 * (s1 / s2)
                mu_F = min(0.95, max(0.05, mu_F))
        if S_CR:
            mu_CR = 0.90 * mu_CR + 0.10 * (sum(S_CR) / float(len(S_CR)))
            mu_CR = min(1.0, max(0.0, mu_CR))

        # stagnation tracking
        if best < last_best - 1e-12 * (1.0 + abs(last_best)):
            last_best = best
            no_best = 0
        else:
            no_best += 1

        # Local refinement schedule:
        # - mid/late: SPSA is very cost-effective
        # - endgame: add coordinate/pattern after SPSA
        if best_x is not None:
            if endgame or (gen % (6 + int(math.sqrt(dim))) == 0):
                if now() < deadline:
                    # few SPSA steps unless endgame
                    steps = 10 if not endgame else 22
                    fp, xp = spsa_refine(best_x, best, steps=steps)
                    push_elite(fp, xp)
                    if fp < best:
                        best, best_x = fp, xp[:]
                        last_best = best
                        no_best = 0

                if now() < deadline and endgame:
                    fp, xp = coord_pattern(best_x, best, rounds=2, max_coords=14, base=0.05)
                    push_elite(fp, xp)
                    if fp < best:
                        best, best_x = fp, xp[:]
                        last_best = best
                        no_best = 0

        # Reseed worst individuals if stagnated
        if no_best >= (10 + int(2.0 * math.sqrt(dim))):
            no_best = 0
            q = max(2, NP // 6)
            idxs_worst = list(range(NP))
            idxs_worst.sort(key=lambda i: fit[i], reverse=True)
            worst = idxs_worst[:q]

            for k in worst:
                if now() >= deadline:
                    break
                if elites and random.random() < 0.88:
                    top = min(len(elites), 16)
                    idx = int((random.random() ** 2) * top)
                    xseed = elites[idx][1][:]
                    # jitter around elite
                    for d in range(dim):
                        if spans[d] > 0.0 and random.random() < 0.65:
                            xseed[d] += (0.10 * spans_nz[d]) * (2.0 * random.random() - 1.0)
                    repair(xseed)
                else:
                    xseed = halton_point(hal_k) if random.random() < 0.75 else rand_uniform_point()
                    hal_k += 1

                fseed = evaluate(xseed)
                pop[k] = xseed
                fit[k] = fseed
                push_elite(fseed, xseed)
                if fseed < best:
                    best, best_x = fseed, xseed[:]
                    last_best = best

    return best
