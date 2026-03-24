import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no numpy).

    Key upgrades vs the provided best (DE + simple local search):
      - Uses JADE-style adaptive Differential Evolution (current-to-pbest + archive)
        for stronger, less-parameter-sensitive global search.
      - Opposition-based + Halton + random initialization for better early coverage.
      - Lightweight budget-aware local refinement (stochastic coordinate search) on elites.
      - Diversity maintenance (restart part of population if stagnating).
      - Strict time checks to respect max_time.

    Returns:
        best (float): best objective value found.
    """

    # -------------------------- helpers --------------------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(x))

    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, bases):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            s = spans[d]
            if s == 0.0:
                x[d] = lo
            else:
                u = halton_value(k, bases[d])
                x[d] = lo + u * s
        return x

    def random_point():
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            s = spans[d]
            x[d] = lo if s == 0.0 else (lo + random.random() * s)
        return x

    def opposite_point(x):
        # opposition around center of bounds: x' = lo + hi - x
        xo = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                xo[d] = lo
            else:
                xo[d] = clamp(lo + hi - x[d], lo, hi)
        return xo

    def rand_cauchy(loc, scale):
        # Cauchy(loc, scale): loc + scale * tan(pi*(u-0.5))
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def rand_normal(mu, sigma):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def ensure_in_bounds(u, x_parent):
        # fast "bounce-back": if out of bounds, move halfway toward parent
        for d in range(dim):
            lo, hi = bounds[d]
            if spans[d] == 0.0:
                u[d] = lo
            else:
                if u[d] < lo:
                    u[d] = 0.5 * (lo + x_parent[d])
                elif u[d] > hi:
                    u[d] = 0.5 * (hi + x_parent[d])
                # still might be out if parent out (shouldn't), so clamp
                u[d] = clamp(u[d], lo, hi)
        return u

    def local_refine(x, fx, deadline, intensity):
        # stochastic coordinate search, very cheap; intensity in [0,1] (later -> smaller steps)
        bestx = list(x)
        bestf = fx

        base = (0.18 * (1.0 - 0.80 * intensity) + 0.008)  # ~0.188 -> ~0.008
        step = [base * s for s in spans]
        shrink = 0.55
        expand = 1.20

        iters = max(10, min(70, 6 * dim))
        for _ in range(iters):
            if time.time() >= deadline:
                break
            improved = False
            coords = list(range(dim))
            random.shuffle(coords)
            for d in coords:
                if time.time() >= deadline:
                    break
                if spans[d] == 0.0:
                    continue
                sd = step[d]
                if sd <= 0.0:
                    continue
                lo, hi = bounds[d]

                # 2 tries, signed; occasional larger jump
                for _try in range(2):
                    direction = -1.0 if random.random() < 0.5 else 1.0
                    scale = 1.0 if random.random() < 0.80 else (0.30 + 1.70 * random.random())
                    cand = list(bestx)
                    cand[d] = clamp(cand[d] + direction * sd * scale, lo, hi)
                    if cand[d] == bestx[d]:
                        continue
                    f = eval_f(cand)
                    if f < bestf:
                        bestf = f
                        bestx = cand
                        step[d] = min(step[d] * expand, spans[d])
                        improved = True
                        break

            if not improved:
                for d in range(dim):
                    if spans[d] > 0.0:
                        step[d] *= shrink
                        min_sd = 1e-14 * (spans[d] if spans[d] > 0 else 1.0)
                        if step[d] < min_sd:
                            step[d] = min_sd
        return bestx, bestf

    # -------------------------- main --------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    spans = []
    for lo, hi in bounds:
        if hi < lo:
            raise ValueError("Each bound must be (low, high) with high >= low")
        spans.append(hi - lo)

    if all(s == 0.0 for s in spans):
        x = [bounds[d][0] for d in range(dim)]
        return eval_f(x)

    random.seed()

    # Population sizing: modest but effective for time-bounded optimization
    NP = max(14, min(60, 10 + 4 * dim))

    bases = first_primes(dim)

    # --- Initialization: Halton + random + opposition (pick better of each pair) ---
    pop = []
    fit = []
    best = float("inf")
    bestx = None

    k = 1
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i < (NP * 2) // 3:
            x = halton_point(k, bases)
            k += 1
        else:
            x = random_point()

        fx = eval_f(x)
        xo = opposite_point(x)
        fxo = eval_f(xo) if time.time() < deadline else float("inf")

        if fxo < fx:
            x, fx = xo, fxo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, bestx = fx, x

    # --- JADE parameters (adaptive) ---
    mu_F = 0.6
    mu_CR = 0.5
    c = 0.1  # learning rate
    p_min = 2.0 / NP

    archive = []  # stores replaced solutions

    # stagnation control
    last_best = best
    stagn_gens = 0

    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        frac = elapsed / max(1e-12, float(max_time))
        if frac > 1.0:
            break

        # pbest fraction increases slightly over time (more exploitation late)
        p = min(0.35, max(p_min, 0.10 + 0.20 * frac))
        pbest_count = max(2, int(round(p * NP)))

        # Precompute indices sorted by fitness for pbest selection
        idx_sorted = list(range(NP))
        idx_sorted.sort(key=lambda i: fit[i])

        S_F = []
        S_CR = []
        weights = []

        improved_any = False

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # Sample CR ~ N(mu_CR, 0.1), clipped to [0,1]
            CRi = rand_normal(mu_CR, 0.1)
            if CRi < 0.0: CRi = 0.0
            elif CRi > 1.0: CRi = 1.0

            # Sample F ~ Cauchy(mu_F, 0.1) until > 0, then clip to 1
            Fi = -1.0
            tries = 0
            while Fi <= 0.0:
                Fi = rand_cauchy(mu_F, 0.1)
                tries += 1
                if tries > 12:
                    Fi = max(1e-3, mu_F)
                    break
            if Fi > 1.0:
                Fi = 1.0

            # choose pbest from top pbest_count
            pbest_idx = idx_sorted[random.randrange(pbest_count)]
            xpbest = pop[pbest_idx]

            # choose r1 from pop, != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # choose r2 from union(pop + archive), != i and != r1 (by identity/index handling)
            union = pop + archive
            # avoid very large union sampling cost; but union size is small (archive bounded below)
            r2_obj = None
            while True:
                cand = union[random.randrange(len(union))]
                if cand is xi or cand is xr1:
                    continue
                r2_obj = cand
                break
            xr2 = r2_obj

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xpbest[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover (binomial)
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    u[d] = bounds[d][0]
                else:
                    if random.random() < CRi or d == jrand:
                        u[d] = v[d]
                    else:
                        u[d] = xi[d]

            u = ensure_in_bounds(u, xi)
            fu = eval_f(u)

            if fu <= fi:
                # successful
                improved_any = True

                # archive the replaced parent
                if len(archive) < NP:
                    archive.append(xi)
                else:
                    archive[random.randrange(NP)] = xi

                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best, bestx = fu, u

                # store parameter updates (weighted by improvement)
                df = abs(fi - fu)
                w = df if df > 0.0 else 1.0
                S_F.append(Fi)
                S_CR.append(CRi)
                weights.append(w)

        # adapt mu_F and mu_CR if there were successes
        if S_F:
            wsum = sum(weights)
            if wsum <= 0.0:
                wsum = 1.0
            # Lehmer mean for F, arithmetic mean for CR (JADE)
            num = 0.0
            den = 0.0
            cr_mean = 0.0
            for Fi, CRi, w in zip(S_F, S_CR, weights):
                num += w * Fi * Fi
                den += w * Fi
                cr_mean += w * CRi
            F_lehmer = (num / den) if den != 0.0 else mu_F
            CR_mean = cr_mean / wsum

            mu_F = (1.0 - c) * mu_F + c * F_lehmer
            mu_CR = (1.0 - c) * mu_CR + c * CR_mean

            # keep reasonable bounds
            if mu_F < 0.05: mu_F = 0.05
            if mu_F > 0.95: mu_F = 0.95
            if mu_CR < 0.05: mu_CR = 0.05
            if mu_CR > 0.95: mu_CR = 0.95

        # local refinement sometimes, more near the end; apply to current best and one other elite
        if bestx is not None and time.time() < deadline:
            p_ref = 0.06 + 0.32 * frac
            if random.random() < p_ref:
                rx, rf = local_refine(bestx, best, deadline, frac)
                if rf < best:
                    best, bestx = rf, rx

        # stagnation handling: if not improving, re-randomize a few worst late/early as needed
        if best < last_best - 1e-15:
            last_best = best
            stagn_gens = 0
        else:
            stagn_gens += 1

        if stagn_gens >= 10 and time.time() < deadline:
            # replace some worst individuals with new samples (Halton + random + opposition)
            replace_n = max(2, NP // 6)
            worst = list(range(NP))
            worst.sort(key=lambda i: fit[i], reverse=True)
            for t in range(replace_n):
                if time.time() >= deadline:
                    return best
                j = worst[t]
                # generate candidate
                if random.random() < 0.65:
                    xnew = halton_point(k, bases); k += 1
                else:
                    xnew = random_point()
                fnew = eval_f(xnew)
                xo = opposite_point(xnew)
                fxo = eval_f(xo) if time.time() < deadline else float("inf")
                if fxo < fnew:
                    xnew, fnew = xo, fxo

                pop[j] = xnew
                fit[j] = fnew
                if fnew < best:
                    best, bestx = fnew, xnew
            archive = []  # reset archive after restart-like event
            stagn_gens = 0

    return best
