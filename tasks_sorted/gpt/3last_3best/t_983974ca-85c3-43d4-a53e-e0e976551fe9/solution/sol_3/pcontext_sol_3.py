import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    What’s improved vs your best (#2):
      - Switches the main global optimizer to a robust DE variant: JADE (adaptive F/CR + current-to-pbest)
      - Uses an external archive (JADE) to maintain diversity and escape local minima
      - Better initialization: mixture of LHS-like + corners + random
      - Adds lightweight "polish" (coordinate pattern search) only on the best solution occasionally
      - Time-aware: avoids heavy operations and respects the deadline tightly

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        try:
            v = func([])
            return float(v) if v is not None else float("inf")
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    max_span = max(spans) if spans else 1.0

    # ---------------- helpers ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def corner_vec(jitter_scale=0.02):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        for i in range(dim):
            x[i] += random.gauss(0.0, jitter_scale * spans[i])
        return clip_inplace(x)

    # simple LHS-like: n points, independent permutation per dimension
    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        invn = 1.0 / n
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][k] + random.random()) * invn
                x[d] = lows[d] + u * spans[d]
            pts.append(x)
        return pts

    # sample from Normal(0,1) clipped to [a,b] via rejection (rarely loops much)
    def randn_clipped(a, b):
        while True:
            z = random.gauss(0.0, 1.0)
            if a <= z <= b:
                return z

    # Cauchy(0,1) sample using tan(pi*(u-0.5))
    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------------- parameters ----------------
    # population size
    pop_size = int(12 + 6 * math.log(dim + 1))
    pop_size = max(18, min(80, pop_size))

    # JADE params (typical)
    p_best_rate = 0.15  # top p% considered for pbest
    c = 0.1             # learning rate for mu_F, mu_CR
    mu_F = 0.6
    mu_CR = 0.9

    # external archive size
    archive_max = pop_size

    # polish settings
    min_step = 1e-15

    def polish(best_x, best_f):
        """Very cheap coordinate pattern search around current best."""
        # try only a subset of coordinates (budget friendly)
        m = min(dim, max(6, int(0.30 * dim)))
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:m]

        # starting step proportional to span, but small
        # (keep it conservative; DE already does global search)
        for i in idxs:
            if time.time() >= deadline:
                break
            step = max(min_step, 0.01 * spans[i])

            base = best_x[i]

            # + step
            best_x[i] = base + step
            if best_x[i] > highs[i]:
                best_x[i] = highs[i]
            f1 = safe_eval(best_x)

            # - step
            best_x[i] = base - step
            if best_x[i] < lows[i]:
                best_x[i] = lows[i]
            f2 = safe_eval(best_x)

            # restore then apply best move
            best_x[i] = base

            if f1 < best_f or f2 < best_f:
                if f1 <= f2:
                    best_x[i] = base + step
                    if best_x[i] > highs[i]:
                        best_x[i] = highs[i]
                    best_f = f1
                else:
                    best_x[i] = base - step
                    if best_x[i] < lows[i]:
                        best_x[i] = lows[i]
                    best_f = f2

        return best_x, best_f

    # ---------------- initialization ----------------
    # dedicate a small portion of time to initialization
    init_until = min(deadline, t0 + 0.18 * max_time)

    pop = []
    fits = []

    # build candidate batch
    n_lhs = max(8, min(pop_size, int(10 + 6 * math.log(dim + 1))))
    candidates = []
    candidates.extend(lhs_points(n_lhs))
    # some corners (good for boundary optima)
    for _ in range(max(2, pop_size // 6)):
        candidates.append(corner_vec(0.02))
    # fill with random
    while len(candidates) < pop_size:
        candidates.append(rand_vec())

    # evaluate until we fill population or time runs out
    best = float("inf")
    best_x = None

    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(6, pop_size // 2):
            break
        f = safe_eval(x)
        pop.append(list(x))
        fits.append(f)
        if f < best:
            best, best_x = f, list(x)
        if len(pop) >= pop_size:
            break

    # if init ended early, fill the rest quickly
    while len(pop) < pop_size and time.time() < deadline:
        x = rand_vec()
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best, best_x = f, list(x)

    if best_x is None:
        best_x = rand_vec()
        best = safe_eval(best_x)

    archive = []  # stores vectors only

    # ---------------- main loop (JADE) ----------------
    gen = 0
    last_improve = time.time()

    while time.time() < deadline:
        gen += 1

        # sort indices by fitness (for p-best selection)
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda i: fits[i])

        # occasional polish
        if gen % 12 == 0 and time.time() < deadline:
            bx = list(best_x)
            bf = best
            bx, bf = polish(bx, bf)
            if bf < best:
                best, best_x = bf, list(bx)
                last_improve = time.time()

        # stagnation: inject diversity (replace a few worst)
        if time.time() - last_improve > max(0.30 * max_time, 0.9):
            worst_k = max(2, pop_size // 7)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted[-1 - t]
                x = corner_vec(0.06) if random.random() < 0.4 else rand_vec()
                f = safe_eval(x)
                pop[wi] = x
                fits[wi] = f
                if f < best:
                    best, best_x = f, list(x)
                    last_improve = time.time()

        SF = []   # successful F
        SCR = []  # successful CR
        dF = []   # improvements for Lehmer mean weighting

        # number of p-best candidates
        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))

        # iterate individuals
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            # sample CR ~ N(mu_CR, 0.1) clipped
            CR = mu_CR + 0.1 * randn_clipped(-2.0, 2.0)
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            # sample F from Cauchy(mu_F, 0.1) until F>0 then clamp to 1
            F = mu_F + 0.1 * cauchy()
            tries = 0
            while F <= 0.0 and tries < 8:
                F = mu_F + 0.1 * cauchy()
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            # choose pbest among top p_num
            pbest_idx = idx_sorted[random.randrange(p_num)]
            xpbest = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from (pop U archive), != i, != r1
            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                # pick either from archive or population with equal probability
                if random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = i
                    while r2 == i or r2 == r1:
                        r2 = random.randrange(pop_size)
                    x2 = pop[r2]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x2 = pop[r2]

            x1 = pop[r1]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d])

            # repair (clip) + small bound-jitter to avoid sticking on edges
            for d in range(dim):
                if v[d] < lows[d]:
                    v[d] = lows[d] + 0.02 * spans[d] * random.random()
                elif v[d] > highs[d]:
                    v[d] = highs[d] - 0.02 * spans[d] * random.random()

            # crossover (binomial)
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]

            fu = safe_eval(u)

            # selection
            if fu <= fi:
                # add replaced parent to archive
                archive.append(list(xi))
                if len(archive) > archive_max:
                    # random deletion
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                SF.append(F)
                SCR.append(CR)
                improvement = fi - fu
                if improvement < 0.0:
                    improvement = 0.0
                dF.append(improvement if improvement > 0.0 else 1e-12)

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve = time.time()

        # update mu_F, mu_CR using successes
        if SF:
            # mu_CR = (1-c)*mu_CR + c*mean(SCR)
            mu_CR = (1.0 - c) * mu_CR + c * (sum(SCR) / float(len(SCR)))

            # mu_F uses Lehmer mean: sum(w*F^2)/sum(w*F) with w proportional to improvement
            num = 0.0
            den = 0.0
            for F, w in zip(SF, dF):
                num += w * F * F
                den += w * F
            if den > 0.0:
                lehmer = num / den
                mu_F = (1.0 - c) * mu_F + c * lehmer

            # keep parameters in sane ranges
            if mu_F < 0.05:
                mu_F = 0.05
            elif mu_F > 0.95:
                mu_F = 0.95

            if mu_CR < 0.05:
                mu_CR = 0.05
            elif mu_CR > 0.98:
                mu_CR = 0.98

    return best
