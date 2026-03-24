import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (no numpy, self-contained).

    Main improvements over your current best (#2: 43.09):
      - Adds *true* diagonal-CMA-ES style covariance learning from successful steps
        (rank-1 + rank-mu update on diagonal only; very cheap, very effective).
      - Keeps DE/current-to-best and coordinate pattern search as complementary operators.
      - Uses bound reflection (not just clamp) to reduce boundary-sticking and preserve step geometry.
      - Uses a small "ask/tell" batch: generate multiple candidates cheaply, evaluate few best-scoring
        ones (score = predicted improvement via z-norm step length heuristic).
      - Stronger, cleaner restart triggers (stagnation + sigma collapse), with near-best re-seeding.

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- helpers --------------------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect(v, lo, hi):
        # billiard reflection into [lo,hi]
        if lo == hi:
            return lo
        span = hi - lo
        y = (v - lo) % (2.0 * span)
        if y > span:
            y = 2.0 * span - y
        return lo + y

    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Box-Muller Gaussian
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Halton sequence for seeding (low discrepancy)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    def pick3(n, exclude):
        # pick 3 distinct indices != exclude
        while True:
            a = random.randrange(n)
            if a != exclude:
                break
        while True:
            b = random.randrange(n)
            if b != exclude and b != a:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != a and c != b:
                break
        return a, b, c

    # -------------------- setup --------------------
    start = time.time()
    max_time = float(max_time) if max_time is not None else 0.0
    deadline = start + max(0.0, max_time)

    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # population sizes
    pop_size = max(12, min(44, 6 * dim))
    elite_size = max(4, min(14, pop_size // 2))

    # diag-CMA state: mean m, diag std sigma_d, global sigma_g
    sigma_d = [0.20 * s for s in span]
    min_sigma_d = [1e-15 * s for s in span]
    max_sigma_d = [0.85 * s for s in span]
    sigma_g = 1.0

    # evolution paths (diag only)
    pc = [0.0] * dim
    ps = [0.0] * dim

    # learning rates (lightweight diagonal-CMA choices)
    # These are stable defaults; do not depend on numpy.
    mu = elite_size
    # recombination weights (positive, sum to 1)
    w = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
    w_sum = sum(w) if sum(w) > 0 else float(mu)
    w = [wi / w_sum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # CMA coefficients
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # DE parameters (kept as secondary operator)
    F_base = 0.55
    CR_base = 0.85

    # bookkeeping
    best = float("inf")
    best_x = None
    last_improve_t = start
    stagnation = max(0.35, 0.12 * max_time)

    # init mean: center of bounds
    mean_x = [(bounds[j][0] + bounds[j][1]) * 0.5 for j in range(dim)]

    # -------------------- initialization: Halton + random + opposition --------------------
    shift = [random.random() for _ in range(dim)]
    pop = []
    k = 1
    init_budget = max(pop_size, 14 * dim)

    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1

        fx = eval_f(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, list(x)
            last_improve_t = time.time()

        if time.time() >= deadline:
            break

        if random.random() < 0.55:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, list(xo)
                last_improve_t = time.time()
        i += 1

    if not pop:
        return eval_f(rand_vec())

    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_size]
    best, best_x = pop[0][0], list(pop[0][1])

    # initialize mean to weighted mean of elites
    elites0 = pop[:elite_size]
    mean_x = [0.0] * dim
    for i in range(len(elites0)):
        wi = w[i] if i < len(w) else 0.0
        xi = elites0[i][1]
        for j in range(dim):
            mean_x[j] += wi * xi[j]

    # -------------------- local coordinate pattern --------------------
    def coord_pattern(x, fx, step_scale):
        nonlocal best, best_x, last_improve_t
        m = 1 if dim == 1 else min(dim, 4)
        idxs = random.sample(range(dim), m)
        for j in idxs:
            lo, hi = bounds[j]
            step = step_scale * sigma_g * sigma_d[j]
            if step <= 0.0:
                continue

            xp = list(x)
            xp[j] = clamp(xp[j] + step, lo, hi)
            fp = eval_f(xp)
            if fp < fx:
                if fp < best:
                    best, best_x = fp, list(xp)
                    last_improve_t = time.time()
                return xp, fp

            xm = list(x)
            xm[j] = clamp(xm[j] - step, lo, hi)
            fm = eval_f(xm)
            if fm < fx:
                if fm < best:
                    best, best_x = fm, list(xm)
                    last_improve_t = time.time()
                return xm, fm
        return x, fx

    def sigma_too_small():
        cnt = 0
        for j in range(dim):
            if sigma_d[j] <= (5e-14 * span[j]):
                cnt += 1
        return cnt >= max(1, int(0.75 * dim))

    # -------------------- main loop --------------------
    # keep some recent successful normalized steps for diag covariance updates
    # (for cmu update we only need z^2 averaged; keep as list of z vectors)
    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0 if deadline <= start else (now - start) / (deadline - start)
        if tfrac < 0.0:
            tfrac = 0.0
        elif tfrac > 1.0:
            tfrac = 1.0

        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]
        if pop[0][0] < best:
            best, best_x = pop[0][0], list(pop[0][1])
            last_improve_t = time.time()

        elites = pop[:elite_size]

        # restart if stagnating or sigma collapse
        if (now - last_improve_t) > stagnation or sigma_too_small():
            # reset diag-CMA paths and widen steps moderately
            pc = [0.0] * dim
            ps = [0.0] * dim
            sigma_g = 1.0
            sigma_d = [0.28 * s for s in span]

            # keep elites, inject near-best and global
            new_pop = elites[:]
            inject = max(3, pop_size // 2)
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.70:
                    xr = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        xr.append(reflect(best_x[j] + gauss() * (0.45 * span[j]), lo, hi))
                else:
                    xr = rand_vec()
                fr = eval_f(xr)
                new_pop.append((fr, xr))
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

            while len(new_pop) < pop_size and time.time() < deadline:
                if random.random() < 0.5:
                    xr = rand_vec()
                else:
                    xr = halton_point(k, shift)
                    k += 1
                fr = eval_f(xr)
                new_pop.append((fr, xr))
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

            new_pop.sort(key=lambda t: t[0])
            pop = new_pop[:pop_size]

            # reset mean to best
            mean_x = list(pop[0][1])
            continue

        # --- Update mean from elites (recombination) ---
        old_mean = list(mean_x)
        mean_x = [0.0] * dim
        for i in range(mu):
            wi = w[i]
            xi = elites[i][1]
            for j in range(dim):
                mean_x[j] += wi * xi[j]

        # --- Diagonal-CMA updates (ps, pc, sigma_d, sigma_g) ---
        # y = (m_new - m_old) / (sigma_g * sigma_d)
        y = [0.0] * dim
        inv_sg = 1.0 / max(1e-300, sigma_g)
        for j in range(dim):
            denom = max(1e-300, sigma_d[j])
            y[j] = (mean_x[j] - old_mean[j]) * inv_sg / denom

        # update ps
        # ps = (1-cs) ps + sqrt(cs*(2-cs)*mueff) * y
        c = math.sqrt(cs * (2.0 - cs) * mueff)
        ps_norm2 = 0.0
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + c * y[j]
            ps_norm2 += ps[j] * ps[j]

        # expected norm of N(0,I)
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        # update sigma_g
        ps_norm = math.sqrt(ps_norm2)
        sigma_g *= math.exp((cs / damps) * ((ps_norm / max(1e-300, chi_n)) - 1.0))
        if sigma_g < 0.05:
            sigma_g = 0.05
        if sigma_g > 4.0:
            sigma_g = 4.0

        # update pc (use hsig)
        hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (1.0 + 5.0 * tfrac)))) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
        cc_fac = math.sqrt(cc * (2.0 - cc) * mueff)
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * cc_fac * y[j]

        # rank-mu update term for diagonal covariance: E[z^2] from elites steps
        # z_i = (x_i - old_mean) / (sigma_g * sigma_d)
        # Here we approximate using elites around old_mean (stable).
        z2_bar = [0.0] * dim
        for i in range(mu):
            xi = elites[i][1]
            wi = w[i]
            for j in range(dim):
                denom = max(1e-300, sigma_d[j])
                z = (xi[j] - old_mean[j]) * inv_sg / denom
                z2_bar[j] += wi * (z * z)

        # update sigma_d as sqrt(diag(C)) (diag-only covariance)
        for j in range(dim):
            # diag(C) update: (1 - c1 - cmu) * C + c1*pc^2 + cmu*z2_bar
            # use current sigma_d^2 as C
            Cjj = sigma_d[j] * sigma_d[j]
            Cjj = (1.0 - c1 - cmu) * Cjj + c1 * (pc[j] * pc[j]) + cmu * z2_bar[j]
            if Cjj < (min_sigma_d[j] * min_sigma_d[j]):
                Cjj = min_sigma_d[j] * min_sigma_d[j]
            if Cjj > (max_sigma_d[j] * max_sigma_d[j]):
                Cjj = max_sigma_d[j] * max_sigma_d[j]
            sigma_d[j] = math.sqrt(Cjj)

        # --- Candidate generation: small batch, evaluate a few ---
        # We mix: (1) CMA sample around mean, (2) DE/current-to-best, (3) coord-pattern near best
        pool = 10 if dim <= 8 else 14
        pool = int(pool + 8 * tfrac)  # slightly more late
        eval_top = 3 if tfrac < 0.7 else 4
        if eval_top > pool:
            eval_top = pool

        candidates = []

        # choose a current target for DE ops
        idx = random.randrange(len(pop))
        fx, x = pop[idx]

        for _ in range(pool):
            r = random.random()
            if r < (0.55 - 0.25 * tfrac):
                op = "CMA"
            elif r < (0.85 - 0.10 * tfrac):
                op = "DE"
            else:
                op = "COORD"

            if op == "CMA":
                # sample from diagonal Gaussian around mean_x
                cand = [0.0] * dim
                z_norm2 = 0.0
                for j in range(dim):
                    lo, hi = bounds[j]
                    z = gauss()
                    z_norm2 += z * z
                    step = sigma_g * sigma_d[j] * z
                    cand[j] = reflect(mean_x[j] + step, lo, hi)
                # heuristic score: prefer smaller predicted step late (local refinement),
                # larger step early (exploration)
                z_norm = math.sqrt(z_norm2)
                score = (z_norm * (0.30 + 0.70 * (1.0 - tfrac)))  # smaller is better late
                candidates.append((score, cand))

            elif op == "DE" and best_x is not None and len(pop) >= 4:
                a, b, c = pick3(len(pop), idx)
                xb = pop[b][1]
                xc = pop[c][1]
                F = F_base + 0.25 * (random.random() - 0.5)
                if F < 0.25:
                    F = 0.25
                if F > 0.95:
                    F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10:
                    CR = 0.10
                if CR > 0.98:
                    CR = 0.98

                v = [0.0] * dim
                for j in range(dim):
                    lo, hi = bounds[j]
                    vj = x[j] + F * (best_x[j] - x[j]) + F * (xb[j] - xc[j])
                    v[j] = reflect(vj, lo, hi)

                u = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    u[j] = v[j] if (random.random() < CR or j == jrand) else x[j]
                # score: prefer closer-to-best late
                d2 = 0.0
                for j in range(dim):
                    t = u[j] - best_x[j]
                    d2 += (t / span[j]) * (t / span[j])
                candidates.append((d2, u))

            else:
                # coordinate-based proposal around best (no evaluation inside)
                base = best_x if best_x is not None else x
                cand = list(base)
                m = 1 if dim == 1 else min(dim, 3)
                idxs = random.sample(range(dim), m)
                step_scale = 0.85 - 0.60 * tfrac
                for j in idxs:
                    lo, hi = bounds[j]
                    step = step_scale * sigma_g * sigma_d[j]
                    cand[j] = reflect(cand[j] + (step if random.random() < 0.5 else -step), lo, hi)
                # score: tiny preference for smaller move late
                score = m * (0.5 + 0.5 * tfrac)
                candidates.append((score, cand))

        candidates.sort(key=lambda t: t[0])

        # Evaluate top few, accept improvements into population
        improved_any = False
        for p in range(eval_top):
            if time.time() >= deadline:
                break
            cand = candidates[p][1]
            fc = eval_f(cand)

            # replace worst or target if better
            if fc <= fx:
                pop[idx] = (fc, cand)
                fx, x = fc, cand
                improved_any = True

            # also allow replacing worst for diversity
            if fc < pop[-1][0]:
                pop[-1] = (fc, cand)

            if fc < best:
                best, best_x = fc, list(cand)
                last_improve_t = time.time()
                improved_any = True

        # occasional explicit coord refinement on best late
        if tfrac > 0.60 and best_x is not None and random.random() < 0.22 and time.time() < deadline:
            xb = list(best_x)
            fb = float(best)
            xb2, fb2 = coord_pattern(xb, fb, 0.55)
            if fb2 < best:
                best, best_x = fb2, list(xb2)
                last_improve_t = time.time()
                pop.sort(key=lambda t: t[0])
                pop[-1] = (fb2, xb2)

        # mild random injection early
        if random.random() < (0.012 * (1.0 - tfrac) + 0.002):
            xr = rand_vec()
            fr = eval_f(xr)
            pop.sort(key=lambda t: t[0])
            ridx = random.randrange(max(elite_size, 1), len(pop))
            if fr < pop[ridx][0] or random.random() < 0.20:
                pop[ridx] = (fr, xr)
                if fr < best:
                    best, best_x = fr, list(xr)
                    last_improve_t = time.time()

        # If we didn't improve for a bit, slightly increase exploration (undo over-contraction)
        if (not improved_any) and random.random() < 0.10:
            sigma_g *= 1.03
            if sigma_g > 4.0:
                sigma_g = 4.0

    return best
