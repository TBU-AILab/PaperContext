import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained; stdlib only).

    Core ideas (robust + fast under unknown objectives):
      - Low-discrepancy-ish seeding (stratified + opposition) to get a strong initial best
      - Two-track search:
          (A) CMA-ES-like diagonal evolution strategy with elite recombination (global guidance)
          (B) Trust-region coordinate/local random-direction refinement around current best
      - Heavy-tailed mutations + periodic restarts to escape local minima
      - Strict time-guarding: never starts expensive blocks when time is too low

    Returns:
      best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    if dim <= 0:
        return float(func([]))

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] == 0.0:
            spans[i] = 1.0  # avoid divide-by-zero / degenerate scaling

    # ---------- helpers ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def evalf(x):
        return float(func(x))

    def randn():
        # approx N(0,1): sum of 12 uniforms - 6
        return sum(random.random() for _ in range(12)) - 6.0

    def cauchy():
        # Standard Cauchy using tan(pi*(u-0.5)), clamped
        u = random.random()
        if u <= 1e-15:
            u = 1e-15
        elif u >= 1.0 - 1e-15:
            u = 1.0 - 1e-15
        v = math.tan(math.pi * (u - 0.5))
        if v > 80.0:
            v = 80.0
        elif v < -80.0:
            v = -80.0
        return v

    def rand_uniform():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def time_left():
        return deadline - time.time()

    # ---------- initialization (stratified + opposition) ----------
    # Keep this moderate; leave time for exploitation.
    init_n = max(16, 8 * dim)
    # If plenty of time, seed more; if little time, seed less.
    if max_time < 0.25:
        init_n = max(6, 2 * dim)
    elif max_time > 2.0:
        init_n = max(init_n, 24 + 10 * dim)

    best = float("inf")
    best_x = None

    # Always evaluate center first (often decent)
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    clip_inplace(center)
    best = evalf(center)
    best_x = center[:]

    # Stratified (LHS-ish): independent permutation per dim
    m = init_n
    strata = list(range(m))
    perms = []
    for _ in range(dim):
        p = strata[:]
        random.shuffle(p)
        perms.append(p)

    for k in range(m):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            u = (perms[i][k] + random.random()) / m
            x[i] = lows[i] + u * (highs[i] - lows[i])
        f = evalf(x)
        if f < best:
            best, best_x = f, x[:]

        xo = opposite(x)
        clip_inplace(xo)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    # ---------- ES distribution state (diagonal) ----------
    mean = best_x[:]
    sigma = [0.30 * (highs[i] - lows[i]) for i in range(dim)]
    for i in range(dim):
        if sigma[i] <= 0.0:
            sigma[i] = 1.0

    sig_floor = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]
    sig_ceil = [0.80 * (highs[i] - lows[i]) if highs[i] != lows[i] else 1.0 for i in range(dim)]

    # ES hyperparameters (tuned for robustness)
    alpha_m = 0.35
    alpha_s = 0.25

    # ---------- local trust region around best ----------
    tr = [0.18 * (highs[i] - lows[i]) for i in range(dim)]
    tr_min = [1e-14 * (highs[i] - lows[i] if highs[i] != lows[i] else 1.0) for i in range(dim)]

    # ---------- small elite archive (memory) ----------
    archive = []  # list of (f, x)
    archive_cap = max(12, 3 * dim)

    def add_archive(f, x):
        archive.append((f, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > archive_cap:
            del archive[archive_cap:]

    add_archive(best, best_x)

    # ---------- main loop ----------
    no_improve = 0
    it = 0

    while True:
        if time.time() >= deadline:
            return best

        it += 1
        tl = time_left()
        if tl <= 0.0:
            return best

        # Decide evaluation budget per generation (time/dim aware)
        # Keep small when time is low; moderate otherwise.
        base_batch = 8 + 4 * dim
        if tl < 0.08:
            lam = max(6, 2 * dim)
        elif tl > 1.0 and dim <= 16:
            lam = int(base_batch * 1.6)
        else:
            lam = base_batch

        # ---------- (A) ES generation ----------
        pop = []
        half = (lam + 1) // 2

        # Occasionally pull mean slightly toward best (prevents drift)
        if random.random() < 0.20:
            mean = [0.80 * mean[i] + 0.20 * best_x[i] for i in range(dim)]

        for _ in range(half):
            if time.time() >= deadline:
                return best

            # heavy-tailed with small probability (escape)
            heavy = (random.random() < 0.12)
            z = [randn() for _ in range(dim)]

            x1 = [0.0] * dim
            x2 = [0.0] * dim
            for i in range(dim):
                step = sigma[i] * (cauchy() if heavy else z[i])
                x1[i] = mean[i] + step
                x2[i] = mean[i] - step  # antithetic
            clip_inplace(x1)
            clip_inplace(x2)

            f1 = evalf(x1)
            pop.append((f1, x1))
            if f1 < best:
                best, best_x = f1, x1[:]
                add_archive(best, best_x)
                no_improve = 0

            if len(pop) < lam:
                f2 = evalf(x2)
                pop.append((f2, x2))
                if f2 < best:
                    best, best_x = f2, x2[:]
                    add_archive(best, best_x)
                    no_improve = 0

        pop.sort(key=lambda t: t[0])

        # select elites; mix in archive for stability
        mu = max(4, lam // 5)
        elites = pop[:mu]
        if archive:
            take = min(len(archive), max(1, mu // 3))
            elites = elites[:max(1, mu - take)] + archive[:take]
            elites.sort(key=lambda t: t[0])
            elites = elites[:mu]

        # rank weights (log)
        weights = []
        wsum = 0.0
        for r in range(mu):
            w = math.log(mu + 1.0) - math.log(r + 1.0)
            weights.append(w)
            wsum += w
        if wsum <= 0.0:
            wsum = 1.0

        new_mean = [0.0] * dim
        for w, (f, x) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                new_mean[i] += ww * x[i]

        new_var = [0.0] * dim
        for w, (f, x) in zip(weights, elites):
            ww = w / wsum
            for i in range(dim):
                d = x[i] - new_mean[i]
                new_var[i] += ww * d * d

        new_sigma = [0.0] * dim
        for i in range(dim):
            s = math.sqrt(max(new_var[i], sig_floor[i] * sig_floor[i]))
            # clamp
            if s < sig_floor[i]:
                s = sig_floor[i]
            elif s > sig_ceil[i]:
                s = sig_ceil[i]
            new_sigma[i] = s

        # smooth updates
        for i in range(dim):
            mean[i] = (1.0 - alpha_m) * mean[i] + alpha_m * new_mean[i]
            sigma[i] = (1.0 - alpha_s) * sigma[i] + alpha_s * new_sigma[i]
            if sigma[i] < sig_floor[i]:
                sigma[i] = sig_floor[i]
            elif sigma[i] > sig_ceil[i]:
                sigma[i] = sig_ceil[i]

        # ---------- (B) local trust-region refinement around best ----------
        # Do more when stagnating; cheap coordinate + occasional random direction.
        local_rounds = 2 if no_improve < 12 else 6
        for _ in range(local_rounds):
            if time.time() >= deadline:
                return best

            if random.random() < 0.78:
                # coordinate try
                i = random.randrange(dim)
                h = tr[i]
                if h <= tr_min[i]:
                    continue

                base = best_x[:]
                # try +/-h (random order)
                if random.random() < 0.5:
                    dirs = (1.0, -1.0)
                else:
                    dirs = (-1.0, 1.0)

                improved = False
                for sgn in dirs:
                    cand = base[:]
                    cand[i] += sgn * h
                    clip_inplace(cand)
                    f = evalf(cand)
                    if f < best:
                        best, best_x = f, cand[:]
                        add_archive(best, best_x)
                        no_improve = 0
                        tr[i] = min(tr[i] * 1.20, highs[i] - lows[i])
                        improved = True
                        break
                if not improved:
                    tr[i] = max(tr[i] * 0.65, tr_min[i])
            else:
                # random-direction small step within TR scale
                avg_tr = sum(tr) / dim
                # normalized random direction
                d = [randn() for _ in range(dim)]
                n2 = sum(v * v for v in d)
                if n2 > 0.0:
                    inv = 1.0 / math.sqrt(n2)
                    d = [v * inv for v in d]
                scale = avg_tr * (0.10 + 0.50 * random.random())
                cand = [best_x[i] + scale * d[i] for i in range(dim)]
                clip_inplace(cand)
                f = evalf(cand)
                if f < best:
                    best, best_x = f, cand[:]
                    add_archive(best, best_x)
                    no_improve = 0

        # ---------- stagnation logic / restarts ----------
        no_improve += 1

        # periodic sigma inflation (escape)
        if no_improve % 18 == 0:
            for i in range(dim):
                sigma[i] = min(sig_ceil[i], sigma[i] * 1.35)

        # partial restart: move mean to best (or random) and reset TR a bit
        if no_improve % 55 == 0:
            if random.random() < 0.75:
                mean = best_x[:]
            else:
                mean = rand_uniform()
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.22 * (highs[i] - lows[i]))
                tr[i] = max(tr[i], 0.12 * (highs[i] - lows[i]))
            no_improve = 0
