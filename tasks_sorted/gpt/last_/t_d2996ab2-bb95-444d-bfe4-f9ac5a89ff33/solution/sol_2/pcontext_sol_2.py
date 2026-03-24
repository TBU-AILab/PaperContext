import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (self-contained, no external libs).

    Improvements vs previous version:
      - Uses a lightweight surrogate idea: keeps a history of evaluated points and
        proposes new samples biased toward good regions (softmax over ranks) +
        Gaussian sampling around elites.
      - Adds CMA-ES-like diagonal adaptation (per-dimension sigma) driven by
        successful steps (not full covariance; still fast, robust, library-free).
      - Mixes global exploration (Latin-ish stratified sampling + random) with
        local refinement (pattern/coordinate + stochastic).
      - Stronger restart logic and better time slicing to keep progress under
        strict max_time.
    Returns:
      best (float) best objective value found.
    """

    # ---------- basic helpers ----------
    eps = 1e-12

    def clip(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def in_bounds(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(x):
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if not math.isfinite(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [max(hi[i] - lo[i], 0.0) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def center_vec():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    # Stratified sample per dimension (cheap "Latin-ish" without permutations storage)
    # For each dim i, pick one of m bins at random, then uniform within the bin.
    def stratified_vec(m):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
                continue
            b = random.randrange(m)
            a = lo[i] + (b / m) * span[i]
            c = lo[i] + ((b + 1) / m) * span[i]
            x[i] = random.uniform(a, c)
        return x

    def weighted_choice(weights):
        s = sum(weights)
        if s <= 0:
            return random.randrange(len(weights))
        r = random.random() * s
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                return i
        return len(weights) - 1

    # ---------- time control ----------
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    def time_left():
        return time.time() < deadline

    # ---------- state ----------
    best = float("inf")
    x_best = center_vec()
    best = safe_eval(x_best)

    # History: keep (f, x). Keep limited to control overhead.
    hist = [(best, x_best[:])]
    HIST_MAX = 4000

    # Elite set for local sampling
    ELITE_MAX = max(8, 3 * dim)

    # Diagonal "sigma" (step size per dimension)
    sigma0 = [0.25 * span[i] if span[i] > eps else 0.0 for i in range(dim)]

    # ---------- initialization ----------
    # Spend a chunk early on diversified sampling
    init_budget = max(20, 12 * dim)
    m_bins = max(4, int(math.sqrt(init_budget)) + 1)

    for k in range(init_budget):
        if not time_left():
            return best
        if k < init_budget // 2:
            x = stratified_vec(m_bins)
        else:
            x = rand_uniform_vec()
        fx = safe_eval(x)
        hist.append((fx, x))
        if fx < best:
            best, x_best = fx, x[:]

    # ---------- main loop ----------
    # We'll do repeated "generations" mixing:
    #  (A) elite-centered Gaussian proposals with diagonal sigma adaptation
    #  (B) coordinate/pattern local steps from a chosen elite
    #  (C) occasional global stratified/random restarts
    gen = 0
    no_global_improve = 0

    while time_left():
        gen += 1

        # Keep history size bounded
        if len(hist) > HIST_MAX:
            hist.sort(key=lambda t: t[0])
            hist = hist[:HIST_MAX]

        # Sort and define elites
        hist.sort(key=lambda t: t[0])
        elites = hist[:min(ELITE_MAX, len(hist))]
        if elites and elites[0][0] < best:
            best, x_best = elites[0][0], elites[0][1][:]

        # ---- choose a base point (biased toward better ranks) ----
        # Soft weights: w_r = exp(-r / tau)
        tau = max(2.0, 0.35 * len(elites))
        weights = [math.exp(-r / tau) for r in range(len(elites))]
        base_idx = weighted_choice(weights) if elites else 0
        x0 = elites[base_idx][1][:] if elites else rand_uniform_vec()
        f0 = elites[base_idx][0] if elites else safe_eval(x0)

        # Maintain a working sigma (start from sigma0, shrink slowly over time but reset on restarts)
        # Use mild annealing with generations, but not too aggressive.
        anneal = 1.0 / (1.0 + 0.015 * gen)
        sigma = [max(1e-15, sigma0[i] * anneal) for i in range(dim)]

        # If stagnating globally, inflate sigma to jump out.
        if no_global_improve >= 8:
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.35 * span[i])
        elif no_global_improve >= 4:
            for i in range(dim):
                sigma[i] = max(sigma[i], 0.18 * span[i])

        # ---- (A) elite-centered sampling batch ----
        # Create N proposals; accept all evaluations into history, update best.
        # Successful steps slightly increase sigma on changed coordinates; failures shrink.
        N = max(12, 6 * dim)
        successes = [0] * dim
        trials = [0] * dim

        for _ in range(N):
            if not time_left():
                return best

            # With some probability pick a different elite as mean to maintain multi-modality
            if elites and random.random() < 0.35:
                j = weighted_choice(weights)
                mean = elites[j][1]
            else:
                mean = x0

            x = mean[:]
            # Sample diagonal Gaussian; occasionally heavy-tail
            heavy = (random.random() < 0.08)
            for i in range(dim):
                if span[i] <= eps:
                    x[i] = lo[i]
                    continue
                step = random.gauss(0.0, sigma[i])
                if heavy:
                    # Cauchy-ish perturbation: tan(pi(u-0.5)) scaled
                    u = random.random()
                    step += 0.25 * span[i] * math.tan(math.pi * (u - 0.5))
                if abs(step) > 0.0:
                    trials[i] += 1
                x[i] = clip(x[i] + step, lo[i], hi[i])

            fx = safe_eval(x)
            hist.append((fx, x))

            if fx < best:
                best, x_best = fx, x[:]
                no_global_improve = 0
            # Track coordinate "success" relative to base f0 (cheap signal)
            if fx < f0:
                for i in range(dim):
                    # If coord differs meaningfully from mean, count success
                    if abs(x[i] - mean[i]) > 1e-18:
                        successes[i] += 1

        # Adapt sigma (diagonal)
        for i in range(dim):
            if span[i] <= eps:
                sigma0[i] = 0.0
                continue
            if trials[i] == 0:
                continue
            rate = successes[i] / max(1, trials[i])
            # target success around ~0.2
            if rate > 0.25:
                sigma0[i] = min(0.6 * span[i], sigma0[i] * 1.20)
            elif rate < 0.12:
                sigma0[i] = max(1e-12 * max(span[i], 1.0), sigma0[i] * 0.70)

        # ---- (B) fast coordinate/pattern refinement from current best elite ----
        # Short, greedy coordinate search with adaptive step (pattern search)
        x = elites[0][1][:] if elites else x_best[:]
        fx = elites[0][0] if elites else safe_eval(x)

        # Start local steps from current sigma (clipped)
        step = [min(0.25 * span[i], max(1e-12 * max(span[i], 1.0), 0.8 * sigma0[i]))
                for i in range(dim)]
        shrink = 0.55
        grow = 1.15

        local_iters = max(20, 8 * dim)
        for _ in range(local_iters):
            if not time_left():
                return best
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if span[i] <= eps or step[i] <= 0.0:
                    continue

                xi = x[i]
                # try +/- step
                x1 = x[:]; x1[i] = clip(xi + step[i], lo[i], hi[i])
                f1 = safe_eval(x1)
                x2 = x[:]; x2[i] = clip(xi - step[i], lo[i], hi[i])
                f2 = safe_eval(x2)

                # record
                hist.append((f1, x1))
                hist.append((f2, x2))

                if f1 < fx or f2 < fx:
                    if f1 <= f2:
                        x, fx = x1, f1
                    else:
                        x, fx = x2, f2
                    improved = True
                    step[i] = min(0.5 * span[i], step[i] * grow)
                    if fx < best:
                        best, x_best = fx, x[:]
                        no_global_improve = 0
                else:
                    step[i] = max(1e-12 * max(span[i], 1.0), step[i] * shrink)

            if not improved:
                break

        # ---- (C) global restart injection if stagnating ----
        no_global_improve += 1
        if no_global_improve >= 6 and time_left():
            # Add a few global points (stratified + random) to refresh hist
            for _ in range(4):
                if not time_left():
                    return best
                xg = stratified_vec(max(4, 2 * dim)) if random.random() < 0.6 else rand_uniform_vec()
                fg = safe_eval(xg)
                hist.append((fg, xg))
                if fg < best:
                    best, x_best = fg, xg[:]
                    no_global_improve = 0

    return best
