import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no numpy):

    Hybrid strategy:
      1) Broad seeding with Latin-hypercube-like stratified sampling (+ a few pure randoms)
      2) Multi-start local search:
           - (1+lambda)-ES for exploration (log-normal sigma adaptation)
           - interleaved coordinate pattern search for exploitation (very robust on box-bounded problems)
      3) Restarts with shrinking/expanding neighborhood based on progress

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # --- helpers
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

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

    # Latin-hypercube-ish sampling (cheap; no heavy bookkeeping)
    def lhs_points(n):
        # Create n strata per dimension, permuted independently
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        invn = 1.0 / n
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                # sample uniformly within stratum
                u = (perms[d][k] + random.random()) * invn
                x[d] = lows[d] + u * spans[d]
            pts.append(x)
        return pts

    # --- Phase 1: seeding (stratified + random)
    best = float("inf")
    best_x = None

    # Use a fraction of time for seeding, but ensure at least some attempts
    seed_time = 0.20 * max_time
    seed_until = min(deadline, t0 + seed_time)

    # choose n strata based on dim (avoid too many points in high-d)
    # target a modest batch size; additional randoms fill remaining time
    n_lhs = max(8, min(64, int(10 + 6 * math.log(dim + 1))))
    for x in lhs_points(n_lhs):
        if time.time() >= seed_until:
            break
        f = safe_eval(x)
        if f < best:
            best, best_x = f, x

    while time.time() < seed_until:
        x = rand_vec()
        f = safe_eval(x)
        if f < best:
            best, best_x = f, x

    if best_x is None:
        best_x = rand_vec()
        best = safe_eval(best_x)

    # Keep a small elite pool for restarts (unique-ish by fitness order)
    elite = [(best, list(best_x))]
    ELITE_MAX = 6

    def elite_add(f, x):
        nonlocal elite
        elite.append((f, list(x)))
        elite.sort(key=lambda t: t[0])
        # trim
        if len(elite) > ELITE_MAX:
            elite = elite[:ELITE_MAX]

    # --- Phase 2: hybrid ES + coordinate pattern search
    # ES params
    lam = max(6, 6 + int(3 * math.log(dim + 1)))  # a bit more offspring than before
    tau = 1.0 / math.sqrt(2.0 * dim)
    tau0 = 1.0 / math.sqrt(2.0 * math.sqrt(dim))
    min_sigma = 1e-15
    max_sigma = max(spans) if dim > 0 else 1.0

    # Start from best seed
    parent_x = list(best_x)
    parent_f = best

    # Initial step scales
    sigma = [max(1e-12, 0.25 * s) for s in spans]  # slightly broader start

    # Pattern search step (per-dimension)
    step = [max(1e-12, 0.05 * s) for s in spans]

    # Stagnation/restart logic
    no_improve = 0
    hard_stag = 25 + 8 * dim
    soft_stag = 8 + 2 * dim

    # how often to run pattern-search passes
    # (pattern search is evaluation-heavy but very effective at polishing)
    pattern_every = 3
    it = 0

    def pattern_search(x, fx):
        """
        Coordinate pattern search with adaptive step sizes.
        Tries +/- along each coordinate; keeps improvements.
        """
        nonlocal best, best_x
        improved_any = False

        # random order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        for idx in order:
            if time.time() >= deadline:
                break

            si = step[idx]
            if si <= min_sigma:
                continue

            base = x[idx]

            # try + step
            x[idx] = base + si
            if x[idx] > highs[idx]:
                x[idx] = highs[idx]
            f1 = safe_eval(x)

            # try - step
            x[idx] = base - si
            if x[idx] < lows[idx]:
                x[idx] = lows[idx]
            f2 = safe_eval(x)

            # restore base then apply best move if any
            x[idx] = base

            if f1 < fx or f2 < fx:
                if f1 <= f2:
                    x[idx] = base + si
                    if x[idx] > highs[idx]:
                        x[idx] = highs[idx]
                    fx = f1
                else:
                    x[idx] = base - si
                    if x[idx] < lows[idx]:
                        x[idx] = lows[idx]
                    fx = f2
                improved_any = True

                if fx < best:
                    best = fx
                    best_x = list(x)
                    elite_add(fx, x)
            else:
                # no gain: slightly shrink this coordinate step
                step[idx] = max(min_sigma, step[idx] * 0.7)

        if improved_any:
            # if we made progress, slightly expand steps to keep moving
            for i in range(dim):
                step[i] = min(max_sigma, step[i] * 1.15)
        return x, fx

    while time.time() < deadline:
        it += 1

        # Occasional exploitation pass
        if it % pattern_every == 0:
            parent_x, parent_f = pattern_search(parent_x, parent_f)

        if time.time() >= deadline:
            break

        # --- (1+lambda)-ES step
        best_off_x = parent_x
        best_off_f = parent_f
        best_off_sigma = sigma

        global_n = random.gauss(0.0, 1.0)

        # Slightly bias offspring toward best_x sometimes (helps on rugged landscapes)
        use_bias = (random.random() < 0.35)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # mutate sigma (log-normal)
            child_sigma = [0.0] * dim
            for i in range(dim):
                coord_n = random.gauss(0.0, 1.0)
                s = sigma[i] * math.exp(tau0 * global_n + tau * coord_n)
                if s < min_sigma:
                    s = min_sigma
                elif s > max_sigma:
                    s = max_sigma
                child_sigma[i] = s

            # mutate point
            child_x = [0.0] * dim
            if use_bias:
                # combine parent and global best, then perturb
                # (keeps search centered but still explores)
                for i in range(dim):
                    center = 0.6 * parent_x[i] + 0.4 * best_x[i]
                    child_x[i] = center + random.gauss(0.0, child_sigma[i])
            else:
                for i in range(dim):
                    child_x[i] = parent_x[i] + random.gauss(0.0, child_sigma[i])

            clip_inplace(child_x)
            child_f = safe_eval(child_x)

            if child_f < best_off_f:
                best_off_f = child_f
                best_off_x = child_x
                best_off_sigma = child_sigma

        # selection
        if best_off_f < parent_f:
            parent_x = list(best_off_x)
            parent_f = best_off_f
            sigma = list(best_off_sigma)
            no_improve = 0

            if parent_f < best:
                best = parent_f
                best_x = list(parent_x)
                elite_add(best, best_x)

            # when improving, allow a bit larger exploration
            for i in range(dim):
                sigma[i] = min(max_sigma, sigma[i] * 1.02)
        else:
            no_improve += 1
            # contract exploration and steps to focus
            for i in range(dim):
                sigma[i] = max(min_sigma, sigma[i] * 0.92)

        # --- restart policies
        if no_improve >= soft_stag and time.time() < deadline:
            # light restart: jump to a random elite member and jitter
            ef, ex = elite[random.randrange(len(elite))]
            parent_x = list(ex)
            # jitter scaled to current sigma/step
            for i in range(dim):
                jitter = (0.5 * sigma[i] + 0.5 * step[i])
                if jitter > 0:
                    parent_x[i] += random.gauss(0.0, jitter)
            clip_inplace(parent_x)
            parent_f = safe_eval(parent_x)
            no_improve = 0

            if parent_f < best:
                best = parent_f
                best_x = list(parent_x)
                elite_add(best, best_x)

        if no_improve >= hard_stag and time.time() < deadline:
            # hard restart: new random point; reset scales
            parent_x = rand_vec()
            parent_f = safe_eval(parent_x)
            sigma = [max(1e-12, 0.30 * s) for s in spans]
            step = [max(1e-12, 0.08 * s) for s in spans]
            no_improve = 0

            if parent_f < best:
                best = parent_f
                best_x = list(parent_x)
                elite_add(best, best_x)

    return best
