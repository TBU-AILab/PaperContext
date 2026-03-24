import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Self-contained optimizer (stdlib only) aimed at good results under tight time limits.

    Strategy:
      - Latin-hypercube-like initial sampling to seed good starting points
      - Multi-start local search with:
          * (1+1)-ES style success-based step-size adaptation (global scalar sigma)
          * occasional coordinate-only proposals (helps on axis-aligned structure)
          * heavy-tailed steps sometimes (Cauchy) to escape shallow basins
      - Regular restarts from:
          * best-so-far neighborhood
          * random global points
    Returns:
      best (float): minimum fitness found within max_time seconds
    """

    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # ---------- helpers ----------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def reflect_1d(x, lo, hi):
        # Reflect across bounds; handles large excursions robustly
        if lo == hi:
            return lo
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    def reflect_vec(vec):
        out = vec[:]
        for i in range(dim):
            out[i] = reflect_1d(out[i], lows[i], highs[i])
        return out

    def evaluate(vec):
        return float(func(vec))

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Latin-hypercube-like batch: for each dim, sample N stratified bins and permute
    def lhs_batch(n):
        if n <= 1:
            return [rand_uniform_vec()]
        per_dim = []
        for i in range(dim):
            lo, sp = lows[i], spans[i]
            if sp <= 0.0:
                per_dim.append([lo] * n)
                continue
            bins = [(k + random.random()) / n for k in range(n)]
            random.shuffle(bins)
            per_dim.append([lo + sp * u for u in bins])
        pts = []
        for k in range(n):
            pts.append([per_dim[i][k] for i in range(dim)])
        return pts

    # ---------- initial seeding ----------
    best = float("inf")
    best_x = None

    # Seed count scales mildly with dim but stays small for tiny max_time
    seed_n = max(8, min(60, 10 + 2 * dim))
    seeds = lhs_batch(seed_n)

    # Also include center point (often good for normalized problems)
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    seeds.append(center)

    # Evaluate seeds
    for x in seeds:
        if time.time() >= deadline:
            return best
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_uniform_vec()
        best = evaluate(best_x)

    # ---------- main loop: repeated local searches ----------
    # Global step-size (relative) and per-dimension scaling from spans
    # Start a bit larger than your previous code to explore faster, then adapt down.
    base_rel = 0.22
    min_rel = 1e-12
    max_rel = 0.75

    # Local-search controls
    # Success-based adaptation akin to 1/5th rule (but smoothed):
    # If we get enough successes in a window -> increase sigma, else decrease.
    window = max(20, 10 + 2 * dim)
    target_rate = 0.20
    adapt_strength = 0.35  # higher => faster response

    # Restart controls
    stall_limit = max(120, 40 + 8 * dim)  # iterations without best improvement
    hard_restart_limit = max(400, 120 + 20 * dim)

    # Proposal mixing
    p_coord = 0.30        # coordinate-only move probability
    p_heavy = 0.12        # heavy-tailed step probability
    p_best_restart = 0.70 # restart around best vs uniform

    # Multiple starts: keep trying until time is up
    # Start each local search either at best_x or a fresh point
    current = best_x[:]
    fcur = best
    rel_sigma = base_rel

    # Tracking
    it = 0
    succ = 0
    tried = 0
    no_best_improve = 0

    while True:
        if time.time() >= deadline:
            return best

        it += 1
        tried += 1

        # Build a candidate
        cand = current[:]

        # Choose step distribution
        # Gaussian usually, sometimes Cauchy (heavy tail) to jump out.
        heavy = (random.random() < p_heavy)

        if random.random() < p_coord:
            j = random.randrange(dim)
            scale = spans[j] * rel_sigma
            if scale > 0.0:
                if heavy:
                    # Cauchy step: tan(pi*(u-0.5))
                    u = random.random()
                    step = scale * math.tan(math.pi * (u - 0.5))
                else:
                    step = random.gauss(0.0, scale)
                cand[j] = cand[j] + step
        else:
            for i in range(dim):
                scale = spans[i] * rel_sigma
                if scale <= 0.0:
                    continue
                if heavy:
                    u = random.random()
                    step = scale * math.tan(math.pi * (u - 0.5))
                else:
                    step = random.gauss(0.0, scale)
                cand[i] = cand[i] + step

        cand = reflect_vec(cand)
        fcand = evaluate(cand)

        # (1+1) selection: accept if improves current
        if fcand <= fcur:
            current, fcur = cand, fcand
            succ += 1
        # Track global best
        if fcand < best:
            best, best_x = fcand, cand[:]
            no_best_improve = 0
        else:
            no_best_improve += 1

        # Step-size adaptation every 'window' tries
        if tried >= window:
            rate = succ / float(tried)
            # multiplicative update; centered so that rate>target increases sigma
            # clamp to keep sane
            factor = math.exp(adapt_strength * (rate - target_rate))
            rel_sigma *= factor
            if rel_sigma < min_rel:
                rel_sigma = min_rel
            elif rel_sigma > max_rel:
                rel_sigma = max_rel
            succ = 0
            tried = 0

        # Restart logic:
        # - if we haven't improved global best for a while, restart
        # - or if sigma becomes tiny and we're stuck
        if (no_best_improve >= stall_limit and rel_sigma <= 0.02) or no_best_improve >= hard_restart_limit:
            if best_x is not None and random.random() < p_best_restart:
                # Restart near best with moderate sigma
                current = best_x[:]
                # Jitter around best to diversify basin exploration
                jitter_rel = max(0.06, rel_sigma * 1.5)
                for i in range(dim):
                    sc = spans[i] * jitter_rel
                    if sc > 0.0:
                        current[i] = current[i] + random.gauss(0.0, sc)
                current = reflect_vec(current)
                fcur = evaluate(current)
                rel_sigma = max(0.08, base_rel * 0.75)
            else:
                # Global restart (uniform)
                current = rand_uniform_vec()
                fcur = evaluate(current)
                rel_sigma = base_rel

            if fcur < best:
                best, best_x = fcur, current[:]

            no_best_improve = 0
