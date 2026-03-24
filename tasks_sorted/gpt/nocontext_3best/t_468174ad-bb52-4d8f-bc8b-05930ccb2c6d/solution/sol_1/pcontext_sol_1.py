import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Key upgrades vs the given baseline:
      - Better global exploration: opposition-based sampling + LHS-style batches
      - Stronger exploitation: stochastic local search with per-dimension step sizes
      - Occasional quasi-Newton-ish behavior: quadratic 1D fit along coordinates
      - Simulated-annealing style acceptance to escape local minima
      - Budget-aware: adapts #initial samples and performs cheap moves most of the time

    Returns:
      best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0.0 else 0.0 for s in spans]

    # ---- helpers ----
    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        # Opposition point around the center of bounds: x' = low + high - x
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # Cheap approx normal(0,1) using sum of uniforms
    def randn():
        return (sum(random.random() for _ in range(12)) - 6.0)

    def time_left():
        return deadline - time.time()

    # ---- initialization: LHS batches + opposition-based selection ----
    # Keep initialization modest; scale with dim and time.
    # If time is tiny, do a few random points.
    init_n = max(12, 8 * dim)
    if max_time < 0.2:
        init_n = max(4, 2 * dim)
    elif max_time > 3.0:
        init_n = max(init_n, 16 + 10 * dim)

    best = float("inf")
    best_x = None

    # LHS-ish: permuted strata per dim, but in 2-3 batches to reduce correlation
    batches = 2 if init_n < 60 else 3
    per_batch = max(4, init_n // batches)

    for _ in range(batches):
        if time.time() >= deadline:
            return best

        m = per_batch
        strata = list(range(m))
        perms = []
        for _d in range(dim):
            p = strata[:]
            random.shuffle(p)
            perms.append(p)

        for k in range(m):
            if time.time() >= deadline:
                return best

            x = [0.0] * dim
            for i in range(dim):
                u = (perms[i][k] + random.random()) / m
                x[i] = lows[i] + u * spans[i]

            # Opposition-based pair: evaluate both x and opposite(x); keep best
            f1 = evaluate(x)
            if f1 < best:
                best, best_x = f1, x[:]

            xo = opposite(x)
            clip(xo)
            f2 = evaluate(xo)
            if f2 < best:
                best, best_x = f2, xo[:]

    if best_x is None:
        best_x = rand_uniform_vec()
        best = evaluate(best_x)

    # ---- local search state ----
    # Per-dimension step sizes
    step = [0.35 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_step = [1e-14 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
    max_step = [spans[i] if spans[i] > 0 else step[i] for i in range(dim)]

    # Simulated annealing temperature based on observed scale
    # Start with something proportional to |best|+1 then cool.
    T0 = max(1.0, abs(best) + 1.0)
    T = T0

    # Stagnation
    no_improve = 0
    # These values are conservative and robust
    shrink_every = 30
    restart_every = 220

    # Cache: last accepted point (may be worse than best), helps SA walk
    cur_x = best_x[:]
    cur_f = best

    # ---- coordinate quadratic step (1D parabolic interpolation) ----
    # Given f(-h), f(0), f(+h), estimate minimizer along that axis:
    # x* = x0 + h * (f(-h) - f(+h)) / (2*(f(-h) - 2f(0) + f(+h)))
    def coord_quadratic_move(i, base_x, base_f, h):
        if h <= min_step[i]:
            return None

        x_minus = base_x[:]
        x_plus = base_x[:]
        x_minus[i] = x_minus[i] - h
        x_plus[i] = x_plus[i] + h
        clip(x_minus)
        clip(x_plus)

        f_minus = evaluate(x_minus)
        if time.time() >= deadline:
            return None
        f_plus = evaluate(x_plus)
        if time.time() >= deadline:
            return None

        denom = (f_minus - 2.0 * base_f + f_plus)
        if denom == 0.0:
            return None

        # Proposed step along axis
        t = 0.5 * (f_minus - f_plus) / denom  # in units of h
        # Cap t to avoid wild extrapolation
        if t > 2.5:
            t = 2.5
        elif t < -2.5:
            t = -2.5

        cand = base_x[:]
        cand[i] = cand[i] + t * h
        clip(cand)
        f_cand = evaluate(cand)
        return cand, f_cand

    # ---- main loop ----
    while time.time() < deadline:
        # Cool temperature smoothly as time passes
        elapsed = time.time() - t0
        frac = elapsed / max_time if max_time > 0 else 1.0
        # Exponential-ish cooling (never hits 0)
        T = T0 * (0.02 + (1.0 - frac) ** 2)

        improved = False
        made_move = False

        r = random.random()

        # 1) Mostly: cheap coordinate / random-direction proposals
        if r < 0.60:
            i = random.randrange(dim)
            h = step[i]
            if h > min_step[i]:
                # Try +/- with slight bias to explore both
                sgn = -1.0 if random.random() < 0.5 else 1.0
                cand = cur_x[:]
                cand[i] = cand[i] + sgn * h
                clip(cand)
                f = evaluate(cand)

                made_move = True

                # SA acceptance
                if f <= cur_f:
                    cur_x, cur_f = cand, f
                else:
                    # accept with probability exp(-(f-cur_f)/T)
                    if T > 0 and random.random() < math.exp(-(f - cur_f) / T):
                        cur_x, cur_f = cand, f

                if cur_f < best:
                    best, best_x = cur_f, cur_x[:]
                    improved = True

        # 2) Sometimes: random direction step (multi-dim)
        elif r < 0.85:
            # random unit direction
            dir_vec = [randn() for _ in range(dim)]
            norm2 = sum(d * d for d in dir_vec)
            if norm2 > 0.0:
                invn = 1.0 / math.sqrt(norm2)
                dir_vec = [d * invn for d in dir_vec]

            avg_step = sum(step) / dim if dim else 1.0
            scale = avg_step * (0.15 + 0.85 * random.random())
            cand = [cur_x[i] + scale * dir_vec[i] for i in range(dim)]
            clip(cand)
            f = evaluate(cand)

            made_move = True
            if f <= cur_f or (T > 0 and random.random() < math.exp(-(f - cur_f) / T)):
                cur_x, cur_f = cand, f
                if cur_f < best:
                    best, best_x = cur_f, cur_x[:]
                    improved = True

        # 3) Rare: coordinate quadratic fit (costs 2-3 evals, but can jump to good minima)
        else:
            i = random.randrange(dim)
            h = step[i]
            if h > min_step[i] and time_left() > 0.001:
                res = coord_quadratic_move(i, cur_x, cur_f, h)
                if res is not None:
                    cand, f = res
                    made_move = True
                    if f <= cur_f or (T > 0 and random.random() < math.exp(-(f - cur_f) / T)):
                        cur_x, cur_f = cand, f
                        if cur_f < best:
                            best, best_x = cur_f, cur_x[:]
                            improved = True

        if not made_move:
            # If steps are tiny or something degenerate, do a cheap global sample
            cand = rand_uniform_vec()
            f = evaluate(cand)
            cur_x, cur_f = cand, f
            if f < best:
                best, best_x = f, cand[:]
                improved = True

        # ---- adapt step sizes ----
        if improved:
            no_improve = 0
            # Gentle expansion encourages exploration near success
            for i in range(dim):
                # Expand more if we're not near bounds (heuristic)
                slack = min(best_x[i] - lows[i], highs[i] - best_x[i]) * inv_spans[i] if spans[i] else 1.0
                grow = 1.06 if slack > 0.15 else 1.02
                step[i] = min(step[i] * grow, max_step[i] if max_step[i] > 0 else step[i])
        else:
            no_improve += 1
            if no_improve % shrink_every == 0:
                for i in range(dim):
                    step[i] = max(step[i] * 0.55, min_step[i])

            # ---- restart / kick ----
            if no_improve % restart_every == 0:
                if time.time() >= deadline:
                    return best

                # Mix: global restart, local restart around best, and "opposition kick"
                rr = random.random()
                if rr < 0.30:
                    cur_x = rand_uniform_vec()
                elif rr < 0.80:
                    cur_x = [best_x[i] + (2.0 * random.random() - 1.0) * step[i] for i in range(dim)]
                    clip(cur_x)
                else:
                    cur_x = opposite(best_x)
                    # small noise to avoid symmetry traps
                    for i in range(dim):
                        if spans[i] > 0:
                            cur_x[i] += (2.0 * random.random() - 1.0) * 0.02 * spans[i]
                    clip(cur_x)

                cur_f = evaluate(cur_x)
                if cur_f < best:
                    best, best_x = cur_f, cur_x[:]
                    no_improve = 0

    return best
