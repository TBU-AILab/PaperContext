import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs).

    Strategy (hybrid, time-aware):
      1) Initial global exploration via LHS-like stratified sampling.
      2) Maintain an elite set; periodically intensify around elites.
      3) Local optimizer: stochastic coordinate search + occasional pair moves
         with a simple 1/5-success-rule style step adaptation.
      4) Restarts: mixture of global, best-elite jitter, and multi-elite jitter.

    Returns:
      best (float): minimum fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [bounds[i][0] for i in range(dim)]
    highs = [bounds[i][1] for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # handle degenerate spans
    spans = [s if s != 0 else 1.0 for s in spans]

    def now():
        return time.time()

    def clip_vec(x):
        return [highs[i] if x[i] > highs[i] else (lows[i] if x[i] < lows[i] else x[i]) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    def lhs_points(n):
        # LHS-like: permutation of bins per dimension
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = []
            for d in range(dim):
                u = (perms[d][i] + random.random()) / n
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    def tri_noise(scale):
        # centered triangular noise in [-scale, scale]
        return (random.random() - random.random()) * scale

    # --- elite management ---
    elite_size = max(3, min(12, 2 + int(math.sqrt(dim) * 2)))
    elites = []  # list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_size:
            elites = elites[:elite_size]

    # --- initialization (global) ---
    best = float("inf")
    best_x = None

    # Use time-aware initial sample count
    n0 = max(8, min(60, 10 + int(4 * math.sqrt(dim))))
    for x in lhs_points(n0):
        if now() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

    if best_x is None:
        x0 = rand_vec()
        best = eval_f(x0)
        return best

    # --- local search state ---
    x = best_x[:]
    fx = best

    # Initial step per dimension: 20% range (more aggressive than previous), capped.
    step = [0.20 * spans[i] for i in range(dim)]
    # Minimum step relative to range
    min_step = [1e-14 * spans[i] for i in range(dim)]
    # Maximum step cap
    max_step = [0.75 * spans[i] for i in range(dim)]

    # Success-rule style adaptation
    succ = 0
    att = 0
    adapt_every = 20  # adapt frequently

    # Stagnation / restart
    last_best_time = now()
    stall_seconds = max(0.03, 0.10 * max_time)

    # To reduce evaluation waste: prefer random dimension order each sweep
    dims = list(range(dim))

    # helper: propose around a center with per-dim scale
    def jitter_around(center, base_scale):
        y = center[:]
        for i in range(dim):
            s = base_scale[i]
            y[i] += tri_noise(s)
        return clip_vec(y)

    # helper: choose an elite (biased to best)
    def pick_elite():
        # rank-based selection: more weight to better elites
        if not elites:
            return best_x[:]
        # weights ~ 1/(rank+1)
        r = random.random()
        total = 0.0
        weights = []
        for k in range(len(elites)):
            w = 1.0 / (k + 1.0)
            weights.append(w)
            total += w
        r *= total
        acc = 0.0
        for k, w in enumerate(weights):
            acc += w
            if acc >= r:
                return elites[k][1][:]
        return elites[0][1][:]

    # --- main loop ---
    while now() < deadline:
        # periodic intensification around elites
        if random.random() < 0.08 and elites:
            center = pick_elite()
            # use smaller jitter than current step to refine
            base_scale = [max(min_step[i], 0.5 * step[i]) for i in range(dim)]
            y = jitter_around(center, base_scale)
            fy = eval_f(y)
            if fy < fx:
                x, fx = y, fy
                succ += 1
            att += 1
            if fy < best:
                best, best_x = fy, y[:]
                last_best_time = now()
            push_elite(fy, y)

        # coordinate/pair stochastic search
        random.shuffle(dims)
        improved_in_sweep = False

        # budget per sweep: try coordinate moves + occasional pair moves
        # (pair moves help with interaction between variables)
        for idx, d in enumerate(dims):
            if now() >= deadline:
                return best

            # choose move type
            if dim >= 2 and random.random() < 0.20:
                # pair move
                d2 = dims[(idx + 1) % dim]
                s1 = step[d]
                s2 = step[d2]
                y = x[:]
                y[d] += tri_noise(s1)
                y[d2] += tri_noise(s2)
            else:
                # coordinate move: try signed step or random within step
                s = step[d]
                y = x[:]
                if random.random() < 0.5:
                    y[d] += s
                else:
                    y[d] -= s
                # add small stochasticity
                y[d] += tri_noise(0.25 * s)

            y = clip_vec(y)
            fy = eval_f(y)
            att += 1

            if fy < fx:
                x, fx = y, fy
                succ += 1
                improved_in_sweep = True

                if fy < best:
                    best, best_x = fy, y[:]
                    last_best_time = now()
                push_elite(fy, y)

        # adapt step sizes every so often using success ratio
        if att >= adapt_every:
            rate = succ / float(att) if att else 0.0
            # 1/5 success rule heuristic: target ~0.2
            if rate > 0.22:
                # increase steps
                for i in range(dim):
                    step[i] = min(max_step[i], step[i] * 1.25)
            elif rate < 0.18:
                # decrease steps
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.75)
            succ = 0
            att = 0

        # additional contraction if no improvement in a sweep
        if not improved_in_sweep:
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.85)

        # restart on stall or if steps got too small
        tiny = True
        for i in range(dim):
            if step[i] > 50.0 * min_step[i]:
                tiny = False
                break

        if tiny or (now() - last_best_time) > stall_seconds:
            r = random.random()
            if r < 0.30:
                # global restart
                x = rand_vec()
            elif r < 0.75 and elites:
                # restart around best elite with moderate jitter
                center = elites[0][1]
                base_scale = [max(min_step[i], 0.30 * spans[i]) for i in range(dim)]
                x = jitter_around(center, base_scale)
            else:
                # combine two elites (interpolation) + jitter
                if len(elites) >= 2:
                    a = elites[0][1]
                    b = pick_elite()
                    t = random.random()
                    x = [a[i] * t + b[i] * (1.0 - t) for i in range(dim)]
                    x = clip_vec(x)
                else:
                    x = best_x[:]
                base_scale = [max(min_step[i], 0.15 * spans[i]) for i in range(dim)]
                x = jitter_around(x, base_scale)

            fx = eval_f(x)
            push_elite(fx, x)

            if fx < best:
                best, best_x = fx, x[:]
                last_best_time = now()

            # reset steps to be exploratory again, but not too large
            for i in range(dim):
                step[i] = min(max_step[i], max(0.10 * spans[i], step[i]))

    return best
