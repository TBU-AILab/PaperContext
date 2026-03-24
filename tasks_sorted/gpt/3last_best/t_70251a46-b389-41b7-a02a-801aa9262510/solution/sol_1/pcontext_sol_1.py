import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization (self-contained, no externals).

    Hybrid strategy:
      1) Low-discrepancy global exploration (Halton sequence) to cover bounds well.
      2) (1+λ)-ES style local search around current best with adaptive sigma
         using the classic 1/5 success rule.
      3) Occasional coordinate/pattern refinement and randomized restarts.

    Returns:
      best (float): best (minimum) fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return deadline - time.time()

    # --- Bounds / scaling ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Avoid zero spans
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    # --- Halton sequence for strong global coverage ---
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

    primes = first_primes(max(1, dim))

    def halton(index, base):
        # Radical inverse in given base
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        # k starts at 1
        x = [0.0] * dim
        for j in range(dim):
            u = halton(k, primes[j])
            x[j] = lows[j] + u * spans[j]
        return x

    # --- Init ---
    # Start from a couple of random points + a few Halton points quickly
    best_x = [lows[i] + random.random() * spans[i] for i in range(dim)]
    best = eval_f(best_x)

    if time_left() <= 0:
        return best

    # A small batch of Halton points (fast and good coverage)
    # scale with dim but keep modest
    H = max(16, min(200, 20 + 8 * dim))
    for k in range(1, H + 1):
        if time_left() <= 0:
            return best
        x = halton_point(k)
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    # --- Main optimization loop: adaptive ES + occasional coordinate refine ---
    # sigma in normalized space (relative to span)
    # start moderately large, then adapt
    sigma = 0.20
    sigma_min = 1e-12
    sigma_max = 0.60

    # λ offspring per generation
    lam = max(8, min(40, 4 + 2 * dim))

    # 1/5 success rule bookkeeping
    succ = 0
    trials = 0

    # coordinate refinement parameters
    coord_step = [0.10 * s for s in spans]
    coord_min = [1e-12 * (1.0 + s) for s in spans]

    # restart controls
    since_improve = 0
    halton_index = H + 1

    # helper: gaussian without external libs (Box-Muller)
    spare = [None]
    def gauss():
        if spare[0] is not None:
            z = spare[0]
            spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        spare[0] = z1
        return z0

    # Keep a tiny elite pool for occasional recombination
    elite_x = [best_x[:]]
    elite_f = [best]

    def push_elite(x, f):
        # keep up to 5 best unique-ish
        # (cheap uniqueness check via coarse rounding)
        key = tuple(int((x[i]-lows[i])/(spans[i]+1e-300)*1e6) for i in range(dim))
        for ex in elite_x:
            ekey = tuple(int((ex[i]-lows[i])/(spans[i]+1e-300)*1e6) for i in range(dim))
            if ekey == key:
                return
        elite_x.append(x[:])
        elite_f.append(f)
        # sort and trim
        idx = sorted(range(len(elite_f)), key=lambda i: elite_f[i])
        elite_x[:] = [elite_x[i] for i in idx[:5]]
        elite_f[:] = [elite_f[i] for i in idx[:5]]

    push_elite(best_x, best)

    while time_left() > 0:
        # --- Generate offspring around a parent (best or recombined elite) ---
        # Parent selection: mostly best, sometimes random elite, sometimes blend
        r = random.random()
        if len(elite_x) >= 2 and r < 0.15:
            a = random.randrange(len(elite_x))
            b = random.randrange(len(elite_x))
            if a == b:
                parent = elite_x[a][:]
            else:
                # blend
                w = random.random()
                pa = elite_x[a]
                pb = elite_x[b]
                parent = [w * pa[i] + (1.0 - w) * pb[i] for i in range(dim)]
        elif len(elite_x) >= 1 and r < 0.35:
            parent = elite_x[random.randrange(len(elite_x))][:]
        else:
            parent = best_x[:]

        # Evaluate λ candidates; keep best of the batch
        batch_best_x = None
        batch_best_f = float("inf")

        # sigma is relative; convert to per-dimension absolute scale
        # Use slightly heavier tails sometimes to escape basins
        heavy = (random.random() < 0.10)

        for _ in range(lam):
            if time_left() <= 0:
                return best

            x = parent[:]  # mutate copy
            if heavy:
                # Cauchy-like: g / max(eps, |h|) gives heavier tails
                for i in range(dim):
                    g = gauss()
                    h = gauss()
                    step = (g / max(1e-9, abs(h))) * (sigma * spans[i])
                    x[i] += step
            else:
                for i in range(dim):
                    x[i] += gauss() * (sigma * spans[i])

            clip_inplace(x)
            f = eval_f(x)

            if f < batch_best_f:
                batch_best_f = f
                batch_best_x = x

        # Selection
        trials += 1
        if batch_best_f < best:
            best = batch_best_f
            best_x = batch_best_x
            push_elite(best_x, best)
            succ += 1
            since_improve = 0
        else:
            since_improve += 1

        # 1/5 success rule: update every few generations
        if trials >= 10:
            rate = succ / float(trials)
            # If success rate > 1/5, increase sigma, else decrease
            if rate > 0.2:
                sigma *= 1.25
            else:
                sigma *= 0.82
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max
            succ = 0
            trials = 0

        # --- Occasional coordinate/pattern refinement around incumbent ---
        if since_improve % 7 == 0:
            # try a quick randomized coordinate search with current coord_step
            coords = list(range(dim))
            random.shuffle(coords)
            improved = False
            for j in coords:
                if time_left() <= 0:
                    return best

                sj = coord_step[j]
                if sj < coord_min[j]:
                    continue

                x1 = best_x[:]
                x1[j] += sj
                clip_inplace(x1)
                f1 = eval_f(x1)
                if f1 < best:
                    best, best_x = f1, x1
                    push_elite(best_x, best)
                    improved = True
                    continue

                x2 = best_x[:]
                x2[j] -= sj
                clip_inplace(x2)
                f2 = eval_f(x2)
                if f2 < best:
                    best, best_x = f2, x2
                    push_elite(best_x, best)
                    improved = True
                    continue

            if improved:
                # small expansion to exploit ridge directions
                for j in range(dim):
                    coord_step[j] *= 1.08
            else:
                # contract
                for j in range(dim):
                    coord_step[j] *= 0.70
                    if coord_step[j] < coord_min[j]:
                        coord_step[j] = coord_min[j]

        # --- Restart / inject global samples if stuck ---
        if since_improve >= 25:
            since_improve = 0

            # inject a couple of Halton points + one random
            inject = 3
            for _ in range(inject):
                if time_left() <= 0:
                    return best
                if random.random() < 0.67:
                    x = halton_point(halton_index)
                    halton_index += 1
                else:
                    x = [lows[i] + random.random() * spans[i] for i in range(dim)]
                f = eval_f(x)
                if f < best:
                    best, best_x = f, x
                    push_elite(best_x, best)

            # also re-widen sigma a bit to escape local basin
            sigma = min(sigma_max, max(sigma, 0.12))

    return best
