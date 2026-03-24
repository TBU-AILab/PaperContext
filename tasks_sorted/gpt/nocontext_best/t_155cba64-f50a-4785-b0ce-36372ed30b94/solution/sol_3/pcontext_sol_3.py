import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Hybrid strategy:
      1) fast low-discrepancy init (Halton) + random
      2) CMA-ES-like diagonal adaptation (mean + per-dimension sigma)
      3) 1/5th success rule to stabilize global step-size
      4) opportunistic coordinate/pattern local search around incumbent
      5) restarts with partial re-initialization when stagnating

    func: callable(list[float]) -> float
    returns: best (minimum) fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --------------------- utilities ---------------------
    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def repair(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    # keep evaluation robust (avoid propagating NaN/inf)
    def eval_x(x):
        xx = repair(x)
        v = func(xx)
        try:
            v = float(v)
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        return v, xx

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Halton sequence
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def halton_index(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(max(1, dim))

    def halton_point(k):
        x = []
        for j in range(dim):
            u = halton_index(k, primes[j])
            lo, hi = bounds[j]
            x.append(lo + (hi - lo) * u)
        return x

    ranges = [(bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 1.0 for i in range(dim)]
    avg_range = (sum(ranges) / dim) if dim > 0 else 1.0

    # --------------------- initialization ---------------------
    best = float("inf")
    best_x = rand_point()

    # More aggressive init than previous code (still time-bounded)
    n_init = max(24, min(220, 18 * dim + 40))
    halton_skip = 11

    k = 1 + halton_skip
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        x = halton_point(k)
        k += 1
        v, xx = eval_x(x)
        if v < best:
            best, best_x = v, xx

    # A few pure random points to break structure
    for _ in range(max(8, dim)):
        if time.time() >= deadline:
            return best
        v, xx = eval_x(rand_point())
        if v < best:
            best, best_x = v, xx

    # --------------------- diagonal CMA-ES-ish state ---------------------
    # mean starts at best
    mean = best_x[:]

    # population
    lam = max(16, 8 * dim)
    mu = max(4, lam // 4)

    # recombination weights (log)
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # per-dimension sigma (diagonal)
    # start moderately large to explore; then adapt quickly via success + diag update
    sig = [max(1e-12, 0.25 * ranges[j]) for j in range(dim)]
    sig_min = [max(1e-15, 1e-12 * (ranges[j] if ranges[j] > 0 else 1.0)) for j in range(dim)]
    sig_max = [max(1e-12, 0.60 * ranges[j]) for j in range(dim)]

    # global step multiplier (stabilizes scale)
    gsig = 1.0

    # adaptation rates (lightweight)
    # diag learning rate: small but meaningful; higher in small dim
    c_diag = min(0.35, (0.10 + 2.0 / (dim + 10.0)))
    # global step-size learning via 1/5th rule
    cs = 0.20

    # stagnation/restart control
    last_improve_time = time.time()
    best_at_last_improve = best
    stall_seconds = max(0.20, 0.12 * max_time)  # restart if no progress for a while
    tiny_improve = 1e-12

    # --------------------- local search (pattern + coordinate) ---------------------
    def local_refine(x0, v0, eval_budget):
        x = x0[:]
        vbest = v0
        # per-dim step starts relative to current sig, but not too small
        steps = [max(1e-12, 0.5 * gsig * sig[j]) for j in range(dim)]
        shrink = 0.5
        grow = 1.35

        # small pattern-search loop
        for _ in range(eval_budget):
            if time.time() >= deadline:
                break
            j = random.randrange(dim)
            s = steps[j]
            if s <= sig_min[j]:
                continue

            improved = False
            base = x[j]
            for d in (1.0, -1.0):
                if time.time() >= deadline:
                    break
                trial = x[:]
                trial[j] = clip(base + d * s, bounds[j][0], bounds[j][1])
                vv, tt = eval_x(trial)
                if vv < vbest:
                    x, vbest = tt, vv
                    steps[j] = min(sig_max[j], s * grow)
                    improved = True
                    break
            if not improved:
                steps[j] = max(sig_min[j], s * shrink)

        return vbest, x

    # --------------------- main loop ---------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # sample offspring
        off = []
        successes = 0
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # Gaussian perturbation with diagonal sigmas
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            child = [0.0] * dim
            for j in range(dim):
                child[j] = mean[j] + (gsig * sig[j]) * z[j]
            v, xx = eval_x(child)
            if v < best:
                best, best_x = v, xx
                successes += 1
                last_improve_time = time.time()
                best_at_last_improve = best
            off.append((v, xx, z))

        off.sort(key=lambda t: t[0])

        # recombine mean using top mu
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            xi = off[i][1]
            for j in range(dim):
                new_mean[j] += w * xi[j]

        # diagonal sigma adaptation: compare top mu spread around mean
        # update sig based on weighted RMS of (xi - new_mean)
        # This acts like a very cheap diag covariance adaptation.
        for j in range(dim):
            s2 = 0.0
            for i in range(mu):
                w = weights[i]
                d = off[i][1][j] - new_mean[j]
                s2 += w * (d * d)
            target = math.sqrt(max(1e-30, s2)) + 1e-30
            # move sig toward target (with caps)
            sj = sig[j]
            sj = (1.0 - c_diag) * sj + c_diag * target
            if sj < sig_min[j]:
                sj = sig_min[j]
            if sj > sig_max[j]:
                sj = sig_max[j]
            sig[j] = sj

        mean = repair(new_mean)

        # global step-size via 1/5th rule (success among offspring vs incumbent best at gen start)
        # (use successes over lam as proxy)
        ps = successes / float(lam)
        # if ps > 0.2 -> increase, else decrease
        # smooth multiplicative update
        gsig *= math.exp(cs * (ps - 0.2))
        # keep gsig reasonable
        if gsig < 1e-6:
            gsig = 1e-6
        if gsig > 5.0:
            gsig = 5.0

        # occasional local refinement near incumbent (strongly helps on many benchmarks)
        if gen % 7 == 0 and time.time() < deadline:
            # small budget so it doesn't dominate
            v1, x1 = local_refine(best_x, best, eval_budget=18 + 3 * dim)
            if v1 < best:
                best, best_x = v1, x1
                last_improve_time = time.time()
                best_at_last_improve = best

        # restart if stalled in wall-clock time or if steps got too tiny
        stalled = (time.time() - last_improve_time) > stall_seconds
        too_small = all((gsig * sig[j]) <= (3.0 * sig_min[j]) for j in range(dim))
        if stalled or too_small:
            # partial restart: keep best, diversify mean with immigrants, reset gsig moderately
            mean = best_x[:]
            gsig = max(0.8, min(1.6, gsig * 1.25))

            # re-inflate sig a bit (but not full reset)
            for j in range(dim):
                sig[j] = max(sig[j], 0.18 * ranges[j])
                if sig[j] > sig_max[j]:
                    sig[j] = sig_max[j]

            # inject a few random points and move mean to best among them (elitist + diversify)
            for _ in range(max(6, dim // 2)):
                if time.time() >= deadline:
                    return best
                v, xx = eval_x(rand_point())
                if v + tiny_improve < best:
                    best, best_x = v, xx
                    mean = best_x[:]
                    last_improve_time = time.time()
                    best_at_last_improve = best

            # if still stalled, do a broader shake around best
            if stalled and time.time() < deadline:
                for _ in range(max(4, dim)):
                    if time.time() >= deadline:
                        return best
                    x = best_x[:]
                    for j in range(dim):
                        x[j] = clip(x[j] + random.gauss(0.0, 1.0) * (0.35 * ranges[j]),
                                    bounds[j][0], bounds[j][1])
                    v, xx = eval_x(x)
                    if v < best:
                        best, best_x = v, xx
                        mean = best_x[:]
                        last_improve_time = time.time()
                        best_at_last_improve = best

    return best
