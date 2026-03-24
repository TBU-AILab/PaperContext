import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided hybrid:
      - Adds a true *trust-region* local search around the incumbent with adaptive radius
        (very effective once near a basin).
      - Uses *antithetic sampling* (paired +/- noise) to reduce variance and improve progress.
      - Uses *normalized coordinates* internally for stable behavior across heterogeneous bounds.
      - Adds a lightweight *surrogate-like cache* (hash grid) to avoid re-evaluating near-duplicates.
      - Keeps global exploration via low-discrepancy init + evolution-style sampling + restarts.

    func: callable(list[float]) -> float
    returns: best (minimum) fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        # degenerate: just evaluate empty vector if allowed
        try:
            v = float(func([]))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    # ---------- basic helpers ----------
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    # avoid divide-by-zero
    span = [s if s > 0.0 else 1.0 for s in span]

    def clip(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    # normalized <-> real
    def to_u(x):  # x in real -> u in [0,1]
        return [(x[i] - lo[i]) / span[i] for i in range(dim)]

    def to_x(u):  # u in [0,1] -> x in real
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    def repair_u(u):
        return [0.0 if u[i] < 0.0 else (1.0 if u[i] > 1.0 else u[i]) for i in range(dim)]

    # robust eval + small cache to avoid near-duplicates
    # grid quantization in normalized space
    grid = {}
    # choose granularity: finer in small dims, coarser in large dims
    q = 1e-4 if dim <= 6 else (3e-4 if dim <= 15 else 8e-4)

    def key_u(u):
        return tuple(int(uu / q + 0.5) for uu in u)

    def eval_u(u):
        uu = repair_u(u)
        k = key_u(uu)
        if k in grid:
            return grid[k], uu
        x = to_x(uu)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        grid[k] = v
        return v, uu

    def rand_u():
        return [random.random() for _ in range(dim)]

    # ---------- Halton init ----------
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

    primes = first_primes(dim)

    def halton_u(k):
        return [halton_index(k, primes[j]) for j in range(dim)]

    # ---------- initialization ----------
    best = float("inf")
    best_u = rand_u()

    # slightly larger init but still time-safe
    n_init = max(32, min(260, 22 * dim + 48))
    k = 13  # skip
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(halton_u(k))
        k += 1
        if v < best:
            best, best_u = v, uu

    # a few random probes to break LDS structure
    for _ in range(max(10, dim)):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(rand_u())
        if v < best:
            best, best_u = v, uu

    # ---------- evolution-style sampler in normalized space ----------
    # mean around best
    mean = best_u[:]

    # population sizes
    lam = max(18, 10 * dim)
    if lam % 2 == 1:
        lam += 1  # even for antithetic pairs
    mu = max(6, lam // 4)

    # log weights
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    s = sum(weights)
    weights = [w / s for w in weights]

    # diagonal std in normalized coords
    # start moderate; adapt via cheap diag update + success rule
    sig = [0.25] * dim
    sig_min = [1e-6] * dim
    sig_max = [0.50] * dim
    gsig = 1.0

    c_diag = min(0.30, 0.10 + 2.0 / (dim + 10.0))
    cs = 0.22

    # ---------- trust-region local search around incumbent ----------
    # radius in normalized space
    tr = 0.20 if dim <= 10 else 0.12
    tr_min = 1e-6
    tr_max = 0.45

    def trust_refine(u0, v0, budget):
        """Adaptive trust-region search using coordinate + random directions in normalized space."""
        u = u0[:]
        vbest = v0
        radius = tr

        # mixture: coordinate polling + random directions
        for it in range(budget):
            if time.time() >= deadline:
                break

            improved = False

            # half of the time: coordinate poll
            if (it & 1) == 0:
                j = random.randrange(dim)
                step = radius
                for sgn in (1.0, -1.0):
                    if time.time() >= deadline:
                        break
                    uu = u[:]
                    uu[j] = uu[j] + sgn * step
                    vv, uue = eval_u(uu)
                    if vv < vbest:
                        u, vbest = uue, vv
                        improved = True
                        break
            else:
                # random direction on sphere-ish (Gaussian then normalize)
                z = [random.gauss(0.0, 1.0) for _ in range(dim)]
                norm = math.sqrt(sum(zz * zz for zz in z)) + 1e-18
                z = [zz / norm for zz in z]
                # try both directions (pattern)
                for sgn in (1.0, -1.0):
                    if time.time() >= deadline:
                        break
                    uu = [u[i] + sgn * radius * z[i] for i in range(dim)]
                    vv, uue = eval_u(uu)
                    if vv < vbest:
                        u, vbest = uue, vv
                        improved = True
                        break

            # adapt radius
            if improved:
                radius = min(tr_max, radius * 1.35)
            else:
                radius = max(tr_min, radius * 0.62)

            # if radius got tiny, stop early
            if radius <= tr_min * 1.5:
                break

        return vbest, u, radius

    # ---------- restart control ----------
    last_improve = time.time()
    stall_seconds = max(0.20, 0.10 * max_time)

    gen = 0
    while time.time() < deadline:
        gen += 1

        # sample offspring with antithetic pairing
        off = []
        successes = 0

        # keep a snapshot of best for measuring "success"
        best_before = best

        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]

            # child +
            u1 = [mean[j] + (gsig * sig[j]) * z[j] for j in range(dim)]
            v1, uu1 = eval_u(u1)
            off.append((v1, uu1))

            # child -
            u2 = [mean[j] - (gsig * sig[j]) * z[j] for j in range(dim)]
            v2, uu2 = eval_u(u2)
            off.append((v2, uu2))

            if v1 < best:
                best, best_u = v1, uu1
            if v2 < best:
                best, best_u = v2, uu2

        if best + 0.0 < best_before:
            successes = 1  # at least one success this generation
            last_improve = time.time()

        off.sort(key=lambda t: t[0])

        # recombine mean from top mu
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                new_mean[j] += w * ui[j]
        mean = repair_u(new_mean)

        # diag sigma adaptation from elite spread
        for j in range(dim):
            s2 = 0.0
            mj = mean[j]
            for i in range(mu):
                w = weights[i]
                d = off[i][1][j] - mj
                s2 += w * d * d
            target = math.sqrt(max(1e-30, s2)) + 1e-30
            sj = (1.0 - c_diag) * sig[j] + c_diag * target
            if sj < sig_min[j]: sj = sig_min[j]
            if sj > sig_max[j]: sj = sig_max[j]
            sig[j] = sj

        # global step size: 1/5-ish but using "any success" signal for robustness
        ps = 1.0 if successes else 0.0
        # aim around 0.2
        gsig *= math.exp(cs * (ps - 0.2))
        if gsig < 1e-6: gsig = 1e-6
        if gsig > 6.0:  gsig = 6.0

        # periodic trust-region refinement near incumbent
        if (gen % 5 == 0) and time.time() < deadline:
            budget = 22 + 4 * dim
            vloc, uloc, new_tr = trust_refine(best_u, best, budget)
            if vloc < best:
                best, best_u = vloc, uloc
                last_improve = time.time()
                tr = new_tr
                mean = best_u[:]  # intensify around improved point
            else:
                # still update radius gently
                tr = max(tr_min, min(tr_max, 0.85 * tr + 0.15 * new_tr))

        # restart if stalled or steps collapsed
        stalled = (time.time() - last_improve) > stall_seconds
        tiny_steps = all((gsig * sig[j]) < 2e-6 for j in range(dim))
        if stalled or tiny_steps:
            # keep best, diversify mean and inflate sig
            mean = best_u[:]
            gsig = min(2.0, max(0.9, gsig * 1.4))
            tr = min(tr_max, max(0.10, tr * 1.25))

            for j in range(dim):
                sig[j] = min(sig_max[j], max(sig[j], 0.18))

            # inject a few random points
            for _ in range(max(8, dim)):
                if time.time() >= deadline:
                    return best
                v, uu = eval_u(rand_u())
                if v < best:
                    best, best_u = v, uu
                    mean = best_u[:]
                    last_improve = time.time()

    return best
