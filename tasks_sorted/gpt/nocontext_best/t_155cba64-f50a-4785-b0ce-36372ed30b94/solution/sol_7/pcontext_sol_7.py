import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs your current version:
      1) True CMA-ES style adaptation signals (success measured by rank-improvement, not just "any").
      2) Stronger local search: multi-scale coordinate pattern search + occasional quadratic 1D fits.
      3) Better restarts: IPOP-like population growth + sigma reset, plus occasional "center = best"
         intensification and "center = random" diversification.
      4) Better caching: quantized grid + LRU eviction to control memory and speed.
      5) Boundary handling: reflection in normalized space (often better than hard clipping).
    Returns: best (minimum) fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        try:
            v = float(func([]))
            return v if math.isfinite(v) else float("inf")
        except Exception:
            return float("inf")

    # -------------------- bounds / normalization --------------------
    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if not math.isfinite(lo[i]) or not math.isfinite(hi[i]) or span[i] <= 0.0:
            # fallback to something sane
            lo[i], hi[i] = (0.0, 1.0)
            span[i] = 1.0

    def to_x(u):
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    def reflect01(u):
        # reflect into [0,1] (better than clipping for search dynamics)
        out = [0.0] * dim
        for i, v in enumerate(u):
            if v >= 0.0 and v <= 1.0:
                out[i] = v
            else:
                # reflect with period 2: ..., [0,1],[1,0],[0,1],...
                v = v % 2.0
                out[i] = 2.0 - v if v > 1.0 else v
        return out

    # -------------------- small LRU cache on quantized u --------------------
    # Quantization: adapt to dim (coarser in high dim to keep hit rate)
    q = 1e-4 if dim <= 8 else (3e-4 if dim <= 20 else 8e-4)
    cache = {}
    order = []
    max_cache = 12000 if dim <= 15 else 8000

    def key_u(u):
        return tuple(int(v / q + 0.5) for v in u)

    def eval_u(u):
        u = reflect01(u)
        k = key_u(u)
        v = cache.get(k)
        if v is not None:
            return v, u
        x = to_x(u)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        cache[k] = v
        order.append(k)
        if len(order) > max_cache:
            kk = order.pop(0)
            # may have been re-added; safe remove if present
            if kk in cache:
                del cache[kk]
        return v, u

    def rand_u():
        return [random.random() for _ in range(dim)]

    # -------------------- low discrepancy init (Halton) --------------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            r = int(x ** 0.5)
            ok = True
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
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

    # -------------------- initialization --------------------
    best = float("inf")
    best_u = rand_u()

    k = 17
    n_init = max(48, min(320, 26 * dim + 60))
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, u = eval_u(halton_u(k))
        k += 1
        if v < best:
            best, best_u = v, u

    # random shake to avoid LDS resonance
    for _ in range(max(12, dim)):
        if time.time() >= deadline:
            return best
        v, u = eval_u(rand_u())
        if v < best:
            best, best_u = v, u

    # -------------------- CMA-ES-like evolution loop (diag) --------------------
    mean = best_u[:]

    # base population; will grow on restarts (IPOP-ish)
    lam0 = max(20, 10 * dim)
    if lam0 % 2 == 1:
        lam0 += 1

    # initial diagonal sigma in normalized space
    sig = [0.22] * dim
    sig_min = 1e-7
    sig_max = 0.55
    gsig = 1.0

    # learning rates
    c_diag = min(0.35, 0.12 + 2.5 / (dim + 12.0))  # update from elite spread
    cs = 0.18  # global step adaptation from success signal

    # restart bookkeeping
    restart = 0
    last_improve_t = time.time()
    stall_seconds = max(0.25, 0.12 * max_time)

    # -------------------- local search --------------------
    def try_quadratic_1d(u, j, step):
        """
        Fit a parabola through f(x-step), f(x), f(x+step) in 1D and try minimizer.
        Works surprisingly well for cheap local curvature exploitation.
        """
        uj = u[j]
        u0 = u[:]
        v0, _ = eval_u(u0)

        uL = u[:]
        uL[j] = uj - step
        vL, uL = eval_u(uL)

        uR = u[:]
        uR[j] = uj + step
        vR, uR = eval_u(uR)

        # parabola through (-1,vL), (0,v0), (1,vR): vertex at x* = (vL - vR)/(2*(vL - 2v0 + vR))
        denom = (vL - 2.0 * v0 + vR)
        if denom == 0.0 or not math.isfinite(denom):
            return v0, u0
        xstar = 0.5 * (vL - vR) / denom
        # restrict within [-1.5, 1.5] for stability
        if xstar < -1.5:
            xstar = -1.5
        elif xstar > 1.5:
            xstar = 1.5
        uQ = u[:]
        uQ[j] = uj + xstar * step
        vQ, uQ = eval_u(uQ)

        # return best among tested
        vb, ub = v0, u0
        if vL < vb:
            vb, ub = vL, uL
        if vR < vb:
            vb, ub = vR, uR
        if vQ < vb:
            vb, ub = vQ, uQ
        return vb, ub

    def local_refine(u0, v0, budget):
        """
        Multi-scale coordinate/pattern local search in normalized space.
        Uses coordinate steps + occasional 1D quadratic interpolation.
        """
        u = u0[:]
        vbest = v0

        # start from a radius tied to current exploration scale
        base = 0.35 * max(1e-9, gsig * (sum(sig) / dim))
        step = min(0.25, max(1e-4, base))

        for it in range(budget):
            if time.time() >= deadline:
                break

            improved = False
            j = random.randrange(dim)

            # coordinate +/- step
            for sgn in (1.0, -1.0):
                uu = u[:]
                uu[j] = uu[j] + sgn * step
                vv, uue = eval_u(uu)
                if vv < vbest:
                    u, vbest = uue, vv
                    improved = True
                    break

            # occasional 1D quadratic fit when stuck a bit
            if (not improved) and (it % 5 == 0):
                vv, uu = try_quadratic_1d(u, j, step)
                if vv < vbest:
                    u, vbest = uu, vv
                    improved = True

            # pattern move: if improved, try continuing same direction cheaply
            if improved:
                # small acceleration: take one more step same coordinate direction with reduced step
                uu = u[:]
                uu[j] = uu[j] + (0.6 * step if random.random() < 0.7 else 0.0)
                vv, uue = eval_u(uu)
                if vv < vbest:
                    u, vbest = uue, vv
                step = min(0.35, step * 1.25)
            else:
                step = max(1e-7, step * 0.65)

            if step <= 2e-7:
                break

        return vbest, u

    # -------------------- main loop with restarts --------------------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # IPOP-like population growth on each restart
        lam = lam0 * (2 ** restart)
        lam = min(lam, 2400)  # safety
        if lam % 2 == 1:
            lam += 1
        mu = max(6, lam // 4)

        # log weights
        weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        sw = sum(weights)
        weights = [w / sw for w in weights]

        # --- sample offspring (antithetic) ---
        off = []
        best_before = best
        best_in_gen = best

        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            u1 = [mean[j] + (gsig * sig[j]) * z[j] for j in range(dim)]
            v1, u1 = eval_u(u1)
            off.append((v1, u1))
            if v1 < best:
                best, best_u = v1, u1

            u2 = [mean[j] - (gsig * sig[j]) * z[j] for j in range(dim)]
            v2, u2 = eval_u(u2)
            off.append((v2, u2))
            if v2 < best:
                best, best_u = v2, u2

            if v1 < best_in_gen:
                best_in_gen = v1
            if v2 < best_in_gen:
                best_in_gen = v2

        off.sort(key=lambda t: t[0])

        # --- recombine mean from top mu ---
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                new_mean[j] += w * ui[j]
        mean = reflect01(new_mean)

        # --- update diag sig from elite spread ---
        for j in range(dim):
            mj = mean[j]
            s2 = 0.0
            for i in range(mu):
                d = off[i][1][j] - mj
                s2 += weights[i] * d * d
            target = math.sqrt(max(1e-30, s2))
            sj = (1.0 - c_diag) * sig[j] + c_diag * target
            if sj < sig_min:
                sj = sig_min
            elif sj > sig_max:
                sj = sig_max
            sig[j] = sj

        # --- global step-size adaptation (rank-based success) ---
        # success if generation's best beats median of previous generation-ish:
        # approximate using off[mu] as "typical good" threshold
        thresh = off[min(len(off) - 1, max(1, mu))][0]
        success = 1.0 if best_in_gen < thresh else 0.0
        gsig *= math.exp(cs * (success - 0.25))  # target ~25% success
        if gsig < 1e-7:
            gsig = 1e-7
        elif gsig > 8.0:
            gsig = 8.0

        # --- periodic local refine around incumbent ---
        if (gen % 4 == 0) and time.time() < deadline:
            budget = 18 + 5 * dim
            vloc, uloc = local_refine(best_u, best, budget)
            if vloc < best:
                best, best_u = vloc, uloc
                mean = best_u[:]  # intensify
                last_improve_t = time.time()

        if best < best_before:
            last_improve_t = time.time()

        # --- restart criteria ---
        stalled = (time.time() - last_improve_t) > stall_seconds
        tiny = (gsig * (sum(sig) / dim)) < 3e-7
        if stalled or tiny:
            restart += 1

            # alternate intensify/diversify restarts
            if restart % 2 == 1:
                # diversify: random mean, bigger steps
                mean = rand_u()
            else:
                # intensify: go back to best
                mean = best_u[:]

            # reset step sizes
            gsig = min(2.5, max(1.0, gsig * 1.8))
            base = 0.26 if dim <= 10 else 0.20
            sig = [min(sig_max, max(0.08, base)) for _ in range(dim)]

            # sprinkle a few random probes
            for _ in range(max(10, dim)):
                if time.time() >= deadline:
                    return best
                v, u = eval_u(rand_u())
                if v < best:
                    best, best_u = v, u
                    mean = best_u[:]
                    last_improve_t = time.time()

    return best
