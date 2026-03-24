import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs previous version:
      - Proper 1/5-success step-size control using generation success-rate (not just "any success").
      - Add separable Cauchy-heavy-tail mutations (better global jumps / escaping local minima).
      - Add fast coordinate-wise pattern search with per-dimension step sizes (very strong final polish).
      - Better restart logic using both time-since-improvement and *progress* (delta-best).
      - Cache with adaptive quantization (coarser when dim is large) + LRU-like pruning.
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

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if not math.isfinite(span[i]) or span[i] <= 0.0:
            span[i] = 1.0

    def repair_u(u):
        out = u[:]  # in [0,1]
        for i in range(dim):
            x = out[i]
            if x < 0.0: x = 0.0
            elif x > 1.0: x = 1.0
            out[i] = x
        return out

    def to_x(u):
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    # -------------------- cached evaluation --------------------
    # quantization step in normalized space
    # (bigger in high dims to reduce memory; still avoids waste)
    if dim <= 6:
        q = 1e-4
    elif dim <= 15:
        q = 4e-4
    elif dim <= 30:
        q = 1e-3
    else:
        q = 2e-3

    grid = {}
    grid_order = []  # insertion order for pruning
    max_cache = 60000 if dim <= 20 else (35000 if dim <= 60 else 20000)

    def key_u(u):
        return tuple(int(uu / q + 0.5) for uu in u)

    def eval_u(u):
        uu = repair_u(u)
        k = key_u(uu)
        v = grid.get(k)
        if v is not None:
            return v, uu
        x = to_x(uu)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")
        grid[k] = v
        grid_order.append(k)
        # prune occasionally (cheap)
        if len(grid_order) > max_cache + 512:
            # drop oldest ~20%
            drop = len(grid_order) - max_cache
            for i in range(drop):
                kk = grid_order[i]
                # might already be overwritten, but safe
                if kk in grid:
                    del grid[kk]
            grid_order[:] = grid_order[drop:]
        return v, uu

    def rand_u():
        return [random.random() for _ in range(dim)]

    # -------------------- Halton init --------------------
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

    # -------------------- init best --------------------
    best = float("inf")
    best_u = rand_u()

    k = 17
    n_init = max(48, min(340, 26 * dim + 64))
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(halton_u(k))
        k += 1
        if v < best:
            best, best_u = v, uu

    # a few random probes
    for _ in range(max(12, 2 * dim)):
        if time.time() >= deadline:
            return best
        v, uu = eval_u(rand_u())
        if v < best:
            best, best_u = v, uu

    # -------------------- evolution strategy core --------------------
    mean = best_u[:]

    lam = max(22, 12 * dim)
    if lam % 2 == 1:
        lam += 1
    mu = max(6, lam // 4)

    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    # diagonal std in normalized coords
    sig = [0.22] * dim
    sig_min = 1e-7
    sig_max = 0.55
    gsig = 1.0

    c_diag = min(0.28, 0.10 + 2.0 / (dim + 10.0))

    # 1/5 success adaptation (uses success-rate per generation)
    cs = 0.55  # slightly more reactive than before
    target_ps = 0.20

    # heavy-tail mix probability
    p_cauchy = 0.20 if dim <= 20 else 0.12

    def cauchy():
        # standard Cauchy via tan(pi*(u-0.5))
        u = random.random()
        # avoid infinities
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # -------------------- coordinate pattern search polish --------------------
    # per-dim step sizes (normalized)
    coord_step = [0.15] * dim
    coord_min = 5e-7
    coord_max = 0.40

    def coordinate_polish(u0, v0, passes):
        """Greedy coordinate search with adaptive per-dimension steps."""
        u = u0[:]
        vbest = v0
        # shuffle dims each pass
        for _ in range(passes):
            if time.time() >= deadline:
                break
            idx = list(range(dim))
            random.shuffle(idx)
            any_improve = False
            for j in idx:
                if time.time() >= deadline:
                    break
                step = coord_step[j]
                if step < coord_min:
                    continue
                # try both directions
                improved = False
                for sgn in (1.0, -1.0):
                    uu = u[:]
                    uu[j] = uu[j] + sgn * step
                    vv, uue = eval_u(uu)
                    if vv < vbest:
                        u, vbest = uue, vv
                        improved = True
                        any_improve = True
                        break
                if improved:
                    coord_step[j] = min(coord_max, coord_step[j] * 1.25)
                else:
                    coord_step[j] = max(coord_min, coord_step[j] * 0.72)
            if not any_improve:
                break
        return vbest, u

    # -------------------- restart / stall control --------------------
    last_best = best
    last_improve_t = time.time()
    stall_seconds = max(0.25, 0.10 * max_time)

    gen = 0
    while time.time() < deadline:
        gen += 1
        best_before = best
        off = []

        # create offspring; antithetic gaussian + heavy-tail option
        successes = 0
        for _ in range(lam // 2):
            if time.time() >= deadline:
                return best

            use_tail = (random.random() < p_cauchy)
            if use_tail:
                z = [cauchy() for _ in range(dim)]
                # tame extreme tails a bit (still heavy-tail)
                z = [max(-8.0, min(8.0, zz)) for zz in z]
            else:
                z = [random.gauss(0.0, 1.0) for _ in range(dim)]

            # scale vector
            step = [(gsig * sig[j]) * z[j] for j in range(dim)]

            u1 = [mean[j] + step[j] for j in range(dim)]
            v1, uu1 = eval_u(u1)
            off.append((v1, uu1))

            u2 = [mean[j] - step[j] for j in range(dim)]
            v2, uu2 = eval_u(u2)
            off.append((v2, uu2))

            if v1 < best:
                best, best_u = v1, uu1
            if v2 < best:
                best, best_u = v2, uu2

            # success definition: offspring better than parent mean's best of last gen
            if v1 < best_before:
                successes += 1
            if v2 < best_before:
                successes += 1

        off.sort(key=lambda t: t[0])

        # recombine mean from top mu
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                new_mean[j] += w * ui[j]
        mean = repair_u(new_mean)

        # diag sigma update from elite spread
        for j in range(dim):
            mj = mean[j]
            s2 = 0.0
            for i in range(mu):
                d = off[i][1][j] - mj
                s2 += weights[i] * (d * d)
            target = math.sqrt(max(1e-30, s2))
            sj = (1.0 - c_diag) * sig[j] + c_diag * target
            if sj < sig_min:
                sj = sig_min
            elif sj > sig_max:
                sj = sig_max
            sig[j] = sj

        # 1/5 success rule using success rate
        ps = successes / float(lam)
        gsig *= math.exp(cs * (ps - target_ps))
        if gsig < 1e-7:
            gsig = 1e-7
        elif gsig > 7.0:
            gsig = 7.0

        # periodic coordinate polish on incumbent (cheap, strong)
        if gen % 4 == 0 and time.time() < deadline:
            passes = 2 if dim > 30 else 3
            vloc, uloc = coordinate_polish(best_u, best, passes=passes)
            if vloc < best:
                best, best_u = vloc, uloc
                mean = best_u[:]

        # update stall info
        if best < best_before:
            last_improve_t = time.time()
            last_best = best

        # restart if stalled or collapsed steps
        stalled = (time.time() - last_improve_t) > stall_seconds
        tiny = (gsig < 5e-4) or all((gsig * sig[j]) < 2e-6 for j in range(dim))

        # also restart if progress is extremely small for a while
        no_progress = (abs(best_before - best) <= 1e-12) and stalled

        if stalled or tiny or no_progress:
            # keep incumbent; reset mean to it, diversify around it + inject random points
            mean = best_u[:]
            gsig = min(2.5, max(1.0, gsig * 1.8))
            for j in range(dim):
                sig[j] = max(sig[j], 0.20)
                if sig[j] > sig_max:
                    sig[j] = sig_max
                # also reset coordinate steps (helps polish after restart)
                coord_step[j] = max(coord_step[j], 0.10)

            # inject random samples (global)
            inject = max(10, dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                v, uu = eval_u(rand_u())
                if v < best:
                    best, best_u = v, uu
                    mean = best_u[:]
                    last_improve_t = time.time()

    return best
