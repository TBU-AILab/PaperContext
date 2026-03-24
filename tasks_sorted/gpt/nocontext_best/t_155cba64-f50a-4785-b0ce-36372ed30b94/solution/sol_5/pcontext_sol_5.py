import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs previous version:
      - Adds a true *multi-start coordinate pattern search* with adaptive per-dimension steps.
      - Uses a light *CMA-ES-like diagonal sampler* but with better success-based step control.
      - Adds *opposition / reflection* proposals around the current best (cheap diversification).
      - Uses a *two-level cache* (coarse+fine) to reduce redundant evaluations without over-pruning.
      - Allocates time explicitly between global sampling and local polishing near the end.

    Returns: best (minimum fitness found) within max_time seconds.
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
    span = [s if s > 0.0 else 1.0 for s in span]

    def repair_u(u):
        # clamp to [0,1]
        r = [0.0] * dim
        for i, v in enumerate(u):
            if v <= 0.0: r[i] = 0.0
            elif v >= 1.0: r[i] = 1.0
            else: r[i] = v
        return r

    def to_x(u):
        return [lo[i] + span[i] * u[i] for i in range(dim)]

    # Two-level cache to avoid wasted re-evaluations.
    # Coarse prevents obvious repeats; fine helps local search.
    coarse = {}
    fine = {}
    q_coarse = 8e-4 if dim <= 15 else 1.5e-3
    q_fine = 1.2e-4 if dim <= 15 else 2.5e-4

    def key(u, q):
        return tuple(int(v / q + 0.5) for v in u)

    def eval_u(u):
        u = repair_u(u)
        kc = key(u, q_coarse)
        if kc in coarse:
            return coarse[kc], u
        kf = key(u, q_fine)
        if kf in fine:
            v = fine[kf]
            coarse[kc] = v
            return v, u

        x = to_x(u)
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if not math.isfinite(v):
            v = float("inf")

        fine[kf] = v
        coarse[kc] = v
        return v, u

    def rand_u():
        return [random.random() for _ in range(dim)]

    # Halton init (deterministic low-discrepancy)
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r: break
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

    # --- Initialization ---
    best = float("inf")
    best_u = rand_u()

    n_init = max(40, min(340, 26 * dim + 70))
    k = 17
    for _ in range(n_init):
        if time.time() >= deadline:
            return best
        v, u = eval_u(halton_u(k))
        k += 1
        if v < best:
            best, best_u = v, u

    # Some random points too
    for _ in range(max(12, 2 * dim)):
        if time.time() >= deadline:
            return best
        v, u = eval_u(rand_u())
        if v < best:
            best, best_u = v, u

    # --- Global sampler (diag-CMA-ish) ---
    mean = best_u[:]
    lam = max(20, 10 * dim)
    if lam % 2 == 1:
        lam += 1
    mu = max(6, lam // 4)

    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    sig = [0.22] * dim
    sig_min = 1e-7
    sig_max = 0.55
    gsig = 1.0

    c_diag = min(0.28, 0.08 + 2.0 / (dim + 12.0))

    # slightly stronger success control than previous code:
    # - track fraction of offspring improving *best of generation*, not just "any success"
    cs = 0.35

    # Opposition / reflection mixing
    def reflect_about_best(u):
        # u' = 2*best - u, then clamp
        return repair_u([2.0 * best_u[i] - u[i] for i in range(dim)])

    # --- Local pattern search (multi-start, coordinate-wise, adaptive steps) ---
    def local_pattern(u0, v0, max_evals):
        u = u0[:]
        vbest = v0
        # per-dimension step sizes in normalized space
        step = [0.18 if dim <= 10 else 0.12] * dim
        step_min = 2e-7
        step_max = 0.45

        # round-robin coordinates but shuffled blocks
        order = list(range(dim))
        idx = 0
        evals = 0

        # occasional random directions to escape axis-aligned issues
        while evals < max_evals and time.time() < deadline:
            if idx == 0:
                random.shuffle(order)

            j = order[idx]
            idx = (idx + 1) % dim

            improved = False
            sj = step[j]

            # coordinate poll
            for sgn in (1.0, -1.0):
                if evals >= max_evals or time.time() >= deadline:
                    break
                uu = u[:]
                uu[j] = uu[j] + sgn * sj
                vv, uue = eval_u(uu)
                evals += 1
                if vv < vbest:
                    u, vbest = uue, vv
                    improved = True
                    break

            # if no coordinate improvement, try a random small direction sometimes
            if (not improved) and (evals < max_evals) and (random.random() < 0.22):
                z = [random.gauss(0.0, 1.0) for _ in range(dim)]
                norm = math.sqrt(sum(zz * zz for zz in z)) + 1e-18
                z = [zz / norm for zz in z]
                rad = 0.7 * (sum(step) / dim)
                for sgn in (1.0, -1.0):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    uu = [u[i] + sgn * rad * z[i] for i in range(dim)]
                    vv, uue = eval_u(uu)
                    evals += 1
                    if vv < vbest:
                        u, vbest = uue, vv
                        improved = True
                        break

            # adapt per-dimension step
            if improved:
                step[j] = min(step_max, step[j] * 1.35)
            else:
                step[j] = max(step_min, step[j] * 0.60)

            # global early stop if all steps tiny
            if max(step) <= step_min * 2.0:
                break

        return vbest, u

    # --- Main loop with explicit time allocation ---
    # Keep last ~25% time for intensification
    intensify_time = t0 + 0.75 * max_time

    last_best_time = time.time()
    stall_seconds = max(0.20, 0.12 * max_time)
    gen = 0

    while time.time() < deadline:
        gen += 1

        best_before = best
        off = []
        improve_count = 0

        # generate offspring (antithetic) + occasional reflection candidate
        for i in range(lam // 2):
            if time.time() >= deadline:
                return best

            z = [random.gauss(0.0, 1.0) for _ in range(dim)]

            u1 = [mean[j] + (gsig * sig[j]) * z[j] for j in range(dim)]
            v1, uu1 = eval_u(u1)
            off.append((v1, uu1))

            u2 = [mean[j] - (gsig * sig[j]) * z[j] for j in range(dim)]
            v2, uu2 = eval_u(u2)
            off.append((v2, uu2))

            # small chance of reflection proposal (diversify cheaply)
            if (i == 0 or (i % 5 == 0)) and (random.random() < 0.35):
                ur = reflect_about_best(uu1 if v1 < v2 else uu2)
                vr, uur = eval_u(ur)
                off.append((vr, uur))

            # update best
            if v1 < best:
                best, best_u = v1, uu1
            if v2 < best:
                best, best_u = v2, uu2

        if best < best_before:
            last_best_time = time.time()

        # success ratio relative to generation baseline: count how many beat best_before
        for v, _ in off:
            if v < best_before:
                improve_count += 1
        success_rate = improve_count / max(1, len(off))

        off.sort(key=lambda t: t[0])

        # recombine mean
        new_mean = [0.0] * dim
        for i in range(mu):
            w = weights[i]
            ui = off[i][1]
            for j in range(dim):
                new_mean[j] += w * ui[j]
        mean = repair_u(new_mean)

        # adapt diag sig from elite spread around mean
        for j in range(dim):
            mj = mean[j]
            s2 = 0.0
            for i in range(mu):
                w = weights[i]
                d = off[i][1][j] - mj
                s2 += w * d * d
            target = math.sqrt(max(1e-30, s2))
            sj = (1.0 - c_diag) * sig[j] + c_diag * target
            if sj < sig_min: sj = sig_min
            if sj > sig_max: sj = sig_max
            sig[j] = sj

        # global step size control: target ~0.20 success rate
        gsig *= math.exp(cs * (success_rate - 0.20))
        if gsig < 1e-7: gsig = 1e-7
        if gsig > 7.0:  gsig = 7.0

        # periodic local polishing (more aggressive near the end)
        now = time.time()
        if now >= intensify_time:
            # spend more effort locally late
            budget = 40 + 10 * dim
            vloc, uloc = local_pattern(best_u, best, budget)
            if vloc < best:
                best, best_u = vloc, uloc
                mean = best_u[:]
                last_best_time = time.time()
        else:
            if gen % 6 == 0:
                budget = 18 + 4 * dim
                vloc, uloc = local_pattern(best_u, best, budget)
                if vloc < best:
                    best, best_u = vloc, uloc
                    mean = best_u[:]
                    last_best_time = time.time()

        # restart / diversify on stall or collapse
        stalled = (time.time() - last_best_time) > stall_seconds
        tiny = (gsig * max(sig)) < (2e-6 if dim <= 20 else 5e-6)
        if stalled or tiny:
            # reset around best but inject diversity
            mean = best_u[:]
            gsig = min(2.5, max(1.0, gsig * 1.6))
            for j in range(dim):
                sig[j] = max(sig[j], 0.20)
                if sig[j] > sig_max:
                    sig[j] = sig_max

            # inject random points + their opposites
            inj = max(10, dim)
            for _ in range(inj):
                if time.time() >= deadline:
                    return best
                u = rand_u()
                v, uu = eval_u(u)
                if v < best:
                    best, best_u = v, uu
                    mean = best_u[:]
                    last_best_time = time.time()

                uo = repair_u([1.0 - uu[i] for i in range(dim)])
                vo, uuo = eval_u(uo)
                if vo < best:
                    best, best_u = vo, uuo
                    mean = best_u[:]
                    last_best_time = time.time()

    return best
