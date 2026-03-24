import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only), improved for robustness/speed.

    Main changes vs prior:
      - Uses a fast diagonal-separable CMA-style update most of the time (O(n)),
        and only occasionally injects pairwise rotations to learn correlations
        without full eigendecomposition (O(n^2) avoided).
      - Stronger global exploration: scrambled Halton + opposition + rare wide jumps.
      - Better boundary handling: reflection in normalized space.
      - Built-in trust-region local search around incumbent (adaptive per-dim steps).
      - Adaptive population size + IPOP-style restarts on stagnation.
      - Optional evaluation caching with coarse quantization to avoid repeats.
    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    n = int(dim)
    eps = 1e-12

    lo = [float(bounds[i][0]) for i in range(n)]
    hi = [float(bounds[i][1]) for i in range(n)]
    span = [hi[i] - lo[i] for i in range(n)]
    for i in range(n):
        if not (span[i] > 0.0):
            span[i] = 1.0

    def now():
        return time.time()

    def reflect01(u):
        # reflect arbitrary real into [0,1]
        if 0.0 <= u <= 1.0:
            return u
        u = u % 2.0
        if u > 1.0:
            u = 2.0 - u
        return u

    def x_from_u(u):
        return [lo[i] + u[i] * span[i] for i in range(n)]

    # ---- caching (coarse) ----
    cache = {}
    q = 1e-10

    def key_u(u):
        return tuple(int(u[i] / q) for i in range(n))

    def eval_u(u):
        ur = [reflect01(u[i]) for i in range(n)]
        k = key_u(ur)
        v = cache.get(k)
        if v is not None:
            return v, ur
        fx = float(func(x_from_u(ur)))
        cache[k] = fx
        return fx, ur

    # ---- low-discrepancy init: scrambled Halton ----
    def first_primes(m):
        primes = []
        x = 2
        while len(primes) < m:
            r = int(math.isqrt(x))
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

    def vdc(k, base):
        out = 0.0
        denom = 1.0
        while k:
            k, r = divmod(k, base)
            denom *= base
            out += r / denom
        return out

    bases = first_primes(n)
    # per-dimension scramble: additive shift + digit scrambling via random offset
    shift = [random.random() for _ in range(n)]
    salt = [random.randrange(1, 1_000_000) for _ in range(n)]
    hal_k = 1

    def halton_u(k):
        u = [0.0] * n
        for i in range(n):
            u[i] = (vdc(k + salt[i], bases[i]) + shift[i]) % 1.0
        return u

    # ---- utilities ----
    def dot(a, b):
        s = 0.0
        for i in range(n):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    # ---- local search (adaptive coordinate trust region in u-space) ----
    def local_refine(u0, f0, step_u, tries=1):
        u = u0[:]
        f = f0
        improved = False
        for _ in range(tries):
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                s = step_u[i]
                if s <= 1e-18:
                    continue
                # test +/- and also a smaller step if both fail
                best_local = (f, u)
                for scale in (1.0, 0.5):
                    ss = s * scale
                    for sign in (1.0, -1.0):
                        uu = u[:]
                        uu[i] = reflect01(uu[i] + sign * ss)
                        ff, uur = eval_u(uu)
                        if ff < best_local[0]:
                            best_local = (ff, uur)
                        if now() >= deadline:
                            return best_local[1], best_local[0], (best_local[0] < f)
                if best_local[0] < f:
                    f, u = best_local
                    improved = True
                if now() >= deadline:
                    return u, f, improved
        return u, f, improved

    # ---- initialization / incumbent ----
    best = float("inf")
    best_u = None

    init_budget = max(80, 30 * n)
    for _ in range(init_budget):
        if now() >= deadline:
            return best
        r = random.random()
        if r < 0.65:
            u = halton_u(hal_k); hal_k += 1
        elif r < 0.85:
            u = [random.random() for _ in range(n)]
        elif r < 0.95:
            # opposition point (often helps on bounded problems)
            u0 = halton_u(hal_k); hal_k += 1
            u = [1.0 - u0[i] for i in range(n)]
        else:
            # around current best
            if best_u is None:
                u = [random.random() for _ in range(n)]
            else:
                u = [reflect01(best_u[i] + random.gauss(0.0, 0.20)) for i in range(n)]

        fx, ur = eval_u(u)
        if fx < best:
            best, best_u = fx, ur

    if best_u is None:
        return best

    # ---- Separable CMA-ES core with occasional 2D rotation learning ----
    # Mean in u-space
    m = best_u[:]

    # Per-dimension stddev in u-space
    sigma0 = 0.30
    sig = [sigma0] * n

    # Evolution paths for step-size control (separable)
    ps = [0.0] * n

    # Rotation pool (sparse correlation model): list of (i,j,angle)
    # Applied as: [z_i,z_j] <- R(angle)*[z_i,z_j]
    rot = []
    max_rot = min(64, max(8, 2 * n))

    # CMA parameters (separable)
    lam_base = max(10, 4 + int(3 * math.log(n + 1.0)))
    lam = max(lam_base, 8 + (4 * n) // 3)
    mu = lam // 2
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    cs = (mueff + 2.0) / (n + mueff + 5.0)
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # learning rate for per-dim variances
    cvar = min(0.35, (2.0 / (n + 2.0)) * (mueff / (mueff + 2.0)))

    # restart / stagnation controls
    no_imp = 0
    stagnate = 20 + 6 * n
    IPOP_factor = 1

    # local step for refinement
    step_u = [0.10 for _ in range(n)]

    def apply_rotations(z):
        # apply sparse rotations in-place (copy returned)
        if not rot:
            return z
        zz = z[:]
        for (i, j, ang) in rot:
            ci = math.cos(ang)
            si = math.sin(ang)
            a = zz[i]
            b = zz[j]
            zz[i] = ci * a - si * b
            zz[j] = si * a + ci * b
        return zz

    def maybe_update_rotations(elite):
        # elite: list of (fx, z_raw, y, u) sorted
        # learn a few pairwise correlations among best steps y
        if n < 2:
            return
        # choose a few random pairs, estimate sign of covariance in elite, nudge angle
        k_pairs = min(6, 2 + n // 5)
        for _ in range(k_pairs):
            i = random.randrange(n)
            j = random.randrange(n - 1)
            if j >= i:
                j += 1
            cov = 0.0
            for t in range(min(mu, len(elite))):
                y = elite[t][2]
                cov += y[i] * y[j]
            cov /= max(1, min(mu, len(elite)))

            # determine desired rotation direction (very small angles)
            # if cov positive, rotate to align axes slightly; if negative, opposite
            delta = 0.06 * clamp(cov, -1.0, 1.0)
            if abs(delta) < 1e-6:
                continue

            # insert/update a rotation entry for (i,j)
            if len(rot) < max_rot and random.random() < 0.6:
                rot.append((i, j, delta))
            else:
                # modify a random existing rotation
                idx = random.randrange(len(rot))
                ii, jj, ang = rot[idx]
                rot[idx] = (ii, jj, ang * 0.85 + delta)

    gen = 0
    while True:
        if now() >= deadline:
            return best
        gen += 1

        # occasional wide/global injection probability
        p_inject = 0.12 if gen < 10 else 0.06
        p_cauchy = 0.04

        pop = []
        for _ in range(lam):
            if now() >= deadline:
                return best

            if random.random() < p_inject:
                # global candidate from Halton / random / best-perturb
                r = random.random()
                if r < 0.55:
                    u = halton_u(hal_k); hal_k += 1
                elif r < 0.80:
                    u = [random.random() for _ in range(n)]
                else:
                    u = [reflect01(best_u[i] + random.gauss(0.0, 0.25)) for i in range(n)]
                fx, ur = eval_u(u)
                # approximate y and z for learning (z not reliable but ok for ranking)
                y = [(ur[i] - m[i]) / (sig[i] + eps) for i in range(n)]
                z = y[:]  # raw
                pop.append((fx, z, y, ur))
                continue

            # sample z ~ N(0,I), apply sparse rotations
            z = [random.gauss(0.0, 1.0) for _ in range(n)]
            if random.random() < p_cauchy:
                # heavy-tail kick
                c = math.tan(math.pi * (random.random() - 0.5))
                j = random.randrange(n)
                z[j] += 0.75 * c

            z2 = apply_rotations(z)
            y = z2  # in separable model, y=z after rotations
            u = [reflect01(m[i] + sig[i] * y[i]) for i in range(n)]
            fx, ur = eval_u(u)
            pop.append((fx, z, y, ur))

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best = pop[0][0]
            best_u = pop[0][3]
            no_imp = 0
        else:
            no_imp += 1

        # weighted recombination in y-space
        y_w = [0.0] * n
        for k in range(mu):
            yk = pop[k][2]
            wk = w[k]
            for i in range(n):
                y_w[i] += wk * yk[i]

        # update mean
        m_old = m
        m = [reflect01(m_old[i] + sig[i] * y_w[i]) for i in range(n)]

        # step-size path (separable)
        c_sig = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(n):
            ps[i] = (1.0 - cs) * ps[i] + c_sig * y_w[i]

        # global step control factor from path length
        psn = norm(ps)
        g = math.exp((cs / damps) * (psn / (chi_n + eps) - 1.0))
        g = clamp(g, 0.6, 1.6)
        for i in range(n):
            sig[i] = clamp(sig[i] * g, 1e-12, 0.75)

        # per-dimension variance adaptation from elite spread
        # (move sig toward RMS of elite steps)
        rms = [0.0] * n
        for k in range(mu):
            yk = pop[k][2]
            for i in range(n):
                rms[i] += w[k] * (yk[i] * yk[i])
        for i in range(n):
            target = math.sqrt(max(eps, rms[i]))
            # convert target (in z units) to sigma scaling gently
            # if target>1 -> expand; target<1 -> contract
            factor = math.exp(0.5 * (target - 1.0))
            factor = clamp(factor, 0.75, 1.35)
            sig[i] = clamp(sig[i] * (1.0 - cvar + cvar * factor), 1e-12, 0.75)

        # learn a few correlations cheaply
        if gen % 3 == 0:
            maybe_update_rotations(pop)

        # local refinement around best
        if gen % 4 == 0 and best_u is not None and now() < deadline:
            u2, f2, imp = local_refine(best_u, best, step_u, tries=1)
            if f2 < best:
                best, best_u = f2, u2
                no_imp = 0
                # tighten local steps
                for i in range(n):
                    step_u[i] = max(1e-18, step_u[i] * 0.75)
                # pull mean toward best
                m = best_u[:]
            else:
                # slowly shrink anyway
                for i in range(n):
                    step_u[i] = max(1e-18, step_u[i] * 0.92)

        # restart on stagnation (IPOP-style)
        if no_imp >= stagnate:
            no_imp = 0
            IPOP_factor = min(16, IPOP_factor * 2)
            lam = min(400, max(lam_base, lam_base * IPOP_factor))
            mu = lam // 2
            w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            wsum = sum(w)
            w = [wi / wsum for wi in w]
            mueff = 1.0 / sum(wi * wi for wi in w)

            cs = (mueff + 2.0) / (n + mueff + 5.0)
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

            # reset model around incumbent
            m = best_u[:] if best_u is not None else [random.random() for _ in range(n)]
            ps = [0.0] * n
            rot = []
            # inflate step sizes to escape
            for i in range(n):
                sig[i] = clamp(sig[i] * 1.8, 0.08, 0.65)
                step_u[i] = clamp(step_u[i] * 1.6, 0.02, 0.25)

        # if very close to boundary or tiny sig, inject tiny noise to prevent collapse
        if gen % 7 == 0:
            for i in range(n):
                if sig[i] < 1e-6:
                    sig[i] = 1e-6
            if best_u is not None and random.random() < 0.15:
                m = [reflect01(best_u[i] + random.gauss(0.0, 0.02)) for i in range(n)]
