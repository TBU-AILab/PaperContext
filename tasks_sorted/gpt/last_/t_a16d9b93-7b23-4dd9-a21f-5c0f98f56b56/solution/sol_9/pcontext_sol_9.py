import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, stdlib only).

    Key upgrades vs previous:
      1) Better global engine: JADE-style DE (current-to-pbest/1) with external archive.
         - more exploratory than current-to-best
         - self-adapting F/CR via successful parameters
      2) Stronger local refinement: simultaneous perturbation (SPSA-like) + coordinate pattern,
         operating in normalized space with adaptive scales and a short line-search.
      3) Smarter time scheduling: interleaves global and local; increases local intensity near end.
      4) Robustness: safe evaluation, bounded cache, degenerate-dimension handling.
    Returns:
      best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if not (span[i] > 0.0):
            span[i] = 0.0

    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def clamp01(u):
        for i in range(dim):
            if u[i] < 0.0:
                u[i] = 0.0
            elif u[i] > 1.0:
                u[i] = 1.0
        return u

    def u_to_x(u):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] == 0.0:
                x[i] = lo[i]
            else:
                ui = u[i]
                if ui < 0.0:
                    ui = 0.0
                elif ui > 1.0:
                    ui = 1.0
                x[i] = lo[i] + ui * span[i]
        return x

    def rand_u():
        return [random.random() for _ in range(dim)]

    # Box-Muller gaussian
    _has_spare = False
    _spare = 0.0
    def gauss():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    # ------------------------- bounded cache -------------------------
    dec = 6 if dim <= 12 else (5 if dim <= 30 else 4)
    CACHE_MAX = 12000 if dim <= 15 else (8000 if dim <= 50 else 4500)
    cache = {}
    ring = [None] * CACHE_MAX
    ring_pos = 0

    def key_u(u):
        return tuple(round(ui, dec) for ui in u)

    def evaluate_u(u):
        nonlocal ring_pos
        u = clamp01(u[:])
        k = key_u(u)
        v = cache.get(k, None)
        if v is not None:
            return v, u

        x = u_to_x(u)
        try:
            fv = float(func(x))
            if not is_finite(fv):
                fv = float("inf")
        except Exception:
            fv = float("inf")

        old = ring[ring_pos]
        if old is not None and old in cache:
            del cache[old]
        ring[ring_pos] = k
        ring_pos += 1
        if ring_pos >= CACHE_MAX:
            ring_pos = 0

        cache[k] = fv
        return fv, u

    best_u = None
    best_f = float("inf")

    def consider(u):
        nonlocal best_u, best_f
        fu, uu = evaluate_u(u)
        if fu < best_f:
            best_f = fu
            best_u = uu[:]
        return fu, uu

    # ------------------------- Local search (SPSA + coord/pattern) -------------------------
    def local_refine(u0, f0, time_limit):
        """
        Hybrid local refinement in normalized space:
          - SPSA-like random +/-1 gradient probing with decaying step sizes
          - opportunistic coordinate pattern + tiny line-search for final polish
        """
        if u0 is None:
            return f0, u0
        t_end = min(deadline, time.time() + max(0.0, time_limit))

        u = u0[:]
        fu = f0

        # SPSA parameters in normalized coordinates
        a0 = 0.15
        c0 = 0.08
        a_min = 1e-6
        c_min = 1e-6

        k = 0
        # coordinate steps for final phase
        coord_step = 0.06
        coord_min = 1e-8

        def try_u(uc):
            fc, uc2 = evaluate_u(uc)
            return fc, uc2

        # SPSA phase
        while time.time() < t_end:
            rem = t_end - time.time()
            if rem <= 0:
                break

            k += 1
            # decay schedules
            ak = max(a_min, a0 / (1.0 + 0.07 * k))
            ck = max(c_min, c0 / (1.0 + 0.09 * k))

            # random Rademacher vector (+1/-1), skip degenerate dims by setting 0
            delta = [0.0] * dim
            for j in range(dim):
                if span[j] == 0.0:
                    delta[j] = 0.0
                else:
                    delta[j] = 1.0 if random.random() < 0.5 else -1.0

            up = [u[j] + ck * delta[j] for j in range(dim)]
            um = [u[j] - ck * delta[j] for j in range(dim)]
            fp, up = try_u(up)
            fm, um = try_u(um)

            if fp == float("inf") and fm == float("inf"):
                # can't learn anything; random tiny jitter
                uj = u[:]
                for j in range(dim):
                    uj[j] += 0.01 * gauss()
                fj, uj = try_u(uj)
                if fj < fu:
                    u, fu = uj, fj
                continue

            # gradient estimate and update
            # ghat_j ≈ (fp - fm) / (2*ck) * (1/delta_j)
            # with delta_j in {+1,-1}
            step = [0.0] * dim
            diff = (fp - fm) / (2.0 * ck + 1e-18)
            for j in range(dim):
                dj = delta[j]
                if dj != 0.0:
                    step[j] = -ak * diff * dj  # because 1/dj == dj
            ucand = [u[j] + step[j] for j in range(dim)]
            fc, ucand = try_u(ucand)

            if fc < fu:
                u, fu = ucand, fc
                # slight increase of aggressiveness when improving
                a0 = min(0.25, a0 * 1.03)
                c0 = min(0.15, c0 * 1.02)
            else:
                # decay more if stuck
                a0 = max(0.02, a0 * 0.985)
                c0 = max(0.01, c0 * 0.99)

            # if close to end or steps tiny, break to coord polish
            if rem < 0.15 * time_limit or (ak < 5e-4 and ck < 5e-4):
                break

        # Coordinate/pattern polish
        order = list(range(dim))
        no_imp = 0
        while time.time() < t_end:
            random.shuffle(order)
            improved = False

            for j in order:
                if time.time() >= t_end:
                    break
                if span[j] == 0.0:
                    continue
                if coord_step < coord_min:
                    continue

                # test +/- move
                best_local = fu
                best_uc = None
                best_dir = 0.0

                uc = u[:]
                uc[j] += coord_step
                fc, uc = try_u(uc)
                if fc < best_local:
                    best_local, best_uc, best_dir = fc, uc, +1.0

                uc = u[:]
                uc[j] -= coord_step
                fc, uc = try_u(uc)
                if fc < best_local:
                    best_local, best_uc, best_dir = fc, uc, -1.0

                if best_uc is not None:
                    u, fu = best_uc, best_local
                    improved = True

                    # tiny line-search along dir
                    alpha = 2.0
                    while time.time() < t_end and alpha * coord_step <= 0.5:
                        uc = u[:]
                        uc[j] += best_dir * alpha * coord_step
                        fc, uc = try_u(uc)
                        if fc < fu:
                            u, fu = uc, fc
                            alpha *= 1.7
                        else:
                            break

            if improved:
                coord_step = min(0.20, coord_step * 1.15)
                no_imp = 0
            else:
                coord_step *= 0.55
                no_imp += 1
                if no_imp >= 2 and coord_step < 3e-5:
                    break

        return fu, u

    # ------------------------- JADE-style DE with archive -------------------------
    pop = int(18 + 6 * math.sqrt(max(1, dim)))
    pop = max(24, min(110, pop))

    # init with diverse sampling: random + opposition + some corner-ish points
    U = []
    Fvals = []

    def push(u):
        fu, uu = consider(u)
        U.append(uu); Fvals.append(fu)

    # random and opposition
    init_target = pop
    while len(U) < init_target and time.time() < deadline:
        u = rand_u()
        push(u)
        if len(U) < init_target and time.time() < deadline:
            push([1.0 - ui for ui in u])

    # corner-ish
    for _ in range(min(2 * dim, 40)):
        if len(U) >= pop or time.time() >= deadline:
            break
        u = [0.01 if random.random() < 0.5 else 0.99 for _ in range(dim)]
        for __ in range(max(1, dim // 8)):
            u[random.randrange(dim)] = random.random()
        push(u)

    # trim best pop
    idx = list(range(len(U)))
    idx.sort(key=lambda i: Fvals[i])
    idx = idx[:pop]
    U = [U[i] for i in idx]
    Fvals = [Fvals[i] for i in idx]

    # JADE parameters
    mu_F = 0.55
    mu_CR = 0.80
    c_adapt = 0.10
    p_best = 0.20  # top p% candidates

    # archive for JADE
    A = []
    Amax = pop

    def cauchy(loc, scale):
        # inverse CDF cauchy
        u = random.random() - 0.5
        return loc + scale * math.tan(math.pi * u)

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    last_improve_t = time.time()
    last_best = best_f
    gen = 0

    while time.time() < deadline:
        gen += 1
        now = time.time()
        rem = deadline - now
        frac = (now - t0) / max(1e-12, max_time)

        if best_f < last_best - 1e-12:
            last_best = best_f
            last_improve_t = now

        # more local near the end or when stagnant
        stagn = now - last_improve_t
        if (frac > 0.72) or (rem < 0.25 * max_time) or (stagn > 0.18 * max_time and rem > 0.10 * max_time):
            if best_u is not None:
                slice_time = min(rem * 0.60, 0.28 * max_time)
                f2, u2 = local_refine(best_u, best_f, slice_time)
                if f2 < best_f:
                    best_f, best_u = f2, u2[:]
                    # inject into population (replace worst)
                    w = max(range(pop), key=lambda i: Fvals[i])
                    U[w] = best_u[:]
                    Fvals[w] = best_f
            # small jitter injection
            if time.time() < deadline and best_u is not None and random.random() < 0.50:
                uj = best_u[:]
                kick = 0.12 if frac < 0.85 else 0.06
                for j in range(dim):
                    uj[j] += kick * gauss()
                fj, uj = consider(uj)
                w = max(range(pop), key=lambda i: Fvals[i])
                if fj < Fvals[w]:
                    U[w] = uj[:]
                    Fvals[w] = fj
            continue

        # DE generation
        # sort indices by fitness for p-best selection
        order = list(range(pop))
        order.sort(key=lambda i: Fvals[i])
        pcount = max(2, int(p_best * pop))
        pset = order[:pcount]

        SF = []
        SCR = []
        dF = []  # fitness improvements for weighting

        # prepare combined pool for mutation selection (U + A)
        UA = U + A
        UA_len = len(UA)

        # random permutation for iteration
        it = list(range(pop))
        random.shuffle(it)

        for i in it:
            if time.time() >= deadline:
                break

            # sample Fi from cauchy, CRi from normal (approx via gauss)
            Fi = cauchy(mu_F, 0.10)
            # resample until valid
            tries = 0
            while (Fi <= 0.0) and tries < 8:
                Fi = cauchy(mu_F, 0.10)
                tries += 1
            Fi = clip(Fi, 0.05, 1.0)

            CRi = clip(mu_CR + 0.10 * gauss(), 0.0, 1.0)

            xi = U[i]

            # choose pbest
            pbi = random.choice(pset)
            xp = U[pbi]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop)
            x1 = U[r1]

            # choose r2 from UA, distinct from i and r1 (by index if within pop; best-effort)
            # We'll just resample and also avoid exact same object reference.
            r2 = None
            x2 = None
            for _ in range(20):
                k = random.randrange(UA_len)
                cand = UA[k]
                if cand is xi or cand is x1:
                    continue
                x2 = cand
                break
            if x2 is None:
                x2 = UA[random.randrange(UA_len)]

            # mutation: current-to-pbest/1 with archive
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (x1[j] - x2[j])

            # binomial crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    trial[j] = v[j]

            # bound handling: bounce-back toward xi (often better than clamp-only)
            for j in range(dim):
                tj = trial[j]
                if tj < 0.0:
                    trial[j] = 0.5 * (xi[j] + 0.0)
                elif tj > 1.0:
                    trial[j] = 0.5 * (xi[j] + 1.0)
            trial = clamp01(trial)

            ft, trial = evaluate_u(trial)
            if ft <= Fvals[i]:
                # success: add parent to archive, replace
                if len(A) < Amax:
                    A.append(xi[:])
                else:
                    A[random.randrange(Amax)] = xi[:]

                imp = Fvals[i] - ft
                U[i] = trial
                Fvals[i] = ft
                SF.append(Fi)
                SCR.append(CRi)
                dF.append(max(0.0, imp))

                if ft < best_f:
                    best_f = ft
                    best_u = trial[:]
                    last_improve_t = time.time()

        # adapt mu_F, mu_CR using successes
        if SF:
            # weights proportional to improvement (or uniform if all zero)
            s = sum(dF)
            if s <= 0.0:
                w = [1.0 / len(SF)] * len(SF)
            else:
                w = [di / s for di in dF]

            # Lehmer mean for F, weighted arithmetic for CR
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            F_lehmer = num / (den + 1e-18)

            CR_mean = 0.0
            for wi, cri in zip(w, SCR):
                CR_mean += wi * cri

            mu_F = (1.0 - c_adapt) * mu_F + c_adapt * clip(F_lehmer, 0.05, 1.0)
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * clip(CR_mean, 0.0, 1.0)

        # occasional micro-local on the current best individual
        if best_u is not None and (gen % 5 == 0) and rem > 0.18 * max_time:
            f2, u2 = local_refine(best_u, best_f, min(0.02 * max_time, rem * 0.06))
            if f2 < best_f:
                best_f, best_u = f2, u2[:]
                w = max(range(pop), key=lambda k: Fvals[k])
                U[w] = best_u[:]
                Fvals[w] = best_f

        # restart if very stagnant (keep elites, refill around best + random)
        if (time.time() - last_improve_t) > 0.28 * max_time and rem > 0.20 * max_time:
            # keep top 25%
            order = list(range(pop))
            order.sort(key=lambda i: Fvals[i])
            keep = max(6, pop // 4)
            elites = [U[i][:] for i in order[:keep]]
            elitesF = [Fvals[i] for i in order[:keep]]

            U, Fvals = elites, elitesF
            A = []  # clear archive on restart

            # refill
            while len(U) < pop and time.time() < deadline:
                if best_u is not None and random.random() < 0.75:
                    u = best_u[:]
                    # multi-scale
                    sig = 0.20 if random.random() < 0.35 else (0.08 if random.random() < 0.7 else 0.03)
                    sig *= (0.90 - 0.50 * frac)
                    sig = max(0.01, sig)
                    for j in range(dim):
                        u[j] += sig * gauss()
                    u = clamp01(u)
                else:
                    u = rand_u()
                fu, uu = consider(u)
                U.append(uu); Fvals.append(fu)

            last_improve_t = time.time()

    return best_f
