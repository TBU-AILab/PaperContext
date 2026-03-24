import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs previous version:
      - Uses an explicit evaluation budget split: global search (DE) + strong local search (pattern/line search).
      - Local search is upgraded to a coordinate/pattern search with adaptive per-dimension step sizes
        + opportunistic 1D line-search along improving directions.
      - Better handling of bounds/degenerate spans and "bad" (nan/inf/exception) evaluations.
      - Smarter restarts: population is periodically rebuilt around best with multi-scale perturbations.
      - Lighter/faster cache with fixed-size ring eviction.

    Returns:
      best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ------------------------- basic helpers -------------------------
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if not (span[i] > 0.0):
            span[i] = 0.0  # degenerate dimension

    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def clamp01(u):
        for i in range(dim):
            ui = u[i]
            if ui < 0.0:
                u[i] = 0.0
            elif ui > 1.0:
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

    # ------------------------- evaluation cache -------------------------
    # Rounded key cache, ring eviction (fast & bounded memory).
    # This helps local search and DE revisits without heavy overhead.
    dec = 6 if dim <= 12 else (5 if dim <= 30 else 4)
    CACHE_MAX = 9000 if dim <= 20 else (6000 if dim <= 60 else 3500)
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

        # evict old
        old = ring[ring_pos]
        if old is not None and old in cache:
            del cache[old]
        ring[ring_pos] = k
        ring_pos += 1
        if ring_pos >= CACHE_MAX:
            ring_pos = 0

        cache[k] = fv
        return fv, u

    # ------------------------- local search: adaptive pattern + line search -------------------------
    def local_search(u0, f0, time_limit):
        """
        Adaptive coordinate/pattern search in normalized space:
          - maintains per-dimension step sizes
          - tries +/- moves; if improvement found, optionally line-search further along same direction
          - shrinks step sizes on failure, expands slightly on success
        Very robust in bounded black-box settings.
        """
        if u0 is None:
            return f0, u0

        t_end = min(deadline, time.time() + max(0.0, time_limit))

        u = u0[:]
        fu = f0

        # initial per-dimension steps
        base = 0.20
        steps = [base] * dim
        step_min = 1e-8
        step_max = 0.50

        # prioritize more "active" dims sometimes by shuffling order
        order = list(range(dim))
        no_improve_sweeps = 0

        def try_step(dir_vec, alpha):
            uc = u[:]
            for j, dj in dir_vec:
                uc[j] += alpha * dj
            uc = clamp01(uc)
            fc, uc = evaluate_u(uc)
            return fc, uc

        while time.time() < t_end:
            random.shuffle(order)
            improved = False

            for j in order:
                if time.time() >= t_end:
                    break

                if span[j] == 0.0:
                    continue

                sj = steps[j]
                if sj < step_min:
                    continue

                # Try +/- coordinate
                best_fc = fu
                best_uc = None
                best_dir = 0.0

                # + step
                fc, uc = try_step([(j, 1.0)], sj)
                if fc < best_fc:
                    best_fc, best_uc, best_dir = fc, uc, +1.0

                # - step
                fc, uc = try_step([(j, 1.0)], -sj)
                if fc < best_fc:
                    best_fc, best_uc, best_dir = fc, uc, -1.0

                if best_uc is not None:
                    # accept coordinate improvement
                    u, fu = best_uc, best_fc
                    improved = True

                    # opportunistic line search along same coordinate direction
                    # geometric expansion until no improvement / boundary / time
                    alpha = 2.0
                    while time.time() < t_end and alpha * sj <= 0.75:
                        fc2, uc2 = try_step([(j, 1.0)], best_dir * alpha * sj)
                        if fc2 < fu:
                            u, fu = uc2, fc2
                            alpha *= 1.8
                        else:
                            break

                    # step adaptation
                    steps[j] = min(step_max, steps[j] * 1.35)
                else:
                    # no improvement in that dim -> shrink a bit
                    steps[j] = max(step_min, steps[j] * 0.70)

            if improved:
                no_improve_sweeps = 0
            else:
                no_improve_sweeps += 1
                # if a full sweep didn't help, shrink all steps
                for j in range(dim):
                    steps[j] = max(step_min, steps[j] * 0.60)
                if no_improve_sweeps >= 2 and max(steps) < 5e-5:
                    break

        return fu, u

    # ------------------------- global search: DE/current-to-best + restart -------------------------
    pop = int(16 + 5 * math.sqrt(max(1, dim)))
    pop = max(20, min(100, pop))

    U = []
    F = []
    best_u = None
    best_f = float("inf")

    def consider(u):
        nonlocal best_f, best_u
        fu, u = evaluate_u(u)
        if fu < best_f:
            best_f, best_u = fu, u[:]
        return fu, u

    # initialization: random + opposition + some near-corners
    init_n = pop
    for _ in range(init_n):
        if time.time() >= deadline:
            return best_f
        fu, u = consider(rand_u())
        U.append(u); F.append(fu)

        if time.time() >= deadline:
            return best_f
        uo = [1.0 - ui for ui in u]
        fo, uo = consider(uo)
        U.append(uo); F.append(fo)

    # near-corners
    for _ in range(min(3 * dim, 30)):
        if time.time() >= deadline:
            return best_f
        u = [0.02 if random.random() < 0.5 else 0.98 for _ in range(dim)]
        for __ in range(max(1, dim // 7)):
            j = random.randrange(dim)
            u[j] = random.random()
        fu, u = consider(u)
        U.append(u); F.append(fu)

    # trim to pop
    idx = list(range(len(F)))
    idx.sort(key=lambda i: F[i])
    idx = idx[:pop]
    U = [U[i] for i in idx]
    F = [F[i] for i in idx]

    # DE parameter memories
    Fm = 0.65
    Crm = 0.85

    last_best = best_f
    last_improve_t = time.time()
    restarts = 0
    gen = 0

    while time.time() < deadline:
        gen += 1
        now = time.time()
        rem = deadline - now
        elapsed = (now - t0) / max(1e-12, max_time)

        # update stagnation tracker
        if best_f < last_best - 1e-12:
            last_best = best_f
            last_improve_t = now

        # If late in time: emphasize local search
        if rem < 0.30 * max_time or elapsed > 0.72:
            if best_u is not None:
                # spend a meaningful chunk of remaining time locally
                slice_time = min(rem * 0.55, 0.22 * max_time)
                f2, u2 = local_search(best_u, best_f, slice_time)
                if f2 < best_f:
                    best_f, best_u = f2, u2[:]
                    # inject into population (replace worst)
                    w = max(range(pop), key=lambda i: F[i])
                    U[w] = best_u[:]
                    F[w] = best_f

            # small random kicks around best (helps escape shallow basins)
            if time.time() < deadline and best_u is not None and random.random() < 0.40:
                u = best_u[:]
                kick = 0.10 if elapsed < 0.85 else 0.05
                for j in range(dim):
                    u[j] += kick * gauss()
                fu, u = consider(u)
                w = max(range(pop), key=lambda i: F[i])
                if fu < F[w]:
                    U[w] = u[:]
                    F[w] = fu
            continue

        # restart if stagnating and still time left
        stagn = now - last_improve_t
        if stagn > (0.16 + 0.06 * min(5, restarts)) * max_time and rem > 0.12 * max_time:
            restarts += 1
            last_improve_t = now

            # keep elites, refill around best with multi-scale noise + some randoms
            order = list(range(pop))
            order.sort(key=lambda i: F[i])
            keep = max(6, pop // 4)
            elites = [U[i][:] for i in order[:keep]]
            elitesF = [F[i] for i in order[:keep]]

            U, F = elites, elitesF

            # multi-scale sigma schedule
            sigma1 = max(0.03, 0.18 / (1.0 + 0.45 * restarts))
            sigma2 = max(0.01, sigma1 * 0.35)

            while len(U) < pop and time.time() < deadline:
                r = random.random()
                if best_u is not None and r < 0.70:
                    u = best_u[:]
                    sig = sigma1 if random.random() < 0.65 else sigma2
                    for j in range(dim):
                        u[j] += sig * gauss()
                else:
                    u = rand_u()

                fu, u = consider(u)
                U.append(u); F.append(fu)

                # opposition sometimes
                if len(U) < pop and random.random() < 0.20:
                    uo = [1.0 - ui for ui in u]
                    fo, uo = consider(uo)
                    U.append(uo); F.append(fo)

            # after restart, do a quick local polish on best
            if best_u is not None and time.time() < deadline:
                f2, u2 = local_search(best_u, best_f, min(0.05 * max_time, rem * 0.10))
                if f2 < best_f:
                    best_f, best_u = f2, u2[:]

        # one DE generation
        best_i = min(range(pop), key=lambda i: F[i])
        if F[best_i] < best_f:
            best_f = F[best_i]
            best_u = U[best_i][:]

        # sample DE parameters
        def sample_F():
            v = Fm + 0.25 * gauss()
            if v < 0.10: v = 0.10
            if v > 1.00: v = 1.00
            return v

        def sample_Cr():
            v = Crm + 0.20 * gauss()
            if v < 0.00: v = 0.00
            if v > 1.00: v = 1.00
            return v

        indices = list(range(pop))
        random.shuffle(indices)

        succF = []
        succCr = []

        for i in indices:
            if time.time() >= deadline:
                break

            Fi = sample_F()
            Cri = sample_Cr()

            # pick r1,r2 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop)

            xi = U[i]
            xb = U[best_i]
            x1 = U[r1]
            x2 = U[r2]

            # mutation: current-to-best/1
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xb[j] - xi[j]) + Fi * (x1[j] - x2[j])

            # crossover: binomial
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if random.random() < Cri or j == jrand:
                    trial[j] = v[j]
            trial = clamp01(trial)

            ft, trial = evaluate_u(trial)
            if ft <= F[i]:
                U[i] = trial
                F[i] = ft
                succF.append(Fi)
                succCr.append(Cri)
                if ft < best_f:
                    best_f = ft
                    best_u = trial[:]
                    best_i = i

        # update memories (if any success)
        if succF:
            # Lehmer mean for F (good for DE), arithmetic for Cr
            num = 0.0
            den = 0.0
            for f in succF:
                num += f * f
                den += f
            Fm = num / (den + 1e-12)
            if Fm < 0.15: Fm = 0.15
            if Fm > 0.95: Fm = 0.95
            Crm = sum(succCr) / len(succCr)
            if Crm < 0.05: Crm = 0.05
            if Crm > 0.98: Crm = 0.98

        # occasional cheap local improvement mid-run
        if best_u is not None and (gen % 4 == 0) and rem > 0.15 * max_time:
            f2, u2 = local_search(best_u, best_f, min(0.03 * max_time, rem * 0.06))
            if f2 < best_f:
                best_f, best_u = f2, u2[:]
                w = max(range(pop), key=lambda k: F[k])
                U[w] = best_u[:]
                F[w] = best_f

    return best_f
