import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Upgrades vs previous version:
      - Better boundary handling: work in normalized [0,1]^d and map to bounds (no reflection artifacts).
      - Hybrid search:
          * DE (differential evolution) for robust global exploration early/mid.
          * Diagonal-separable NES/CMA-like refinement late (natural-gradient on mean + per-dim std).
          * Lightweight coordinate pattern search at the very end.
      - Adaptive population sizing from remaining time; evaluation-budget aware.
      - Stagnation detection with partial restarts and "opposition" samples.
      - Optional caching with quantization to avoid repeated evaluations.

    Returns:
      best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] if hi[i] > lo[i] else 0.0 for i in range(dim)]

    # map normalized u in [0,1] -> x in bounds
    def u_to_x(u):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= 0.0:
                x[i] = lo[i]
            else:
                # keep strictly inside bounds to avoid weird funcs at exact boundaries
                ui = u[i]
                if ui < 0.0: ui = 0.0
                elif ui > 1.0: ui = 1.0
                x[i] = lo[i] + ui * span[i]
        return x

    def clamp01(u):
        for i in range(dim):
            if u[i] < 0.0: u[i] = 0.0
            elif u[i] > 1.0: u[i] = 1.0
        return u

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

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    # quantized cache in u-space
    cache = {}
    def key_u(u):
        # 2^15-ish buckets per dim is too many; keep moderate
        # adaptive buckets: fewer in high dim
        buckets = 600 if dim <= 10 else (320 if dim <= 30 else 200)
        k = []
        for i in range(dim):
            q = int(u[i] * buckets)
            if q < 0: q = 0
            elif q > buckets: q = buckets
            k.append(q)
        return tuple(k)

    evals = 0
    def evaluate_u(u):
        nonlocal evals
        u = clamp01(u[:])
        k = key_u(u)
        if k in cache:
            return cache[k], u
        x = u_to_x(u)
        try:
            v = float(func(x))
            if not is_finite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        evals += 1
        return v, u

    # ---------- initialization ----------
    # Use a time-aware population size
    # (small enough for expensive funcs, large enough for robustness)
    base_pop = int(10 + 3 * math.sqrt(max(1, dim)))
    pop = max(16, min(72, base_pop))

    # initial sampling (Latin-ish: random + opposition)
    U = []
    F = []
    best_u = None
    best_f = float("inf")

    init_n = pop
    for i in range(init_n):
        if time.time() >= deadline:
            return best_f
        u = rand_u()
        fu, u = evaluate_u(u)
        U.append(u); F.append(fu)
        if fu < best_f:
            best_f = fu; best_u = u[:]

        # opposition point
        if time.time() >= deadline:
            return best_f
        uo = [1.0 - ui for ui in u]
        fo, uo = evaluate_u(uo)
        U.append(uo); F.append(fo)
        if fo < best_f:
            best_f = fo; best_u = uo[:]

    # trim to pop
    idx = list(range(len(F)))
    idx.sort(key=lambda i: F[i])
    idx = idx[:pop]
    U = [U[i] for i in idx]
    F = [F[i] for i in idx]

    last_best_f = best_f
    last_improve_time = time.time()
    restart_count = 0

    # ---------- DE parameters ----------
    # classic DE/rand/1/bin with slight adaptation
    Fm = 0.65
    Cr = 0.85

    # ---------- diagonal NES/CMA-like parameters in u-space ----------
    # mean initialized from best half
    def compute_mean_std():
        mu = max(2, pop // 3)
        # weights (log)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        # sort by F
        order = list(range(pop))
        order.sort(key=lambda i: F[i])
        mean = [0.0] * dim
        for j in range(mu):
            uj = U[order[j]]
            for d in range(dim):
                mean[d] += w[j] * uj[d]
        # per-dim std from selected
        var = [1e-6] * dim
        for d in range(dim):
            m = mean[d]
            acc = 0.0
            for j in range(mu):
                uj = U[order[j]][d]
                diff = uj - m
                acc += w[j] * diff * diff
            var[d] = max(1e-8, acc)
        std = [math.sqrt(v) for v in var]
        return mean, std

    mean_u, std_u = compute_mean_std()

    # learning rates
    lr_mean = 0.25
    lr_std = 0.20

    # ---------- local pattern search in u-space ----------
    def pattern_search_u(u0, f0, budget=40):
        u = u0[:]
        fu = f0
        # step sizes: tied to std, but not too tiny
        step = [max(1e-6, min(0.25, 0.7 * std_u[i])) for i in range(dim)]
        used = 0
        while used < budget and time.time() < deadline:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if used >= budget or time.time() >= deadline:
                    break
                si = step[i]
                if si <= 1e-10:
                    continue
                up = u[:]; up[i] += si
                um = u[:]; um[i] -= si
                fp, up = evaluate_u(up); used += 1
                if fp < fu:
                    u, fu = up, fp
                    improved = True
                    continue
                if used >= budget or time.time() >= deadline:
                    break
                fm, um = evaluate_u(um); used += 1
                if fm < fu:
                    u, fu = um, fm
                    improved = True
            if not improved:
                for i in range(dim):
                    step[i] *= 0.5
                if max(step) < 1e-8:
                    break
        return fu, u

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = (time.time() - t0) / max(1e-12, max_time)

        # stagnation handling
        if best_f < last_best_f - 1e-12:
            last_best_f = best_f
            last_improve_time = time.time()

        stagn = time.time() - last_improve_time
        stagn_lim = (0.22 + 0.08 * min(3, restart_count)) * max_time
        if stagn > stagn_lim:
            restart_count += 1
            last_improve_time = time.time()

            # partial restart: keep top k, refill rest with random/opposition around best
            order = list(range(pop))
            order.sort(key=lambda i: F[i])
            keep = max(6, pop // 4)
            U = [U[i] for i in order[:keep]]
            F = [F[i] for i in order[:keep]]

            while len(U) < pop and time.time() < deadline:
                if random.random() < 0.6 and best_u is not None:
                    # jitter around best in u-space
                    u = best_u[:]
                    # jitter scale grows with restart_count
                    scale = min(0.35, 0.10 + 0.07 * min(4, restart_count))
                    for d in range(dim):
                        u[d] += scale * (random.random() - 0.5)
                    u = clamp01(u)
                else:
                    u = rand_u()

                fu, u = evaluate_u(u)
                U.append(u); F.append(fu)
                if fu < best_f:
                    best_f, best_u = fu, u[:]

                if len(U) < pop and time.time() < deadline and random.random() < 0.35:
                    uo = [1.0 - ui for ui in u]
                    fo, uo = evaluate_u(uo)
                    U.append(uo); F.append(fo)
                    if fo < best_f:
                        best_f, best_u = fo, uo[:]

            mean_u, std_u = compute_mean_std()
            # slightly inflate std to re-explore
            for d in range(dim):
                std_u[d] = min(0.45, max(std_u[d], 0.10))

        # phase schedule: DE early/mid, NES late
        use_de = (elapsed < 0.70)

        # --- DE generation ---
        if use_de:
            # mild adaptation
            Fm = 0.55 + 0.25 * random.random()
            Cr = 0.70 + 0.25 * random.random()

            order = list(range(pop))
            random.shuffle(order)
            for idx_i in order:
                if time.time() >= deadline:
                    break
                # choose r1,r2,r3 distinct
                r1 = r2 = r3 = idx_i
                while r1 == idx_i:
                    r1 = random.randrange(pop)
                while r2 == idx_i or r2 == r1:
                    r2 = random.randrange(pop)
                while r3 == idx_i or r3 == r1 or r3 == r2:
                    r3 = random.randrange(pop)

                a = U[r1]; b = U[r2]; c = U[r3]
                target = U[idx_i]

                # mutant
                v = [0.0] * dim
                for d in range(dim):
                    v[d] = a[d] + Fm * (b[d] - c[d])

                # binomial crossover
                jrand = random.randrange(dim)
                trial = target[:]
                for d in range(dim):
                    if random.random() < Cr or d == jrand:
                        trial[d] = v[d]

                trial = clamp01(trial)
                ft, trial = evaluate_u(trial)
                if ft <= F[idx_i]:
                    U[idx_i] = trial
                    F[idx_i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_u = trial[:]

            mean_u, std_u = compute_mean_std()

        # --- NES/CMA-like refinement ---
        else:
            # update from elite samples by weighted z-scores
            order = list(range(pop))
            order.sort(key=lambda i: F[i])
            mu = max(3, pop // 3)

            # log weights
            w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            ws = sum(w)
            w = [wi / ws for wi in w]

            # compute gradients in standardized coordinates
            g_mean = [0.0] * dim
            g_std = [0.0] * dim

            # protect std
            for d in range(dim):
                std_u[d] = min(0.5, max(1e-4, std_u[d]))

            for j in range(mu):
                uj = U[order[j]]
                for d in range(dim):
                    z = (uj[d] - mean_u[d]) / std_u[d]
                    g_mean[d] += w[j] * z
                    # encourage variance along good directions: (z^2 - 1)
                    g_std[d] += w[j] * (z * z - 1.0)

            # apply updates
            for d in range(dim):
                mean_u[d] += lr_mean * std_u[d] * g_mean[d]
            mean_u = clamp01(mean_u)

            for d in range(dim):
                # multiplicative std update; keep bounded
                std_u[d] *= math.exp(0.5 * lr_std * g_std[d])
                std_u[d] = min(0.45, max(1e-4, std_u[d]))

            # sample a few candidates around updated mean to refresh population
            # replace worst individuals
            n_new = max(6, pop // 3)
            worst = order[-n_new:]
            for wi in worst:
                if time.time() >= deadline:
                    break
                u = [0.0] * dim
                for d in range(dim):
                    u[d] = mean_u[d] + std_u[d] * gauss()
                u = clamp01(u)
                fu, u = evaluate_u(u)
                U[wi] = u
                F[wi] = fu
                if fu < best_f:
                    best_f = fu
                    best_u = u[:]

            # occasional best-centered exploitation
            if best_u is not None and random.random() < 0.25:
                u = best_u[:]
                for d in range(dim):
                    u[d] += 0.25 * std_u[d] * gauss()
                u = clamp01(u)
                fu, u = evaluate_u(u)
                # insert if good
                if fu < max(F):
                    k = max(range(pop), key=lambda i: F[i])
                    U[k] = u
                    F[k] = fu
                if fu < best_f:
                    best_f = fu
                    best_u = u[:]

        # late-stage pattern search
        if best_u is not None and elapsed > 0.85 and (gen % 4 == 0) and (time.time() + 0.005 < deadline):
            bud = 25 if elapsed < 0.93 else 55
            bud += 10 * min(3, restart_count)
            fl, ul = pattern_search_u(best_u, best_f, budget=bud)
            if fl < best_f:
                best_f = fl
                best_u = ul[:]
                # tighten mean/std around best
                mean_u = best_u[:]
                for d in range(dim):
                    std_u[d] = max(1e-4, 0.6 * std_u[d])

    return best_f
