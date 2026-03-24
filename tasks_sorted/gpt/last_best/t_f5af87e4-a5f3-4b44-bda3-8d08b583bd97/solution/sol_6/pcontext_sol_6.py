import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stdlib-only, time-bounded minimizer focused on robustness + fast improvement.

    Core idea: Differential Evolution (current-to-pbest/1/bin) with:
      - diverse init (Halton + stratified + random + opposition)
      - jDE-style self-adaptation of F and CR per individual
      - p-best selection + archive (JADE/SHADE-like) to keep exploration strong
      - bound handling via reflection (better dynamics than clamp)
      - periodic local search (coordinate/pattern + small random directions)
      - stagnation-triggered partial restart

    Returns:
      best (float): best objective value found within max_time
    """
    t0 = time.perf_counter()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            v = float(func([]))
            return v if (not math.isnan(v) and not math.isinf(v)) else float("inf")
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def is_finite(x):
        return not (math.isnan(x) or math.isinf(x))

    def reflect_inplace(x):
        # reflect each coordinate into [lo, hi] with period 2*(hi-lo)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            v = x[i]
            r = hi - lo
            p = 2.0 * r
            y = (v - lo) % p
            if y < 0.0:
                y += p
            if y <= r:
                x[i] = lo + y
            else:
                x[i] = hi - (y - r)

    def eval_f(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        return v if is_finite(v) else float("inf")

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def center_point():
        return [0.5 * (lows[i] + highs[i]) for i in range(dim)]

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # fast ~N(0,1) via 12 uniforms - 6
    def randn():
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    # ---------------- Halton init ----------------
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173
    ]

    def radical_inverse(n, base):
        inv = 1.0 / base
        f = inv
        r = 0.0
        while n > 0:
            n, mod = divmod(n, base)
            r += mod * f
            f *= inv
        return r

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            base = primes[i % len(primes)]
            u = radical_inverse(index, base)
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- population init ----------------
    pop_size = max(22, min(120, 12 + 6 * dim))
    pop = []
    fit = []

    best = float("inf")
    best_x = None

    def push(x):
        nonlocal best, best_x
        reflect_inplace(x)
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best = fx
            best_x = list(x)

    # seed with center + opposite
    if time.perf_counter() < deadline:
        push(center_point())
    if time.perf_counter() < deadline:
        push(opposite_point(center_point()))

    halton_start = 1 + random.randrange(1, 20000)
    bins = max(8, pop_size)

    k = 0
    while len(pop) < pop_size and time.perf_counter() < deadline:
        r = random.random()
        if r < 0.45:
            x = halton_point(halton_start + k)
        elif r < 0.80:
            # stratified-like
            x = [0.0] * dim
            for i in range(dim):
                if spans[i] == 0.0:
                    x[i] = lows[i]
                else:
                    b = (k + 17 * i) % bins
                    u = (b + random.random()) / float(bins)
                    x[i] = lows[i] + u * spans[i]
        else:
            x = rand_point()
        k += 1
        push(list(x))

        # opposition pair sometimes
        if len(pop) < pop_size and time.perf_counter() < deadline and random.random() < 0.65:
            push(opposite_point(x))

    if not pop:
        return float("inf")

    # Ensure diversity if tiny n
    while len(pop) < 6 and time.perf_counter() < deadline:
        push(rand_point())

    # ---------------- DE with archive + self-adapt ----------------
    n = len(pop)

    # per-individual parameters (jDE-ish)
    F = [0.5 + 0.3 * random.random() for _ in range(n)]
    CR = [0.5 + 0.4 * random.random() for _ in range(n)]

    # archive of replaced solutions to improve difference vectors
    archive = []
    archive_cap = max(n, 2 * n)

    # p-best fraction (adaptive-ish)
    pmin, pmax = 0.08, 0.30

    # stagnation control
    last_improve_time = time.perf_counter()
    stall_seconds = max(0.12, 0.18 * float(max_time))

    # ---------------- local search ----------------
    def local_refine(x0, f0, eval_budget=28):
        nonlocal best, best_x
        x = list(x0)
        fx = float(f0)
        step = 0.08
        evals = 0

        # include occasional random direction tries (helps in rotated basins)
        while evals < eval_budget and time.perf_counter() < deadline:
            improved = False

            # coordinate pattern
            coords = list(range(dim))
            random.shuffle(coords)
            for i in coords:
                if evals >= eval_budget or time.perf_counter() >= deadline:
                    break
                if spans[i] == 0.0:
                    continue
                delta = step * safe_spans[i]

                xp = list(x); xp[i] += delta
                reflect_inplace(xp)
                fp = eval_f(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    if fp < best:
                        best, best_x = fp, list(x)
                    continue

                xm = list(x); xm[i] -= delta
                reflect_inplace(xm)
                fm = eval_f(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    if fm < best:
                        best, best_x = fm, list(x)

            # small number of random direction probes
            if evals < eval_budget and time.perf_counter() < deadline:
                if random.random() < 0.35:
                    d = [randn() for _ in range(dim)]
                    # normalize direction
                    norm = math.sqrt(sum(v*v for v in d)) or 1.0
                    for j in range(dim):
                        d[j] /= norm
                    # try +/- along direction
                    scale = step
                    xp = [x[j] + scale * safe_spans[j] * d[j] for j in range(dim)]
                    reflect_inplace(xp)
                    fp = eval_f(xp); evals += 1
                    if fp < fx:
                        x, fx = xp, fp
                        improved = True
                        if fp < best:
                            best, best_x = fp, list(x)
                    elif evals < eval_budget:
                        xm = [x[j] - scale * safe_spans[j] * d[j] for j in range(dim)]
                        reflect_inplace(xm)
                        fm = eval_f(xm); evals += 1
                        if fm < fx:
                            x, fx = xm, fm
                            improved = True
                            if fm < best:
                                best, best_x = fm, list(x)

            if not improved:
                step *= 0.55
                if step < 1e-8:
                    break
        return x, fx

    # Utility: pick r != excluded
    def rand_index_excluding(n, excl_set):
        j = random.randrange(n)
        while j in excl_set:
            j = random.randrange(n)
        return j

    def choose_pbest_index(sorted_idx, p):
        # choose uniformly among top p fraction
        top = max(2, int(math.ceil(p * len(sorted_idx))))
        return sorted_idx[random.randrange(top)]

    # ---------------- main loop ----------------
    gen = 0
    while time.perf_counter() < deadline:
        gen += 1
        n = len(pop)

        # rank for pbest selection
        order = sorted(range(n), key=lambda i: fit[i])
        if fit[order[0]] < best:
            best = fit[order[0]]
            best_x = list(pop[order[0]])

        # time-based schedule for p (more exploitation near end)
        remaining = deadline - time.perf_counter()
        frac_left = remaining / max(1e-12, float(max_time))
        p = pmin + (pmax - pmin) * (1.0 - min(1.0, frac_left * 1.2))

        # for each target
        for i in range(n):
            if time.perf_counter() >= deadline:
                return float(best)

            # jDE parameter adaptation
            if random.random() < 0.12:
                # F: keep in (0,1.2], prefer mid-high
                Fi = 0.25 + 0.95 * random.random()
            else:
                Fi = F[i]
            if random.random() < 0.10:
                CRi = random.random()
            else:
                CRi = CR[i]

            # current-to-pbest/1 with archive
            pbest = choose_pbest_index(order, p)

            r1 = rand_index_excluding(n, {i, pbest})
            # r2 from pop U archive
            use_archive = (len(archive) > 0 and random.random() < 0.5)
            if use_archive:
                r2_is_arch = True
                xr2 = archive[random.randrange(len(archive))]
            else:
                r2_is_arch = False
                r2 = rand_index_excluding(n, {i, pbest, r1})
                xr2 = pop[r2]

            xi = pop[i]
            xpb = pop[pbest]
            xr1 = pop[r1]

            v = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    v[d] = lows[d]
                else:
                    v[d] = xi[d] + Fi * (xpb[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # binomial crossover
            u = list(xi)
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
            reflect_inplace(u)

            fu = eval_f(u)

            if fu <= fit[i]:
                # add replaced to archive
                if len(archive) < archive_cap:
                    archive.append(pop[i])
                else:
                    archive[random.randrange(archive_cap)] = pop[i]

                pop[i] = u
                fit[i] = fu
                F[i] = Fi
                CR[i] = CRi

                if fu < best:
                    best = fu
                    best_x = list(u)
                    last_improve_time = time.perf_counter()

        # shrink archive if needed
        if len(archive) > archive_cap:
            # random downsample
            random.shuffle(archive)
            del archive[archive_cap:]

        now = time.perf_counter()

        # occasional local refinement (more often near end)
        if best_x is not None:
            if (deadline - now) < 0.20 * float(max_time):
                _, _ = local_refine(best_x, best, eval_budget=34)
            elif random.random() < 0.08:
                _, _ = local_refine(best_x, best, eval_budget=16)

        # stagnation: partial restart of worst half, keep best quarter
        if (now - last_improve_time) > stall_seconds and time.perf_counter() < deadline:
            last_improve_time = now
            order = sorted(range(n), key=lambda i: fit[i])
            keep = max(4, n // 4)

            # keep best; reinit worst 1/2 around best or globally
            for idx in order[keep:]:
                if time.perf_counter() >= deadline:
                    break
                if best_x is not None and random.random() < 0.75:
                    x = list(best_x)
                    for d in range(dim):
                        if spans[d] != 0.0:
                            x[d] += (0.22 * safe_spans[d]) * randn()
                    reflect_inplace(x)
                else:
                    x = rand_point()
                pop[idx] = x
                fit[idx] = eval_f(x)
                F[idx] = 0.4 + 0.5 * random.random()
                CR[idx] = random.random()

                if fit[idx] < best:
                    best = fit[idx]
                    best_x = list(pop[idx])

            # refresh archive (old info may be misleading after restart)
            archive.clear()

    return float(best)
