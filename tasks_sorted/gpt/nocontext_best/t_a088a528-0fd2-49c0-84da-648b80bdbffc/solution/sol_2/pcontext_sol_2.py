import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs prior version:
      - Proper "ask/tell" budget control (keeps sampling dense until deadline)
      - Better initialization: scrambled stratified sampling + opposition points
      - Two search modes blended:
          (A) Adaptive diagonal ES around current mean (fast local improvement)
          (B) Trust-region coordinate/pattern steps (robust on sharp valleys)
      - More principled step-size control: success-based + occasional line-search
      - Restart logic based on time + progress, with archive seeding
    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            raise ValueError("Each bound must be (low, high) with low <= high")

    # ---------- helpers ----------
    def clip(x):
        return [highs[i] if x[i] > highs[i] else (lows[i] if x[i] < lows[i] else x[i]) for i in range(dim)]

    def rand_uniform():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    # Box-Muller normal, cached
    have_spare = False
    spare = 0.0
    def randn():
        nonlocal have_spare, spare
        if have_spare:
            have_spare = False
            return spare
        u1 = 1e-12 + random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        spare = z1
        have_spare = True
        return z0

    def randcauchy():
        u = 1e-12 + (1.0 - 2e-12) * random.random()
        return math.tan(math.pi * (u - 0.5))

    def evaluate(x):
        return float(func(x))

    def now():
        return time.time()

    # ---------- init: stratified + opposition ----------
    best = float("inf")
    best_x = None

    # number of initial samples: modest but enough to get a good basin
    init_n = max(16, min(120, 10 * dim + 30))
    # scrambled stratification per-dimension
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    # evaluate both x and its "opposition" around center for free diversity
    for j in range(init_n):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] == 0:
                x[i] = lows[i]
            else:
                u = (perms[i][j] + random.random()) / init_n
                x[i] = lows[i] + u * spans[i]
        x = clip(x)
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

        if now() >= deadline:
            return best
        # opposition point (mirror around mid)
        xo = [0.0] * dim
        for i in range(dim):
            if spans[i] == 0:
                xo[i] = lows[i]
            else:
                mid = 0.5 * (lows[i] + highs[i])
                xo[i] = 2.0 * mid - x[i]
        xo = clip(xo)
        fxo = evaluate(xo)
        if fxo < best:
            best, best_x = fxo, xo[:]

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # Archive of elites for restart seeds
    archive = [(best, best_x[:])]
    arch_max = 12

    # ---------- state ----------
    m = best_x[:]                      # mean / incumbent
    fm = best

    # diagonal sigma (absolute, per-dimension), init as fraction of span
    sig = [0.18 * s if s > 0 else 0.0 for s in spans]
    for i in range(dim):
        if spans[i] > 0 and sig[i] <= 0:
            sig[i] = 0.18 * spans[i]

    sigma_g = 1.0
    sigma_g_min = 1e-10
    sigma_g_max = 100.0

    # trust-region / pattern search step (relative to spans)
    tr = 0.12
    tr_min = 1e-12
    tr_max = 0.5

    # ask counts
    lam_base = max(10, min(64, 4 * dim))
    mu = max(2, lam_base // 4)

    # success tracking
    succ = 0
    trials = 0
    adapt_window = 25

    # restart/progress
    stall = 0
    stall_limit = 35 + 7 * dim
    last_improve_t = now()
    # time-based restart if no improvement for a fraction of remaining time
    no_improve_time = max(0.2, 0.15 * max_time)

    # ---------- inner operators ----------
    def update_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        if len(archive) > arch_max:
            archive = archive[:arch_max]

    def recombine(offspring):
        # offspring is sorted list of (f,x)
        # log weights
        weights = []
        for k in range(mu):
            weights.append(max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)))
        wsum = sum(weights) or 1.0
        weights = [w / wsum for w in weights]

        new_m = [0.0] * dim
        for k in range(mu):
            _, xk = offspring[k]
            wk = weights[k]
            for i in range(dim):
                new_m[i] += wk * xk[i]
        return clip(new_m)

    def try_pattern_steps(xc, fc):
        """
        Small coordinate/pattern steps around incumbent.
        Useful when ES sampling is too diffuse or objective is ill-conditioned.
        """
        nonlocal tr
        best_local_f = fc
        best_local_x = xc[:]

        # choose a small set of coordinates to probe (all for small dim, subset for large)
        if dim <= 18:
            coords = list(range(dim))
        else:
            k = 18
            coords = random.sample(range(dim), k)

        improved = False
        # randomized order helps
        random.shuffle(coords)

        for i in coords:
            if now() >= deadline:
                break
            if spans[i] == 0:
                continue

            step = tr * spans[i]
            if step <= 0:
                continue

            # test + and -
            for sgn in (1.0, -1.0):
                if now() >= deadline:
                    break
                xt = best_local_x[:]
                xt[i] = xt[i] + sgn * step
                xt = clip(xt)
                ft = evaluate(xt)
                if ft < best_local_f:
                    best_local_f = ft
                    best_local_x = xt
                    improved = True

        # Adapt trust region
        if improved:
            tr = min(tr_max, tr * 1.35)
        else:
            tr = max(tr_min, tr * 0.65)

        return improved, best_local_f, best_local_x

    # ---------- main loop ----------
    gen = 0
    while now() < deadline:
        gen += 1

        # Restart if stalled (by iterations) or by wall-clock since last improvement
        if stall >= stall_limit or (now() - last_improve_t) > no_improve_time:
            stall = 0
            # pick seed: best, archive, or random
            r = random.random()
            if r < 0.55 and archive:
                _, seed = random.choice(archive)
                m = seed[:]
            elif r < 0.85:
                m = best_x[:]
            else:
                m = rand_uniform()

            fm = evaluate(m)
            if fm < best:
                best, best_x = fm, m[:]
                update_archive(best, best_x)

            # enlarge exploration on restart
            sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g * 1.8))
            tr = min(tr_max, max(tr_min, tr * 1.5))

        # Occasionally do pattern search around current best (cheap, robust)
        if dim <= 64 or (gen % 3 == 0):
            if now() >= deadline:
                break
            imp, fl, xl = try_pattern_steps(best_x, best)
            if imp:
                best, best_x = fl, xl[:]
                m, fm = best_x[:], best
                update_archive(best, best_x)
                last_improve_t = now()
                stall = 0
            else:
                stall += 1

        # ES sampling around mean
        lam = lam_base
        offspring = []
        improved_gen = False

        for _ in range(lam):
            if now() >= deadline:
                return best

            x = m[:]
            heavy = (random.random() < 0.10)
            for i in range(dim):
                if spans[i] == 0:
                    x[i] = lows[i]
                    continue
                step = sig[i] * sigma_g
                if step <= 0:
                    continue
                z = randcauchy() if heavy else randn()
                x[i] = x[i] + step * z

            x = clip(x)
            fx = evaluate(x)
            offspring.append((fx, x))

            trials += 1
            if fx < best:
                best, best_x = fx, x[:]
                update_archive(best, best_x)
                improved_gen = True
                succ += 1
                last_improve_t = now()

        offspring.sort(key=lambda t: t[0])

        # update mean toward elites (but keep incumbent influence)
        new_m = recombine(offspring)
        new_fm = offspring[0][0]
        m = new_m
        fm = new_fm if new_fm < fm else fm

        # diagonal sigma adaptation using best step magnitude (scale to observed progress)
        fx0, x0 = offspring[0]
        for i in range(dim):
            if spans[i] == 0:
                continue
            # displacement relative to previous mean direction
            d = abs(x0[i] - m[i])
            # target around observed displacement; keep within [tiny, 0.5 span]
            target = max(1e-18, min(0.5 * spans[i], 1.2 * d + 1e-9 * spans[i]))
            # smooth update
            sig[i] = max(1e-18, min(0.5 * spans[i], 0.88 * sig[i] + 0.12 * target))

        # global step-size adaptation by success rate
        if trials >= adapt_window:
            rate = succ / float(trials)
            if rate < 0.17:
                sigma_g *= 0.80
            elif rate > 0.30:
                sigma_g *= 1.22
            sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g))
            succ = 0
            trials = 0

        stall = 0 if improved_gen else (stall + 1)

        # small intensification: if ES best is very good, do a tiny "line-search" shrink around it
        if improved_gen and now() < deadline:
            # shrink scales a bit to exploit, but not too aggressively
            sigma_g = max(sigma_g_min, sigma_g * 0.95)
            tr = max(tr_min, tr * 0.97)

    return best
