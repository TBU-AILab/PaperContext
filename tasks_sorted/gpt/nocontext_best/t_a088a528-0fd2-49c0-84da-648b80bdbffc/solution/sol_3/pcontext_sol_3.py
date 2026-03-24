import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained).

    Improvements vs the provided ES:
      - Opposition-based + stratified init (more diverse, better first best)
      - (μ,λ)-ES with elitist archive + local refinement of the current best
      - Adaptive per-dimension step sizes using success information
      - Coordinate/pattern local search bursts near the best (cheap exploitation)
      - Occasional heavy-tail jumps + restarts with scheduled shrinking

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive")
    if len(bounds) != dim: 
        raise ValueError("bounds length must equal dim_")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for_phi=bounds]  # <-- fix? no external libs; keep simple
#
#Oops—let’s provide the actual working code (no typos) below.
#
import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Key mechanisms:
      - Stratified + opposition initialization
      - ES-style sampling around a moving center with adaptive diagonal steps
      - Local coordinate/pattern search around best_x (fast exploitation)
      - Heavy-tail jumps + restarts when stalled
    Returns:
      best fitness found (float)
    """
    start = time.time()
    deadline = start + float(max_time)

    if dim <= 0:
        raise ValueError("dim must be positive.")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim.")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")

    def clip(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_uniform():
        x = []
        for i in range(dim):
            if spans[i] <= 0:
                x.append(lows[i])
            else:
                x.append(lows[i] + random.random() * spans[i])
        return x

    # --- RNG helpers (fast, no libs) ---
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

    # -------- Initialization: stratified + opposition --------
    best = float("inf")
    best_x = None

    init_n = max(16, min(120, 10 * dim))
    # stratified per dimension using permuted bins
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            if spans[i] <= 0:
                x.append(lows[i])
            else:
                u = (perms[i][j] + random.random()) / init_n
                x.append(lows[i] + u * spans[i])

        # opposition point often helps on bounded problems
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        xo = clip(xo)

        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]
        if time.time() >= deadline:
            return best
        fxo = evaluate(xo)
        if fxo < best:
            best, best_x = fxo, xo[:]

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)

    # --------- Core state ----------
    m = best_x[:]  # search center
    # per-dimension step sizes (start moderate)
    sig = []
    for i in range(dim):
        s = spans[i]
        if s <= 0:
            sig.append(0.0)
        else:
            sig.append(max(1e-12 * s, 0.20 * s))

    sigma_g = 1.0
    sigma_g_min = 1e-8
    sigma_g_max = 100.0

    lam = max(14, min(80, 5 * dim))
    mu = max(3, lam // 5)

    # log weights for recombination
    weights = [max(0.0, math.log(mu + 0.5) - math.log(k + 1.0)) for k in range(mu)]
    wsum = sum(weights) if sum(weights) > 0 else 1.0
    weights = [w / wsum for w in weights]

    # archive of elites for restarts
    archive = [(best, best_x[:])]
    arch_max = 12

    # success counters for step adaptation
    succ = 0
    trials = 0
    adapt_window = 25

    stall = 0
    stall_limit = 18 + 5 * dim

    # ---------- Local refinement (coordinate/pattern search) ----------
    def local_refine(x, fx, budget_evals):
        """Coordinate search around x; returns possibly improved (x, fx)."""
        if budget_evals <= 0:
            return x, fx, 0
        evals = 0
        # initial deltas relative to span
        deltas = []
        for i in range(dim):
            if spans[i] <= 0:
                deltas.append(0.0)
            else:
                deltas.append(max(1e-12 * spans[i], 0.05 * spans[i]))

        improved = True
        while improved and evals < budget_evals and time.time() < deadline:
            improved = False
            # try each coordinate +/- delta
            for i in range(dim):
                if spans[i] <= 0:
                    continue
                d = deltas[i]
                if d <= 0:
                    continue

                # +d
                xp = x[:]
                xp[i] = min(highs[i], x[i] + d)
                fp = evaluate(xp); evals += 1
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue
                if evals >= budget_evals or time.time() >= deadline:
                    break

                # -d
                xm = x[:]
                xm[i] = max(lows[i], x[i] - d)
                fm = evaluate(xm); evals += 1
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

                if evals >= budget_evals or time.time() >= deadline:
                    break

            # if no improvement, shrink deltas
            if not improved:
                for i in range(dim):
                    deltas[i] *= 0.5
                # stop if steps are tiny
                tiny = True
                for i in range(dim):
                    if spans[i] > 0 and deltas[i] > 1e-10 * spans[i]:
                        tiny = False
                        break
                if tiny:
                    break

        return x, fx, evals

    # ---------- Main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # occasional local refine on the current best (cheap exploitation)
        if gen % max(10, 2 * dim) == 0 and time.time() < deadline:
            # keep it small; adapt to remaining time by fixed eval budget
            best_x2, best2, used = local_refine(best_x[:], best, budget_evals=6 * dim)
            if best2 < best:
                best, best_x = best2, best_x2[:]
                archive.append((best, best_x[:]))
                archive.sort(key=lambda t: t[0])
                archive = archive[:arch_max]
                m = best_x[:]
                stall = 0

        # restart logic when stalled
        if stall >= stall_limit:
            stall = 0
            r = random.random()
            if r < 0.50 and archive:
                _, base = random.choice(archive)
                m = base[:]
                sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g * 1.6))
            elif r < 0.85:
                m = best_x[:]
                sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g * 1.3))
            else:
                m = rand_uniform()
                sigma_g = 1.0
            # broaden per-dim a bit on restart
            for i in range(dim):
                if spans[i] > 0:
                    sig[i] = min(0.5 * spans[i], max(sig[i], 0.15 * spans[i]))

        # generate offspring
        offspring = []
        improved_gen = False

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            use_c = (random.random() < 0.10)  # heavy-tail sometimes
            x = m[:]
            for i in range(dim):
                if spans[i] <= 0:
                    x[i] = lows[i]
                    continue
                step = sig[i] * sigma_g
                if step <= 0:
                    continue
                z = randcauchy() if use_c else randn()
                x[i] = x[i] + step * z
                # reflect at bounds (often better than hard clip)
                if x[i] < lows[i]:
                    x[i] = lows[i] + (lows[i] - x[i])
                    if x[i] > highs[i]:
                        x[i] = lows[i]
                elif x[i] > highs[i]:
                    x[i] = highs[i] - (x[i] - highs[i])
                    if x[i] < lows[i]:
                        x[i] = highs[i]

            x = clip(x)
            fx = evaluate(x)
            offspring.append((fx, x))

            trials += 1
            if fx < best:
                best = fx
                best_x = x[:]
                improved_gen = True
                succ += 1
                archive.append((best, best_x[:]))
                archive.sort(key=lambda t: t[0])
                archive = archive[:arch_max]

        offspring.sort(key=lambda t: t[0])

        # recombine top mu to update center
        new_m = [0.0] * dim
        for k in range(mu):
            _, xk = offspring[k]
            wk = weights[k]
            for i in range(dim):
                new_m[i] += wk * xk[i]
        m = clip(new_m)

        # adapt per-dimension sig using successful directions (from best few)
        # compute mean absolute deviation of elites around new m
        dev = [0.0] * dim
        for k in range(mu):
            _, xk = offspring[k]
            for i in range(dim):
                dev[i] += abs(xk[i] - m[i])
        for i in range(dim):
            if spans[i] <= 0:
                continue
            dev[i] /= float(mu)
            # target step ~ elite spread, but bounded
            target = max(1e-12 * spans[i], min(0.35 * spans[i], 1.8 * dev[i] + 1e-15))
            # smooth update
            sig[i] = 0.85 * sig[i] + 0.15 * target
            # keep within limits
            sig[i] = max(1e-12 * spans[i], min(0.5 * spans[i], sig[i]))

        # global step-size adaptation by success rate
        if trials >= adapt_window:
            rate = succ / float(trials)
            if rate < 0.16:
                sigma_g *= 0.80
            elif rate > 0.30:
                sigma_g *= 1.22
            sigma_g = min(sigma_g_max, max(sigma_g_min, sigma_g))
            succ = 0
            trials = 0

        stall = 0 if improved_gen else (stall + 1)

    return best
