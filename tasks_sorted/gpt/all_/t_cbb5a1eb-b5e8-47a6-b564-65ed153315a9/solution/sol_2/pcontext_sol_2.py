import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no external libs).

    What is improved vs the provided algorithms:
      1) Stronger global search: low-discrepancy Halton + opposition points + occasional global probes
      2) Better local search: (mu,lambda)-ES around an archive of elites (not just 1+1)
      3) More robust step-size control: log-normal sigma adaptation + per-dimension scaling memory
      4) Cheap, reliable refinement: coordinate pattern search around the best (bounded cost)
      5) Time-aware scheduling: spends early budget exploring and late budget exploiting

    Returns:
      best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------------- bounds prep ----------------
    if dim <= 0:
        # still call func once if possible
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    # Replace zero spans with 1.0 for step scaling; value itself is fixed by bounds anyway.
    spans_nz = [s if s > 0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Box-Muller N(0,1)
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare = z1
        _has_spare = True
        return z0

    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def mirror_point(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    def opposite_point(x):
        # "Opposition-based" point: x_op = lo + hi - x
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        return mirror_point(xo)

    # ---------------- Halton sequence ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- elite archive ----------------
    # Store small set of best points; sample parents from it.
    elite_size = max(4, min(18, 4 + int(2 * math.sqrt(dim))))
    elites = []  # list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        # Insert keeping sorted; small sizes so O(n) is fine.
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        # If clearly worse than worst and archive full, skip quickly
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        # Insert
        inserted = False
        for i in range(len(elites)):
            if fx < elites[i][0]:
                elites.insert(i, item)
                inserted = True
                break
        if not inserted:
            elites.append(item)
        if len(elites) > elite_size:
            elites.pop()

    def get_best():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # Time-aware init budget
    # Keep it moderate; the rest of time will be iterative ES + refinement.
    init_n = max(16, min(140, 20 + 10 * int(math.sqrt(dim))))
    k_halton = 1

    for _ in range(init_n):
        if now() >= deadline:
            return best
        # mostly Halton, sometimes uniform
        if random.random() < 0.85:
            x = halton_point(k_halton)
            k_halton += 1
        else:
            x = rand_uniform_point()

        fx = evaluate(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        # also try opposition (often helps early)
        if now() >= deadline:
            return best
        xo = opposite_point(x)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]
        push_elite(best, best_x)

    # ---------------- ES parameters ----------------
    # (mu, lambda)-ES with adaptive sigma and per-dimension scaling memory.
    mu = max(2, min(8, 2 + int(math.sqrt(dim))))               # parents
    lam = max(6, min(30, 8 + 3 * int(math.sqrt(dim))))         # offspring per generation
    if lam <= mu:
        lam = mu + 4

    # Global sigma (relative), and per-dimension multipliers to learn anisotropy
    sigma = 0.25
    sigma_min = 1e-12
    sigma_max = 0.9
    # per-dimension scale memory (updated on successful moves)
    scale = [1.0] * dim

    # Success tracking
    gen = 0
    no_best_gens = 0

    # Pattern-search refinement controls (cheap coordinate probe around best)
    # Run it periodically, more often near the end.
    refine_period = max(6, 2 + int(math.sqrt(dim)))
    refine_budget_coords = max(1, min(dim, 8))  # limit cost in high dim

    # ---------------- main loop ----------------
    while now() < deadline:
        gen += 1
        # Select parents from elites (biased to best)
        # Use rank-based sampling: pick among top M where M grows slightly with archive size.
        M = min(len(elites), max(mu, 2 * mu))
        if M <= 0:
            # fallback
            parent = rand_uniform_point()
            f_parent = evaluate(parent)
            push_elite(f_parent, parent)
            best, best_x = get_best()
            continue

        parents = []
        for _ in range(mu):
            # triangular distribution towards 0 (best)
            r = random.random()
            idx = int((r * r) * M)  # bias to smaller indices
            parents.append(elites[idx][1])

        # Generate offspring
        offspring = []
        for _ in range(lam):
            if now() >= deadline:
                break

            # Recombine: choose a base parent
            base = parents[int(random.random() * mu)]
            x = base[:]

            # Mutation:
            # log-normal global sigma update for this child
            # mild (time-stable) adaptation
            tau = 1.0 / math.sqrt(2.0 * dim)
            sigma_child = sigma * math.exp(tau * randn())
            if sigma_child < sigma_min:
                sigma_child = sigma_min
            elif sigma_child > sigma_max:
                sigma_child = sigma_max

            # Apply correlated-ish mutation via shared z + per-dimension z
            z_common = randn() * 0.3
            for i in range(dim):
                if spans[i] == 0:
                    x[i] = lows[i]
                    continue
                z = z_common + randn()
                step = sigma_child * scale[i] * spans_nz[i] * z
                x[i] += step

            mirror_point(x)
            fx = evaluate(x)
            offspring.append((fx, x, sigma_child))

        if not offspring:
            break

        offspring.sort(key=lambda t: t[0])

        # Update elites with top offspring
        for j in range(min(len(offspring), max(2, lam // 2))):
            push_elite(offspring[j][0], offspring[j][1])

        # Update global best
        best_new, best_x_new = get_best()
        improved_best = best_new < best
        if improved_best:
            best, best_x = best_new, best_x_new
            no_best_gens = 0
        else:
            no_best_gens += 1

        # Adapt sigma and per-dimension scale using winner information
        # If best offspring beats current best-of-parents, increase exploration modestly, else decrease.
        # Use median parent fitness estimate from elites subset.
        parent_f_est = elites[min(len(elites) - 1, max(0, mu - 1))][0]
        best_offspring_f, best_offspring_x, best_offspring_sigma = offspring[0]

        if best_offspring_f <= parent_f_est:
            sigma = min(sigma_max, max(sigma_min, best_offspring_sigma * 1.05))
            # learn per-dimension scaling from successful displacement from best_x
            bx = best_x
            dx = [best_offspring_x[i] - bx[i] for i in range(dim)]
            for i in range(dim):
                if spans[i] == 0:
                    scale[i] = 1.0
                    continue
                # encourage scales that match observed improvement move magnitude
                mag = abs(dx[i]) / spans_nz[i]
                # update with cap; keep within [0.15, 3.0]
                s = 0.92 * scale[i] + 0.08 * (1.0 + 4.0 * mag)
                if s < 0.15:
                    s = 0.15
                elif s > 3.0:
                    s = 3.0
                scale[i] = s
        else:
            sigma = max(sigma_min, sigma * 0.92)

        # Occasional global probe / restart if stagnating
        if no_best_gens >= (10 + 2 * int(math.sqrt(dim))):
            no_best_gens = 0
            if random.random() < 0.6:
                # jump to a new Halton point (exploration)
                xg = halton_point(k_halton)
                k_halton += 1
            else:
                # random point
                xg = rand_uniform_point()
            fg = evaluate(xg)
            push_elite(fg, xg)
            if fg < best:
                best, best_x = fg, xg[:]
            # reset sigma moderately
            sigma = min(0.35, max(0.12, sigma))

        # Coordinate refinement around best (cheap exploitation), more likely near the end
        time_left = deadline - now()
        if time_left <= 0:
            break

        endgame = (time_left / float(max_time)) < 0.25 if max_time > 0 else True
        if endgame or (gen % refine_period == 0):
            # Determine step size for refinement from sigma
            # Use a decreasing schedule in endgame
            step_rel = max(1e-6, min(0.25, sigma * (0.7 if endgame else 1.0)))

            # Probe a subset of coordinates to keep bounded cost
            # Choose coords with largest span*scale first (likely impactful)
            idxs = list(range(dim))
            idxs.sort(key=lambda i: spans_nz[i] * scale[i], reverse=True)
            idxs = idxs[:refine_budget_coords]

            x0 = best_x[:]
            f0 = best
            improved_any = False

            for i in idxs:
                if now() >= deadline:
                    break
                if spans[i] == 0:
                    continue

                delta = step_rel * spans_nz[i] * scale[i]
                # try +delta and -delta
                xp = x0[:]
                xp[i] += delta
                mirror_point(xp)
                fp = evaluate(xp)

                xm = x0[:]
                xm[i] -= delta
                mirror_point(xm)
                fm = evaluate(xm)

                if fp < f0 or fm < f0:
                    if fp <= fm:
                        x0, f0 = xp, fp
                    else:
                        x0, f0 = xm, fm
                    improved_any = True

            if improved_any:
                push_elite(f0, x0)
                if f0 < best:
                    best, best_x = f0, x0[:]
                # slightly reduce sigma to exploit around refined best
                sigma = max(sigma_min, sigma * 0.9)

    return best
