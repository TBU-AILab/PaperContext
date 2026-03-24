import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs prior versions:
      - Stronger global exploration via Latin-hypercube batches + opposition points
      - Keep an explicit elite archive and restart from elites (not just best)
      - "CMA-ES-inspired" diagonal ES with:
          * mirrored sampling
          * rank-based mean update
          * path-informed per-dimension scaling
          * robust sigma adaptation using median progress (less noisy than "any improvement")
      - Cheap trust-region local refinement (multi-step coordinate search) on the best elite
      - Soft boundary handling (reflect) to reduce corner-sticking

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ----------------- helpers -----------------
    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def reflect(v, lo, hi):
        # reflect into [lo, hi] (handles mild out-of-bounds smoothly)
        if lo == hi:
            return lo
        w = hi - lo
        y = v
        # map to [0, 2w) then reflect
        y = (y - lo) % (2.0 * w)
        if y < 0:
            y += 2.0 * w
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def project_reflect(x):
        return [reflect(x[i], lows[i], highs[i]) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # Latin hypercube batch sampler (no numpy)
    def lhs_batch(n):
        # For each dimension: create a random permutation of n strata
        perms = []
        for i in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        batch = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                u = (perms[i][k] + random.random()) / float(n)
                x[i] = lows[i] + u * spans[i]
            batch.append(x)
        return batch

    # Keep a small archive of elites (x, f) sorted by f
    elite_cap = max(6, 2 + int(2.0 * math.sqrt(dim + 1.0)))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        # insert if good
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ----------------- initialization (stronger global) -----------------
    # Use a few LHS batches + opposition to find a good basin quickly.
    best = float("inf")
    best_x = None

    init_batches = 2 if dim <= 20 else 1
    init_n = max(12, 6 * dim)

    for _ in range(init_batches):
        if time.time() >= deadline:
            return best
        batch = lhs_batch(init_n)
        for x in batch:
            if time.time() >= deadline:
                return best
            f = evaluate(x)
            if f < best:
                best, best_x = f, x
            push_elite(f, x)

            # opposition sample (cheap extra global coverage)
            if time.time() >= deadline:
                return best
            xo = opposite(x)
            fo = evaluate(xo)
            if fo < best:
                best, best_x = fo, xo
            push_elite(fo, xo)

    if best_x is None:
        best_x = rand_vec()
        best = evaluate(best_x)
        push_elite(best, best_x)

    # ----------------- ES state -----------------
    mean = best_x[:]

    # global step multiplier (dimension-aware)
    sigma = 0.22 if dim <= 20 else 0.18

    # diagonal scales (absolute units)
    diag = [0.25 * spans_safe[i] for i in range(dim)]
    min_diag = [1e-15 * spans_safe[i] for i in range(dim)]
    max_diag = [3.0 * spans_safe[i] for i in range(dim)]

    # population size + selection
    lam = max(10, 6 + int(4.0 * math.log(dim + 1.0)))
    if lam % 2 == 1:
        lam += 1
    mu = max(2, lam // 2)

    # log weights
    weights = [math.log(mu + 0.5) - math.log(k + 1.0) for k in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    # evolution path
    path = [0.0] * dim
    c_path = 0.75  # smoother memory

    # sigma adaptation using robust progress signal
    # maintain EMA of "normalized improvement" of best candidate vs median candidate
    prog_ema = 0.0
    prog_beta = 0.12

    # stagnation + restart controls
    it = 0
    no_improve = 0
    best_at_restart = best
    patience = max(70, 28 * dim)

    # local polish controls
    polish_every = max(18, 4 * dim)

    # ----------------- local refinement (trust-region coord search) -----------------
    def local_polish(xc, fc):
        nonlocal best, best_x, mean
        x = xc[:]
        f = fc
        # start step from current exploration radii, but make it smaller for precision
        base_steps = [(0.45 * sigma) * diag[i] for i in range(dim)]
        # ensure useful steps relative to span
        for i in range(dim):
            if base_steps[i] < 1e-12 * spans_safe[i]:
                base_steps[i] = 1e-12 * spans_safe[i]

        # a few passes, halving step when not improving
        for _pass in range(2):
            if time.time() >= deadline:
                break
            improved_any = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                step = base_steps[j]
                # try up to 2 reductions per coordinate
                for _ in range(3):
                    xp = x[:]
                    xm = x[:]
                    xp[j] = clamp(xp[j] + step, lows[j], highs[j])
                    xm[j] = clamp(xm[j] - step, lows[j], highs[j])

                    fp = evaluate(xp)
                    if fp < f:
                        x, f = xp, fp
                        improved_any = True
                        continue

                    if time.time() >= deadline:
                        break

                    fm = evaluate(xm)
                    if fm < f:
                        x, f = xm, fm
                        improved_any = True
                        continue

                    step *= 0.5
                    if step <= 1e-15 * spans_safe[j]:
                        break

            if not improved_any:
                break

        if f < best:
            best, best_x = f, x
            mean = best_x[:]
            push_elite(f, x)
        return x, f

    # ----------------- main loop -----------------
    while time.time() < deadline:
        it += 1

        # Occasionally inject a small global LHS mini-batch for diversity
        if it % max(25, 6 * dim) == 0:
            n = max(6, dim // 2)
            for x in lhs_batch(n):
                if time.time() >= deadline:
                    return best
                f = evaluate(x)
                if f < best:
                    best, best_x = f, x
                    mean = best_x[:]
                push_elite(f, x)

        # Generate candidates (mirrored)
        candidates = []  # (f, x, z)
        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                break
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]
            x1 = [mean[i] + sigma * diag[i] * z[i] for i in range(dim)]
            x2 = [mean[i] - sigma * diag[i] * z[i] for i in range(dim)]
            x1 = project_reflect(x1)
            x2 = project_reflect(x2)

            f1 = evaluate(x1)
            candidates.append((f1, x1, z))
            push_elite(f1, x1)
            if f1 < best:
                best, best_x = f1, x1

            if time.time() >= deadline:
                break

            f2 = evaluate(x2)
            candidates.append((f2, x2, [-zi for zi in z]))
            push_elite(f2, x2)
            if f2 < best:
                best, best_x = f2, x2

        if not candidates:
            break

        candidates.sort(key=lambda t: t[0])
        elites_gen = candidates[:mu]

        # Update mean (rank-based recombination)
        new_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * elites_gen[k][1][i]
            new_mean[i] = s
        mean = project_reflect(new_mean)

        # Update path and diag based on z-mean
        z_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * elites_gen[k][2][i]
            z_mean[i] = s

        for i in range(dim):
            path[i] = c_path * path[i] + (1.0 - c_path) * z_mean[i]

        # Per-dim scaling: expand where path is consistently large, contract otherwise (gentle)
        for i in range(dim):
            t = abs(path[i])
            # bounded multiplier ~ [0.92, 1.12]
            mult = 1.0 + 0.16 * (t - 0.8) / (1.0 + t)
            if mult < 0.92:
                mult = 0.92
            elif mult > 1.12:
                mult = 1.12
            di = diag[i] * mult
            if di < min_diag[i]:
                di = min_diag[i]
            elif di > max_diag[i]:
                di = max_diag[i]
            diag[i] = di

        # Robust sigma update:
        # progress signal = (median - best_in_gen) / (|median|+eps)  (higher is better)
        med = candidates[len(candidates) // 2][0]
        best_gen = candidates[0][0]
        denom = abs(med) + 1e-12
        prog = (med - best_gen) / denom
        prog_ema = (1.0 - prog_beta) * prog_ema + prog_beta * prog

        if prog_ema > 0.03:
            sigma *= 1.035
        elif prog_ema < 0.005:
            sigma *= 0.965
        else:
            sigma *= 0.995  # slight anneal for stability

        if sigma < 1e-15:
            sigma = 1e-15
        if sigma > 2.5:
            sigma = 2.5

        # Stagnation tracking
        if best < best_at_restart:
            best_at_restart = best
            no_improve = 0
        else:
            no_improve += 1

        # Periodic local polish on current best elite
        if it % polish_every == 0 and time.time() < deadline:
            local_polish(best_x, best)

        # Restart if stuck: start from a random elite + jitter, or global random
        if no_improve >= patience and time.time() < deadline:
            no_improve = 0
            best_at_restart = best

            use_elite = (len(elites) > 0 and random.random() < 0.75)
            if use_elite:
                # pick from top few elites, bias to best
                kmax = min(len(elites), max(2, elite_cap // 2))
                idx = int((random.random() ** 2) * kmax)  # bias to small index
                anchor = elites[idx][1][:]
            else:
                anchor = rand_vec()

            # jitter anchor
            mean = anchor[:]
            for i in range(dim):
                mean[i] = clamp(mean[i] + random.gauss(0.0, 0.12 * spans_safe[i]), lows[i], highs[i])

            # reset exploration (slightly smaller to focus around elite basins)
            sigma = 0.26 if use_elite else 0.30
            diag = [0.28 * spans_safe[i] for i in range(dim)]
            path = [0.0] * dim
            prog_ema = 0.0

            # quick opposition trial on restart anchor
            xop = opposite(mean)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop
                mean = best_x[:]

    return best
