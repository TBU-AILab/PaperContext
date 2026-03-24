import random
import time
import math

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        return float(func(x))

    # ---------- Phase 1: low-discrepancy-ish global sampling (stratified) ----------
    # Stratify each dimension into m bins; pick random bin per dim per sample.
    # This usually beats pure random for the same number of evals.
    best = float("inf")
    best_x = None

    # time-aware choice of samples: small if dim large, larger if dim small
    m = 8 if dim <= 10 else (6 if dim <= 30 else 4)

    # Do a first probe to ensure best_x exists
    x0 = rand_point()
    f0 = eval_point(x0)
    best, best_x = f0, x0

    # stratified samples
    # each sample picks a random bin index in each dimension; within-bin uniform
    init_samples = max(40, 30 * dim)
    for _ in range(init_samples - 1):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            bi = random.randrange(m)
            u = (bi + random.random()) / m
            x.append(lows[i] + u * spans[i])
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x

    # ---------- Phase 2: multi-start local improvement (Pattern search + adaptive noise) ----------
    # Coordinate pattern search is robust without gradients.
    # We run multiple short "intensification" runs from top candidates, with restarts.

    # Keep a small elite pool (x, f). This is cheap and helps restart intelligently.
    elite = [(best, list(best_x))]

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        # keep small pool
        if len(elite) > 8:
            del elite[8:]

    # Seed elite with a few more good points
    extra_elite_tries = max(10, 10 * dim)
    for _ in range(extra_elite_tries):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_point(x)
        if fx < elite[-1][0] if len(elite) >= 8 else True:
            elite_add(fx, x)
            if fx < best:
                best, best_x = fx, list(x)

    # Pattern search parameters
    base_step = [0.25 * spans[i] for i in range(dim)]
    min_step = [1e-12 * spans[i] for i in range(dim)]

    # Evaluate budget / time: use repeated short runs until time expires
    while time.time() < deadline:
        # pick a start: mostly from elite, sometimes random to escape
        if random.random() < 0.80 and elite:
            _, x = random.choice(elite[:min(4, len(elite))])
            x = list(x)
        else:
            x = rand_point()

        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
            elite_add(fx, x)

        # local step sizes start moderate; adapt per-dimension
        step = [s for s in base_step]

        # small "temperature" for occasional worse acceptance (helps on rugged surfaces)
        T0 = 1e-6 + 0.01 * (abs(fx) + 1.0)
        T = T0

        # short local run
        no_improve = 0
        # Limit iterations per run so we can restart often under time limits
        it_limit = 60 + 20 * dim

        for _ in range(it_limit):
            if time.time() >= deadline:
                return best

            improved = False

            # Randomize coordinate order to reduce bias
            idxs = list(range(dim))
            random.shuffle(idxs)

            # Try +/- step moves on a subset of coordinates each iteration
            # (full coordinate sweep can be too expensive in high dim)
            if dim <= 10:
                trial_coords = idxs
            else:
                k = max(4, dim // 6)
                trial_coords = idxs[:k]

            for i in trial_coords:
                si = step[i]
                if si <= min_step[i]:
                    continue

                # Try both directions; randomized order
                if random.random() < 0.5:
                    dirs = (-1.0, 1.0)
                else:
                    dirs = (1.0, -1.0)

                best_local_fx = fx
                best_local_xi = None

                for d in dirs:
                    xi = clamp(x[i] + d * si, i)
                    if xi == x[i]:
                        continue
                    cand = x[:]  # copy
                    cand[i] = xi

                    cand_f = eval_point(cand)
                    if cand_f < best_local_fx:
                        best_local_fx = cand_f
                        best_local_xi = xi

                    # global update
                    if cand_f < best:
                        best, best_x = cand_f, cand
                        elite_add(cand_f, cand)

                if best_local_xi is not None:
                    x[i] = best_local_xi
                    fx = best_local_fx
                    improved = True

            if improved:
                no_improve = 0
                # slightly grow steps when making progress (but keep bounded)
                for i in trial_coords:
                    step[i] *= 1.05
                    # cap to avoid explosive steps
                    if step[i] > 0.5 * spans[i]:
                        step[i] = 0.5 * spans[i]
            else:
                no_improve += 1
                # shrink steps when stuck
                for i in trial_coords:
                    step[i] *= 0.6
                    if step[i] < min_step[i]:
                        step[i] = min_step[i]

                # occasional stochastic jump around current point (small)
                # accept if better, or sometimes if slightly worse (annealing-like)
                if random.random() < 0.35:
                    cand = x[:]
                    # perturb a few dims
                    k = 1 if dim == 1 else (1 + random.randrange(min(dim, 4)))
                    for _p in range(k):
                        j = random.randrange(dim)
                        cand[j] = clamp(cand[j] + (2.0 * random.random() - 1.0) * (0.15 * spans[j]), j)
                    cand_f = eval_point(cand)
                    if cand_f < fx:
                        x, fx = cand, cand_f
                        improved = True
                        no_improve = 0
                        if cand_f < best:
                            best, best_x = cand_f, list(cand)
                            elite_add(cand_f, cand)
                    else:
                        # SA-style acceptance (very conservative)
                        # Avoid overflow/underflow issues
                        df = cand_f - fx
                        if df <= 0:
                            acc = True
                        else:
                            # probability exp(-df/T)
                            if T > 0:
                                # clamp exponent to safe range
                                expo = -df / T
                                if expo < -50:
                                    acc = False
                                else:
                                    acc = (random.random() < math.exp(expo))
                            else:
                                acc = False
                        if acc:
                            x, fx = cand, cand_f

                # cool temperature
                T *= 0.95

            # If steps are tiny and no improvement, stop this local run early
            if no_improve > (10 + dim // 2):
                tiny = True
                for i in range(dim):
                    if step[i] > (1e-9 * spans[i]):
                        tiny = False
                        break
                if tiny:
                    break

        # Add end-of-run point to elite
        if fx < elite[-1][0] if len(elite) >= 8 else True:
            elite_add(fx, x)
            if fx < best:
                best, best_x = fx, list(x)

    return best
