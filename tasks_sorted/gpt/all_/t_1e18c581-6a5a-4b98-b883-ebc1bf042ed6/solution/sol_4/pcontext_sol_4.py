import random, time, math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs the provided ES variants:
      1) Better global coverage early: scrambled Halton + opposition + cheap local probing.
      2) Robust local convergence: Powell-like coordinate pattern search with adaptive steps.
      3) Multi-start with elite archive: repeatedly intensify around multiple good basins.
      4) Cross-Entropy / CEM-style sampling around elites: learns a diagonal distribution
         from top samples (more stable than noisy sigma rules on many problems).
      5) Reflection boundary handling (less corner-sticking than hard clamp).

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        if y > w:
            y = 2.0 * w - y
        return lo + y

    def proj_reflect(x):
        return [reflect(x[i], lows[i], highs[i]) for i in range(dim)]

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def evaluate(x):
        return float(func(x))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---- Halton (low discrepancy) ----
    def nth_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
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

    bases = nth_primes(dim)
    halton_i = 1
    shift = [random.random() for _ in range(dim)]  # Cranley-Patterson rotation

    def radical_inverse(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    def halton_vec():
        nonlocal halton_i
        k = halton_i
        halton_i += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (radical_inverse(k, bases[i]) + shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- elite archive ----------------
    elite_cap = max(10, 4 + int(3.0 * math.sqrt(dim + 1.0)))
    elites = []  # list of (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ---------------- local pattern search ----------------
    def pattern_search(x0, f0, base_steps, iters=2):
        """
        Powell-like coordinate search with adaptive per-dim steps.
        Very effective as a "polish" and also as a basin-refiner after sampling.
        """
        x = x0[:]
        f = f0
        steps = base_steps[:]
        for _ in range(iters):
            if time.time() >= deadline:
                break
            improved_any = False
            for j in range(dim):
                if time.time() >= deadline:
                    break
                step = steps[j]
                # Try a few halvings if needed
                for _try in range(4):
                    xp = x[:]
                    xm = x[:]
                    xp[j] = clamp(xp[j] + step, lows[j], highs[j])
                    xm[j] = clamp(xm[j] - step, lows[j], highs[j])

                    fp = evaluate(xp)
                    if fp < f:
                        x, f = xp, fp
                        improved_any = True
                        # modest expand for that coordinate
                        steps[j] = min(steps[j] * 1.25, spans_safe[j])
                        break

                    if time.time() >= deadline:
                        break

                    fm = evaluate(xm)
                    if fm < f:
                        x, f = xm, fm
                        improved_any = True
                        steps[j] = min(steps[j] * 1.25, spans_safe[j])
                        break

                    step *= 0.5
                    if step <= 1e-15 * spans_safe[j]:
                        break
                    steps[j] = step
            if not improved_any:
                break
        return x, f, steps

    # ---------------- initialization (global) ----------------
    best = float("inf")
    best_x = None

    init_n = max(24, 10 * dim)
    # global points: halton + random + opposition
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        if random.random() < 0.80:
            x = halton_vec()
        else:
            x = rand_vec()

        f = evaluate(x)
        if f < best:
            best, best_x = f, x
        push_elite(f, x)

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

    # light initial polish (often big win for rugged functions)
    base_steps = [0.10 * spans_safe[i] for i in range(dim)]
    best_x, best, base_steps = pattern_search(best_x, best, base_steps, iters=1)
    push_elite(best, best_x)

    # ---------------- main CEM / elite-driven loop ----------------
    # sampling budget per "generation"
    pop = max(18, 8 + int(6.0 * math.log(dim + 1.0)))
    keep = max(4, pop // 4)

    # diagonal distribution parameters (mean, std) that we update from top samples
    mean = best_x[:]
    std = [0.25 * spans_safe[i] for i in range(dim)]
    min_std = [1e-15 * spans_safe[i] for i in range(dim)]
    max_std = [1.5 * spans_safe[i] for i in range(dim)]

    # smoothing (CEM)
    alpha_mean = 0.35
    alpha_std = 0.25

    # restart/intensification controls
    last_best = best
    stall = 0
    patience = max(60, 22 * dim)
    polish_every = max(20, 5 * dim)
    it = 0

    while time.time() < deadline:
        it += 1

        # choose an anchor: best or one of the top elites (multi-basin)
        if elites and random.random() < 0.55:
            kmax = min(len(elites), max(3, elite_cap // 2))
            idx = int((random.random() ** 2) * kmax)  # bias to best
            anchor = elites[idx][1]
        else:
            anchor = best_x

        # sample population around current distribution, with some anchor-mixing
        samples = []
        for _ in range(pop):
            if time.time() >= deadline:
                break

            # occasionally inject a fresh global point
            if random.random() < 0.12:
                x = halton_vec() if random.random() < 0.85 else rand_vec()
            else:
                x = [0.0] * dim
                # mix anchor and mean, then perturb
                mix = 0.65 + 0.30 * random.random()  # bias towards anchor
                for i in range(dim):
                    m = mix * anchor[i] + (1.0 - mix) * mean[i]
                    x[i] = m + random.gauss(0.0, std[i])
                x = proj_reflect(x)

            f = evaluate(x)
            samples.append((f, x))
            push_elite(f, x)
            if f < best:
                best, best_x = f, x

        if not samples:
            break

        samples.sort(key=lambda t: t[0])
        top = samples[:keep]

        # update mean/std from top samples (diagonal)
        new_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(len(top)):
                s += top[k][1][i]
            new_mean[i] = s / float(len(top))

        new_std = [0.0] * dim
        for i in range(dim):
            m = new_mean[i]
            s2 = 0.0
            for k in range(len(top)):
                d = top[k][1][i] - m
                s2 += d * d
            # robust-ish: avoid std collapse too fast
            v = s2 / float(max(1, len(top) - 1))
            new_std[i] = math.sqrt(v + 1e-30)

        # smooth updates
        for i in range(dim):
            mean[i] = reflect((1.0 - alpha_mean) * mean[i] + alpha_mean * new_mean[i], lows[i], highs[i])
            si = (1.0 - alpha_std) * std[i] + alpha_std * new_std[i]
            if si < min_std[i]:
                si = min_std[i]
            if si > max_std[i]:
                si = max_std[i]
            std[i] = si

        # periodic polish of incumbent (strong exploitation)
        if it % polish_every == 0 and time.time() < deadline:
            # steps tied to current std (acts like trust region)
            steps = [max(1e-15 * spans_safe[i], 0.9 * std[i]) for i in range(dim)]
            best_x, best, _ = pattern_search(best_x, best, steps, iters=2)
            push_elite(best, best_x)
            mean = best_x[:]  # intensify around polished best

        # stall / restart handling
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # restart distribution around a good elite or global point
            if elites and random.random() < 0.75:
                kmax = min(len(elites), max(3, elite_cap // 2))
                idx = int((random.random() ** 2) * kmax)
                mean = elites[idx][1][:]
            else:
                mean = halton_vec() if random.random() < 0.8 else rand_vec()

            # reset std moderately large, then shrink again via CEM
            std = [0.35 * spans_safe[i] for i in range(dim)]
            # quick opposition check at restart mean
            xop = opposite(mean)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop

    return best
