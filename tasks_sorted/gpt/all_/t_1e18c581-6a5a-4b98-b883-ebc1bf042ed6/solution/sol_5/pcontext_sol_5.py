import random, time, math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Main upgrades vs previous versions:
      - Stronger early global coverage: scrambled Halton + opposition + small LHS batches
      - Robust multi-start: elite archive + basin selection + restart-on-stall
      - Better local exploitation: adaptive Hooke-Jeeves / pattern search around elites
      - Better distribution learning: diagonal CEM with weighted elites + soft trust-region
      - Safer boundaries: reflection projection (reduces corner sticking)

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

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

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------------- low discrepancy: Halton ----------------
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

    bases = nth_primes(max(1, dim))
    h_index = 1
    h_shift = [random.random() for _ in range(dim)]  # Cranley-Patterson rotation

    def radical_inverse(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    def halton_vec():
        nonlocal h_index
        k = h_index
        h_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (radical_inverse(k, bases[i]) + h_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------------- LHS mini-batch (cheap diversity injection) ----------------
    def lhs_batch(n):
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

    # ---------------- elite archive ----------------
    elite_cap = max(12, 6 + int(4.0 * math.sqrt(dim + 1.0)))
    elites = []  # list of (f, x) sorted

    def push_elite(f, x):
        nonlocal elites
        if len(elites) < elite_cap or f < elites[-1][0]:
            elites.append((f, x[:]))
            elites.sort(key=lambda t: t[0])
            if len(elites) > elite_cap:
                elites.pop()

    # ---------------- local optimizer: Hooke-Jeeves pattern search ----------------
    def hooke_jeeves(x0, f0, step0, iters=2):
        # Exploratory moves + pattern move; shrink step when stuck.
        x = x0[:]
        f = f0
        step = step0[:]  # per-dimension steps
        min_step = [1e-15 * spans_safe[i] for i in range(dim)]

        def explore(xb, fb):
            xx = xb[:]
            ff = fb
            for j in range(dim):
                if time.time() >= deadline:
                    break
                sj = step[j]
                if sj <= min_step[j]:
                    continue

                # + direction
                xp = xx[:]
                xp[j] = clamp(xp[j] + sj, lows[j], highs[j])
                fp = evaluate(xp)
                if fp < ff:
                    xx, ff = xp, fp
                    continue

                if time.time() >= deadline:
                    break

                # - direction
                xm = xx[:]
                xm[j] = clamp(xm[j] - sj, lows[j], highs[j])
                fm = evaluate(xm)
                if fm < ff:
                    xx, ff = xm, fm
            return xx, ff

        for _ in range(iters):
            if time.time() >= deadline:
                break

            xb, fb = x[:], f
            xn, fn = explore(xb, fb)
            if fn < fb:
                # pattern move
                while time.time() < deadline:
                    xpatt = [clamp(xn[i] + (xn[i] - xb[i]), lows[i], highs[i]) for i in range(dim)]
                    fpatt = evaluate(xpatt)
                    if fpatt < fn:
                        xb, fb = xn, fn
                        xn, fn = explore(xpatt, fpatt)
                        if fn >= fpatt:
                            xn, fn = xpatt, fpatt
                    else:
                        break
                x, f = xn, fn
            else:
                # shrink steps
                shrunk = 0
                for j in range(dim):
                    if step[j] > min_step[j]:
                        step[j] *= 0.5
                        if step[j] < min_step[j]:
                            step[j] = min_step[j]
                        shrunk += 1
                if shrunk == 0:
                    break

        return x, f, step

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    init_n = max(30, 12 * dim)
    # Mix: Halton + random + opposition
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.80 else rand_vec()
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

    # quick initial local improvement
    base_step = [0.12 * spans_safe[i] for i in range(dim)]
    best_x, best, base_step = hooke_jeeves(best_x, best, base_step, iters=1)
    push_elite(best, best_x)

    # ---------------- main loop: diagonal CEM + intensification ----------------
    pop = max(22, 10 + int(8.0 * math.log(dim + 1.0)))
    keep = max(5, pop // 5)

    mean = best_x[:]
    std = [0.30 * spans_safe[i] for i in range(dim)]
    min_std = [1e-15 * spans_safe[i] for i in range(dim)]
    max_std = [2.0 * spans_safe[i] for i in range(dim)]

    # CEM smoothing
    a_mean = 0.45
    a_std = 0.28

    # Weighted elites (more stable than plain top-k averaging)
    def elite_weights(k):
        ws = [math.log(k + 0.5) - math.log(i + 1.0) for i in range(k)]
        s = sum(ws)
        return [w / s for w in ws]

    it = 0
    stall = 0
    last_best = best
    patience = max(70, 25 * dim)
    polish_every = max(18, 5 * dim)
    inject_every = max(30, 8 * dim)

    while time.time() < deadline:
        it += 1

        # occasional LHS injection to fight premature convergence
        if it % inject_every == 0:
            n = max(6, dim // 2)
            for x in lhs_batch(n):
                if time.time() >= deadline:
                    return best
                f = evaluate(x)
                if f < best:
                    best, best_x = f, x
                    mean = best_x[:]
                push_elite(f, x)

        # choose an anchor from elites (multi-basin)
        if elites and random.random() < 0.70:
            kmax = min(len(elites), max(4, elite_cap // 2))
            idx = int((random.random() ** 2.2) * kmax)  # bias to better
            anchor = elites[idx][1]
        else:
            anchor = best_x

        # sample population
        samples = []
        for _ in range(pop):
            if time.time() >= deadline:
                break

            if random.random() < 0.10:
                x = halton_vec() if random.random() < 0.85 else rand_vec()
            else:
                # blend mean+anchor, then perturb
                mix = 0.55 + 0.40 * random.random()
                x = [0.0] * dim
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
        w = elite_weights(len(top))

        # update mean/std from weighted top
        new_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(len(top)):
                s += w[k] * top[k][1][i]
            new_mean[i] = s

        new_std = [0.0] * dim
        for i in range(dim):
            m = new_mean[i]
            s2 = 0.0
            for k in range(len(top)):
                d = top[k][1][i] - m
                s2 += w[k] * (d * d)
            new_std[i] = math.sqrt(s2 + 1e-30)

        # smooth update + bounds
        for i in range(dim):
            mean[i] = reflect((1.0 - a_mean) * mean[i] + a_mean * new_mean[i], lows[i], highs[i])
            si = (1.0 - a_std) * std[i] + a_std * new_std[i]
            if si < min_std[i]:
                si = min_std[i]
            if si > max_std[i]:
                si = max_std[i]
            std[i] = si

        # periodic intensification: polish best and a random elite
        if it % polish_every == 0 and time.time() < deadline:
            steps = [max(1e-15 * spans_safe[i], 0.8 * std[i]) for i in range(dim)]
            best_x, best, _ = hooke_jeeves(best_x, best, steps, iters=2)
            push_elite(best, best_x)
            mean = best_x[:]

            if elites and time.time() < deadline:
                idx = int((random.random() ** 1.8) * min(len(elites), max(4, elite_cap // 2)))
                ex = elites[idx][1]
                ef = elites[idx][0]
                ex2, ef2, _ = hooke_jeeves(ex, ef, steps, iters=1)
                push_elite(ef2, ex2)
                if ef2 < best:
                    best, best_x = ef2, ex2
                    mean = best_x[:]

        # stall / restart
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        if stall >= patience and time.time() < deadline:
            stall = 0
            # restart mean from a good elite or global point; widen std
            if elites and random.random() < 0.80:
                kmax = min(len(elites), max(4, elite_cap // 2))
                idx = int((random.random() ** 2.0) * kmax)
                mean = elites[idx][1][:]
            else:
                mean = halton_vec() if random.random() < 0.8 else rand_vec()

            std = [0.38 * spans_safe[i] for i in range(dim)]
            # try opposition at restart mean
            xop = opposite(mean)
            fop = evaluate(xop)
            push_elite(fop, xop)
            if fop < best:
                best, best_x = fop, xop
                mean = best_x[:]

    return best
