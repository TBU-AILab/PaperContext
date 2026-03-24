import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Upgrade over previous:
      - Adds a fast local optimizer: (1+lambda) Evolution Strategy + 1/5th success rule,
        running throughout (not only at the end), seeded from best-so-far.
      - Keeps a trimmed JADE/L-SHADE-style DE as the global engine (current-to-pbest/1 + archive).
      - Uses occasional "opposition / lens" restart around the current best to escape plateaus.
      - Uses lighter bookkeeping and a more consistent boundary handling.

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    # ---------------- utils ----------------
    def time_left():
        return time.time() < deadline

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def reflect_into_bounds(x):
        # reflection folding into [lo,hi]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            span = hi - lo
            if span <= 0.0:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                u = (v - lo) % (2.0 * span)
                if u < 0.0:
                    u += 2.0 * span
                if u > span:
                    u = 2.0 * span - u
                v = lo + u
            x[i] = v
        return x

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def choose_distinct(k, n, banned):
        banned = set(banned)
        out = []
        # rejection then fallback shuffle
        tries = 0
        while len(out) < k and tries < 60 * k:
            r = random.randrange(n)
            tries += 1
            if r in banned:
                continue
            banned.add(r)
            out.append(r)
        if len(out) < k:
            pool = [i for i in range(n) if i not in banned]
            random.shuffle(pool)
            out.extend(pool[:(k - len(out))])
        return out

    # A cheap hashable key for caching
    def key_of(x):
        # rounding reduces cache misses from tiny float differences
        return tuple(int((x[i] - lows[i]) / (spans[i] + 1e-300) * 1000000.0) for i in range(dim))

    # ---------------- small cache ----------------
    # Helps when local search revisits near-identical points
    cache = {}
    cache_max = 20000

    def eval_cached(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = safe_eval(x)
        if len(cache) < cache_max:
            cache[k] = fx
        else:
            # random eviction
            for _ in range(3):
                cache.pop(next(iter(cache)), None)
            cache[k] = fx
        return fx

    # ---------------- global DE setup (JADE/L-SHADE-ish) ----------------
    NP_init = max(28, min(140, 10 * dim + 40))
    NP_min = max(10, min(30, 3 * dim + 8))
    NP = NP_init

    H = max(8, min(25, NP_init // 3))
    MF = [0.55] * H
    MCR = [0.5] * H
    mem_ptr = 0

    archive = []
    archive_max = NP_init

    p_min, p_max = 0.05, 0.25

    pop = [rand_vec() for _ in range(NP)]
    fit = [float("inf")] * NP

    best = float("inf")
    best_x = None
    for i in range(NP):
        if not time_left():
            return best
        fi = eval_cached(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    def target_np():
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        return int(round(NP_init - frac * (NP_init - NP_min)))

    # ---------------- local ES: (1+lambda) with 1/5th success rule ----------------
    # seed from current best, re-seed occasionally
    es_x = best_x[:] if best_x is not None else rand_vec()
    es_f = best
    # initial sigma proportional to span
    es_sigma = 0.25
    es_lambda = 4 if dim <= 10 else 6
    es_success = 0
    es_tries = 0
    es_last_improve_t = time.time()

    def es_step():
        nonlocal es_x, es_f, es_sigma, es_success, es_tries, es_last_improve_t, best, best_x
        # create lambda candidates around es_x
        base = es_x
        base_f = es_f

        improved = False
        best_cand = None
        best_cand_f = base_f

        # correlated step: random direction + coordinate noise
        for _ in range(es_lambda):
            if not time_left():
                return
            cand = base[:]
            # direction scaling with sigma and spans
            # use both global direction and per-coordinate noise for robustness
            g = randn()
            for j in range(dim):
                cand[j] = cand[j] + (es_sigma * spans[j]) * (0.60 * g + 0.40 * randn())
            reflect_into_bounds(cand)
            fc = eval_cached(cand)
            if fc < best_cand_f:
                best_cand_f = fc
                best_cand = cand

        es_tries += 1
        if best_cand is not None and best_cand_f < base_f:
            es_x = best_cand
            es_f = best_cand_f
            es_success += 1
            improved = True
            es_last_improve_t = time.time()
            if best_cand_f < best:
                best = best_cand_f
                best_x = best_cand[:]
        else:
            # keep parent
            pass

        # update sigma using 1/5 rule in small windows
        if es_tries >= 10:
            rate = es_success / float(es_tries)
            # if success rate > 1/5 increase sigma else decrease
            if rate > 0.20:
                es_sigma *= 1.25
            else:
                es_sigma *= 0.82
            es_sigma = max(1e-6, min(0.6, es_sigma))
            es_success = 0
            es_tries = 0

        # if ES stagnates, re-seed around global best or random
        if time.time() - es_last_improve_t > 0.20 * max_time:
            if best_x is not None and random.random() < 0.85:
                es_x = best_x[:]
                # moderate reset sigma
                es_sigma = min(0.35, max(0.05, es_sigma * 1.4))
            else:
                es_x = rand_vec()
                es_sigma = 0.35
            es_f = eval_cached(es_x)
            es_last_improve_t = time.time()

    # ---------------- main loop ----------------
    gen = 0
    stagn = 0
    last_best = best

    while time_left():
        gen += 1

        # interleave local ES steps frequently (cheap and strong near minima)
        # more ES later, but still run early
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        es_steps = 1 if frac < 0.4 else (2 if frac < 0.8 else 3)
        for _ in range(es_steps):
            if not time_left():
                return best
            es_step()

        # rank
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]
        if best < last_best - 1e-12:
            stagn = 0
            last_best = best
        else:
            stagn += 1

        # occasional "lens/opposition" restart around best to escape plateaus
        if stagn >= max(15, 4 + dim) and best_x is not None:
            k = max(2, NP // 6)
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                x = pop[wi][:]
                # lens transform wrt best: x' = best + alpha*(best - x) + noise
                alpha = 0.7 + 0.6 * random.random()
                for j in range(dim):
                    x[j] = best_x[j] + alpha * (best_x[j] - pop[wi][j]) + (0.02 * spans[j]) * randn()
                reflect_into_bounds(x)
                pop[wi] = x
                fit[wi] = eval_cached(x)
            archive.clear()
            stagn = 0

        # compute p for pbest selection
        p = p_max - (p_max - p_min) * min(1.0, max(0.0, frac))
        pcount = max(2, int(math.ceil(p * NP)))

        # memory update buffers
        SCR, SF, dF = [], [], []

        union = pop + archive
        union_n = len(union)

        # DE generation
        for i in range(NP):
            if not time_left():
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = clamp01(mu_cr + 0.10 * randn())

            F = cauchy(mu_f, 0.12)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 18:
                F = cauchy(mu_f, 0.12)
                tries += 1
            if F <= 0.0:
                F = 0.2 + 0.4 * random.random()
            if F > 1.0:
                F = 1.0
            F = min(1.0, max(1e-6, F))

            pbest_idx = idx_sorted[random.randrange(pcount)]
            x_pbest = pop[pbest_idx]

            r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]
            # choose r2 from union, avoiding duplicates with pop indices when possible
            r2u = None
            for _ in range(50):
                cand = random.randrange(union_n)
                if cand < NP and cand in (i, pbest_idx, r1):
                    continue
                r2u = cand
                break
            if r2u is None:
                r2u = random.randrange(union_n)

            x_r1 = pop[r1]
            x_r2 = union[r2u]

            donor = [0.0] * dim
            for j in range(dim):
                donor[j] = xi[j] + F * (x_pbest[j] - xi[j]) + F * (x_r1[j] - x_r2[j])

            # binomial crossover
            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]
            reflect_into_bounds(trial)

            ftrial = eval_cached(trial)
            if ftrial <= fi:
                # archive update
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = trial
                fit[i] = ftrial

                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]
                    # also sync ES parent sometimes
                    if random.random() < 0.5:
                        es_x = best_x[:]
                        es_f = best

                df = fi - ftrial
                if df < 0.0:
                    df = 0.0
                SCR.append(CR)
                SF.append(F)
                dF.append(df)

        # update parameter memories
        if dF:
            wsum = sum(dF)
            if wsum <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            num = 0.0
            den = 0.0
            for wi, fi_ in zip(w, SF):
                num += wi * fi_ * fi_
                den += wi * fi_
            mf = (num / den) if den > 1e-12 else 0.5

            MCR[mem_ptr] = clamp01(mcr)
            MF[mem_ptr] = min(1.0, max(1e-6, mf))
            mem_ptr = (mem_ptr + 1) % H

        # population size reduction
        targ = target_np()
        if targ < NP:
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            keep = idx_sorted[:targ]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = targ
            archive_max = max(NP, NP_min)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

    return best
