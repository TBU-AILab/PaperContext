import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    Key improvements over the best provided DE/rand/1:
      - Robust evaluation wrapper (exception + NaN/inf safe)
      - Better initialization: LHS-like stratification + opposition
      - Stronger DE core: current-to-pbest/1 with optional archive (JADE)
      - Success-history adaptation of F and CR (SHADE-style memory)
      - Cheap local refinement around best (coordinate + gaussian), time-scheduled
      - Stagnation-triggered partial restart of worst individuals
      - Faster index sampling (avoids building idx lists each trial)

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- degenerate dimension ----
    if dim <= 0:
        try:
            v = float(func([]))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0.0:
            raise ValueError("Each bound must satisfy low <= high.")

    def now():
        return time.time()

    def clip_val(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def is_finite(v):
        return not (v is None or math.isnan(v) or math.isinf(v))

    def eval_f(x):
        try:
            v = float(func(x))
            return v if is_finite(v) else float("inf")
        except Exception:
            return float("inf")

    # ---- population sizing: modest but not tiny ----
    pop_size = max(14, min(64, 10 + 6 * dim))

    # ---- LHS-like init + opposition ----
    perms = []
    for d in range(dim):
        p = list(range(pop_size))
        random.shuffle(p)
        perms.append(p)

    def sample_lhs(i):
        x = [0.0] * dim
        for d in range(dim):
            if spans[d] == 0.0:
                x[d] = lows[d]
            else:
                k = perms[d][i]
                u = (k + random.random()) / float(pop_size)
                x[d] = lows[d] + u * spans[d]
        return x

    pop = []
    fit = []
    best = float("inf")
    best_x = None

    for i in range(pop_size):
        if now() >= deadline:
            return best
        x = sample_lhs(i)
        xo = [clip_val(lows[d] + highs[d] - x[d], d) for d in range(dim)]
        fx = eval_f(x)
        fo = eval_f(xo)
        if fo < fx:
            x, fx = xo, fo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # ---- SHADE memories + archive (JADE) ----
    H = 8
    M_F = [0.5] * H
    M_CR = [0.8] * H
    mem_idx = 0

    archive = []
    archive_max = pop_size

    pmin, pmax = 0.08, 0.25  # p-best fraction range

    def clamp01(v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def approx_normal():
        # sum of 4 uniforms - 2: mean~0, cheap "normal-ish"
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    def sample_CR(mu):
        return clamp01(mu + 0.10 * approx_normal())

    def sample_F(mu):
        # Cauchy-like around mu; resample to keep (0,1]
        for _ in range(10):
            u = random.random()
            f = mu + 0.10 * math.tan(math.pi * (u - 0.5))
            if 0.0 < f <= 1.0:
                return f
        # fallback clamp-ish
        if mu <= 0.0:
            return 0.1
        if mu > 1.0:
            return 1.0
        return mu

    # ---- local refinement around best (very cheap) ----
    def local_refine(x0, f0, max_evals):
        if x0 is None or max_evals <= 0:
            return x0, f0
        x = x0[:]
        f = f0

        steps = [0.10 * (spans[i] if spans[i] > 0.0 else 1.0) for i in range(dim)]
        min_steps = [1e-12 * (spans[i] if spans[i] > 0.0 else 1.0) for i in range(dim)]

        evals = 0
        while evals < max_evals and now() < deadline:
            improved = False

            # coordinate +/- moves
            for i in range(dim):
                if evals >= max_evals or now() >= deadline:
                    break
                if spans[i] == 0.0 or steps[i] < min_steps[i]:
                    continue
                s = steps[i]

                xp = x[:]
                xp[i] = clip_val(xp[i] + s, i)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= max_evals or now() >= deadline:
                    break

                xm = x[:]
                xm[i] = clip_val(xm[i] - s, i)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            # a couple gaussian probes
            if evals < max_evals and now() < deadline:
                for _ in range(2):
                    if evals >= max_evals or now() >= deadline:
                        break
                    xt = x[:]
                    for i in range(dim):
                        if spans[i] == 0.0:
                            continue
                        xt[i] = clip_val(xt[i] + random.gauss(0.0, 0.30) * steps[i], i)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved = True

            if not improved:
                for i in range(dim):
                    steps[i] *= 0.5

        return x, f

    # ---- helper: pick random index not equal to forbidden ----
    def rand_not(n, a):
        r = random.randrange(n - 1)
        return r + 1 if r >= a else r

    def rand_not2(n, a, b):
        # sample uniformly from [0..n-1]\{a,b}
        r = random.randrange(n - 2)
        # map r to an index skipping a,b
        lo, hi = (a, b) if a < b else (b, a)
        if r >= lo:
            r += 1
        if r >= hi:
            r += 1
        return r

    stagn = 0
    last_best = best

    while now() < deadline:
        remaining = deadline - now()

        # time-aware refinement
        if best_x is not None:
            if remaining > 0.35 * max_time:
                budget = max(2 * dim, 8)
            else:
                budget = max(8 * dim, 24)
            bx, bf = local_refine(best_x, best, budget)
            if bf < best:
                best, best_x = bf, bx

        # sort for p-best pool
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda k: fit[k])

        p = pmin + (pmax - pmin) * random.random()
        p_num = max(2, int(math.ceil(p * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        S_F, S_CR, S_df = [], [], []

        for i in range(pop_size):
            if now() >= deadline:
                return best

            rmem = random.randrange(H)
            Fi = sample_F(M_F[rmem])
            CRi = sample_CR(M_CR[rmem])

            pbest = random.choice(pbest_pool)

            r1 = rand_not(pop_size, i)

            # r2 from union(pop \ {i,r1} + archive) with 50/50 split if archive exists
            if archive and random.random() < 0.5:
                if random.random() < (len(archive) / float(len(archive) + (pop_size - 2))):
                    x_r2 = archive[random.randrange(len(archive))]
                else:
                    r2 = rand_not2(pop_size, i, r1)
                    x_r2 = pop[r2]
            else:
                r2 = rand_not2(pop_size, i, r1)
                x_r2 = pop[r2]

            x_i = pop[i]
            x_pbest = pop[pbest]
            x_r1 = pop[r1]

            # current-to-pbest/1 mutation
            v = [0.0] * dim
            for d in range(dim):
                vd = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])
                v[d] = clip_val(vd, d)

            # binomial crossover
            u = [0.0] * dim
            jrand = random.randrange(dim)
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CRi) else x_i[d]

            fu = eval_f(u)

            if fu <= fit[i]:
                # archive parent
                if len(archive) < archive_max:
                    archive.append(x_i[:])
                else:
                    archive[random.randrange(archive_max)] = x_i[:]

                df = fit[i] - fu
                pop[i] = u
                fit[i] = fu

                if df > 0.0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(df)

                if fu < best:
                    best, best_x = fu, u[:]

        # update SHADE memories
        if S_F:
            w_sum = sum(S_df)
            if w_sum <= 0.0:
                w_sum = 1.0

            mean_cr = 0.0
            for cr, w in zip(S_CR, S_df):
                mean_cr += cr * (w / w_sum)

            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, S_df):
                ww = w / w_sum
                num += ww * (fval * fval)
                den += ww * fval
            mean_f = (num / den) if den > 1e-18 else M_F[mem_idx]

            M_CR[mem_idx] = mean_cr
            M_F[mem_idx] = mean_f
            mem_idx = (mem_idx + 1) % H

        # stagnation logic + partial restart
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            stagn = 0
            last_best = best
        else:
            stagn += 1

        if stagn >= 14 and (deadline - now()) > 0.20 * max_time:
            stagn = 0
            # restart worst third
            idx_worst = list(range(pop_size))
            idx_worst.sort(key=lambda k: fit[k], reverse=True)
            k_restart = max(1, pop_size // 3)

            for k in idx_worst[:k_restart]:
                if now() >= deadline:
                    break
                if best_x is not None and random.random() < 0.85:
                    xnew = best_x[:]
                    for d in range(dim):
                        if spans[d] == 0.0:
                            continue
                        xnew[d] = clip_val(xnew[d] + random.gauss(0.0, 0.18) * spans[d], d)
                else:
                    xnew = [lows[d] + random.random() * spans[d] if spans[d] > 0.0 else lows[d] for d in range(dim)]
                fnew = eval_f(xnew)
                pop[k] = xnew
                fit[k] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            archive = []

    return best
