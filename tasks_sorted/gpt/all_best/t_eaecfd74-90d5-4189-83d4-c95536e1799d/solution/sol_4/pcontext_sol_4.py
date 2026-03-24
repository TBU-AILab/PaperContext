import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (no external libraries).

    What’s improved vs the basic DE/rand/1:
      - Better initialization: stratified (LHS-like) + opposition points
      - Much stronger DE variant: current-to-pbest/1 with external archive (JADE)
      - Self-adaptive parameters via success-history memory (SHADE/LSHADE-style)
      - Safer / more robust evaluation (handles exceptions, NaN/inf)
      - Cheap late-stage local improvement around best (coordinate + small Gaussian)
      - Stagnation-triggered partial restart of worst individuals

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
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

    def ensure_bounds(vec):
        for i in range(dim):
            vec[i] = clip_val(vec[i], i)
        return vec

    def is_finite(v):
        return not (v is None or math.isnan(v) or math.isinf(v))

    def eval_f(x):
        # Robust evaluation wrapper
        try:
            v = float(func(x))
            return v if is_finite(v) else float("inf")
        except Exception:
            return float("inf")

    # ---------- population size (balanced for speed/quality) ----------
    # A bit larger than the simple DE to improve robustness, capped for time.
    pop_size = max(14, min(72, 12 + 6 * dim))

    # ---------- LHS-like initialization + opposition ----------
    # Build a cheap stratified sampler: permute bins per dimension.
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
        xo = [lows[d] + highs[d] - x[d] for d in range(dim)]
        ensure_bounds(xo)

        fx = eval_f(x)
        fo = eval_f(xo)
        if fo < fx:
            x, fx = xo, fo

        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # ---------- SHADE memory for F and CR ----------
    H = 8
    M_F = [0.5] * H
    M_CR = [0.8] * H
    mem_idx = 0

    # External archive (JADE idea)
    archive = []
    archive_max = pop_size

    # p-best fraction range for current-to-pbest/1
    pmin, pmax = 0.08, 0.25

    def clamp01(v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def approx_normal():
        # CLT approximation: sum of 4 uniforms - 2 => mean ~0
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    def sample_CR(mu):
        return clamp01(mu + 0.10 * approx_normal())

    def sample_F(mu):
        # Cauchy-like heavy tail around mu; keep in (0, 1]
        for _ in range(8):
            u = random.random()
            f = mu + 0.10 * math.tan(math.pi * (u - 0.5))
            if 0.0 < f <= 1.0:
                return f
        # fallback
        if mu <= 0.0:
            return 0.1
        if mu > 1.0:
            return 1.0
        return mu

    # ---------- lightweight local refinement around best ----------
    def local_refine(x0, f0, max_evals):
        if x0 is None or max_evals <= 0:
            return x0, f0
        x = x0[:]
        f = f0

        steps = [0.12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
        min_steps = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

        evals = 0
        while evals < max_evals and now() < deadline:
            improved = False

            # Coordinate search (+/- step)
            for i in range(dim):
                if evals >= max_evals or now() >= deadline:
                    break
                if spans[i] == 0.0:
                    continue
                if steps[i] < min_steps[i]:
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

            # A couple gaussian probes
            if evals < max_evals and now() < deadline:
                for _ in range(2):
                    if evals >= max_evals or now() >= deadline:
                        break
                    xt = x[:]
                    for i in range(dim):
                        if spans[i] == 0.0:
                            continue
                        xt[i] = clip_val(xt[i] + random.gauss(0.0, 0.35) * steps[i], i)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved = True

            if not improved:
                for i in range(dim):
                    steps[i] *= 0.5

        return x, f

    # ---------- main optimization loop ----------
    stagn = 0
    last_best = best

    while now() < deadline:
        remaining = deadline - now()

        # Time-aware exploitation: refine best more near the end
        if best_x is not None:
            if remaining > 0.35 * max_time:
                budget = max(2 * dim, 8)
            else:
                budget = max(6 * dim, 18)
            bx, bf = local_refine(best_x, best, budget)
            if bf < best:
                best, best_x = bf, bx

        # Rank for p-best pool
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda k: fit[k])

        p = pmin + (pmax - pmin) * random.random()
        p_num = max(2, int(math.ceil(p * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        # Success collections for memory update
        S_F, S_CR, S_df = [], [], []

        for i in range(pop_size):
            if now() >= deadline:
                return best

            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            pbest = random.choice(pbest_pool)
            x_i = pop[i]
            x_pbest = pop[pbest]

            # r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            x_r1 = pop[r1]

            # r2 from population excluding i,r1 or from archive (50/50)
            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                union_n = (pop_size - 2) + len(archive)
                pick = random.randrange(union_n)
                if pick < (pop_size - 2):
                    cnt = -1
                    r2 = None
                    for j in range(pop_size):
                        if j == i or j == r1:
                            continue
                        cnt += 1
                        if cnt == pick:
                            r2 = j
                            break
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[pick - (pop_size - 2)]
            else:
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = random.randrange(pop_size)
                x_r2 = pop[r2]

            # Mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])
                v[d] = clip_val(v[d], d)

            # Binomial crossover
            u = [0.0] * dim
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                u[d] = v[d] if (d == jrand or random.random() < CRi) else x_i[d]

            fu = eval_f(u)

            # Selection + archive update
            if fu <= fit[i]:
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
                    best = fu
                    best_x = u[:]

        # Update memories (weighted; Lehmer mean for F)
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
                ww = (w / w_sum)
                num += ww * (fval * fval)
                den += ww * fval
            mean_f = (num / den) if den > 1e-18 else M_F[mem_idx]

            M_CR[mem_idx] = mean_cr
            M_F[mem_idx] = mean_f
            mem_idx = (mem_idx + 1) % H

        # Stagnation tracking + partial restart
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            stagn = 0
            last_best = best
        else:
            stagn += 1

        if stagn >= 16 and (deadline - now()) > 0.20 * max_time:
            stagn = 0
            idx_worst = list(range(pop_size))
            idx_worst.sort(key=lambda k: fit[k], reverse=True)

            k_restart = max(1, pop_size // 3)
            for k in idx_worst[:k_restart]:
                if now() >= deadline:
                    break
                if best_x is not None and random.random() < 0.80:
                    xnew = best_x[:]
                    for d in range(dim):
                        if spans[d] == 0.0:
                            continue
                        xnew[d] = clip_val(xnew[d] + random.gauss(0.0, 0.20) * spans[d], d)
                else:
                    xnew = [lows[d] + random.random() * spans[d] if spans[d] > 0 else lows[d] for d in range(dim)]
                fnew = eval_f(xnew)
                pop[k] = xnew
                fit[k] = fnew
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]

            archive = []

    return best
