import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs):

    Core: Differential Evolution with modern, strong components:
      - LSHADE-style adaptation (success-history memory for F/CR) + current-to-pbest/1
      - Optional "archive" to increase diversity (JADE idea)
      - Bound handling by repair (clip)
      - Periodic lightweight local refinement around best (coordinate + Gaussian)
      - Stagnation-triggered partial restart
      - Time-aware: always returns best-so-far within max_time

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- bounds ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low<=high).")

    def now():
        return time.time()

    def clip_val(x, i):
        lo = lows[i]
        hi = highs[i]
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def ensure_bounds(vec):
        for i in range(dim):
            vec[i] = clip_val(vec[i], i)
        return vec

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def is_finite(v):
        return not (v is None or math.isnan(v) or math.isinf(v))

    def eval_f(x):
        try:
            v = func(x)
            v = float(v)
            return v if is_finite(v) else float("inf")
        except Exception:
            return float("inf")

    # ---------- population sizing ----------
    # Slightly larger than your best DE for better robustness, still fast.
    pop_size = max(16, min(70, 12 + 6 * dim))

    # ---------- initialize ----------
    pop = []
    fit = []
    best = float("inf")
    best_x = None

    # Opposition-based init (often improves start)
    for _ in range(pop_size):
        if now() >= deadline:
            return best
        x = rand_vec()
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        ensure_bounds(xo)
        fx = eval_f(x)
        fo = eval_f(xo)
        if fo < fx:
            x, fx = xo, fo
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # ---------- LSHADE/JADE-style adaptation memory ----------
    # Memory of successful F and CR; sample around them each trial.
    H = 6  # small memory works well under tight time budgets
    M_F = [0.5] * H
    M_CR = [0.8] * H
    mem_idx = 0

    # Archive for diversity (stores replaced parents)
    archive = []
    archive_max = pop_size

    # p-best fraction range (top p of population used as attractors)
    pmin, pmax = 0.08, 0.25

    # ---------- local refinement (cheap) ----------
    def local_refine(x0, f0, budget):
        if x0 is None:
            return x0, f0
        x = x0[:]
        f = f0

        steps = [0.12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
        min_steps = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

        evals = 0
        while evals < budget and now() < deadline:
            improved = False

            # Coordinate search (+/- step)
            for i in range(dim):
                if evals >= budget or now() >= deadline:
                    break
                if spans[i] == 0:
                    continue
                s = steps[i]
                if s < min_steps[i]:
                    continue

                xp = x[:]
                xp[i] = clip_val(xp[i] + s, i)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= budget or now() >= deadline:
                    break

                xm = x[:]
                xm[i] = clip_val(xm[i] - s, i)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            # A couple Gaussian probes
            if evals < budget and now() < deadline:
                for _ in range(2):
                    if evals >= budget or now() >= deadline:
                        break
                    xt = x[:]
                    for i in range(dim):
                        if spans[i] == 0:
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

    # ---------- main loop ----------
    stagn = 0
    last_best = best

    # Helper to sample CR from normal around memory; clamp [0,1]
    def sample_CR(mu):
        # approximate normal via sum of uniforms (CLT), no external libs
        # mean 0, variance ~ 1
        z = (random.random() + random.random() + random.random() + random.random() - 2.0)
        cr = mu + 0.10 * z
        if cr < 0.0:
            return 0.0
        if cr > 1.0:
            return 1.0
        return cr

    # Helper to sample F from "Cauchy-like" heavy tail around mu (fallback loop)
    def sample_F(mu):
        # Cauchy-ish: mu + scale * tan(pi*(u-0.5))
        # Use math.tan; resample if out of (0,1]
        for _ in range(6):
            u = random.random()
            f = mu + 0.10 * math.tan(math.pi * (u - 0.5))
            if 0.0 < f <= 1.0:
                return f
        # fallback
        f = mu
        if f <= 0.0:
            f = 0.1
        if f > 1.0:
            f = 1.0
        return f

    while now() < deadline:
        remaining = deadline - now()

        # Time-aware exploitation: refine best periodically, more near the end
        if best_x is not None and remaining > 0:
            if remaining > 0.35 * max_time:
                budget = max(2 * dim, 8)
            else:
                budget = max(5 * dim, 16)
            bx, bf = local_refine(best_x, best, budget)
            if bf < best:
                best, best_x = bf, bx

        # Rank indices by fitness for p-best selection
        idx_sorted = list(range(pop_size))
        idx_sorted.sort(key=lambda k: fit[k])

        p = pmin + (pmax - pmin) * random.random()
        p_num = max(2, int(math.ceil(p * pop_size)))
        pbest_pool = idx_sorted[:p_num]

        # Collect successful parameters for memory update
        S_F = []
        S_CR = []
        S_df = []  # fitness improvements for weighting

        # Iterate population
        for i in range(pop_size):
            if now() >= deadline:
                return best

            # choose memory slot
            r = random.randrange(H)
            Fi = sample_F(M_F[r])
            CRi = sample_CR(M_CR[r])

            # pick pbest
            pbest = random.choice(pbest_pool)
            x_i = pop[i]
            x_pbest = pop[pbest]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from union(pop \ {i,r1} + archive); must be distinct from i,r1
            use_archive = (len(archive) > 0 and random.random() < 0.5)
            if use_archive:
                # union size
                union_size = (pop_size - 2) + len(archive)
                # sample index from union
                pick = random.randrange(union_size)
                if pick < (pop_size - 2):
                    # map pick to population index excluding i and r1
                    # build two excluded without building full list too often
                    # simple loop since pop_size is small/moderate
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

            x_r1 = pop[r1]

            # Mutation: current-to-pbest/1 with archive option
            v = [0.0] * dim
            for d in range(dim):
                v[d] = x_i[d] + Fi * (x_pbest[d] - x_i[d]) + Fi * (x_r1[d] - x_r2[d])
                v[d] = clip_val(v[d], d)

            # Binomial crossover
            u = [0.0] * dim
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
                else:
                    u[d] = x_i[d]

            fu = eval_f(u)

            # Selection
            if fu <= fit[i]:
                # push parent into archive
                if len(archive) < archive_max:
                    archive.append(x_i[:])
                else:
                    # replace random
                    archive[random.randrange(archive_max)] = x_i[:]

                df = fit[i] - fu
                pop[i] = u
                fit[i] = fu

                # record successes
                if df > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(df)

                if fu < best:
                    best = fu
                    best_x = u[:]

        # Update memory using Lehmer mean for F and weighted mean for CR
        if S_F:
            w_sum = sum(S_df) if S_df else 1.0
            # weighted means
            mean_cr = 0.0
            for cr, w in zip(S_CR, S_df):
                mean_cr += cr * (w / w_sum)

            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            num = 0.0
            den = 0.0
            for fval, w in zip(S_F, S_df):
                ww = (w / w_sum)
                num += ww * (fval * fval)
                den += ww * fval
            mean_f = num / den if den > 0 else M_F[mem_idx]

            # write into memory
            M_CR[mem_idx] = mean_cr
            M_F[mem_idx] = mean_f
            mem_idx = (mem_idx + 1) % H

        # Stagnation detection & partial restart
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            stagn = 0
            last_best = best
        else:
            stagn += 1

        if stagn >= 18 and (deadline - now()) > 0.20 * max_time:
            stagn = 0
            # restart worst 30-50% around best + random
            idx_sorted = list(range(pop_size))
            idx_sorted.sort(key=lambda k: fit[k], reverse=True)
            k_restart = max(1, pop_size // 3)
            worst = idx_sorted[:k_restart]
            for k in worst:
                if now() >= deadline:
                    break
                if best_x is not None and random.random() < 0.75:
                    xnew = best_x[:]
                    for d in range(dim):
                        if spans[d] == 0:
                            continue
                        # modest radius around best
                        xnew[d] = clip_val(xnew[d] + random.gauss(0.0, 0.18) * spans[d], d)
                else:
                    xnew = rand_vec()
                fnew = eval_f(xnew)
                pop[k] = xnew
                fit[k] = fnew
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]

            # also clear archive to re-diversify properly
            archive = []

    return best
