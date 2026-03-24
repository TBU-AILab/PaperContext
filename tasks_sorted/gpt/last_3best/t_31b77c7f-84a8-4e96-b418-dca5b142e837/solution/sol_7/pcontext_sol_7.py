import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    Main upgrades (vs the provided DE/LSHADE-ish code):
      - Uses a compact CMA-ES core (diagonal + limited full covariance rank-one update)
        that is very strong on continuous problems and works well under tight budgets.
      - Adds restarts with increasing population (IPOP-style) + elitist seeding from
        best solutions seen so far (robust on multimodal functions).
      - Keeps a DE/current-to-pbest/1 "rescue" phase when CMA sampling collapses or
        the objective is very rugged, providing strong global exploration.
      - Uses deterministic-ish low-discrepancy (scrambled Halton) + LHS-like seeding.
      - Uses a light evaluation cache in normalized space to avoid wasted duplicate calls.
      - Always respects max_time.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    # ---------------------------- helpers ----------------------------

    def now():
        return time.time()

    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def reflect_coord(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            else:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scale = [(w if w > 0.0 else 1.0) for w in widths]
    centers = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect_coord(y[i], lo, hi)
        return y

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ---- normalized cache (coarse quantization) ----
    cache = {}
    # choose quant relatively coarse; too fine reduces cache hits and costs time
    q = 2.0e-12

    def key_of(x):
        k = []
        for i in range(dim):
            lo, hi = bounds[i]
            if hi == lo:
                k.append(0)
            else:
                u = (x[i] - lo) / (hi - lo)
                k.append(int(round(u / q)))
        return tuple(k)

    def eval_f(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        fx = float(func(x))
        cache[k] = fx
        return fx

    # ---------------------------- seeding ----------------------------

    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            ok = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(dim)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def vdc_scrambled(n, base):
        v = 0.0
        denom = 1.0
        perm = digit_perm[base]
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += perm[rem] / denom
        return v

    def halton_point(index):
        x = []
        for i in range(dim):
            u = vdc_scrambled(index, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    def opposition(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            y.append(lo + hi - x[i])
        return y

    best = float("inf")
    best_x = centers[:]

    # initial design: center + LHS-like + Halton + random + opposition
    # (kept moderate so we don't burn all time on init)
    seed_n = max(40, min(240, 14 * dim + 60))
    cand = [centers[:]]

    lhs_n = max(12, seed_n // 3)
    strata = []
    for i in range(dim):
        idx = list(range(lhs_n))
        random.shuffle(idx)
        strata.append(idx)
    for k in range(lhs_n):
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            u = (strata[i][k] + random.random()) / lhs_n
            x.append(lo + u * (hi - lo))
        cand.append(x)

    halton_n = max(12, seed_n // 3)
    offset = random.randint(1, 50000)
    for k in range(1, halton_n + 1):
        cand.append(halton_point(offset + k))

    while len(cand) < seed_n:
        cand.append(rand_uniform_vec())

    for x in cand[:max(12, len(cand) // 6)]:
        cand.append(opposition(x))

    elite_pool = []  # keep (f,x) sorted, limited
    elite_max = max(12, min(80, 6 * dim))

    def push_elite(fx, x):
        nonlocal elite_pool
        elite_pool.append((fx, x[:]))
        elite_pool.sort(key=lambda t: t[0])
        if len(elite_pool) > elite_max:
            elite_pool = elite_pool[:elite_max]

    for x in cand:
        if now() >= deadline:
            return best
        x = reflect_into_bounds(x)
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x[:]
        push_elite(fx, x)

    # ---------------------------- CMA-ES core ----------------------------

    # Minimal linear algebra (no numpy)
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def mat_vec(M, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Mi = M[i]
            for j in range(n):
                s += Mi[j] * v[j]
            out[i] = s
        return out

    def outer(u, v):
        n = len(u)
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            ui = u[i]
            row = M[i]
            for j in range(n):
                row[j] = ui * v[j]
        return M

    def mat_add_inplace(A, B, alpha=1.0):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            Bi = B[i]
            for j in range(n):
                Ai[j] += alpha * Bi[j]

    def mat_scale_inplace(A, alpha):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            for j in range(n):
                Ai[j] *= alpha

    def symmetrize_inplace(A):
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                v = 0.5 * (A[i][j] + A[j][i])
                A[i][j] = v
                A[j][i] = v

    # Cholesky decomposition of SPD matrix; add jitter on failure
    def chol_spd(A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 1e-18:
                        return None
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / (L[j][j] + 1e-30)
        return L

    def sample_gauss_vec(n):
        # standard normals using Box-Muller
        out = [0.0] * n
        i = 0
        while i < n:
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(-2.0 * math.log(max(1e-300, u1)))
            z0 = r * math.cos(2.0 * math.pi * u2)
            z1 = r * math.sin(2.0 * math.pi * u2)
            out[i] = z0
            if i + 1 < n:
                out[i + 1] = z1
            i += 2
        return out

    def rank_one_update(C, v, alpha):
        # C <- (1-alpha)C + alpha * v v^T
        mat_scale_inplace(C, (1.0 - alpha))
        ov = outer(v, v)
        mat_add_inplace(C, ov, alpha=alpha)
        symmetrize_inplace(C)

    # We run multiple restarts of CMA; within each we may fall back to DE-rescue
    restart = 0
    base_pop = 4 + int(3 * math.log(dim + 1.0))
    base_pop = max(8, min(40, base_pop))

    # ---------------------------- DE rescue (small, fast) ----------------------------

    def de_rescue(best_x, best_f, time_frac):
        # short DE/current-to-pbest/1 with archive for exploration
        if now() >= deadline:
            return best_f, best_x

        NP = max(18, min(70, 6 * dim + 10))
        # build initial pop around elites + random
        pop = []
        fit = []
        # take some elites
        take = min(len(elite_pool), NP // 2)
        for k in range(take):
            pop.append(elite_pool[k][1][:])
        while len(pop) < NP:
            if random.random() < 0.55 and len(elite_pool) > 0:
                # around a random elite
                _, xb = elite_pool[random.randrange(min(len(elite_pool), 12))]
                x = xb[:]
                for d in range(dim):
                    x[d] += random.gauss(0.0, 0.25 * scale[d])
                pop.append(reflect_into_bounds(x))
            else:
                pop.append(rand_uniform_vec())

        for x in pop:
            if now() >= deadline:
                return best_f, best_x
            fx = eval_f(x)
            fit.append(fx)
            if fx < best_f:
                best_f, best_x = fx, x[:]
                push_elite(fx, x)

        archive = []
        arch_max = 2 * NP

        t_end = t0 + time_frac * float(max_time)
        gen = 0
        muF, muCR = 0.55, 0.6
        while now() < deadline and now() < t_end:
            gen += 1
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            pcount = max(2, int(0.2 * NP))
            succF = []
            succCR = []
            idxs = list(range(NP))
            random.shuffle(idxs)
            for i in idxs:
                if now() >= deadline or now() >= t_end:
                    break
                # sample F/CR
                # Cauchy-like for F
                for _ in range(12):
                    u = random.random()
                    F = muF + 0.1 * math.tan(math.pi * (u - 0.5))
                    if F > 0:
                        break
                if F <= 0:
                    F = 0.3
                if F > 1:
                    F = 1.0
                CR = random.gauss(muCR, 0.12)
                if CR < 0: CR = 0.0
                if CR > 1: CR = 1.0

                pbest = order[random.randrange(pcount)]
                # r1
                r1 = i
                while r1 == i:
                    r1 = random.randrange(NP)
                # r2 in pop U archive
                union = NP + len(archive)
                r2u = random.randrange(union) if union > 0 else random.randrange(NP)
                tries = 0
                while tries < 8 and (r2u == i or r2u == r1):
                    r2u = random.randrange(union)
                    tries += 1

                x_i = pop[i]
                x_p = pop[pbest]
                x_r1 = pop[r1]
                x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

                mutant = [0.0] * dim
                for d in range(dim):
                    mutant[d] = x_i[d] + F * (x_p[d] - x_i[d]) + F * (x_r1[d] - x_r2[d])
                mutant = reflect_into_bounds(mutant)

                trial = x_i[:]
                jrand = random.randrange(dim)
                for d in range(dim):
                    if d == jrand or random.random() < CR:
                        trial[d] = mutant[d]
                trial = reflect_into_bounds(trial)

                f_trial = eval_f(trial)
                if f_trial <= fit[i]:
                    archive.append(x_i[:])
                    if len(archive) > arch_max:
                        del archive[random.randrange(len(archive))]
                    pop[i] = trial
                    fit[i] = f_trial
                    succF.append(F)
                    succCR.append(CR)
                    if f_trial < best_f:
                        best_f, best_x = f_trial, trial[:]
                        push_elite(f_trial, trial)

            if succF:
                # Lehmer mean for F
                num = 0.0
                den = 0.0
                for f in succF:
                    num += f * f
                    den += f
                muF = 0.9 * muF + 0.1 * (num / (den + 1e-12))
                muCR = 0.9 * muCR + 0.1 * (sum(succCR) / float(len(succCR)))

        return best_f, best_x

    # ---------------------------- main multi-restart loop ----------------------------

    # allocate CMA most of time; DE rescue kicks in intermittently
    while now() < deadline:
        # restart-dependent population (IPOP-like)
        lam = min(64 + 4 * dim, int(base_pop * (2 ** restart)))
        lam = max(8, min(lam, 160))
        mu = max(2, lam // 2)

        # weights (log)
        w = []
        for i in range(1, mu + 1):
            w.append(math.log(mu + 0.5) - math.log(i))
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # init mean: from best / elites / center
        if restart == 0:
            mean = best_x[:]
        else:
            if len(elite_pool) > 0 and random.random() < 0.75:
                mean = elite_pool[random.randrange(min(len(elite_pool), 10))][1][:]
            else:
                mean = centers[:]

        # init sigma: based on domain size, shrink with restart a bit
        avgw = sum(scale) / float(dim)
        sigma = 0.22 * avgw * (0.6 ** min(6, restart))
        sigma = max(sigma, 1e-12 * (avgw + 1.0))

        # covariance init: identity * 1
        C = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            C[i][i] = 1.0

        # learning rates (rank-one only; stable and cheap)
        c1 = min(0.2, (2.0 / ((dim + 1.3) ** 2 + mueff)))
        cc = min(0.4, (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim))
        cs = min(0.4, (mueff + 2.0) / (dim + mueff + 5.0))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # evolution paths
        ps = [0.0] * dim
        pc = [0.0] * dim

        # expected norm of N(0,I)
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # cholesky cache
        L = None
        last_chol_gen = -1
        gen = 0
        stall = 0
        last_best = best

        # number of generations before restart trigger
        max_gens = 6 + int(12 * math.sqrt(dim + 1.0)) + 6 * restart

        while now() < deadline and gen < max_gens:
            gen += 1

            if best < last_best - 1e-15:
                last_best = best
                stall = 0
            else:
                stall += 1

            # occasional DE rescue if stalling and time remains
            if stall > 8 and (gen % 6 == 0) and (now() < deadline - 0.02):
                # use a small slice of remaining time
                time_frac = min(0.98, (now() - t0 + 0.12 * max_time) / max_time)
                best, best_x = de_rescue(best_x, best, time_frac)
                stall = 0
                mean = best_x[:]

            # recompute Cholesky occasionally or if sigma changed a lot
            if L is None or (gen - last_chol_gen) >= 4:
                # add jitter until SPD
                jitter = 1e-12
                for _ in range(6):
                    A = [[C[i][j] for j in range(dim)] for i in range(dim)]
                    for i in range(dim):
                        A[i][i] += jitter
                    Ltry = chol_spd(A)
                    if Ltry is not None:
                        L = Ltry
                        last_chol_gen = gen
                        break
                    jitter *= 10.0
                if L is None:
                    # fallback to identity if numerical issues
                    L = [[0.0] * dim for _ in range(dim)]
                    for i in range(dim):
                        L[i][i] = 1.0

            # sample offspring
            arz = []
            arx = []
            arf = []
            for k in range(lam):
                if now() >= deadline:
                    return best
                z = sample_gauss_vec(dim)
                y = mat_vec(L, z)
                x = [mean[i] + sigma * y[i] for i in range(dim)]
                x = reflect_into_bounds(x)
                fx = eval_f(x)

                arz.append(z)
                arx.append(x)
                arf.append(fx)

                if fx < best:
                    best = fx
                    best_x = x[:]
                push_elite(fx, x)

            # sort by fitness
            idx = list(range(lam))
            idx.sort(key=lambda i: arf[i])

            # new mean
            old_mean = mean[:]
            mean = [0.0] * dim
            for j in range(mu):
                xj = arx[idx[j]]
                wj = w[j]
                for i in range(dim):
                    mean[i] += wj * xj[i]
            mean = reflect_into_bounds(mean)

            # compute y = (mean-old_mean)/sigma in coordinate space
            y = [(mean[i] - old_mean[i]) / (sigma + 1e-30) for i in range(dim)]

            # update ps (approx; without C^{-1/2}, still works as step-size controller)
            for i in range(dim):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * y[i]

            # sigma adaptation
            norm_ps = math.sqrt(max(1e-300, dot(ps, ps)))
            sigma *= math.exp((cs / damps) * (norm_ps / (chi_n + 1e-30) - 1.0))

            # update pc
            for i in range(dim):
                pc[i] = (1.0 - cc) * pc[i] + math.sqrt(cc * (2.0 - cc) * mueff) * y[i]

            # rank-one covariance update using pc
            rank_one_update(C, pc, c1)

            # keep C reasonably conditioned (diagonal lower bound)
            for i in range(dim):
                if C[i][i] < 1e-16:
                    C[i][i] = 1e-16

            # very small local coordinate search around best (cheap intensification)
            if now() < deadline and (gen % 4 == 0 or stall > 10):
                x = best_x[:]
                f = best
                # try a few coordinates
                tries = min(dim, 8)
                coords = list(range(dim))
                random.shuffle(coords)
                for t in range(tries):
                    if now() >= deadline:
                        break
                    j = coords[t]
                    step = 0.12 * scale[j] * (0.6 ** min(8, restart))
                    if step <= 0:
                        continue
                    lo, hi = bounds[j]
                    xp = x[:]; xm = x[:]
                    xp[j] = clamp(xp[j] + step, lo, hi)
                    xm[j] = clamp(xm[j] - step, lo, hi)
                    fp = eval_f(xp)
                    fm = eval_f(xm)
                    if fp < f or fm < f:
                        if fp <= fm:
                            x, f = xp, fp
                        else:
                            x, f = xm, fm
                if f < best:
                    best, best_x = f, x[:]
                    push_elite(f, x)
                    mean = best_x[:]
                    stall = 0

            # restart criteria (stall + very small sigma)
            if stall > 18 or sigma < 1e-14 * (avgw + 1.0):
                break

        restart += 1

    return best
