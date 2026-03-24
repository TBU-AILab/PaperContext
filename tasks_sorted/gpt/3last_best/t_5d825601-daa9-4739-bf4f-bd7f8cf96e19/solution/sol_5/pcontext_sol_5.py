import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Further-improved time-bounded black-box minimizer (stdlib-only).

    Improvements over the best provided algorithm:
      1) Better bound handling: fast "mirror modulo" reflection (no while loops).
      2) Evaluation caching (rounded key) to avoid re-evaluating near-duplicates.
      3) Tri-strategy ensemble with bandit adaptation:
            - current-to-pbest/1 (exploit)
            - rand/1 (explore)
            - best/1 (strong exploit late / when confident)
      4) Eigenvector-free correlation handling: occasional 2D-rotation crossover
         (cheap, helps when variables are coupled).
      5) Stronger local search around best:
            - adaptive coordinate line-search with step halving
            - sparse random pattern steps
      6) More principled stagnation rescue: re-seed worst with mixture of
         opposition-to-best, around-best Cauchy kicks, and pure random.
      7) Dynamic budgets: local search frequency and strength increase with time progress.

    Returns best fitness found within max_time seconds.
    """

    t0 = time.time()
    end_time = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0.0 else 0.0 for s in spans]

    # ---------------- timing ----------------
    def timed_out():
        return time.time() >= end_time

    def frac_time():
        return min(1.0, max(0.0, (time.time() - t0) / (max_time + 1e-12)))

    # ---------------- robust eval + cache ----------------
    # Cache key rounding: coarse early (promote diversity), finer later (avoid duplicates).
    eval_cache = {}
    cache_order = []  # for simple eviction
    cache_cap = 12000  # keep bounded

    def safe_eval(x):
        # Create time-dependent rounding to reduce cache collisions early.
        ft = frac_time()
        # digits: 4 early -> 8 late
        digits = 4 + int(4 * ft)
        key = tuple(round(float(xi), digits) for xi in x)
        if key in eval_cache:
            return eval_cache[key]
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if math.isnan(v) or math.isinf(v):
            v = float("inf")
        eval_cache[key] = v
        cache_order.append(key)
        if len(cache_order) > cache_cap:
            # evict a chunk
            for _ in range(200):
                if not cache_order:
                    break
                k = cache_order.pop(0)
                eval_cache.pop(k, None)
        return v

    # ---------------- vectors ----------------
    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposition(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def reflect_fast_inplace(x):
        # Mirror using modulo in [lo, hi] with period 2*span
        for i in range(dim):
            lo = lows[i]
            s = spans[i]
            if s <= 0.0:
                x[i] = lo
                continue
            hi = highs[i]
            period = 2.0 * s
            y = (x[i] - lo) % period  # in [0,2s)
            if y > s:
                y = period - y
            x[i] = lo + y
            # numerical safety
            if x[i] < lo: x[i] = lo
            if x[i] > hi: x[i] = hi
        return x

    def cauchy(scale=1.0):
        u = random.random()
        return scale * math.tan(math.pi * (u - 0.5))

    # ---------------- SHADE memories ----------------
    def sample_F(mu):
        # Cauchy around mu; ensure (0,1]
        for _ in range(12):
            f = mu + 0.1 * cauchy(1.0)
            if f > 0.0:
                return 1.0 if f > 1.0 else f
        return 0.6 if mu <= 0.0 else (1.0 if mu > 1.0 else mu)

    def sample_CR(mu):
        cr = mu + 0.1 * random.gauss(0.0, 1.0)
        if cr < 0.0: return 0.0
        if cr > 1.0: return 1.0
        return cr

    # ---------------- initialization ----------------
    # Slightly larger initial pop helps global search; cap to control overhead.
    NP_init = max(30, 16 * dim)
    NP_init = min(NP_init, 180)
    NP_min = max(10, 4 * dim)
    if NP_min > NP_init:
        NP_min = max(10, NP_init // 2)

    H = 14
    M_F = [0.6] * H
    M_CR = [0.5] * H
    mem_idx = 0

    p_best_rate = 0.12
    p_best_rate = max(0.05, min(0.30, p_best_rate))

    archive = []
    archive_cap = NP_init

    # Seeding: random + opposition + around-center sprays
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
    pop = []
    while len(pop) < NP_init:
        x = rand_vec()
        pop.append(x)
        if len(pop) < NP_init:
            xo = opposition(x)
            reflect_fast_inplace(xo)
            pop.append(xo)
        if len(pop) < NP_init and random.random() < 0.35:
            xs = center[:]
            # mild random spray near center
            for d in range(dim):
                xs[d] += (random.random() * 2.0 - 1.0) * 0.15 * spans[d]
            reflect_fast_inplace(xs)
            pop.append(xs)
    pop = pop[:NP_init]

    fit = [float("inf")] * len(pop)
    best = float("inf")
    best_x = None

    for i in range(len(pop)):
        if timed_out():
            return best
        fi = safe_eval(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # ---------------- local search ----------------
    ls_sigma = 0.14
    ls_sigma_min = 1e-8
    ls_sigma_max = 0.50

    def local_refine(xb, fb, ft):
        """Adaptive coordinate line-search with step halving + sparse patterns."""
        nonlocal ls_sigma
        if xb is None:
            return xb, fb
        x = xb[:]
        f = fb

        # budgets increase over time, but remain small
        coord_budget = min(dim, 4 + int(10 * ft))
        pattern_budget = 1 + int(3 * ft)

        # Coordinate line search (halve step if not improving)
        idxs = list(range(dim))
        random.shuffle(idxs)
        for t in range(coord_budget):
            if timed_out():
                break
            d = idxs[t % dim]
            if spans[d] <= 0.0:
                continue
            step = ls_sigma * spans[d]
            if step <= 0.0:
                continue

            improved = False
            # small line search: try +/- step, then halve once if fail
            for attempt in range(2):
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[d] = xt[d] + sgn * step
                    reflect_fast_inplace(xt)
                    ftv = safe_eval(xt)
                    if ftv < f:
                        x, f = xt, ftv
                        improved = True
                        break
                if improved:
                    break
                step *= 0.5
                if step < 1e-15:
                    break

            if improved:
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.90)
            else:
                ls_sigma = min(ls_sigma_max, ls_sigma * 1.03)

        # Sparse random pattern moves
        for _ in range(pattern_budget):
            if timed_out():
                break
            xt = x[:]
            prob = 0.08 + 0.18 * (1.0 - ft)  # denser earlier
            for d in range(dim):
                if random.random() < prob:
                    xt[d] += (random.random() * 2.0 - 1.0) * ls_sigma * spans[d]
            reflect_fast_inplace(xt)
            ftv = safe_eval(xt)
            if ftv < f:
                x, f = xt, ftv
                ls_sigma = max(ls_sigma_min, ls_sigma * 0.93)

        return x, f

    # ---------------- crossover helpers ----------------
    def binomial_crossover(x, v, cr):
        jrand = random.randrange(dim)
        u = [0.0] * dim
        for d in range(dim):
            u[d] = v[d] if (d == jrand or random.random() < cr) else x[d]
        return u

    def rotate_2d_crossover(x, v, cr):
        # Start from binomial, then rotate 1-2 random planes on (u-x)
        u = binomial_crossover(x, v, cr)
        if dim < 2:
            reflect_fast_inplace(u)
            return u
        k = 1 if dim < 10 else 2
        for _ in range(k):
            a = random.randrange(dim)
            b = random.randrange(dim)
            if a == b:
                continue
            theta = (random.random() * 2.0 - 1.0) * (math.pi / 3.0)
            cth = math.cos(theta)
            sth = math.sin(theta)
            da = u[a] - x[a]
            db = u[b] - x[b]
            ra = cth * da - sth * db
            rb = sth * da + cth * db
            u[a] = x[a] + ra
            u[b] = x[b] + rb
        reflect_fast_inplace(u)
        return u

    # ---------------- bandit strategies ----------------
    # 0: current-to-pbest/1
    # 1: rand/1
    # 2: best/1 (aggressive exploitation, especially later)
    strat_w = [1.0, 1.0, 0.6]
    strat_decay = 0.90
    minw = 1e-8

    no_improve_gens = 0
    stagnation_limit = max(10, 3 * dim)

    gen = 0
    while not timed_out():
        gen += 1
        ft = frac_time()

        # linear population size reduction
        NP_target = int(round(NP_init - ft * (NP_init - NP_min)))
        if NP_target < NP_min:
            NP_target = NP_min

        NP = len(pop)
        if NP_target < NP:
            keep = sorted(range(NP), key=lambda i: fit[i])[:NP_target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = NP_target
            archive_cap = max(NP, NP_init // 2)
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

        # ranking
        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        p_count = max(2, int(math.ceil(p_best_rate * NP)))

        # success histories
        S_F, S_CR, S_df = [], [], []
        improved_gen = False
        strat_gain = [0.0, 0.0, 0.0]

        def pick_pop(excl):
            while True:
                j = random.randrange(NP)
                if j not in excl:
                    return j

        for i in range(NP):
            if timed_out():
                return best

            xi, fi = pop[i], fit[i]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # choose strategy
            sw = strat_w[0] + strat_w[1] + strat_w[2]
            rsel = random.random() * sw
            if rsel < strat_w[0]:
                strat = 0
            elif rsel < strat_w[0] + strat_w[1]:
                strat = 1
            else:
                strat = 2

            if strat == 0:
                # current-to-pbest/1
                pbest = pop[idx_sorted[random.randrange(p_count)]]
                r1 = pick_pop({i})
                use_arch = (archive and random.random() < 0.5)
                if use_arch:
                    combined_n = NP + len(archive)
                    while True:
                        j = random.randrange(combined_n)
                        if j == i or j == r1:
                            continue
                        xr2 = pop[j] if j < NP else archive[j - NP]
                        break
                else:
                    r2 = pick_pop({i, r1})
                    xr2 = pop[r2]
                xr1 = pop[r1]
                v = [xi[d] + F * (pbest[d] - xi[d]) + F * (xr1[d] - xr2[d]) for d in range(dim)]

            elif strat == 1:
                # rand/1 (diversity)
                r1 = pick_pop({i})
                r2 = pick_pop({i, r1})
                x1 = pop[r1]
                x2 = pop[r2]
                use_arch = (archive and random.random() < 0.35)
                if use_arch:
                    x3 = archive[random.randrange(len(archive))]
                else:
                    r3 = pick_pop({i, r1, r2})
                    x3 = pop[r3]
                v = [x1[d] + F * (x2[d] - x3[d]) for d in range(dim)]

            else:
                # best/1 (more often later)
                if best_x is None:
                    best_base = pop[idx_sorted[0]]
                else:
                    best_base = best_x
                r1 = pick_pop({i})
                use_arch = (archive and random.random() < 0.4)
                if use_arch:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_pop({i, r1})
                    x2 = pop[r2]
                x1 = pop[r1]
                v = [best_base[d] + F * (x1[d] - x2[d]) for d in range(dim)]

            reflect_fast_inplace(v)

            # crossover: sometimes rotate planes to cope with correlations
            if dim >= 2 and random.random() < (0.10 + 0.10 * (1.0 - ft)):
                u = rotate_2d_crossover(xi, v, CR)
            else:
                u = binomial_crossover(xi, v, CR)
                reflect_fast_inplace(u)

            # mild best-injection late
            if best_x is not None and random.random() < (0.006 + 0.06 * ft):
                d = random.randrange(dim)
                u[d] = 0.82 * u[d] + 0.18 * best_x[d]
                reflect_fast_inplace(u)

            fu = safe_eval(u)

            if fu <= fi:
                archive.append(xi[:])
                if len(archive) > archive_cap:
                    archive.pop(random.randrange(len(archive)))

                pop[i] = u
                fit[i] = fu

                df = fi - fu
                if df > 0.0:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(df)
                    strat_gain[strat] += df

                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_gen = True

        # update SHADE memories
        if S_F:
            wsum = sum(S_df)
            if wsum <= 0.0:
                weights = [1.0 / float(len(S_df))] * len(S_df)
            else:
                inv = 1.0 / wsum
                weights = [df * inv for df in S_df]

            meanCR = 0.0
            for w, cr in zip(weights, S_CR):
                meanCR += w * cr

            num = 0.0
            den = 0.0
            for w, f in zip(weights, S_F):
                num += w * f * f
                den += w * f
            meanF = (num / den) if den != 0.0 else M_F[mem_idx]

            M_CR[mem_idx] = min(1.0, max(0.0, meanCR))
            M_F[mem_idx] = min(1.0, max(0.05, meanF))
            mem_idx = (mem_idx + 1) % H

        # bandit update
        for s in range(3):
            strat_w[s] = strat_w[s] * strat_decay + (strat_gain[s] + 1e-12)
            if strat_w[s] < minw:
                strat_w[s] = minw

        # local improvement schedule
        if best_x is not None and not timed_out():
            p_ref = 0.05 + 0.45 * ft
            if improved_gen or random.random() < p_ref:
                bx, bf = local_refine(best_x, best, ft)
                if bf < best:
                    best, best_x = bf, bx
                    improved_gen = True

        # stagnation rescue
        if improved_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= stagnation_limit and not timed_out():
            no_improve_gens = 0
            NP = len(pop)
            k = max(2, NP // 5)
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]

            for wi in worst:
                if timed_out():
                    return best
                rmode = random.random()
                if best_x is not None and rmode < 0.40:
                    # opposition-to-best (jumps to other side)
                    x = opposition(best_x)
                    # add small noise
                    rad = (0.18 * (1.0 - ft) + 0.02)
                    for d in range(dim):
                        if random.random() < 0.25:
                            x[d] += (random.random() * 2.0 - 1.0) * rad * spans[d]
                    reflect_fast_inplace(x)
                elif best_x is not None and rmode < 0.80:
                    # around-best heavy-tailed kick
                    x = best_x[:]
                    kick = (0.20 * (1.0 - ft) + 0.03)
                    m = 1 + int(0.12 * dim)
                    for _ in range(m):
                        d = random.randrange(dim)
                        x[d] += cauchy(1.0) * kick * spans[d]
                    reflect_fast_inplace(x)
                else:
                    x = rand_vec()

                fx = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best, best_x = fx, x[:]

            # thin archive
            if len(archive) > archive_cap:
                random.shuffle(archive)
                archive = archive[:archive_cap]

    return best
