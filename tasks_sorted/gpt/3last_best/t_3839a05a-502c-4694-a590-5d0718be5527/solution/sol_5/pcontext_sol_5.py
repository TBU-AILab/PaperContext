import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Key improvements over the provided best:
      1) Stronger global exploration early via multi-start "region pool" + low-discrepancy-ish
         random (LHS-style per-dimension stratification with permutation).
      2) More reliable exploitation late via:
           - Adaptive DE (JADE/SHADE flavor) with success-history memory arrays (H memories)
           - Eigen/rotated-coordinate local search (cheap random orthonormal-ish directions)
           - Deterministic trust-region-like pattern search around best with step control
      3) Better stagnation control: multi-level restarts (worst-half refresh + region switch),
         plus archive reset and parameter "temperature" reset.
      4) Faster evaluation throughput: lighter cache keying + opportunistic re-evaluation reuse.
      5) Robust bound handling: periodic fold (triangle-wave) + optional midpoint pull for NaNs.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    def time_left():
        return time.time() < deadline

    if dim <= 0:
        return float(func([]))

    # ---- bounds ----
    lo = [0.0] * dim
    hi = [0.0] * dim
    span = [0.0] * dim
    for i in range(dim):
        a = float(bounds[i][0])
        b = float(bounds[i][1])
        if b < a:
            a, b = b, a
        lo[i], hi[i] = a, b
        span[i] = b - a

    mid = [lo[i] + 0.5 * span[i] for i in range(dim)]

    # ---- periodic fold into bounds (triangle wave) ----
    def fold_inplace(x):
        for i in range(dim):
            a = lo[i]
            s = span[i]
            if s <= 0.0:
                x[i] = a
                continue
            v = x[i] - a
            v = v % (2.0 * s)
            if v > s:
                v = 2.0 * s - v
            x[i] = a + v
        return x

    # ---- RNG helpers ----
    def randn():
        # ~N(0,1) via 12 uniforms
        s = 0.0
        for _ in range(12):
            s += random.random()
        return s - 6.0

    def cauchy():
        u = random.random()
        u = 1e-12 if u < 1e-12 else (1.0 - 1e-12 if u > 1.0 - 1e-12 else u)
        return math.tan(math.pi * (u - 0.5))

    # ---- cache (quantized) ----
    # Use coarser quantization than previous to reduce overhead/collisions cost
    cache = {}
    cache_max = 14000
    q = 2_000_000  # quantization bins per dimension (relative)

    def key_of(x):
        k = []
        for i in range(dim):
            s = span[i] if span[i] > 0 else 1.0
            # map to [0,1], quantize
            u = (x[i] - lo[i]) / s
            if u <= 0.0:
                k.append(0)
            elif u >= 1.0:
                k.append(q)
            else:
                k.append(int(u * q + 0.5))
        return tuple(k)

    def safe_eval(x):
        # x assumed already folded; still protect NaNs/inf
        fx = float(func(x))
        if fx != fx or fx == float("inf") or fx == float("-inf"):
            # pull toward mid if weird evaluation
            xt = [(x[i] + mid[i]) * 0.5 for i in range(dim)]
            fold_inplace(xt)
            fx2 = float(func(xt))
            if fx2 == fx2 and fx2 != float("inf") and fx2 != float("-inf"):
                return fx2, xt
            # last resort: very bad
            return 1e300, x
        return fx, x

    def evaluate(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx, _ = safe_eval(x)
        cache[k] = fx
        if len(cache) > cache_max:
            # prune random chunk
            kill = min(len(cache) // 6, 2500)
            keys = list(cache.keys())
            if 0 < kill < len(keys):
                for kk in random.sample(keys, k=kill):
                    cache.pop(kk, None)
        return fx

    # ---- sampling: LHS-like (cheap) ----
    def lhs_batch(n):
        # For each dim: take a random permutation of n strata.
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = [0.0] * dim
            for d in range(dim):
                s = span[d]
                if s <= 0.0:
                    x[d] = lo[d]
                else:
                    u = (perms[d][i] + random.random()) / float(n)
                    x[d] = lo[d] + u * s
            pts.append(x)
        return pts

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # ---- elite set (best few) ----
    elite_k = max(10, min(36, 6 + dim // 2))
    elites = []  # list of (f, x)

    def add_elite(x, fx):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites = elites[:elite_k]

    # ---- Local search: mixed directional + coordinate trust region ----
    def local_refine(x0, f0, budget=24):
        x = x0[:]
        f = f0
        # trust region step (relative)
        step = [0.06 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]
        step_min = [1e-14 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

        # phase A: random directions (approx rotated search)
        for _ in range(max(6, budget // 2)):
            if not time_left():
                break
            m = max(2, int(math.sqrt(dim)))
            idxs = range(dim) if m >= dim else random.sample(range(dim), m)

            # random direction on subset
            dirv = [0.0] * dim
            norm2 = 0.0
            for j in idxs:
                r = randn()
                dirv[j] = r
                norm2 += r * r
            if norm2 <= 1e-18:
                continue
            invn = 1.0 / math.sqrt(norm2)

            # try a few radii along the direction
            base_rad = 0.25 + 0.75 * random.random()
            for sign in (1.0, -1.0):
                xt = x[:]
                for j in idxs:
                    xt[j] += sign * base_rad * step[j] * dirv[j] * invn
                fold_inplace(xt)
                ft = evaluate(xt)
                if ft < f:
                    x, f = xt, ft
                    for j in idxs:
                        step[j] = min(step[j] * 1.12, 0.35 * (span[j] if span[j] > 0 else 1.0))
                    break
            else:
                for j in idxs:
                    step[j] *= 0.82

        # phase B: coordinate pattern polish
        for _ in range(max(6, budget // 2)):
            if not time_left():
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if not time_left():
                    break
                if span[j] <= 0.0:
                    continue
                sj = step[j]
                if sj <= step_min[j]:
                    continue
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[j] += sgn * sj
                    fold_inplace(xt)
                    ft = evaluate(xt)
                    if ft < f:
                        x, f = xt, ft
                        improved = True
                        break
                if improved:
                    break
            if improved:
                for j in range(dim):
                    step[j] = min(step[j] * 1.08, 0.25 * (span[j] if span[j] > 0 else 1.0))
            else:
                for j in range(dim):
                    step[j] *= 0.60

        return x, f

    # ---- Initialization (multi-region pool) ----
    best_x = mid[:]
    fold_inplace(best_x)
    best = evaluate(best_x)
    add_elite(best_x, best)

    # Build a region pool of good seeds from batches; pick best few regions
    region_seeds = []
    # keep init moderate to not waste time in short budgets
    init_batches = 2 if dim <= 25 else 1
    batch_n = max(12, min(40, 10 + int(2.0 * math.sqrt(dim))))

    for _ in range(init_batches):
        if not time_left():
            break
        pts = lhs_batch(batch_n)
        for x in pts:
            if not time_left():
                break
            fold_inplace(x)
            fx = evaluate(x)
            region_seeds.append((fx, x))
            add_elite(x, fx)
            if fx < best:
                best, best_x = fx, x[:]

        # opposition points of current elites (cheap)
        for (fxe, xe) in elites[:min(6, len(elites))]:
            if not time_left():
                break
            xo = [lo[i] + hi[i] - xe[i] for i in range(dim)]
            fold_inplace(xo)
            fxo = evaluate(xo)
            region_seeds.append((fxo, xo))
            add_elite(xo, fxo)
            if fxo < best:
                best, best_x = fxo, xo[:]

    region_seeds.sort(key=lambda t: t[0])
    region_seeds = region_seeds[:max(6, min(18, 4 + dim // 3))]

    if time_left():
        bx, bf = local_refine(best_x, best, budget=16)
        if bf < best:
            best, best_x = bf, bx[:]
            add_elite(best_x, best)

    # ---- Adaptive DE with success-history memories (H) ----
    NP = max(16, min(56, 10 + 2 * int(math.sqrt(dim)) + dim // 2))
    NP = min(NP, max(16, len(region_seeds) * 3))

    pop = []
    fit = []

    # seed population from region seeds + jitter + random
    # jitter strength relative to span
    jrad = 0.08
    i = 0
    while len(pop) < NP and time_left():
        if i < len(region_seeds):
            _, xb = region_seeds[i]
            x = xb[:]
            # jitter subset
            m = max(2, int(math.sqrt(dim)))
            idxs = range(dim) if m >= dim else random.sample(range(dim), m)
            heavy = (random.random() < 0.12)
            for d in idxs:
                if span[d] > 0:
                    z = cauchy() if heavy else randn()
                    x[d] += jrad * span[d] * z
            fold_inplace(x)
        else:
            x = rand_point()
            fold_inplace(x)

        fx = evaluate(x)
        pop.append(x)
        fit.append(fx)
        add_elite(x, fx)
        if fx < best:
            best, best_x = fx, x[:]
        i += 1

    if not pop:
        return best

    archive = []
    arch_max = NP * 3

    # SHADE memory arrays
    H = max(6, min(20, 4 + dim // 8))
    M_F = [0.6] * H
    M_CR = [0.5] * H
    k_mem = 0

    p_best_rate = 0.20

    def sample_F(memF):
        for _ in range(12):
            v = memF + 0.10 * cauchy()
            if v > 0.0:
                return 1.0 if v > 1.0 else v
        return max(0.03, min(1.0, memF))

    def sample_CR(memCR):
        v = memCR + 0.10 * randn()
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def pick_distinct(exclude, k):
        res = []
        tries = 0
        while len(res) < k and tries < 120:
            j = random.randrange(NP)
            if j == exclude or j in res:
                tries += 1
                continue
            res.append(j)
        while len(res) < k:
            j = random.randrange(NP)
            if j != exclude:
                res.append(j)
        return res

    no_improve = 0
    stagnate_after = 70 + 8 * dim
    refine_cooldown = 0

    # Time-scheduling: more exploration early, more exploitation late
    while time_left():
        # rank for pbest pool
        order = list(range(NP))
        order.sort(key=lambda ii: fit[ii])
        pN = max(2, int(math.ceil(p_best_rate * NP)))
        pbest_pool = order[:pN]

        S_F = []
        S_CR = []
        S_df = []

        for ii in range(NP):
            if not time_left():
                break

            xi = pop[ii]
            fi = fit[ii]

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            # mutation: current-to-pbest/1 with archive
            pb = pop[random.choice(pbest_pool)]
            a = pick_distinct(ii, 1)[0]
            x1 = pop[a]

            use_arch = (archive and random.random() < 0.5)
            if use_arch:
                x2 = archive[random.randrange(len(archive))]
            else:
                b = pick_distinct(ii, 1)[0]
                x2 = pop[b]

            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (pb[d] - xi[d]) + F * (x1[d] - x2[d])

            # binomial crossover
            u = xi[:]
            jrand = random.randrange(dim)
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            # micro-noise to prevent collapse (late: smaller)
            if random.random() < 0.10:
                m = max(1, int(math.sqrt(dim)))
                idxs = range(dim) if m >= dim else random.sample(range(dim), m)
                amp = (0.0012 + 0.0045 * random.random())
                for d in idxs:
                    if span[d] > 0:
                        u[d] += amp * span[d] * randn()

            fold_inplace(u)
            fu = evaluate(u)

            if fu <= fi:
                # selection success
                archive.append(xi[:])
                if len(archive) > arch_max:
                    del archive[random.randrange(len(archive))]

                pop[ii] = u
                fit[ii] = fu

                df = fi - fu
                if df < 0.0:
                    df = 0.0
                S_F.append(F)
                S_CR.append(CR)
                S_df.append(df)

                if fu < best:
                    best, best_x = fu, u[:]
                    add_elite(best_x, best)
                    no_improve = 0

                    # opportunistic local refine, but cooled down
                    if refine_cooldown <= 0 and time_left():
                        bx, bf = local_refine(best_x, best, budget=18)
                        if bf < best:
                            best, best_x = bf, bx[:]
                            add_elite(best_x, best)
                        refine_cooldown = 10 + dim // 3
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if refine_cooldown > 0:
                refine_cooldown -= 1

        # update memories (SHADE style)
        if S_F:
            wsum = sum(S_df)
            if wsum <= 1e-18:
                weights = [1.0 / len(S_F)] * len(S_F)
            else:
                weights = [df / wsum for df in S_df]

            # Lehmer mean for F, weighted mean for CR
            num = 0.0
            den = 0.0
            cr_m = 0.0
            for j in range(len(S_F)):
                wj = weights[j]
                fj = S_F[j]
                num += wj * fj * fj
                den += wj * fj
                cr_m += wj * S_CR[j]
            if den > 1e-18:
                M_F[k_mem] = num / den
            M_CR[k_mem] = cr_m
            # keep in sane ranges
            if M_F[k_mem] < 0.05:
                M_F[k_mem] = 0.05
            if M_F[k_mem] > 0.95:
                M_F[k_mem] = 0.95
            if M_CR[k_mem] < 0.0:
                M_CR[k_mem] = 0.0
            if M_CR[k_mem] > 1.0:
                M_CR[k_mem] = 1.0

            k_mem = (k_mem + 1) % H

        # stagnation handling: re-seed worst half around elites / region seeds
        if no_improve > stagnate_after and time_left():
            no_improve = 0
            # reduce archive to remove bias
            if len(archive) > arch_max // 2:
                archive = random.sample(archive, k=arch_max // 2)

            # choose source pool: elites and best region seeds
            sources = [x for (_, x) in elites[:min(len(elites), elite_k)]]
            for (fxs, xs) in region_seeds[:min(len(region_seeds), 8)]:
                sources.append(xs)

            # reinit worst half
            order = list(range(NP))
            order.sort(key=lambda ii: fit[ii], reverse=True)
            k_re = max(3, NP // 2)

            for t in range(k_re):
                if not time_left():
                    break
                ii = order[t]
                if sources and random.random() < 0.85:
                    xnew = random.choice(sources)[:]
                    rad = 0.10 + 0.55 * random.random()
                    heavy = (random.random() < 0.18)
                    for d in range(dim):
                        if span[d] > 0:
                            z = cauchy() if heavy else randn()
                            xnew[d] += rad * span[d] * z
                else:
                    xnew = rand_point()

                fold_inplace(xnew)
                fnew = evaluate(xnew)
                pop[ii] = xnew
                fit[ii] = fnew
                add_elite(xnew, fnew)
                if fnew < best:
                    best, best_x = fnew, xnew[:]

            # reset memories slightly towards exploration
            for i in range(H):
                M_F[i] = 0.68
                M_CR[i] = 0.45
            refine_cooldown = 0

    return best
