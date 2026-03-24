import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libraries).

    Core idea (robust in limited time):
      1) Strong global search: L-SHADE/JADE-style DE (current-to-pbest/1 + archive)
         with success-history memory for F/CR and linear population reduction.
      2) Stronger local search than the previous versions:
         - (a) "Rosenbrock-style" adaptive random subspace search around best
             (keeps an internal orthonormal basis, adapts per-direction steps).
         - (b) Occasional coordinate pattern search + quadratic interpolation.
         This significantly improves exploitation late in the run and helps
         close the gap after DE finds a good basin.
      3) Better time usage:
         - Keep DE running but allocate a growing fraction of evaluations to local
           improvement as time progresses.
      4) Safer cache:
         - Coarser normalized quantization (much larger q than 1e-10) to actually
           get cache hits and avoid pointless repeats, but not too coarse.

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float(func([]))

    # ------------------------- helpers -------------------------

    def now():
        return time.time()

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    widths = []
    scale = []
    centers = []
    for i in range(dim):
        lo, hi = bounds[i]
        w = hi - lo
        widths.append(w)
        s = w if w > 0.0 else 1.0
        scale.append(s)
        centers.append((lo + hi) * 0.5)
    avgw = sum(scale) / float(dim)

    def reflect_coord(x, lo, hi):
        if lo == hi:
            return lo
        # reflect repeatedly if far out
        while x < lo or x > hi:
            if x < lo:
                x = lo + (lo - x)
            else:
                x = hi - (x - hi)
        return clamp(x, lo, hi)

    def reflect_into_bounds(x):
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect_coord(y[i], lo, hi)
        return y

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ------------------------- cache -------------------------
    # Prior q=1e-10 was too fine to be useful for typical noisy/continuous funcs.
    # Use a coarser normalized grid so repeated points / near-identical points hit.
    cache = {}
    q = 1e-6  # normalized quantization resolution

    def key_of(x):
        k = []
        for i in range(dim):
            lo, hi = bounds[i]
            if hi == lo:
                k.append(0)
            else:
                u = (x[i] - lo) / (hi - lo)
                k.append(int(u / q))
        return tuple(k)

    def eval_f(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        return fx

    def diversity(pop, best_x):
        if not pop:
            return 0.0
        s = 0.0
        for x in pop:
            d = 0.0
            for i in range(dim):
                d += abs(x[i] - best_x[i]) / (scale[i] + 1e-12)
            s += d / dim
        return s / len(pop)

    # ------------------------- seeding (scrambled Halton + stratified) -------------------------

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

    # ------------------------- local search 1: coord pattern + quadratic probe -------------------------

    coord_step = [0.15 * s for s in scale]
    coord_min = [1e-14 * (s + 1.0) for s in scale]
    coord_max = [0.75 * s for s in scale]

    def local_coord_quadratic(x0, f0, budget_evals):
        x = x0[:]
        f = f0
        used = 0
        while used < budget_evals and now() < deadline:
            idxs = list(range(dim))
            random.shuffle(idxs)
            m = min(dim, max(2, dim // 3))
            improved = False
            for t in range(m):
                if used >= budget_evals or now() >= deadline:
                    break
                j = idxs[t]
                s = coord_step[j]
                if s < coord_min[j]:
                    continue
                lo, hi = bounds[j]

                xp = x[:]
                xm = x[:]
                xp[j] = clamp(xp[j] + s, lo, hi)
                xm[j] = clamp(xm[j] - s, lo, hi)

                fp = eval_f(xp); used += 1
                if used >= budget_evals or now() >= deadline:
                    if fp < f:
                        x, f = xp, fp
                        improved = True
                    break
                fm = eval_f(xm); used += 1

                if fp < f or fm < f:
                    if fp <= fm:
                        x, f = xp, fp
                    else:
                        x, f = xm, fm
                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.25)
                    improved = True
                else:
                    # quadratic interpolation around x[j]
                    if random.random() < 0.30 and used < budget_evals and now() < deadline:
                        denom = (fm - 2.0 * f + fp)
                        if denom != 0.0:
                            tstar = 0.5 * (fm - fp) / denom
                            if -1.0 <= tstar <= 1.0:
                                xv = x[:]
                                xv[j] = clamp(xv[j] + tstar * s, lo, hi)
                                fv = eval_f(xv); used += 1
                                if fv < f:
                                    x, f = xv, fv
                                    coord_step[j] = min(coord_max[j], coord_step[j] * 1.15)
                                    improved = True
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.70)

            if not improved:
                for j in range(dim):
                    coord_step[j] = max(coord_min[j], coord_step[j] * 0.90)
                break
        return f, x

    # ------------------------- local search 2: Rosenbrock-style adaptive subspace search -------------------------
    # Maintains an orthonormal basis B and per-direction steps alpha.
    # Tries +/- alpha_k along each direction; if improvement, expand, else shrink.
    # Periodically re-orthonormalizes and can rotate basis with random perturbations.

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def norm(a):
        return math.sqrt(dot(a, a))

    def axpy(a, x, y):
        # a*x + y
        return [a * xi + yi for xi, yi in zip(x, y)]

    def gram_schmidt(vectors):
        B = []
        for v in vectors:
            u = v[:]
            for b in B:
                proj = dot(u, b)
                if proj != 0.0:
                    u = axpy(-proj, b, u)
            nu = norm(u)
            if nu > 1e-18:
                B.append([ui / nu for ui in u])
        # If rank-deficient, complete with coordinate axes
        if len(B) < dim:
            for k in range(dim):
                e = [0.0] * dim
                e[k] = 1.0
                u = e
                for b in B:
                    proj = dot(u, b)
                    if proj != 0.0:
                        u = axpy(-proj, b, u)
                nu = norm(u)
                if nu > 1e-18:
                    B.append([ui / nu for ui in u])
                if len(B) >= dim:
                    break
        return B[:dim]

    # initial basis: coordinate axes (fast)
    B = [[1.0 if i == j else 0.0 for i in range(dim)] for j in range(dim)]
    alpha = [0.12 * s for s in scale]
    alpha_min = [1e-14 * (s + 1.0) for s in scale]
    alpha_max = [0.9 * s for s in scale]

    def rosenbrock_local(x0, f0, budget_evals, progress):
        x = x0[:]
        f = f0
        used = 0

        # more aggressive expansions late
        expand = 1.35 if progress > 0.6 else 1.25
        shrink = 0.62 if progress > 0.6 else 0.70

        # use a subset of directions each call (cheaper)
        # but include more directions late
        k_dir = min(dim, max(2, int(0.35 * dim + 2 + 0.35 * dim * progress)))

        # shuffled direction order
        dir_idx = list(range(dim))
        random.shuffle(dir_idx)
        dir_idx = dir_idx[:k_dir]

        for idx in dir_idx:
            if used + 1 >= budget_evals or now() >= deadline:
                break

            dvec = B[idx]
            step = alpha[idx]
            if step < alpha_min[idx]:
                continue

            # try +step
            xp = x[:]
            for i in range(dim):
                xp[i] += step * dvec[i]
            xp = reflect_into_bounds(xp)
            fp = eval_f(xp); used += 1

            if fp < f:
                x, f = xp, fp
                alpha[idx] = min(alpha_max[idx], alpha[idx] * expand)
                continue

            if used + 1 >= budget_evals or now() >= deadline:
                alpha[idx] = max(alpha_min[idx], alpha[idx] * shrink)
                break

            # try -step
            xm = x[:]
            for i in range(dim):
                xm[i] -= step * dvec[i]
            xm = reflect_into_bounds(xm)
            fm = eval_f(xm); used += 1

            if fm < f:
                x, f = xm, fm
                alpha[idx] = min(alpha_max[idx], alpha[idx] * expand)
            else:
                alpha[idx] = max(alpha_min[idx], alpha[idx] * shrink)

        # occasional basis refresh/rotation
        if used > 0 and random.random() < (0.10 + 0.15 * progress):
            # create perturbed directions biased toward successful larger steps
            vecs = []
            # include current basis directions with random mixing
            for _ in range(min(dim, 6)):
                v = [0.0] * dim
                # sparse mix
                for _t in range(3):
                    j = random.randrange(dim)
                    sgn = -1.0 if random.random() < 0.5 else 1.0
                    w = (0.5 + random.random()) * (alpha[j] / (scale[j] + 1e-12))
                    for i in range(dim):
                        v[i] += sgn * w * B[j][i]
                # small random noise
                for i in range(dim):
                    v[i] += 0.05 * (random.random() - 0.5)
                vecs.append(v)
            vecs.extend(B)
            newB = gram_schmidt(vecs)
            # blend old/new a bit (keep stability)
            B[:] = newB

        return f, x

    # ------------------------- DE (L-SHADE style) -------------------------

    NP_max = max(30, min(140, 12 * dim + 20))
    NP_min = max(12, min(40, 4 * dim + 8))
    NP = NP_max

    seed_n = max(NP, min(800, 18 * dim + 80))
    candidates = [centers[:]]

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
        candidates.append(x)

    halton_n = max(12, seed_n // 3)
    offset = random.randint(1, 60000)
    for k in range(1, halton_n + 1):
        candidates.append(halton_point(offset + k))

    while len(candidates) < seed_n:
        candidates.append(rand_uniform_vec())

    for x in candidates[:max(16, len(candidates) // 6)]:
        candidates.append(opposition(x))

    best = float("inf")
    best_x = centers[:]

    scored = []
    for x in candidates:
        if now() >= deadline:
            return best
        x = reflect_into_bounds(x)
        fx = eval_f(x)
        scored.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    scored.sort(key=lambda t: t[0])
    pop = [scored[i][1][:] for i in range(min(NP, len(scored)))]
    while len(pop) < NP and now() < deadline:
        x = reflect_into_bounds(rand_uniform_vec())
        fx = eval_f(x)
        pop.append(x)
        if fx < best:
            best, best_x = fx, x[:]

    fit = [eval_f(x) for x in pop]
    for i in range(NP):
        if fit[i] < best:
            best, best_x = fit[i], pop[i][:]

    archive = []
    archive_max = 2 * NP_max

    H = 10
    M_F = [0.6] * H
    M_CR = [0.6] * H
    k_mem = 0

    def sample_F(mf):
        for _ in range(25):
            u = random.random()
            F = mf + 0.1 * math.tan(math.pi * (u - 0.5))
            if F > 0.0:
                return 1.0 if F > 1.0 else F
        return max(0.05, min(1.0, mf))

    def sample_CR(mcr):
        cr = random.gauss(mcr, 0.1)
        if cr < 0.0:
            return 0.0
        if cr > 1.0:
            return 1.0
        return cr

    gen = 0
    stagn = 0
    last_best = best
    last_improve_time = now()
    T = max(1e-9, float(max_time))

    while now() < deadline:
        gen += 1

        if best < last_best - 1e-15:
            last_best = best
            stagn = 0
            last_improve_time = now()
        else:
            stagn += 1

        progress = (now() - t0) / T
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0

        # population reduction
        target_NP = int(round(NP_max - progress * (NP_max - NP_min)))
        if target_NP < NP_min:
            target_NP = NP_min
        if target_NP < NP:
            order = list(range(NP))
            order.sort(key=lambda i: fit[i])
            keep_idx = order[:target_NP]
            pop = [pop[i] for i in keep_idx]
            fit = [fit[i] for i in keep_idx]
            NP = target_NP
            if len(archive) > 2 * NP:
                archive = archive[-(2 * NP):]

        order = list(range(NP))
        order.sort(key=lambda i: fit[i])
        pfrac = 0.25 - 0.12 * progress
        if pfrac < 0.08:
            pfrac = 0.08
        pcount = max(2, int(pfrac * NP))

        S_F, S_CR, W = [], [], []

        idxs = list(range(NP))
        random.shuffle(idxs)

        for i in idxs:
            if now() >= deadline:
                return best

            r = random.randrange(H)
            F = sample_F(M_F[r])
            CR = sample_CR(M_CR[r])

            pbest = order[random.randrange(pcount)]
            x_i = pop[i]
            x_p = pop[pbest]

            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)

            union = NP + len(archive)
            if union <= 2:
                r2u = random.randrange(NP)
            else:
                for _ in range(12):
                    r2u = random.randrange(union)
                    if r2u != i and r2u != r1:
                        break

            x_r1 = pop[r1]
            x_r2 = pop[r2u] if r2u < NP else archive[r2u - NP]

            mutant = [0.0] * dim
            for d in range(dim):
                jitter = 0.0005 * scale[d] * (random.random() - 0.5)
                mutant[d] = (x_i[d]
                             + F * (x_p[d] - x_i[d])
                             + F * (x_r1[d] - x_r2[d])
                             + jitter)
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
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                df = abs(fit[i] - f_trial)
                S_F.append(F)
                S_CR.append(CR)
                W.append(df if df > 0.0 else 1e-12)

                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best:
                    best, best_x = f_trial, trial[:]

        if S_F:
            wsum = sum(W) + 1e-18
            numF = 0.0
            denF = 0.0
            meanCR = 0.0
            for f, cr, w in zip(S_F, S_CR, W):
                ww = w / wsum
                numF += ww * f * f
                denF += ww * f
                meanCR += ww * cr
            new_MF = numF / (denF + 1e-18)
            new_MCR = meanCR
            if new_MF <= 0.0:
                new_MF = 0.5
            if new_MF > 1.0:
                new_MF = 1.0
            if new_MCR < 0.0:
                new_MCR = 0.0
            if new_MCR > 1.0:
                new_MCR = 1.0
            M_F[k_mem] = new_MF
            M_CR[k_mem] = new_MCR
            k_mem = (k_mem + 1) % H

        # ------------------------- local refinement (more purposeful) -------------------------
        # Allocate more local work later; also kick in when stagnant.
        if now() < deadline and (gen % 3 == 0 or stagn > 10):
            # budget in evaluations
            base_budget = 6 + min(18, dim)
            extra = int((10 + min(24, dim)) * progress)
            local_budget = base_budget + extra
            if stagn > 25:
                local_budget += min(20, dim)

            # Split budget between rosenbrock and coord/quadratic
            b1 = int(0.65 * local_budget)
            b2 = local_budget - b1

            f_loc, x_loc = rosenbrock_local(best_x, best, b1, progress)
            if f_loc < best:
                best, best_x = f_loc, x_loc[:]
                stagn = 0
                last_improve_time = now()

            if now() < deadline and b2 > 0:
                f_loc2, x_loc2 = local_coord_quadratic(best_x, best, b2)
                if f_loc2 < best:
                    best, best_x = f_loc2, x_loc2[:]
                    stagn = 0
                    last_improve_time = now()

        # restart/diversify if stuck + low diversity
        if now() < deadline:
            div = diversity(pop, best_x)
            stuck_time = now() - last_improve_time
            if (stagn > 45 and div < 0.06) or (stuck_time > 0.40 * T and div < 0.04):
                order = list(range(NP))
                order.sort(key=lambda i: fit[i])
                keep = max(4, NP // 4)
                elites = [pop[order[k]][:] for k in range(keep)]
                elites_fit = [fit[order[k]] for k in range(keep)]

                pop = elites[:]
                fit = elites_fit[:]

                refill = NP - keep
                offset = random.randint(1, 90000)
                for k in range(refill):
                    if now() >= deadline:
                        return best
                    r = random.random()
                    if r < 0.30:
                        x = rand_uniform_vec()
                    elif r < 0.65:
                        x = halton_point(offset + k + 1)
                    else:
                        x = best_x[:]
                        # heavy-tailed around best
                        for d in range(dim):
                            sd = (0.20 + 0.50 * random.random()) * scale[d]
                            if random.random() < 0.10:
                                sd *= 3.0
                            x[d] += random.gauss(0.0, sd)
                    x = reflect_into_bounds(x)
                    fx = eval_f(x)
                    pop.append(x)
                    fit.append(fx)
                    if fx < best:
                        best, best_x = fx, x[:]

                if len(archive) > 2 * NP:
                    archive = archive[-(2 * NP):]

                # relax local steps/basis after restart
                for d in range(dim):
                    coord_step[d] = max(coord_step[d], 0.22 * scale[d])
                    alpha[d] = max(alpha[d], 0.18 * scale[d])

                stagn = 0
                last_best = best
                last_improve_time = now()

    return best
