import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def reflect_1d(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # ---------- Halton (scrambled) ----------
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton_scrambled(index, base, perm):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * perm[i % base]
            i //= base
        return r

    primes = first_primes(dim)
    digit_perm = {}
    for b in set(primes):
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    k_hal = 1
    def halton_scrambled_vec(k):
        x = []
        for i in range(dim):
            b = primes[i]
            u = halton_scrambled(k, b, digit_perm[b])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------------- elite archive ----------------
    archive = []  # list of (f, x)
    archive_cap = 24  # keep small; we'll use proper local refinements

    def norm_l1(a, b):
        d = 0.0
        for i in range(dim):
            d += abs(a[i] - b[i]) / span[i]
        return d / max(1, dim)

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        pruned = []
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 1e-4:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    # ---------------- best so far ----------------
    best_x = rand_vec()
    best = eval_f(best_x)
    push_archive(best, best_x)

    # ============================================================
    # NEW: Budget-aware multi-scale local optimizer:
    #  - coordinate pattern search with adaptive steps
    #  - occasional random direction probes
    #  - "second-order-ish" 1D parabolic refinement along a coordinate
    # Designed to be cheap and robust without gradients.
    # ============================================================
    def polish(x0, f0, max_evals):
        x = x0[:]
        f = f0
        evals = 0

        step = [0.20 * s for s in span]
        min_step = [1e-14 * s for s in span]

        # For very small dim, be more aggressive
        if dim <= 5:
            for j in range(dim):
                step[j] = 0.30 * span[j]

        # "active set" permutation to reduce bias
        perm = list(range(dim))
        random.shuffle(perm)

        # coordinate + random dirs
        while evals < max_evals and time.time() < deadline:
            improved = False

            # ---- coordinate loop (pattern search) ----
            for jj in perm:
                if evals >= max_evals or time.time() >= deadline:
                    break

                lo, hi = bounds[jj]
                sj = step[jj]
                if sj <= min_step[jj]:
                    continue

                x0j = x[jj]
                # try +/- step
                cand_best = None

                # plus
                xp = x[:]
                xp[jj] = reflect_1d(x0j + sj, lo, hi)
                fp = eval_f(xp); evals += 1
                cand_best = (fp, xp)

                # minus
                if evals < max_evals and time.time() < deadline:
                    xm = x[:]
                    xm[jj] = reflect_1d(x0j - sj, lo, hi)
                    fm = eval_f(xm); evals += 1
                    if fm < cand_best[0]:
                        cand_best = (fm, xm)

                if cand_best[0] < f:
                    x, f = cand_best[1], cand_best[0]
                    improved = True

                    # --- quick parabolic refine on this coordinate (optional) ---
                    # Evaluate mid (current) and two sides around it and compute parabola minimum
                    if evals + 2 <= max_evals and time.time() < deadline:
                        # center at current x[jj]
                        c = x[jj]
                        a = 0.5 * sj
                        xL = x[:]; xL[jj] = reflect_1d(c - a, lo, hi)
                        fL = eval_f(xL); evals += 1
                        xR = x[:]; xR[jj] = reflect_1d(c + a, lo, hi)
                        fR = eval_f(xR); evals += 1
                        fC = f

                        # Fit parabola through (-a,fL),(0,fC),(+a,fR)
                        denom = (fL - 2.0 * fC + fR)
                        if abs(denom) > 1e-30:
                            delta = 0.5 * a * (fL - fR) / denom  # argmin shift
                            if abs(delta) <= 2.0 * a:
                                xQ = x[:]
                                xQ[jj] = reflect_1d(c + delta, lo, hi)
                                fQ = eval_f(xQ); evals += 1
                                if fQ < f:
                                    x, f = xQ, fQ
                                    improved = True
                else:
                    # no improvement on this coord => shrink step slightly
                    step[jj] *= 0.75

            # ---- random direction probes (escape tiny ridges) ----
            if evals < max_evals and time.time() < deadline:
                probes = 2 if dim <= 8 else 3
                for _ in range(probes):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    # sparse random direction
                    nnz = 1 if dim == 1 else (2 if dim < 10 else 3)
                    idxs = random.sample(range(dim), nnz)
                    d = [0.0] * dim
                    for j in idxs:
                        d[j] = random.choice((-1.0, 1.0))
                    # length scale tied to median step
                    med_step = sorted(step)[len(step)//2]
                    alpha = (0.5 + random.random()) * med_step
                    xt = x[:]
                    for j in idxs:
                        lo, hi = bounds[j]
                        xt[j] = reflect_1d(xt[j] + alpha * d[j], lo, hi)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved = True

            if improved:
                # modestly expand steps on success (promotes faster convergence on smooth basins)
                for j in range(dim):
                    step[j] *= 1.05
                    if step[j] > 0.5 * span[j]:
                        step[j] = 0.5 * span[j]
            else:
                # global shrink if nothing worked
                tiny = True
                for j in range(dim):
                    step[j] *= 0.70
                    if step[j] < min_step[j]:
                        step[j] = min_step[j]
                    if step[j] > 10.0 * min_step[j]:
                        tiny = False
                if tiny:
                    break

        return f, x

    # ============================================================
    # Improved main optimizer: L-SHADE-like DE with linear pop size reduction
    # + p-best mutation + external archive + periodic multi-start + intensive polishing
    # ============================================================

    # --- Initialization: more diversified & slightly larger budget ---
    init_budget = max(300, 120 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best
        if random.random() < 0.85:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        f = eval_f(x)
        if f < best:
            best, best_x = f, x[:]
        push_archive(f, x)

    # early polish top few
    for i in range(min(6, len(archive))):
        if time.time() >= deadline:
            return best
        f0, x0 = archive[i]
        f2, x2 = polish(x0, f0, max_evals=max(40, 10 * dim))
        if f2 < best:
            best, best_x = f2, x2
        push_archive(f2, x2)

    # --- L-SHADE parameters ---
    NP0 = max(40, min(180, 18 * dim))
    NPmin = max(12, 4 * dim)
    H = 12  # memory size

    MF = [0.6] * H
    MCR = [0.6] * H
    k_mem = 0

    # population: seed with elites + LDS + random
    pop = []
    pop_f = []
    for i in range(NP0):
        if time.time() >= deadline:
            return best
        if i < len(archive) and random.random() < 0.7:
            x = archive[i][1][:]
        elif i % 2 == 0:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        f = eval_f(x)
        pop.append(x)
        pop_f.append(f)
        if f < best:
            best, best_x = f, x[:]

    A = []
    A_cap = NP0

    def cauchy_pos(loc, scale):
        for _ in range(20):
            u = random.random() - 0.5
            v = loc + scale * math.tan(math.pi * u)
            if v > 0:
                return v
        return max(1e-8, loc)

    def clip01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    eval_counter = 0
    total_evals_target = None  # unknown; time-based
    last_improve_t = time.time()

    # main loop
    while True:
        if time.time() >= deadline:
            return best

        NP = len(pop)
        # sort indices by fitness
        order = sorted(range(NP), key=lambda i: pop_f[i])

        # update best
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            last_improve_t = time.time()

        # choose p fraction
        p = 0.08 + 0.12 * random.random()
        pbest_count = max(2, int(p * NP))

        SF = []
        SCR = []
        dF = []

        new_pop = pop[:]      # will overwrite selectively
        new_pop_f = pop_f[:]

        # mutation/crossover for each individual
        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = pop_f[i]

            # select memory index
            r = random.randrange(H)
            Fi = cauchy_pos(MF[r], 0.1)
            if Fi > 1.0:
                Fi = 1.0
            CRi = clip01(MCR[r] + 0.1 * random.gauss(0.0, 1.0))

            # choose pbest
            pbest_idx = order[random.randrange(pbest_count)]
            xp = pop[pbest_idx]

            # choose r1 != i
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            xr1 = pop[r1]

            # choose r2 from union(pop, A), avoiding i, r1, pbest when possible
            total = NP + len(A)
            while True:
                r2 = random.randrange(total)
                if r2 < NP:
                    if r2 != i and r2 != r1 and r2 != pbest_idx:
                        xr2 = pop[r2]
                        break
                else:
                    xr2 = A[r2 - NP]
                    break

            # current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xp[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    lo, hi = bounds[d]
                    u[d] = reflect_1d(v[d], lo, hi)
                else:
                    u[d] = xi[d]

            fu = eval_f(u)
            eval_counter += 1

            # selection
            if fu <= fi:
                # archive
                if len(A) < A_cap:
                    A.append(xi[:])
                else:
                    A[random.randrange(A_cap)] = xi[:]

                new_pop[i] = u
                new_pop_f[i] = fu

                SF.append(Fi)
                SCR.append(CRi)
                dF.append(fi - fu)

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = time.time()
            else:
                new_pop[i] = xi
                new_pop_f[i] = fi

        pop, pop_f = new_pop, new_pop_f

        # update memories (weighted)
        if SF:
            s = sum(dF)
            if s <= 0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [df / s for df in dF]

            # Lehmer mean for F
            num = 0.0
            den = 0.0
            for wi, fi in zip(w, SF):
                num += wi * fi * fi
                den += wi * fi
            if den > 0:
                MF[k_mem] = num / den
            else:
                MF[k_mem] = MF[k_mem]

            # weighted mean for CR
            MCR[k_mem] = sum(wi * cri for wi, cri in zip(w, SCR))
            k_mem = (k_mem + 1) % H

        # linear population size reduction (time-agnostic heuristic)
        # we reduce when stagnating a bit, keeping best individuals
        if len(pop) > NPmin:
            # if no improvement for a short while, shrink
            if time.time() - last_improve_t > 0.12 * max_time:
                newNP = max(NPmin, int(0.85 * len(pop)))
                if newNP < len(pop):
                    order = sorted(range(len(pop)), key=lambda i: pop_f[i])
                    keep = order[:newNP]
                    pop = [pop[i] for i in keep]
                    pop_f = [pop_f[i] for i in keep]
                    A_cap = len(pop)
                    if len(A) > A_cap:
                        A = A[:A_cap]
                last_improve_t = time.time()  # avoid repeated rapid shrink

        # periodic targeted polishing of current best + few elites
        if time.time() < deadline:
            if random.random() < 0.35:
                f2, x2 = polish(best_x, best, max_evals=max(50, 12 * dim))
                if f2 < best:
                    best, best_x = f2, x2
            else:
                # polish one random top individual
                order = sorted(range(len(pop)), key=lambda i: pop_f[i])
                idx = order[random.randrange(min(5, len(order)))]
                f2, x2 = polish(pop[idx], pop_f[idx], max_evals=max(25, 6 * dim))
                if f2 < pop_f[idx]:
                    pop[idx], pop_f[idx] = x2, f2
                    if f2 < best:
                        best, best_x = f2, x2

        # multi-start injection near elites if still time and stalled
        if time.time() >= deadline:
            return best
        if random.random() < 0.10 and len(pop) >= 8:
            # replace a few worst with gaussian around best/elite
            order = sorted(range(len(pop)), key=lambda i: pop_f[i])
            worst_n = max(1, len(pop) // 8)
            for kk in range(worst_n):
                if time.time() >= deadline:
                    return best
                wi = order[-1 - kk]
                base = best_x if random.random() < 0.7 else pop[order[random.randrange(min(5, len(pop)))]]
                xnew = [0.0] * dim
                for d in range(dim):
                    lo, hi = bounds[d]
                    sd = (0.03 + 0.15 * random.random()) * span[d]
                    xnew[d] = reflect_1d(base[d] + random.gauss(0.0, 1.0) * sd, lo, hi)
                fnew = eval_f(xnew)
                pop[wi] = xnew
                pop_f[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
#
#
