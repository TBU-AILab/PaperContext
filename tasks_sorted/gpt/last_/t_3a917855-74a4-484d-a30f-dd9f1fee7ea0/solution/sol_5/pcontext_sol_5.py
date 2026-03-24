import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

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

    # ---------- Halton sequence ----------
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

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(dim)
    k_hal = 1

    def halton_vec(k):
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------- Sobol-like fast scramble for diversity (no external libs) ----------
    # We approximate a low-discrepancy "scrambled Halton" using digit permutations per base.
    # This improves coverage vs plain Halton in many practical cases.
    digit_perm = {}
    for b in set(primes):
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        f = 1.0
        r = 0.0
        i = index
        perm = digit_perm[base]
        while i > 0:
            f /= base
            r += f * perm[i % base]
            i //= base
        return r

    def halton_scrambled_vec(k):
        x = []
        for i in range(dim):
            u = halton_scrambled(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------------- elite archive (larger + diversity) ----------------
    archive = []  # list of (f, x)
    archive_cap = 32

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
        # keep diversity among the best
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 5e-4:
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

    # ---------------- local search: randomized pattern + line search ----------------
    def local_polish(x0, f0, max_evals):
        x = x0[:]
        f = f0
        evals = 0

        # per-dim step sizes start moderate; shrink on failure
        step = [0.12 * s for s in span]
        min_step = [1e-12 * s for s in span]

        # random orthogonal-ish directions: coordinate + sparse random
        while evals < max_evals and time.time() < deadline:
            improved_any = False
            # try a few direction trials
            trials = min(8 + dim, 24)
            for _ in range(trials):
                if evals >= max_evals or time.time() >= deadline:
                    break

                # build a direction vector
                dvec = [0.0] * dim
                if random.random() < 0.65:
                    j = random.randrange(dim)
                    dvec[j] = 1.0 if random.random() < 0.5 else -1.0
                    scale = step[j]
                else:
                    # sparse random direction
                    nnz = 1 + (random.randrange(3) if dim >= 3 else 0)
                    idxs = random.sample(range(dim), nnz)
                    for j in idxs:
                        dvec[j] = random.choice((-1.0, 1.0))
                    # typical scale across chosen dims
                    scale = 0.0
                    for j in idxs:
                        scale += step[j]
                    scale /= max(1, nnz)

                # 1D line search along direction with a couple of radii
                for mul in (1.0, 0.5, 2.0):
                    if evals >= max_evals or time.time() >= deadline:
                        break
                    a = mul * scale
                    xt = x[:]
                    for j in range(dim):
                        if dvec[j] != 0.0:
                            lo, hi = bounds[j]
                            xt[j] = reflect_1d(xt[j] + a * dvec[j], lo, hi)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved_any = True
                        break

            if not improved_any:
                # shrink steps; stop if all tiny
                done = True
                for j in range(dim):
                    step[j] *= 0.7
                    if step[j] < min_step[j]:
                        step[j] = min_step[j]
                    if step[j] > 5.0 * min_step[j]:
                        done = False
                if done:
                    break

        return f, x

    # ---------------- initialization: heavy low-discrepancy + elite polish ----------------
    init_budget = max(200, 80 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best
        # mix scrambled Halton and pure random
        if random.random() < 0.75:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        f = eval_f(x)
        if f < best:
            best, best_x = f, x[:]
            push_archive(best, best_x)

    # quick polish a few best points early
    for i in range(min(4, len(archive))):
        if time.time() >= deadline:
            return best
        f0, x0 = archive[i]
        f2, x2 = local_polish(x0, f0, max_evals=max(20, 6 * dim))
        if f2 < best:
            best, best_x = f2, x2
            push_archive(best, best_x)

    # ============================================================
    # Main optimizer: JADE-style DE/current-to-pbest/1 with
    # external archive + parameter adaptation + restarts
    # ============================================================

    NP = max(24, min(120, 12 * dim))
    pop = []
    pop_f = []

    # DE parameter memories (JADE-style)
    mu_F = 0.6
    mu_CR = 0.6

    # external archive for DE (different from elite archive)
    A = []
    A_cap = NP

    # seed population with best known + low-discrepancy
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i == 0:
            x = best_x[:]
        elif i < len(archive) and i < NP // 5:
            x = archive[i][1][:]
        elif i % 3 == 0:
            x = halton_scrambled_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        f = eval_f(x)
        pop.append(x)
        pop_f.append(f)
        if f < best:
            best, best_x = f, x[:]
            push_archive(best, best_x)

    def cauchy_positive(loc, scale):
        # sample from Cauchy(loc, scale) until > 0
        for _ in range(12):
            u = random.random() - 0.5
            v = loc + scale * math.tan(math.pi * u)
            if v > 0:
                return v
        return max(1e-6, loc)

    def pick_idx_excluding(excl_set, n):
        out = []
        seen = set(excl_set)
        # allow selecting from pop indices only here
        while len(out) < n:
            j = random.randrange(NP)
            if j not in seen:
                seen.add(j)
                out.append(j)
        return out

    def pick_from_union(excl_set):
        # pick vector from pop or external archive A
        # JADE uses union(pop, A)
        total = NP + len(A)
        while True:
            r = random.randrange(total)
            if r < NP:
                if r not in excl_set:
                    return pop[r]
            else:
                return A[r - NP]

    stall = 0
    last_best = best

    while True:
        if time.time() >= deadline:
            return best

        order = sorted(range(NP), key=lambda i: pop_f[i])
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            push_archive(best, best_x)

        if best < last_best - 1e-15:
            stall = 0
            last_best = best
        else:
            stall += 1

        p = 0.05 + 0.20 * random.random()
        pbest_count = max(2, int(p * NP))

        SF = []
        SCR = []
        df = []

        for i in range(NP):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = pop_f[i]

            # choose pbest
            pbest_idx = order[random.randrange(pbest_count)]
            xp = pop[pbest_idx]

            # adaptive parameters
            Fi = cauchy_positive(mu_F, 0.1)
            if Fi > 1.0:
                Fi = 1.0
            CRi = mu_CR + 0.1 * random.gauss(0.0, 1.0)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # mutation: current-to-pbest/1 with archive
            r1, = pick_idx_excluding({i, pbest_idx}, 1)
            xr1 = pop[r1]
            xr2 = pick_from_union({i, pbest_idx, r1})

            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (xp[d] - xi[d]) + Fi * (xr1[d] - xr2[d])

            # crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    lo, hi = bounds[d]
                    u[d] = reflect_1d(v[d], lo, hi)
                else:
                    u[d] = xi[d]

            fu = eval_f(u)

            # selection + archive update
            if fu <= fi:
                # add replaced parent to external archive
                if len(A) < A_cap:
                    A.append(xi[:])
                else:
                    A[random.randrange(A_cap)] = xi[:]

                pop[i] = u
                pop_f[i] = fu

                if fu < best:
                    best, best_x = fu, u[:]
                    push_archive(best, best_x)
                    # polish improvements (small, time-safe)
                    if time.time() < deadline:
                        f2, x2 = local_polish(best_x, best, max_evals=max(12, 3 * dim))
                        if f2 < best:
                            best, best_x = f2, x2
                            push_archive(best, best_x)

                # success memories
                SF.append(Fi)
                SCR.append(CRi)
                df.append(fi - fu)

        # update mu_F, mu_CR (JADE)
        if SF:
            # Lehmer mean for F weighted by improvement
            wsum = sum(df) if sum(df) > 0 else float(len(df))
            if wsum <= 0:
                wsum = float(len(df))
                weights = [1.0 / wsum] * len(df)
            else:
                weights = [d / wsum for d in df]

            num = 0.0
            den = 0.0
            for w, fval in zip(weights, SF):
                num += w * (fval * fval)
                den += w * fval
            if den > 0:
                mu_F = 0.9 * mu_F + 0.1 * (num / den)

            mu_CR = 0.9 * mu_CR + 0.1 * (sum(SCR) / float(len(SCR)))

        # restart/diversify on stall
        if stall > (10 + 3 * dim):
            stall = 0
            # re-seed worst portion using elites + scrambled Halton
            worst = max(2, NP // 4)
            for k in range(worst):
                if time.time() >= deadline:
                    return best
                wi = order[-1 - k]
                if archive and random.random() < 0.8:
                    base = archive[random.randrange(len(archive))][1]
                    xnew = [0.0] * dim
                    for d in range(dim):
                        lo, hi = bounds[d]
                        sd = (0.15 + 0.25 * random.random()) * span[d]
                        xnew[d] = reflect_1d(base[d] + random.gauss(0.0, 1.0) * sd, lo, hi)
                else:
                    xnew = halton_scrambled_vec(k_hal); k_hal += 1
                fnew = eval_f(xnew)
                pop[wi] = xnew
                pop_f[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    push_archive(best, best_x)
