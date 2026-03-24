import random
import math
import time

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-9

    # ---------------- helpers ----------------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_1d(v, lo, hi):
        # mirror reflection to keep continuity at bounds
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

    def ensure_bounds(x):
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = clamp(x[i], lo, hi)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # ---------- Halton sequence (for better coverage than pure random) ----------
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

    def halton_vec(k):
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # ---------------- small elite archive ----------------
    archive = []  # list of (f, x)
    archive_cap = 16

    def norm_l1(a, b):
        d = 0.0
        for i in range(dim):
            d += abs(a[i] - b[i]) / span[i]
        return d / max(1, dim)

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])

        # prune near-duplicates to keep diversity among elites
        pruned = []
        for f, v in archive:
            ok = True
            for _, v2 in pruned:
                if norm_l1(v, v2) < 1e-3:
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

    # ---------------- stronger initialization: Halton + a few randoms ----------------
    k_hal = 1
    init_budget = max(80, 60 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline - eps_time:
            return best
        x = halton_vec(k_hal)
        k_hal += 1
        f = eval_f(x)
        if f < best:
            best, best_x = f, x
            push_archive(best, best_x)

    for _ in range(max(20, 10 * dim)):
        if time.time() >= deadline - eps_time:
            return best
        x = rand_vec()
        f = eval_f(x)
        if f < best:
            best, best_x = f, x
            push_archive(best, best_x)

    # ============================================================
    # Core optimizer: Differential Evolution (current-to-pbest/1)
    # + occasional "opposition" / extrapolation
    # + small local polish (coordinate + gaussian) on improvements
    # ============================================================

    # population size: moderate to keep evaluations efficient
    NP = max(18, min(80, 10 * dim))
    pop = []
    pop_f = []

    # seed population: mix halton + random + elites
    for i in range(NP):
        if time.time() >= deadline - eps_time:
            return best
        if archive and i < min(len(archive), NP // 4):
            x = archive[i][1][:]
        elif i % 3 == 0:
            x = halton_vec(k_hal); k_hal += 1
        else:
            x = rand_vec()
        f = eval_f(x)
        pop.append(x)
        pop_f.append(f)
        if f < best:
            best, best_x = f, x[:]
            push_archive(best, best_x)

    def pick_distinct(n, exclude):
        # pick n distinct indices not in exclude
        s = set(exclude)
        out = []
        while len(out) < n:
            j = random.randrange(NP)
            if j not in s:
                s.add(j)
                out.append(j)
        return out

    def maybe_polish(x0, f0, budget):
        # lightweight local improvement (time-safe)
        x = x0[:]
        f = f0
        step = [0.03 * s for s in span]
        idx = list(range(dim))
        evals = 0
        while evals < budget and time.time() < deadline - eps_time:
            random.shuffle(idx)
            improved = False
            for i in idx:
                if evals >= budget or time.time() >= deadline - eps_time:
                    break
                lo, hi = bounds[i]
                xi = x[i]
                si = step[i]
                if si <= 0:
                    continue

                # try +/- step
                best_i = xi
                best_f = f
                for d in (1.0, -1.0):
                    xt = x[:]
                    xt[i] = reflect_1d(xi + d * si, lo, hi)
                    ft = eval_f(xt); evals += 1
                    if ft < best_f:
                        best_f = ft
                        best_i = xt[i]
                    if evals >= budget or time.time() >= deadline - eps_time:
                        break

                if best_f < f:
                    x[i] = best_i
                    f = best_f
                    improved = True
                else:
                    step[i] *= 0.85  # shrink if unproductive
                    if step[i] < 1e-12 * span[i]:
                        step[i] = 1e-12 * span[i]

            if not improved:
                # small gaussian shake around current point
                xt = x[:]
                for i in range(dim):
                    lo, hi = bounds[i]
                    xt[i] = reflect_1d(xt[i] + random.gauss(0.0, 1.0) * 0.15 * step[i], lo, hi)
                ft = eval_f(xt); evals += 1
                if ft < f:
                    x, f = xt, ft
        return f, x

    # DE parameters (self-adaptive-ish via randomization)
    gen = 0
    stall = 0
    last_best = best

    while True:
        if time.time() >= deadline - eps_time:
            return best

        # sort indices by fitness for pbest selection
        order = sorted(range(NP), key=lambda i: pop_f[i])
        # keep best in archive
        if pop_f[order[0]] < best:
            best = pop_f[order[0]]
            best_x = pop[order[0]][:]
            push_archive(best, best_x)

        # stall handling
        if best < last_best - 1e-15:
            stall = 0
            last_best = best
        else:
            stall += 1

        p = 0.2 + 0.3 * random.random()  # pbest fraction
        pbest_count = max(2, int(p * NP))

        # generation loop
        for i in range(NP):
            if time.time() >= deadline - eps_time:
                return best

            xi = pop[i]
            fi = pop_f[i]

            # choose pbest
            pbest_idx = order[random.randrange(pbest_count)]
            xp = pop[pbest_idx]

            # mutation indices
            r1, r2 = pick_distinct(2, exclude=(i, pbest_idx))

            # randomized control parameters (helps robustness)
            F = 0.45 + 0.5 * random.random()        # ~[0.45, 0.95]
            CR = 0.15 + 0.85 * random.random()      # ~[0.15, 1.0]

            # current-to-pbest/1 mutation: v = x + F*(xp-x) + F*(xr1-xr2)
            xr1 = pop[r1]
            xr2 = pop[r2]
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xp[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    lo, hi = bounds[d]
                    u[d] = reflect_1d(v[d], lo, hi)
                else:
                    u[d] = xi[d]

            fu = eval_f(u)

            # occasional extrapolation/opposition move if trial is good-ish
            if fu < fi and random.random() < 0.15 and time.time() < deadline - eps_time:
                # extrapolate beyond xi towards u
                beta = 1.0 + 0.5 * random.random()
                u2 = [0.0] * dim
                for d in range(dim):
                    lo, hi = bounds[d]
                    u2[d] = reflect_1d(xi[d] + beta * (u[d] - xi[d]), lo, hi)
                fu2 = eval_f(u2)
                if fu2 < fu:
                    u, fu = u2, fu2

            # selection
            if fu <= fi:
                pop[i] = u
                pop_f[i] = fu
                if fu < best:
                    best, best_x = fu, u[:]
                    push_archive(best, best_x)

                    # quick polish on true improvements
                    if time.time() < deadline - eps_time:
                        f2, x2 = maybe_polish(best_x, best, budget=max(6, 2 * dim))
                        if f2 < best:
                            best, best_x = f2, x2
                            push_archive(best, best_x)

        gen += 1

        # mild restart/diversification when stalling
        if stall > (8 + 3 * dim):
            stall = 0

            # replace a fraction of worst individuals by new points around elites + halton
            worst_count = max(2, NP // 5)
            for k in range(worst_count):
                if time.time() >= deadline - eps_time:
                    return best
                wi = order[-1 - k]

                if archive and random.random() < 0.75:
                    base = archive[random.randrange(len(archive))][1]
                    # perturb around elite
                    xnew = [0.0] * dim
                    for d in range(dim):
                        lo, hi = bounds[d]
                        sd = 0.08 * span[d]
                        xnew[d] = reflect_1d(base[d] + random.gauss(0.0, 1.0) * sd, lo, hi)
                else:
                    # space-filling injection
                    xnew = halton_vec(k_hal); k_hal += 1

                fnew = eval_f(xnew)
                pop[wi] = xnew
                pop_f[wi] = fnew
                if fnew < best:
                    best, best_x = fnew, xnew[:]
                    push_archive(best, best_x)
