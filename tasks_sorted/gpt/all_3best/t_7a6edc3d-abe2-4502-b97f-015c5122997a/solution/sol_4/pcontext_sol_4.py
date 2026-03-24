import random
import time
import math

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------------- helpers ----------------
    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def repair(x):
        # bounce-back reflection then clamp (handles big violations robustly)
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect until in-range (rarely loops more than once)
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                elif v > hi:
                    v = hi - (v - hi)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def eval_point(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Halton low-discrepancy sequence
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(index, bases):
        u = [van_der_corput(index, b) for b in bases]
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------------- incumbent ----------------
    x0 = rand_point()
    best_x = list(x0)
    best = eval_point(best_x)

    # ---------------- Phase 1: strong seeding (Halton + opposition + jitter) ----------------
    bases = first_primes(dim)

    # keep seeding modest but effective
    seed_n = max(100, 40 * dim)
    elite_cap = 18
    elite = []  # list of (f,x)

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            del elite[elite_cap:]

    # seed with halton + opposition
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best
        x = halton_point(k, bases)
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        elite_add(fx, x)

        if time.time() >= deadline:
            return best
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        fo = eval_point(xo)
        if fo < best:
            best, best_x = fo, list(xo)
        elite_add(fo, xo)

    # add a bit of random + jitter-around-best (often helps)
    extra = max(30, 12 * dim)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        if random.random() < 0.5:
            x = rand_point()
        else:
            # jitter around current best
            x = best_x[:]
            for i in range(dim):
                x[i] = clamp(x[i] + (2.0 * random.random() - 1.0) * 0.15 * spans[i], i)
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        if (len(elite) < elite_cap) or (fx < elite[-1][0]):
            elite_add(fx, x)

    # ---------------- Phase 2: JADE-like DE with p-best + archive + polishing ----------------
    # population size tuned for time-limited black-box optimization
    pop_size = max(14, min(44, 5 * dim + 10))

    pop = []
    # seed population: best elites, then halton, then random
    for i in range(pop_size):
        if time.time() >= deadline:
            return best
        if i < len(elite):
            x = elite[i][1][:]
        elif i < len(elite) + 6:
            x = halton_point(seed_n + 1 + i, bases)
        else:
            x = rand_point()
        fx = eval_point(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(fx, x)

    # JADE parameters (adapted online)
    mu_F = 0.6
    mu_CR = 0.9
    c_adapt = 0.1

    # external archive to increase diversity
    archive = []
    archive_cap = pop_size

    def pick_distinct_indices(n, exclude_set):
        # returns one index in [0,n) not in exclude_set
        while True:
            r = random.randrange(n)
            if r not in exclude_set:
                return r

    def current_to_pbest_mutation(i, F, pbest_x, r1_x, r2_x):
        xi = pop[i][0]
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xi[d] + F * (pbest_x[d] - xi[d]) + F * (r1_x[d] - r2_x[d])
        return repair(v)

    def bin_crossover(x, v, CR):
        u = x[:]
        jrand = random.randrange(dim)
        for d in range(dim):
            if d == jrand or random.random() < CR:
                u[d] = v[d]
        # final clamp
        for d in range(dim):
            if u[d] < lows[d]:
                u[d] = lows[d]
            elif u[d] > highs[d]:
                u[d] = highs[d]
        return u

    def cauchy(mu, gamma=0.1):
        # mu + gamma * tan(pi*(u-0.5))
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def normal(mu, sigma=0.1):
        return mu + sigma * randn()

    def local_polish_11(x, fx, evals=18):
        # 1+1 ES with decreasing step; sparse perturbations for high dim
        cur = x[:]
        cur_f = fx
        step = [0.06 * spans[i] for i in range(dim)]
        for _ in range(evals):
            if time.time() >= deadline:
                break
            cand = cur[:]
            k = 1 if dim == 1 else (2 if dim <= 12 else max(2, dim // 12))
            for _t in range(k):
                j = random.randrange(dim)
                cand[j] = clamp(cand[j] + randn() * step[j], j)
            cf = eval_point(cand)
            if cf < cur_f:
                cur, cur_f = cand, cf
            for j in range(dim):
                step[j] *= 0.90
        return cur, cur_f

    gen = 0
    while time.time() < deadline:
        gen += 1
        pop.sort(key=lambda t: t[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]

        # occasional polish
        if gen % 10 == 0 and time.time() < deadline:
            bx, bf = local_polish_11(best_x, best, evals=10)
            if bf < best:
                best, best_x = bf, bx[:]
                elite_add(bf, bx)

        # p-best fraction (smaller over time for exploitation)
        p = 0.20 if dim <= 20 else 0.15
        pcount = max(2, int(math.ceil(p * pop_size)))

        SF = []
        SCR = []

        # build union pool for r2 (pop + archive)
        union = [ind[0] for ind in pop] + [a for a in archive]

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]
            # choose pbest from top pcount
            pb = pop[random.randrange(pcount)][0]

            # sample parameters
            Fi = cauchy(mu_F, 0.08)
            # resample if non-positive
            tries = 0
            while Fi <= 0.0 and tries < 6:
                Fi = cauchy(mu_F, 0.08)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(mu_CR, 0.08)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose r1 from pop != i
            r1 = pick_distinct_indices(pop_size, {i})
            r1x = pop[r1][0]

            # choose r2 from union != i and != r1 (by identity; approximate by index region)
            # We'll just reject if it equals xi or r1x by reference equality check is hard; use index logic
            # Use indices: union[0:pop_size]=pop, union[pop_size:]=archive
            while True:
                r2 = random.randrange(len(union))
                # avoid picking i or r1 from pop part
                if r2 == i or r2 == r1:
                    continue
                r2x = union[r2]
                break

            v = current_to_pbest_mutation(i, Fi, pb, r1x, r2x)
            u = bin_crossover(xi, v, CRi)
            fu = eval_point(u)

            if fu <= fi:
                # successful: push replaced xi into archive
                if len(archive) < archive_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_cap)] = xi[:]

                pop[i] = [u, fu]
                if fu < best:
                    best, best_x = fu, u[:]
                    elite_add(fu, u)

                SF.append(Fi)
                SCR.append(CRi)
            else:
                # occasional soft restart if really bad and stagnating
                if random.random() < 0.01:
                    xr = rand_point()
                    fr = eval_point(xr)
                    if fr < pop[i][1]:
                        pop[i] = [xr, fr]
                        if fr < best:
                            best, best_x = fr, xr[:]
                            elite_add(fr, xr)

        # adapt mu_F and mu_CR from successes
        if SF:
            # Lehmer mean for F, arithmetic mean for CR (JADE-style)
            num = sum(f * f for f in SF)
            den = sum(f for f in SF)
            if den > 1e-12:
                mean_lehmer_F = num / den
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * mean_lehmer_F
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))

        # keep archive size in check (already capped, but shrink if pop_size changed)
        if len(archive) > archive_cap:
            archive = archive[:archive_cap]

    return best
