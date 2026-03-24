import random
import time
import math

def run(func, dim, bounds, max_time):
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0.0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def eval_point(x):
        return float(func(x))

    def clamp(v, i):
        lo, hi = lows[i], highs[i]
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect_in_bounds(x):
        # robust reflection for out-of-bound values
        y = x[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect until within [lo,hi]
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                else:
                    v = hi - (v - hi)
            # numeric safety
            if v < lo: v = lo
            if v > hi: v = hi
            y[i] = v
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Halton sequence (for good initial coverage)
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

    # ---------- incumbent ----------
    best_x = rand_point()
    best = eval_point(best_x)

    # ---------- Phase 1: stronger seeding (Halton + opposition + small jitter) ----------
    bases = first_primes(dim)

    # keep seeding meaningful but not too expensive
    seed_n = max(120, 50 * dim)  # slightly stronger than before
    elite_cap = 24
    elite = []  # list of (f, x)

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            del elite[elite_cap:]

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

    # jitter around best and a few randoms
    extra = max(40, 14 * dim)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        if random.random() < 0.65:
            x = best_x[:]
            for i in range(dim):
                x[i] = clamp(x[i] + (2.0 * random.random() - 1.0) * 0.12 * spans[i], i)
        else:
            x = rand_point()
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        if len(elite) < elite_cap or fx < elite[-1][0]:
            elite_add(fx, x)

    # ---------- Phase 2: L-SHADE-ish DE (linear pop reduction) + archive + polishing ----------
    pop_init = max(18, min(60, 6 * dim + 12))
    pop_min = max(8, min(24, 2 * dim + 6))

    # init population from elites then halton then random
    pop = []
    for i in range(pop_init):
        if time.time() >= deadline:
            return best
        if i < len(elite):
            x = elite[i][1][:]
        elif i < len(elite) + 8:
            x = halton_point(seed_n + 10 + i, bases)
        else:
            x = rand_point()
        fx = eval_point(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(fx, x)

    archive = []
    archive_cap = pop_init

    # JADE-style parameter adaptation with a small memory (L-SHADE flavor)
    H = 6
    M_F = [0.6] * H
    M_CR = [0.9] * H
    mem_idx = 0

    def weighted_lehmer_mean(vals, weights):
        num = 0.0
        den = 0.0
        for v, w in zip(vals, weights):
            num += w * v * v
            den += w * v
        return (num / den) if den > 1e-12 else None

    def mutate_current_to_pbest_1(xi, pbest, r1, r2, F):
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (r1[d] - r2[d])
        return reflect_in_bounds(v)

    def bin_cross(x, v, CR):
        u = x[:]
        jrand = random.randrange(dim)
        for d in range(dim):
            if d == jrand or random.random() < CR:
                u[d] = v[d]
        # cheap safety clamp
        for d in range(dim):
            if u[d] < lows[d]:
                u[d] = lows[d]
            elif u[d] > highs[d]:
                u[d] = highs[d]
        return u

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def normal(loc, scale):
        return loc + scale * randn()

    def local_polish_sparse(x, fx, evals=14):
        # sparse 1+1 ES; fast and often beneficial late
        cur = x[:]
        cur_f = fx
        step = [0.05 * spans[i] for i in range(dim)]
        for _ in range(evals):
            if time.time() >= deadline:
                break
            cand = cur[:]
            k = 1 if dim == 1 else (2 if dim <= 16 else max(2, dim // 14))
            for _t in range(k):
                j = random.randrange(dim)
                cand[j] = clamp(cand[j] + randn() * step[j], j)
            cf = eval_point(cand)
            if cf < cur_f:
                cur, cur_f = cand, cf
            # anneal
            for j in range(dim):
                step[j] *= 0.88
        return cur, cur_f

    gen = 0
    while time.time() < deadline and len(pop) >= pop_min:
        gen += 1

        # sort by fitness
        pop.sort(key=lambda it: it[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]

        # periodic polish (a bit more frequent than previous)
        if gen % 8 == 0 and time.time() < deadline:
            bx, bf = local_polish_sparse(best_x, best, evals=10)
            if bf < best:
                best, best_x = bf, bx[:]
                elite_add(bf, bx)

        # linear population size reduction (L-SHADE concept)
        t = (time.time() - t0) / max(1e-12, float(max_time))
        target_np = int(round(pop_init - (pop_init - pop_min) * min(1.0, max(0.0, t))))
        if target_np < pop_min:
            target_np = pop_min
        if len(pop) > target_np:
            pop = pop[:target_np]  # keep best ones
        npop = len(pop)
        if npop <= 2:
            break

        # p-best set size
        p = 0.18 if dim <= 20 else 0.12
        pcount = max(2, int(math.ceil(p * npop)))

        # union for r2 selection
        union = [ind[0] for ind in pop] + archive
        union_len = len(union)

        SF, SCR, dF = [], [], []  # successful F, CR, and weights (delta f)

        for i in range(npop):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]

            # choose memory index
            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            # sample F (cauchy) and CR (normal)
            Fi = cauchy(muF, 0.08)
            tries = 0
            while Fi <= 0.0 and tries < 8:
                Fi = cauchy(muF, 0.08)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(muCR, 0.08)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # choose pbest from top pcount
            pbest = pop[random.randrange(pcount)][0]

            # r1 from pop distinct from i
            while True:
                r1 = random.randrange(npop)
                if r1 != i:
                    break
            r1x = pop[r1][0]

            # r2 from union distinct from i and r1 (indices only safe for pop-part)
            while True:
                r2 = random.randrange(union_len)
                if r2 == i or r2 == r1:
                    continue
                r2x = union[r2]
                break

            v = mutate_current_to_pbest_1(xi, pbest, r1x, r2x, Fi)
            u = bin_cross(xi, v, CRi)
            fu = eval_point(u)

            if fu <= fi:
                # archive old xi
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
                dF.append(max(0.0, fi - fu))
            else:
                # very small chance of re-randomize a loser (diversity)
                if random.random() < 0.006 and time.time() < deadline:
                    xr = rand_point()
                    fr = eval_point(xr)
                    if fr < pop[i][1]:
                        pop[i] = [xr, fr]
                        if fr < best:
                            best, best_x = fr, xr[:]
                            elite_add(fr, xr)

        # update memory if we had successes
        if SF:
            # weights proportional to improvement
            s = sum(dF)
            if s <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [df / s for df in dF]

            new_muF = weighted_lehmer_mean(SF, w)
            if new_muF is None:
                new_muF = sum(SF) / float(len(SF))
            new_muCR = 0.0
            for cr, ww in zip(SCR, w):
                new_muCR += ww * cr

            # write into memory
            M_F[mem_idx] = 0.1 * 0.0 + 0.9 * new_muF  # slight smoothing/bias to stability
            M_CR[mem_idx] = 0.1 * 0.0 + 0.9 * new_muCR
            mem_idx = (mem_idx + 1) % H

        # keep archive aligned with current population size
        archive_cap = npop
        if len(archive) > archive_cap:
            archive = archive[:archive_cap]

    return best
