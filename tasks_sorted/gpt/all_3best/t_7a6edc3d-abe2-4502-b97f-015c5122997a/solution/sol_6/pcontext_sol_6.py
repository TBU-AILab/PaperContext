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

    # ---------------- helpers ----------------
    def clamp(v, i):
        lo, hi = lows[i], highs[i]
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def repair_reflect(x):
        # robust reflection into bounds (handles large violations)
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
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        return float(func(x))

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Halton (low discrepancy) for seeding
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

    # ---------------- incumbent ----------------
    best_x = rand_point()
    best = eval_point(best_x)

    # ---------------- Phase 1: stronger seeding (Halton + opposition + jitter) ----------------
    bases = first_primes(dim)

    # More aggressive seeding (but still bounded so it won't eat all time)
    seed_n = max(160, 60 * dim)
    elite_cap = 32
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

    # Extra probing: mix random, jitter-around-best, and midpoints between elites
    extra = max(60, 18 * dim)
    for _ in range(extra):
        if time.time() >= deadline:
            return best

        r = random.random()
        if r < 0.50:
            # jitter around best
            x = best_x[:]
            for i in range(dim):
                x[i] = clamp(x[i] + (2.0 * random.random() - 1.0) * 0.18 * spans[i], i)
        elif r < 0.75 and len(elite) >= 2:
            # midpoint/extrapolation between two good points (often boosts exploitation)
            a = elite[random.randrange(min(10, len(elite)))][1]
            b = elite[random.randrange(min(10, len(elite)))][1]
            w = 0.5 + 0.35 * (2.0 * random.random() - 1.0)  # ~ [0.15,0.85]
            x = [clamp(w * a[i] + (1.0 - w) * b[i], i) for i in range(dim)]
        else:
            x = rand_point()

        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)
        if len(elite) < elite_cap or fx < elite[-1][0]:
            elite_add(fx, x)

    # ---------------- Phase 2: Improved JADE/LSHADE hybrid DE + archive + "best-2" refine ----------------
    # Keep pop moderate; use slight linear reduction over time.
    pop_init = max(20, min(70, 6 * dim + 14))
    pop_min = max(10, min(28, 2 * dim + 8))

    pop = []
    for i in range(pop_init):
        if time.time() >= deadline:
            return best
        if i < len(elite):
            x = elite[i][1][:]
        elif i < len(elite) + 10:
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

    # memory for parameters (JADE/L-SHADE style)
    H = 8
    M_F = [0.55] * H
    M_CR = [0.85] * H
    mem_idx = 0

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def normal(loc, scale):
        return loc + scale * randn()

    def bin_cross(x, v, CR):
        u = x[:]
        jrand = random.randrange(dim)
        for d in range(dim):
            if d == jrand or random.random() < CR:
                u[d] = v[d]
        for d in range(dim):
            if u[d] < lows[d]:
                u[d] = lows[d]
            elif u[d] > highs[d]:
                u[d] = highs[d]
        return u

    def mutate_current_to_pbest_1(xi, pbest, r1, r2, F):
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xi[d] + F * (pbest[d] - xi[d]) + F * (r1[d] - r2[d])
        return repair_reflect(v)

    def weighted_lehmer_mean(vals, weights):
        num = 0.0
        den = 0.0
        for v, w in zip(vals, weights):
            num += w * v * v
            den += w * v
        return (num / den) if den > 1e-12 else None

    def local_polish_best2(x, fx, y, fy, evals=18):
        # best-of-two guided sparse ES:
        # sample around x plus a small drift toward y (if y exists)
        cur = x[:]
        cur_f = fx
        step = [0.055 * spans[i] for i in range(dim)]
        for _ in range(evals):
            if time.time() >= deadline:
                break
            cand = cur[:]
            # sparse
            k = 1 if dim == 1 else (2 if dim <= 16 else max(2, dim // 14))
            for _t in range(k):
                j = random.randrange(dim)
                drift = 0.0
                if y is not None:
                    drift = 0.25 * (y[j] - cur[j])
                cand[j] = clamp(cand[j] + drift + randn() * step[j], j)
            cf = eval_point(cand)
            if cf < cur_f:
                cur, cur_f = cand, cf
            for j in range(dim):
                step[j] *= 0.89
        return cur, cur_f

    gen = 0
    while time.time() < deadline and len(pop) >= pop_min:
        gen += 1
        pop.sort(key=lambda it: it[1])

        # global best update
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]

        # time-based population reduction
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        target_np = int(round(pop_init - (pop_init - pop_min) * min(1.0, max(0.0, tfrac))))
        if target_np < pop_min:
            target_np = pop_min
        if len(pop) > target_np:
            pop = pop[:target_np]
        npop = len(pop)
        if npop <= 3:
            break

        # periodic best polishing with best-2 guidance
        if gen % 7 == 0 and time.time() < deadline:
            second = pop[1] if npop > 1 else None
            y = second[0] if second else None
            fy = second[1] if second else None
            bx, bf = local_polish_best2(best_x, best, y, fy, evals=10)
            if bf < best:
                best, best_x = bf, bx[:]
                elite_add(bf, bx)

        # p-best selection size (slightly smaller late)
        p = 0.22 if dim <= 20 else 0.14
        pcount = max(2, int(math.ceil(p * npop)))

        union = [ind[0] for ind in pop] + archive
        union_len = len(union)

        SF, SCR, dF = [], [], []

        for i in range(npop):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            Fi = cauchy(muF, 0.09)
            tries = 0
            while Fi <= 0.0 and tries < 8:
                Fi = cauchy(muF, 0.09)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            CRi = normal(muCR, 0.09)
            if CRi < 0.0:
                CRi = 0.0
            elif CRi > 1.0:
                CRi = 1.0

            # pick pbest among top pcount
            pbest = pop[random.randrange(pcount)][0]

            # r1 from pop != i
            while True:
                r1 = random.randrange(npop)
                if r1 != i:
                    break
            r1x = pop[r1][0]

            # r2 from union != i and != r1 (index-safe for pop part)
            while True:
                r2 = random.randrange(union_len)
                if r2 == i or r2 == r1:
                    continue
                r2x = union[r2]
                break

            v = mutate_current_to_pbest_1(xi, pbest, r1x, r2x, Fi)

            # occasional "current-to-best" injection for exploitation (cheap hybridization)
            if random.random() < 0.12:
                for d in range(dim):
                    v[d] = clamp(0.85 * v[d] + 0.15 * best_x[d], d)

            u = bin_cross(xi, v, CRi)
            fu = eval_point(u)

            if fu <= fi:
                # archive old
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
                # slightly stronger diversity kick when losing
                if random.random() < 0.012:
                    # elite-guided random mix (safer than pure random in late stage)
                    if elite:
                        ex = elite[random.randrange(min(8, len(elite)))][1]
                        xr = [clamp(0.65 * ex[d] + 0.35 * (lows[d] + random.random() * spans[d]), d) for d in range(dim)]
                    else:
                        xr = rand_point()
                    fr = eval_point(xr)
                    if fr < pop[i][1]:
                        pop[i] = [xr, fr]
                        if fr < best:
                            best, best_x = fr, xr[:]
                            elite_add(fr, xr)

        # adapt memories
        if SF:
            s = sum(dF)
            if s <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [df / s for df in dF]

            new_muF = weighted_lehmer_mean(SF, w)
            if new_muF is None:
                new_muF = sum(SF) / float(len(SF))
            new_muCR = sum(cr * ww for cr, ww in zip(SCR, w))

            # mild smoothing; keep within sane ranges
            M_F[mem_idx] = min(0.98, max(0.05, 0.2 * M_F[mem_idx] + 0.8 * new_muF))
            M_CR[mem_idx] = min(0.98, max(0.0, 0.2 * M_CR[mem_idx] + 0.8 * new_muCR))
            mem_idx = (mem_idx + 1) % H

        archive_cap = npop
        if len(archive) > archive_cap:
            archive = archive[:archive_cap]

    return best
