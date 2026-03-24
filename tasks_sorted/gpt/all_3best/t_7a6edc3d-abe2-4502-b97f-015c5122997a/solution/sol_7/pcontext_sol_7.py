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

    # ---------------- basic helpers ----------------
    def eval_point(x):
        return float(func(x))

    def clamp(v, i):
        lo, hi = lows[i], highs[i]
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def repair_reflect(x):
        # robust reflect into bounds
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect repeatedly for big violations
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

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------------- Halton seeding helpers ----------------
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

    # ---------------- elite structure ----------------
    elite_cap = 40
    elite = []  # list of (f, x)

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            del elite[elite_cap:]

    # ---------------- incumbent init ----------------
    best_x = rand_point()
    best = eval_point(best_x)
    elite_add(best, best_x)

    # ---------------- Phase 1: strong global coverage ----------------
    bases = first_primes(dim)

    # time-aware seeding: spend up to ~22% of time on broad coverage
    # (but still safe if max_time is tiny: hard cap on eval count)
    seed_target = max(180, 70 * dim)
    seed_cap = 2200  # absolute cap on seeding evals (including opposition etc.)
    used = 0

    k = 1
    while used < seed_target and used < seed_cap and time.time() < deadline:
        x = halton_point(k, bases)
        k += 1
        fx = eval_point(x); used += 1
        if fx < best:
            best, best_x = fx, list(x)
        elite_add(fx, x)
        if time.time() >= deadline or used >= seed_cap:
            break

        # opposition
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        fo = eval_point(xo); used += 1
        if fo < best:
            best, best_x = fo, list(xo)
        elite_add(fo, xo)

        if time.time() >= deadline or used >= seed_cap:
            break

        # quasi-opposition around center (often helps on bounded domains)
        xc = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]
        xqo = [clamp(xc[i] + (xc[i] - x[i]) * (0.6 + 0.8 * random.random()), i) for i in range(dim)]
        fq = eval_point(xqo); used += 1
        if fq < best:
            best, best_x = fq, list(xqo)
        elite_add(fq, xqo)

        # light jitter around best
        if time.time() >= deadline or used >= seed_cap:
            break
        xb = best_x[:]
        for i in range(dim):
            xb[i] = clamp(xb[i] + (2.0 * random.random() - 1.0) * 0.12 * spans[i], i)
        fb = eval_point(xb); used += 1
        if fb < best:
            best, best_x = fb, list(xb)
        elite_add(fb, xb)

    # ---------------- Phase 2: multi-operator DE (ensemble) + archive ----------------
    # L-SHADE-like with:
    # - current-to-pbest/1 (JADE)
    # - occasional rand/1 or best/1 injection
    # - linear pop reduction
    # - archive for r2
    pop_init = max(24, min(84, 7 * dim + 18))
    pop_min = max(10, min(30, 2 * dim + 8))

    # init population preferentially from elite and Halton tail
    pop = []
    for i in range(pop_init):
        if time.time() >= deadline:
            return best
        if i < len(elite):
            x = elite[i][1][:]
        elif i < len(elite) + 12:
            x = halton_point(k + i + 1, bases)
        else:
            x = rand_point()
        fx = eval_point(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]
        elite_add(fx, x)

    archive = []
    archive_cap = pop_init

    # parameter memories
    H = 10
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
        return repair_reflect(v)

    def mutate_rand_1(xa, xb, xc, F):
        v = [0.0] * dim
        for d in range(dim):
            v[d] = xa[d] + F * (xb[d] - xc[d])
        return repair_reflect(v)

    def mutate_best_1(bestx, xb, xc, F):
        v = [0.0] * dim
        for d in range(dim):
            v[d] = bestx[d] + F * (xb[d] - xc[d])
        return repair_reflect(v)

    def local_polish_sparse(x, fx, evals=14):
        cur = x[:]
        cur_f = fx
        step = [0.05 * spans[i] for i in range(dim)]
        for _ in range(evals):
            if time.time() >= deadline:
                break
            cand = cur[:]
            k_ = 1 if dim == 1 else (2 if dim <= 16 else max(2, dim // 14))
            for __ in range(k_):
                j = random.randrange(dim)
                cand[j] = clamp(cand[j] + randn() * step[j], j)
            cf = eval_point(cand)
            if cf < cur_f:
                cur, cur_f = cand, cf
            for j in range(dim):
                step[j] *= 0.88
        return cur, cur_f

    gen = 0
    last_best = best
    stagn_gens = 0

    while time.time() < deadline and len(pop) >= pop_min:
        gen += 1
        pop.sort(key=lambda it: it[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]
            elite_add(best, best_x)

        # stagnation tracking
        if best < last_best - 1e-12:
            last_best = best
            stagn_gens = 0
        else:
            stagn_gens += 1

        # linear population reduction
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        target_np = int(round(pop_init - (pop_init - pop_min) * min(1.0, max(0.0, tfrac))))
        if target_np < pop_min:
            target_np = pop_min
        if len(pop) > target_np:
            pop = pop[:target_np]
        npop = len(pop)
        if npop <= 3:
            break

        # periodic local polish (more frequent when stagnating)
        if (gen % 7 == 0) or (stagn_gens >= 10 and gen % 3 == 0):
            bx, bf = local_polish_sparse(best_x, best, evals=10)
            if bf < best:
                best, best_x = bf, bx[:]
                elite_add(bf, bx)
                stagn_gens = 0

        # p-best size (shrinks over time)
        p0 = 0.25 if dim <= 20 else 0.16
        p = max(0.08, p0 * (1.0 - 0.5 * min(1.0, tfrac)))
        pcount = max(2, int(math.ceil(p * npop)))

        union = [ind[0] for ind in pop] + archive
        union_len = len(union)

        SF, SCR, dF = [], [], []

        # operator probabilities (adapt a bit with time/stagnation)
        # early: more exploration, late: more pbest
        prob_pbest = 0.70 + 0.20 * min(1.0, tfrac)
        prob_best1 = 0.05 + (0.10 if stagn_gens >= 12 else 0.0)
        prob_rand1 = max(0.05, 1.0 - prob_pbest - prob_best1)

        for i in range(npop):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            Fi = cauchy(muF, 0.09)
            tries = 0
            while Fi <= 0.0 and tries < 10:
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

            # index picking helpers
            def pick_pop_index(exclude1=-1, exclude2=-1):
                while True:
                    rr = random.randrange(npop)
                    if rr != exclude1 and rr != exclude2:
                        return rr

            def pick_union(exclude_pop_i=-1, exclude_pop_r1=-1):
                while True:
                    rr = random.randrange(union_len)
                    if rr == exclude_pop_i or rr == exclude_pop_r1:
                        continue
                    return union[rr]

            # choose operator
            op_r = random.random()
            if op_r < prob_pbest:
                pbest = pop[random.randrange(pcount)][0]
                r1 = pick_pop_index(exclude1=i)
                r1x = pop[r1][0]
                r2x = pick_union(exclude_pop_i=i, exclude_pop_r1=r1)
                v = mutate_current_to_pbest_1(xi, pbest, r1x, r2x, Fi)

                # small best injection late
                if random.random() < (0.08 + 0.10 * tfrac):
                    for d in range(dim):
                        v[d] = clamp(0.90 * v[d] + 0.10 * best_x[d], d)

            elif op_r < prob_pbest + prob_best1 and npop >= 3:
                r1 = pick_pop_index(exclude1=i)
                r2 = pick_pop_index(exclude1=i, exclude2=r1)
                v = mutate_best_1(best_x, pop[r1][0], pop[r2][0], Fi)
            else:
                # rand/1
                a = pick_pop_index(exclude1=i)
                b = pick_pop_index(exclude1=i, exclude2=a)
                c = pick_pop_index(exclude1=i, exclude2=b)
                v = mutate_rand_1(pop[a][0], pop[b][0], pop[c][0], Fi)

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
                    stagn_gens = 0

                SF.append(Fi)
                SCR.append(CRi)
                dF.append(max(0.0, fi - fu))
            else:
                # stronger "loser rescue" when stagnating
                if stagn_gens >= 14 and random.random() < 0.03 and elite:
                    ex = elite[random.randrange(min(10, len(elite)))][1]
                    xr = [clamp(0.75 * ex[d] + 0.25 * (lows[d] + random.random() * spans[d]), d) for d in range(dim)]
                    fr = eval_point(xr)
                    if fr < pop[i][1]:
                        pop[i] = [xr, fr]
                        if fr < best:
                            best, best_x = fr, xr[:]
                            elite_add(fr, xr)
                            stagn_gens = 0

        # adapt memories (JADE/L-SHADE style)
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

            # smoothing
            M_F[mem_idx] = min(0.98, max(0.05, 0.25 * M_F[mem_idx] + 0.75 * new_muF))
            M_CR[mem_idx] = min(0.98, max(0.0, 0.25 * M_CR[mem_idx] + 0.75 * new_muCR))
            mem_idx = (mem_idx + 1) % H

        archive_cap = npop
        if len(archive) > archive_cap:
            archive = archive[:archive_cap]

    return best
