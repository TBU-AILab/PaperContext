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

    def repair_reflect(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect repeatedly (handles far violations)
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                else:
                    v = hi - (v - hi)
            if v < lo: v = lo
            elif v > hi: v = hi
            y[i] = v
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Halton sequence (low-discrepancy) for seeding
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

    # ---------- elite pool ----------
    elite_cap = 48
    elite = []  # list of (f, x)

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_cap:
            del elite[elite_cap:]

    # ---------- incumbent ----------
    best_x = rand_point()
    best = eval_point(best_x)
    elite_add(best, best_x)

    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    # ---------- Phase 1: stronger, time-aware coverage ----------
    bases = first_primes(dim)
    seed_deadline = t0 + 0.23 * float(max_time)
    seed_cap = max(500, min(5200, 160 * dim + 800))
    used = 0
    k = 1

    while time.time() < deadline and time.time() < seed_deadline and used < seed_cap:
        x = halton_point(k, bases); k += 1
        fx = eval_point(x); used += 1
        if fx < best:
            best, best_x = fx, list(x)
        elite_add(fx, x)
        if time.time() >= deadline or time.time() >= seed_deadline or used >= seed_cap:
            break

        # opposition
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        fo = eval_point(xo); used += 1
        if fo < best:
            best, best_x = fo, list(xo)
        elite_add(fo, xo)
        if time.time() >= deadline or time.time() >= seed_deadline or used >= seed_cap:
            break

        # quasi-opposition about center
        xqo = [clamp(center[i] + (center[i] - x[i]) * (0.6 + 0.9 * random.random()), i) for i in range(dim)]
        fq = eval_point(xqo); used += 1
        if fq < best:
            best, best_x = fq, list(xqo)
        elite_add(fq, xqo)
        if time.time() >= deadline or time.time() >= seed_deadline or used >= seed_cap:
            break

        # recombination among elites (midpoint/extrapolation)
        if len(elite) >= 2:
            a = elite[random.randrange(min(12, len(elite)))][1]
            b = elite[random.randrange(min(12, len(elite)))][1]
            w = 0.5 + 0.45 * (2.0 * random.random() - 1.0)
            xm = [clamp(w * a[i] + (1.0 - w) * b[i], i) for i in range(dim)]
            fm = eval_point(xm); used += 1
            if fm < best:
                best, best_x = fm, list(xm)
            elite_add(fm, xm)

    # ---------- Phase 2: L-SHADE/JADE DE with archive + two-stage polish ----------
    pop_init = max(28, min(96, 8 * dim + 20))
    pop_min = max(10, min(34, 2 * dim + 8))

    pop = []
    for i in range(pop_init):
        if time.time() >= deadline:
            return best
        if i < len(elite):
            x = elite[i][1][:]
        elif i < len(elite) + 18:
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

    H = 12
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

    # --- local refinement: (A) sparse ES, then (B) coordinate try +/- ---
    def local_polish_combo(x, fx, evals=18):
        cur = x[:]
        cur_f = fx

        # A) sparse Gaussian search
        step = [0.06 * spans[i] for i in range(dim)]
        for _ in range(max(1, evals // 2)):
            if time.time() >= deadline:
                break
            cand = cur[:]
            kdim = 1 if dim == 1 else (2 if dim <= 16 else max(2, dim // 16))
            for __ in range(kdim):
                j = random.randrange(dim)
                cand[j] = clamp(cand[j] + randn() * step[j], j)
            cf = eval_point(cand)
            if cf < cur_f:
                cur, cur_f = cand, cf
            for j in range(dim):
                step[j] *= 0.90

        # B) coordinate +/- (cheap pattern hook)
        step2 = [0.03 * spans[i] for i in range(dim)]
        for _ in range(evals - max(1, evals // 2)):
            if time.time() >= deadline:
                break
            j = random.randrange(dim)
            s = step2[j]
            if s <= 1e-15 * spans[j]:
                continue
            base = cur[j]
            # try both directions
            cand1 = cur[:]; cand1[j] = clamp(base - s, j)
            f1 = eval_point(cand1)
            if f1 < cur_f:
                cur, cur_f = cand1, f1
                step2[j] *= 1.10
                continue
            cand2 = cur[:]; cand2[j] = clamp(base + s, j)
            f2 = eval_point(cand2)
            if f2 < cur_f:
                cur, cur_f = cand2, f2
                step2[j] *= 1.10
            else:
                step2[j] *= 0.70

        return cur, cur_f

    gen = 0
    last_best = best
    stagn = 0

    while time.time() < deadline and len(pop) >= pop_min:
        gen += 1
        pop.sort(key=lambda it: it[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], pop[0][0][:]
            elite_add(best, best_x)

        if best < last_best - 1e-12:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        target_np = int(round(pop_init - (pop_init - pop_min) * min(1.0, max(0.0, tfrac))))
        if target_np < pop_min:
            target_np = pop_min
        if len(pop) > target_np:
            pop = pop[:target_np]
        npop = len(pop)
        if npop <= 3:
            break

        # polishing schedule: more aggressive late and/or when stagnating
        if (gen % 6 == 0) or (stagn >= 9 and gen % 3 == 0) or (tfrac > 0.80 and gen % 2 == 0):
            bx, bf = local_polish_combo(best_x, best, evals=12 if tfrac < 0.75 else 18)
            if bf < best:
                best, best_x = bf, bx[:]
                elite_add(bf, bx)
                stagn = 0

        p0 = 0.28 if dim <= 20 else 0.18
        p = max(0.06, p0 * (1.0 - 0.60 * min(1.0, tfrac)))
        pcount = max(2, int(math.ceil(p * npop)))

        union = [ind[0] for ind in pop] + archive
        union_len = len(union)

        SF, SCR, dF = [], [], []

        # operator mix: exploration early, exploitation late; add best/1 when stuck
        prob_pbest = 0.66 + 0.26 * min(1.0, tfrac)
        prob_best1 = 0.03 + (0.14 if stagn >= 12 else 0.0)
        prob_rand1 = max(0.06, 1.0 - prob_pbest - prob_best1)

        def pick_pop_index(ex1=-1, ex2=-1):
            while True:
                r = random.randrange(npop)
                if r != ex1 and r != ex2:
                    return r

        def pick_union_excluding(pop_i, pop_r1):
            while True:
                r = random.randrange(union_len)
                if r == pop_i or r == pop_r1:
                    continue
                return union[r]

        for i in range(npop):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]

            r = random.randrange(H)
            muF = M_F[r]
            muCR = M_CR[r]

            Fi = cauchy(muF, 0.085)
            tries = 0
            while Fi <= 0.0 and tries < 10:
                Fi = cauchy(muF, 0.085); tries += 1
            if Fi <= 0.0: Fi = 0.5
            if Fi > 1.0: Fi = 1.0

            CRi = normal(muCR, 0.085)
            if CRi < 0.0: CRi = 0.0
            if CRi > 1.0: CRi = 1.0

            op_r = random.random()
            if op_r < prob_pbest:
                pbest = pop[random.randrange(pcount)][0]
                r1 = pick_pop_index(ex1=i)
                r1x = pop[r1][0]
                r2x = pick_union_excluding(i, r1)
                v = mutate_current_to_pbest_1(xi, pbest, r1x, r2x, Fi)

                # stronger best injection late (helps final squeezing)
                if random.random() < (0.10 + 0.16 * tfrac):
                    inj = 0.08 + 0.18 * tfrac
                    for d in range(dim):
                        v[d] = clamp((1.0 - inj) * v[d] + inj * best_x[d], d)

            elif op_r < prob_pbest + prob_best1:
                r1 = pick_pop_index(ex1=i)
                r2 = pick_pop_index(ex1=i, ex2=r1)
                v = mutate_best_1(best_x, pop[r1][0], pop[r2][0], Fi)
            else:
                a = pick_pop_index(ex1=i)
                b = pick_pop_index(ex1=i, ex2=a)
                c = pick_pop_index(ex1=i, ex2=b)
                v = mutate_rand_1(pop[a][0], pop[b][0], pop[c][0], Fi)

            u = bin_cross(xi, v, CRi)
            fu = eval_point(u)

            if fu <= fi:
                # archive replaced
                if len(archive) < archive_cap:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_cap)] = xi[:]

                pop[i] = [u, fu]
                if fu < best:
                    best, best_x = fu, u[:]
                    elite_add(fu, u)
                    stagn = 0

                SF.append(Fi)
                SCR.append(CRi)
                dF.append(max(0.0, fi - fu))
            else:
                # stagnation rescue: elite-centered gaussian sample
                if stagn >= 13 and elite and random.random() < 0.05:
                    ex = elite[random.randrange(min(14, len(elite)))][1]
                    xr = [0.0] * dim
                    for d in range(dim):
                        xr[d] = ex[d] + randn() * (0.07 * spans[d]) + 0.02 * (center[d] - ex[d])
                    xr = repair_reflect(xr)
                    fr = eval_point(xr)
                    if fr < pop[i][1]:
                        pop[i] = [xr, fr]
                        if fr < best:
                            best, best_x = fr, xr[:]
                            elite_add(fr, xr)
                            stagn = 0

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

            # more responsive memory update
            M_F[mem_idx] = min(0.98, max(0.05, 0.20 * M_F[mem_idx] + 0.80 * new_muF))
            M_CR[mem_idx] = min(0.98, max(0.00, 0.20 * M_CR[mem_idx] + 0.80 * new_muCR))
            mem_idx = (mem_idx + 1) % H

        archive_cap = npop
        if len(archive) > archive_cap:
            while len(archive) > archive_cap:
                archive.pop(random.randrange(len(archive)))

    return best
