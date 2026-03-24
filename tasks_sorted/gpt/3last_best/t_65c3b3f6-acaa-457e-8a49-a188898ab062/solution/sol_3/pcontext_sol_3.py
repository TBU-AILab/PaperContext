import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libraries).

    Improvements over the provided DE+pattern-search:
      - Uses "L-SHADE-ish" DE: current-to-pbest/1 with an external archive
      - Success-history adaptation of F and CR (memory of good parameters)
      - p-best selection (not always the single best) to reduce premature convergence
      - Archive promotes diversity and helps escape local minima
      - Robust bound handling (reflect then clip)
      - Lightweight final local coordinate search (budgeted)
      - Strict time checks everywhere

    Returns:
      best (float): best objective value found within time limit.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0 else 1.0 for s in span]

    def now():
        return time.time()

    def clip_reflect(val, d):
        a = lo[d]
        b = hi[d]
        if a == b:
            return a
        # reflect once (good for big jumps), then clip
        if val < a:
            val = a + (a - val)
        elif val > b:
            val = b - (val - b)
        if val < a:
            return a
        if val > b:
            return b
        return val

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---- low-discrepancy init: scrambled Halton (fast) + opposition ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    primes = first_primes(max(1, dim))

    digit_perm = []
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm.append(perm)

    def halton_scrambled_value(index, base, perm):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            digit = i % base
            r += f * perm[digit]
            i //= base
        # keep in [0,1]; small tweak for tiny bases
        if base <= 2:
            return r
        # perm digits are 0..base-1 so r in [0,1); normalize very mildly
        return min(1.0, max(0.0, r))

    def halton_point(k):  # k>=1
        x = []
        for d in range(dim):
            u = halton_scrambled_value(k, primes[d], digit_perm[d])
            x.append(lo[d] + u * span_safe[d])
        return x

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---- DE parameters / sizing ----
    pop_size = int(16 + 4.0 * math.sqrt(max(1, dim)))
    pop_size = max(18, min(72, pop_size))
    if pop_size < 6:
        pop_size = 6

    # archive (external) size
    arc_max = pop_size

    # L-SHADE memories
    H = 6
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    # p-best fraction range
    pmin, pmax = 0.08, 0.25

    # ---- init population ----
    pop = []  # list of [x, fx]
    best = float("inf")
    best_x = None

    k = 1
    while len(pop) < pop_size and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

        if len(pop) < pop_size and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo[:]

    while len(pop) < pop_size and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

    if not pop:
        return float("inf")

    # external archive for replaced solutions
    archive = []  # list of x (only vectors)

    def pick_distinct_index(n, banned_set):
        j = random.randrange(n)
        while j in banned_set:
            j = random.randrange(n)
        return j

    def rand_cauchy(mu, gamma):
        # standard Cauchy: mu + gamma * tan(pi*(u-0.5))
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def clamp01(v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    # ---- lightweight local search (coordinate) ----
    ls_step = [0.08 * span_safe[i] for i in range(dim)]
    ls_min = [1e-12 * span_safe[i] for i in range(dim)]

    def local_refine(x0, f0, passes=2):
        x = x0[:]
        fx = f0
        for _ in range(passes):
            improved = False
            for d in range(dim):
                if now() >= deadline:
                    return x, fx
                step = ls_step[d]
                if step <= ls_min[d]:
                    continue
                xd = x[d]
                for sgn in (-1.0, 1.0):
                    y = x[:]
                    y[d] = clip_reflect(xd + sgn * step, d)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break
            if not improved:
                break
        return x, fx

    # ---- main loop ----
    it = 0
    last_improve_t = now()
    no_improve_window = max(0.7, max_time / 7.0) if max_time > 0 else 0.7

    while True:
        if now() >= deadline:
            return best

        # sort by fitness
        pop.sort(key=lambda t: t[1])

        # p adapts over time: start more exploratory, end more exploit
        frac = (now() - t0) / max(1e-9, max_time)
        frac = clamp01(frac)
        p = pmax - (pmax - pmin) * frac  # decrease p with time (more exploitation later)
        pbest_count = max(2, int(math.ceil(p * len(pop))))

        # union size for mutation (pop + archive)
        union = [ind[0] for ind in pop] + archive

        # success histories for updating MF/MCR
        S_F = []
        S_CR = []
        S_dF = []  # weights (improvement)

        # iterate individuals
        n = len(pop)
        for i in range(n):
            if now() >= deadline:
                return best

            xi, fxi = pop[i][0], pop[i][1]

            # pick memory slot r
            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            # sample CR from normal, F from cauchy (as in SHADE)
            CR = mu_cr + 0.1 * random.gauss(0.0, 1.0)
            CR = clamp01(CR)

            # F: keep resampling until > 0
            F = rand_cauchy(mu_f, 0.1)
            tries = 0
            while F <= 0.0 and tries < 8:
                F = rand_cauchy(mu_f, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            # select pbest among top pbest_count
            pbest_idx = random.randrange(pbest_count)
            xpbest = pop[pbest_idx][0]

            # choose r1 from pop, r2 from union (pop+archive), all distinct
            banned = {i, pbest_idx}
            r1 = pick_distinct_index(n, banned)
            banned.add(r1)

            # for r2 choose from union indices; must not be xi itself by index if within pop
            union_n = len(union)
            # build banned union indices: i, pbest_idx, r1 (only if within pop part)
            banned_union = set()
            for idx in (i, pbest_idx, r1):
                if 0 <= idx < n:
                    banned_union.add(idx)
            r2u = random.randrange(union_n)
            tries2 = 0
            while r2u in banned_union and tries2 < 20:
                r2u = random.randrange(union_n)
                tries2 += 1

            xr1 = pop[r1][0]
            xr2 = union[r2u]

            # current-to-pbest/1:
            # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = clip_reflect(v[d], d)

            fu = evaluate(u)

            # selection + archive update
            if fu <= fxi:
                # push replaced into archive
                archive.append(xi[:])
                if len(archive) > arc_max:
                    # random removal
                    del archive[random.randrange(len(archive))]

                pop[i][0], pop[i][1] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = now()

                # record successes for memory update
                df = (fxi - fu)
                if df < 0:
                    df = 0.0
                S_F.append(F)
                S_CR.append(CR)
                S_dF.append(df if df > 0.0 else 1e-12)

        # update memories if we had successes
        if S_F:
            # weighted Lehmer mean for F; weighted arithmetic mean for CR
            w_sum = sum(S_dF)
            if w_sum <= 0.0:
                w_sum = float(len(S_dF))

            # MCR update
            mcr_new = 0.0
            for cr, w in zip(S_CR, S_dF):
                mcr_new += (w / w_sum) * cr

            # MF update (Lehmer mean): sum(w*f^2)/sum(w*f)
            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_dF):
                num += w * (f * f)
                den += w * f
            mf_new = (num / den) if den > 0.0 else MF[mem_idx]

            # smooth + store
            MCR[mem_idx] = 0.9 * MCR[mem_idx] + 0.1 * mcr_new
            MF[mem_idx] = 0.9 * MF[mem_idx] + 0.1 * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # occasional local refinement late in the run (more valuable near the end)
        if best_x is not None and (it % 10 == 0) and frac > 0.35:
            if now() < deadline:
                xb, fb = local_refine(best_x, best, passes=2)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()
                    for d in range(dim):
                        ls_step[d] = min(0.20 * span_safe[d], ls_step[d] * 1.10)
                else:
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.85)

        # stagnation kick: increase diversity by clearing archive and enlarging memories slightly
        if (now() - last_improve_t) > no_improve_window and max_time > 0:
            last_improve_t = now()
            archive.clear()
            # broaden memories a touch
            for h in range(H):
                MF[h] = min(0.9, max(0.25, MF[h] * 1.05))
                MCR[h] = min(0.9, max(0.05, MCR[h] * 1.02))
            # inject a few random individuals (keep elites)
            pop.sort(key=lambda t: t[1])
            elites = max(2, min(6, pop_size // 8))
            for j in range(elites, len(pop)):
                if now() >= deadline:
                    return best
                if random.random() < 0.35:
                    x = rand_uniform_point()
                    fx = evaluate(x)
                    pop[j] = [x, fx]
                    if fx < best:
                        best, best_x = fx, x[:]
#
