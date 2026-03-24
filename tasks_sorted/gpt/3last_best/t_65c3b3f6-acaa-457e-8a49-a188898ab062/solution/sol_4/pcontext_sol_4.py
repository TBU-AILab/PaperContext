import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (no external libraries).

    Upgrades vs the provided L-SHADE-ish DE:
      1) Evaluation budget awareness (uses time only, but structures work in "chunks")
      2) Better constraint handling: bounce-back reflection (repeated) + tiny jitter
      3) Dual-phase search:
           - Global: L-SHADE style current-to-pbest/1 + archive + history adaptation
           - Exploit: periodic SPX (simplex) / Rosenbrock-like coordinate refinement
      4) Population size reduction over time (L-SHADE idea) for faster late exploitation
      5) Reinitialization on stagnation uses *mixture*:
           - around best (gaussian)
           - around random archive member
           - global uniform
      6) Micro-annealed local search step schedule tied to remaining time

    Returns:
      best (float): best objective value found within time limit.
    """
    t0 = time.time()
    if max_time <= 0:
        return float("inf")
    deadline = t0 + max_time

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    def now():
        return time.time()

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def clip(v, d):
        if v < lo[d]:
            return lo[d]
        if v > hi[d]:
            return hi[d]
        return v

    def reflect_repair(v, d):
        """Repeated reflection into [lo,hi] (handles far out-of-bounds better than single reflect)."""
        a, b = lo[d], hi[d]
        if a == b:
            return a
        # up to a few bounces (enough in practice)
        for _ in range(6):
            if v < a:
                v = a + (a - v)
            elif v > b:
                v = b - (v - b)
            else:
                break
        # final clip
        if v < a:
            v = a
        elif v > b:
            v = b
        return v

    def jitter_inside(x, scale=1e-12):
        # avoid exact-boundary sticking / identical points
        y = x[:]
        for d in range(dim):
            if hi[d] > lo[d]:
                eps = scale * span_safe[d]
                if eps > 0.0:
                    y[d] = clip(y[d] + random.uniform(-eps, eps), d)
        return y

    def rand_uniform_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- scrambled Halton init ----------
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
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        return r

    def halton_point(k):
        x = []
        for d in range(dim):
            u = halton_scrambled_value(k, primes[d], digit_perm[d])
            x.append(lo[d] + u * span_safe[d])
        return x

    def opposite_point(x):
        return [lo[i] + hi[i] - x[i] for i in range(dim)]

    # ---------- DE / SHADE parameters ----------
    # initial population size; will reduce toward min_pop over time
    init_pop = int(18 + 5.0 * math.sqrt(max(1, dim)))
    init_pop = max(22, min(90, init_pop))
    min_pop = max(8, min(24, 6 + int(2.0 * math.sqrt(max(1, dim)))))

    # archive size factor
    arc_factor = 1.0

    # SHADE memories
    H = 8
    MCR = [0.5] * H
    MF = [0.6] * H
    mem_idx = 0

    # p-best schedule
    pmin, pmax = 0.06, 0.25

    def rand_cauchy(mu, gamma):
        u = random.random()
        return mu + gamma * math.tan(math.pi * (u - 0.5))

    def pick_distinct_index(n, banned):
        j = random.randrange(n)
        while j in banned:
            j = random.randrange(n)
        return j

    # ---------- initialize population ----------
    pop = []  # list [x, fx]
    best = float("inf")
    best_x = None

    # oversample a little then downselect (helps)
    target = init_pop
    k = 1
    while len(pop) < target and now() < deadline:
        x = halton_point(k)
        k += 1
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

        if len(pop) < target and now() < deadline:
            xo = opposite_point(x)
            fxo = evaluate(xo)
            pop.append([xo, fxo])
            if fxo < best:
                best, best_x = fxo, xo[:]

    while len(pop) < target and now() < deadline:
        x = rand_uniform_point()
        fx = evaluate(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, x[:]

    if not pop:
        return float("inf")

    # downselect to init_pop best individuals
    pop.sort(key=lambda t: t[1])
    pop = pop[:init_pop]

    archive = []
    last_improve_t = now()

    # ---------- local refinement (Rosenbrock-ish coordinate + adaptive steps) ----------
    ls_step = [0.10 * span_safe[i] for i in range(dim)]
    ls_min = [1e-14 * span_safe[i] for i in range(dim)]

    def local_refine(x0, f0, passes=2):
        x = x0[:]
        fx = f0

        # time-aware step scaling (smaller near end)
        rem = max(0.0, deadline - now())
        frac_rem = rem / max(1e-12, max_time)
        step_scale = 0.35 + 0.65 * frac_rem  # 1.0 early, ~0.35 late

        for _ in range(passes):
            improved_any = False
            # random coordinate order helps in nonseparable problems
            order = list(range(dim))
            random.shuffle(order)
            for d in order:
                if now() >= deadline:
                    return x, fx
                step = ls_step[d] * step_scale
                if step <= ls_min[d]:
                    continue

                xd = x[d]
                # try a small multi-try along coordinate (1x, 2x)
                for sgn in (-1.0, 1.0):
                    for mult in (1.0, 2.0):
                        y = x[:]
                        y[d] = reflect_repair(xd + sgn * mult * step, d)
                        y = jitter_inside(y, scale=1e-15)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved_any = True
                            break
                    if improved_any:
                        break
            if not improved_any:
                break
        return x, fx

    # ---------- main loop ----------
    it = 0
    no_improve_window = max(0.8, max_time / 8.0)
    # evaluation batching: do local refine more often late
    while True:
        t = now()
        if t >= deadline:
            return best

        # progress fraction
        frac = clamp01((t - t0) / max(1e-12, max_time))

        # population size reduction (L-SHADE idea)
        desired_n = int(round(init_pop - (init_pop - min_pop) * frac))
        desired_n = max(min_pop, min(init_pop, desired_n))
        if len(pop) > desired_n:
            pop.sort(key=lambda z: z[1])
            pop = pop[:desired_n]
            # shrink archive max accordingly
            arc_max = int(math.ceil(arc_factor * len(pop)))
            if len(archive) > arc_max:
                # random delete
                while len(archive) > arc_max:
                    del archive[random.randrange(len(archive))]
        else:
            arc_max = int(math.ceil(arc_factor * len(pop)))

        pop.sort(key=lambda z: z[1])
        n = len(pop)
        if n < 4:
            # emergency refill
            while len(pop) < 4 and now() < deadline:
                x = rand_uniform_point()
                fx = evaluate(x)
                pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]
            pop.sort(key=lambda z: z[1])
            n = len(pop)

        # adapt p-best size over time (more exploitation later)
        p = pmax - (pmax - pmin) * frac
        pbest_count = max(2, int(math.ceil(p * n)))

        # union for mutation
        union = [ind[0] for ind in pop] + archive
        union_n = len(union)

        S_F, S_CR, S_w = [], [], []

        for i in range(n):
            if now() >= deadline:
                return best

            xi, fxi = pop[i][0], pop[i][1]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = clamp01(mu_cr + 0.1 * random.gauss(0.0, 1.0))

            F = rand_cauchy(mu_f, 0.1)
            tries = 0
            while F <= 0.0 and tries < 10:
                F = rand_cauchy(mu_f, 0.1)
                tries += 1
            if F <= 0.0:
                F = 0.5
            if F > 1.0:
                F = 1.0

            pbest_idx = random.randrange(pbest_count)
            xpbest = pop[pbest_idx][0]

            banned = {i, pbest_idx}
            r1 = pick_distinct_index(n, banned)
            banned.add(r1)

            # r2 from union (pop+archive) but avoid using same pop indices
            banned_union = set(idx for idx in (i, pbest_idx, r1) if 0 <= idx < n)
            r2u = random.randrange(union_n)
            tries2 = 0
            while r2u in banned_union and tries2 < 25:
                r2u = random.randrange(union_n)
                tries2 += 1

            xr1 = pop[r1][0]
            xr2 = union[r2u]

            # mutation: current-to-pbest/1
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (xpbest[d] - xi[d]) + F * (xr1[d] - xr2[d])

            # crossover + repair
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = reflect_repair(v[d], d)

            u = jitter_inside(u, scale=1e-15)
            fu = evaluate(u)

            if fu <= fxi:
                archive.append(xi[:])
                if len(archive) > arc_max:
                    del archive[random.randrange(len(archive))]

                pop[i][0], pop[i][1] = u, fu

                if fu < best:
                    best, best_x = fu, u[:]
                    last_improve_t = now()

                df = (fxi - fu)
                if df <= 0.0:
                    df = 1e-12
                S_F.append(F)
                S_CR.append(CR)
                S_w.append(df)

        # update memories
        if S_F:
            w_sum = sum(S_w)
            if w_sum <= 0.0:
                w_sum = float(len(S_w))

            mcr_new = 0.0
            for cr, w in zip(S_CR, S_w):
                mcr_new += (w / w_sum) * cr

            num = 0.0
            den = 0.0
            for f, w in zip(S_F, S_w):
                num += w * f * f
                den += w * f
            mf_new = (num / den) if den > 0.0 else MF[mem_idx]

            # a bit less smoothing late (react quicker)
            alpha = 0.08 + 0.10 * frac
            MCR[mem_idx] = (1.0 - alpha) * MCR[mem_idx] + alpha * mcr_new
            MF[mem_idx]  = (1.0 - alpha) * MF[mem_idx]  + alpha * mf_new
            mem_idx = (mem_idx + 1) % H

        it += 1

        # periodic local refinement: more frequent later
        if best_x is not None:
            # every ~12 iterations early, ~6 late
            period = 12 if frac < 0.5 else 6
            if (it % period) == 0 and now() < deadline:
                xb, fb = local_refine(best_x, best, passes=2 if frac < 0.7 else 3)
                if fb < best:
                    best, best_x = fb, xb[:]
                    last_improve_t = now()
                    for d in range(dim):
                        ls_step[d] = min(0.25 * span_safe[d], ls_step[d] * 1.12)
                else:
                    for d in range(dim):
                        ls_step[d] = max(ls_min[d], ls_step[d] * 0.82)

        # stagnation handling: diversified partial restart
        if (now() - last_improve_t) > no_improve_window:
            last_improve_t = now()
            pop.sort(key=lambda z: z[1])

            elite = max(2, min(6, len(pop) // 6))
            kept = pop[:elite]

            # mix restart sources
            new_pop = [[kept[i][0][:], kept[i][1]] for i in range(elite)]

            # restart radius decreases over time
            rad = (0.30 * (1.0 - 0.65 * frac))
            rad = max(0.05, rad)

            # clear some archive but keep a bit
            if len(archive) > 0:
                random.shuffle(archive)
                archive = archive[:max(0, int(0.35 * arc_max))]
            else:
                archive = []

            while len(new_pop) < len(pop) and now() < deadline:
                r = random.random()
                if best_x is not None and r < 0.60:
                    # around best
                    x = []
                    for d in range(dim):
                        sd = rad * span_safe[d]
                        x.append(clip(random.gauss(best_x[d], sd), d))
                elif archive and r < 0.85:
                    # around a random archive point
                    a = archive[random.randrange(len(archive))]
                    x = []
                    for d in range(dim):
                        sd = 0.9 * rad * span_safe[d]
                        x.append(clip(random.gauss(a[d], sd), d))
                else:
                    x = rand_uniform_point()

                x = jitter_inside(x, scale=1e-15)
                fx = evaluate(x)
                new_pop.append([x, fx])
                if fx < best:
                    best, best_x = fx, x[:]

            pop = new_pop

            # slightly broaden memories to encourage exploration after restart
            for h in range(H):
                MF[h] = min(0.95, max(0.20, MF[h] * 1.06))
                MCR[h] = min(0.95, max(0.05, MCR[h] * 1.03))
            # reset local steps a bit upward
            for d in range(dim):
                ls_step[d] = max(ls_step[d], 0.07 * span_safe[d])
