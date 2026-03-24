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

    # --------- helpers ----------
    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def eval_point(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Halton sequence (low-discrepancy) for better global coverage than stratified/binning
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
        # index should start from 1 (0 gives all zeros)
        u = [van_der_corput(index, b) for b in bases]
        return [lows[i] + u[i] * spans[i] for i in range(dim)]

    # --------- core state ----------
    best = float("inf")
    best_x = None

    # Evaluate once quickly so best_x is always defined
    x0 = rand_point()
    f0 = eval_point(x0)
    best, best_x = f0, list(x0)

    # --------- Phase 1: Halton global exploration + light opposition sampling ----------
    bases = first_primes(dim)

    # time-scaled initial sample count (kept conservative; time limit is unknown)
    # but generally improves over random/stratified for the same evals
    init_n = max(80, 35 * dim)

    for k in range(1, init_n + 1):
        if time.time() >= deadline:
            return best

        x = halton_point(k, bases)
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, list(x)

        # Opposition point (often helps on bounded domains with little cost)
        if time.time() >= deadline:
            return best
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        fo = eval_point(xo)
        if fo < best:
            best, best_x = fo, list(xo)

    # Maintain an elite pool for smarter restarts / mixing
    elite = [(best, list(best_x))]

    def elite_add(fx, x):
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        # keep small to reduce overhead
        if len(elite) > 12:
            del elite[12:]

    # Add a few more random probes to diversify elite
    extra = max(20, 10 * dim)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_point(x)
        # admit if good enough
        if len(elite) < 12 or fx < elite[-1][0]:
            elite_add(fx, x)
        if fx < best:
            best, best_x = fx, list(x)

    # --------- Phase 2: DE/current-to-best/1/bin (fast global+local) ----------
    # This tends to beat coordinate pattern search on many black-box functions.
    # Small population; time-limited; uses bound-handling + occasional local polish.

    pop_size = max(12, min(40, 6 * dim))
    pop = []
    # seed population from elite + halton + random
    for i in range(pop_size):
        if i < len(elite):
            x = list(elite[i][1])
        elif i < len(elite) + 5:
            idx = 1 + init_n + i
            x = halton_point(idx, bases)
        else:
            x = rand_point()
        fx = eval_point(x)
        pop.append([x, fx])
        if fx < best:
            best, best_x = fx, list(x)
            elite_add(fx, x)

    # DE parameters (self-adapting-ish via jitter)
    F_base = 0.65
    CR_base = 0.90

    # For a small local refinement occasionally: 1+1 ES around best_x
    def local_polish(x, fx, budget_evals=20):
        # Gaussian step with annealing
        step = [0.08 * spans[i] for i in range(dim)]
        cur = list(x)
        cur_f = fx
        for t in range(budget_evals):
            if time.time() >= deadline:
                break
            cand = cur[:]
            # perturb a few coordinates (sparser in high dim)
            k = 1 if dim == 1 else (2 if dim <= 10 else max(2, dim // 10))
            for _ in range(k):
                j = random.randrange(dim)
                # Box-Muller for normal noise
                u1 = max(1e-12, random.random())
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                cand[j] = clamp(cand[j] + z * step[j], j)
            cand_f = eval_point(cand)
            if cand_f < cur_f:
                cur, cur_f = cand, cand_f
            # anneal step a bit
            for j in range(dim):
                step[j] *= 0.92
        return cur, cur_f

    gen = 0
    while time.time() < deadline:
        gen += 1

        # refresh best index for "current-to-best"
        # (best_x already tracked globally; also track best in pop)
        pop.sort(key=lambda t: t[1])
        if pop[0][1] < best:
            best, best_x = pop[0][1], list(pop[0][0])
            elite_add(best, best_x)

        # occasional local polish on global best (cheap, can pay off)
        if gen % 12 == 0:
            bx, bf = local_polish(best_x, best, budget_evals=12)
            if bf < best:
                best, best_x = bf, list(bx)
                elite_add(best, best_x)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi, fi = pop[i]

            # choose r1,r2 distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = random.sample(idxs, 2)
            x1 = pop[r1][0]
            x2 = pop[r2][0]

            # jitter parameters slightly per-trial (helps robustness)
            F = min(0.95, max(0.25, F_base + (random.random() - 0.5) * 0.2))
            CR = min(0.98, max(0.05, CR_base + (random.random() - 0.5) * 0.2))

            # current-to-best/1 mutation: v = xi + F*(best - xi) + F*(x1 - x2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + F * (best_x[d] - xi[d]) + F * (x1[d] - x2[d])
                # bounce-back bound handling (better than pure clamp in many cases)
                if v[d] < lows[d]:
                    v[d] = lows[d] + (lows[d] - v[d])
                    if v[d] > highs[d]:
                        v[d] = lows[d]
                elif v[d] > highs[d]:
                    v[d] = highs[d] - (v[d] - highs[d])
                    if v[d] < lows[d]:
                        v[d] = highs[d]

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]
                # final clamp (safety)
                if u[d] < lows[d]:
                    u[d] = lows[d]
                elif u[d] > highs[d]:
                    u[d] = highs[d]

            fu = eval_point(u)

            # selection
            if fu <= fi:
                pop[i] = [u, fu]
                if fu < best:
                    best, best_x = fu, list(u)
                    elite_add(fu, u)
            else:
                # tiny chance to inject elite-guided restart to avoid stagnation
                if random.random() < 0.02 and elite:
                    _, ex = random.choice(elite[:min(6, len(elite))])
                    # mix with random point
                    mix = [0.0] * dim
                    a = 0.7
                    for d in range(dim):
                        mix[d] = clamp(a * ex[d] + (1.0 - a) * (lows[d] + random.random() * spans[d]), d)
                    fm = eval_point(mix)
                    if fm < pop[i][1]:
                        pop[i] = [mix, fm]
                        if fm < best:
                            best, best_x = fm, list(mix)
                            elite_add(fm, mix)

    return best
