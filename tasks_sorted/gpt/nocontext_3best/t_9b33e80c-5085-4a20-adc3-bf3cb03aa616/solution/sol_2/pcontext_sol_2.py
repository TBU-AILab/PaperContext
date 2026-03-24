import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization.

    Improvement over the provided codes:
      - Uses a small Differential Evolution (DE/rand/1/bin) population for global search
        (much stronger than pure sampling on many problems).
      - Lightweight local refinement (stochastic coordinate search) applied to the current best.
      - Time-adaptive: explores early, refines more near the deadline.
      - Self-contained: no numpy or external libs.

    Returns:
        best (float): best objective value found within max_time
    """

    # ---------------- helpers ----------------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(x))

    def make_random_point():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            s = spans[i]
            x[i] = lo if s == 0.0 else (lo + random.random() * s)
        return x

    # Halton for better-than-random seeding
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
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

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, bases):
        x = [0.0] * dim
        for d in range(dim):
            lo, hi = bounds[d]
            s = spans[d]
            if s == 0.0:
                x[d] = lo
            else:
                u = halton_value(k, bases[d])
                x[d] = lo + u * s
        return x

    # A small local search around a point
    def local_refine(x, fx, deadline, frac):
        # step scale shrinks as time passes
        base = 0.20 * (1.0 - 0.75 * frac) + 0.015  # ~0.215 -> ~0.015
        step = [base * s for s in spans]

        bestx = list(x)
        bestf = fx

        # iteration count: small but effective
        iters = max(12, min(90, 10 * dim))
        shrink = 0.55
        expand = 1.20

        for _ in range(iters):
            if time.time() >= deadline:
                break

            improved = False
            coords = list(range(dim))
            random.shuffle(coords)

            for d in coords:
                if time.time() >= deadline:
                    break
                if spans[d] == 0.0:
                    continue

                sd = step[d]
                if sd <= 0.0:
                    continue

                lo, hi = bounds[d]

                # Try a few random signed moves (not just +/-sd) for robustness
                # (keeps it cheap: 2-3 evals per coordinate at most)
                for _try in range(2):
                    direction = -1.0 if random.random() < 0.5 else 1.0
                    scale = 1.0 if random.random() < 0.75 else (0.25 + 1.75 * random.random())
                    cand = list(bestx)
                    cand[d] = clamp(cand[d] + direction * sd * scale, lo, hi)
                    if cand[d] == bestx[d]:
                        continue
                    f = eval_f(cand)
                    if f < bestf:
                        bestf = f
                        bestx = cand
                        step[d] = min(step[d] * expand, spans[d])
                        improved = True
                        break

            if not improved:
                # shrink steps if no coordinate helped
                for d in range(dim):
                    if spans[d] > 0.0:
                        step[d] *= shrink
                        # minimal step to keep movement possible
                        min_sd = 1e-14 * (spans[d] if spans[d] > 0 else 1.0)
                        if step[d] < min_sd:
                            step[d] = min_sd

        return bestx, bestf

    # ---------------- main ----------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    spans = []
    for (lo, hi) in bounds:
        if hi < lo:
            raise ValueError("Each bound must be (low, high) with high >= low")
        spans.append(hi - lo)

    # fully fixed point
    if all(s == 0.0 for s in spans):
        x = [bounds[i][0] for i in range(dim)]
        return eval_f(x)

    random.seed()

    # Population size: modest for speed, but enough for DE to work
    pop_size = max(10, min(40, 6 + 3 * dim))

    # Seeding: half Halton, half random (better coverage + randomness)
    bases = first_primes(dim)
    pop = []
    fit = []

    best = float("inf")
    bestx = None

    # Build initial population
    k = 1
    for i in range(pop_size):
        if time.time() >= deadline:
            return best
        if i < pop_size // 2:
            x = halton_point(k, bases)
            k += 1
        else:
            x = make_random_point()

        f = eval_f(x)
        pop.append(x)
        fit.append(f)
        if f < best:
            best = f
            bestx = x

    # Differential Evolution loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = time.time() - t0
        frac = elapsed / max(1e-12, float(max_time))

        # time-adaptive DE parameters:
        # - more exploration early (higher F, higher CR)
        # - more exploitation late (slightly lower F, lower CR)
        F = 0.85 - 0.35 * frac     # 0.85 -> 0.50
        CR = 0.90 - 0.55 * frac    # 0.90 -> 0.35
        if F < 0.40: F = 0.40
        if CR < 0.20: CR = 0.20

        # A "current-to-best" influence very late can help polish
        use_best_pull = (frac > 0.70)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose r1,r2,r3 distinct and not i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = r1
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)
            r3 = r2
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(pop_size)

            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]
            xi = pop[i]

            # mutation
            v = [0.0] * dim
            if use_best_pull and bestx is not None and random.random() < 0.65:
                # current-to-best/1 style (gentle pull to best late in time)
                K = 0.25 + 0.35 * random.random()  # 0.25..0.60
                for d in range(dim):
                    v[d] = xi[d] + K * (bestx[d] - xi[d]) + F * (x2[d] - x3[d])
            else:
                # rand/1
                for d in range(dim):
                    v[d] = x1[d] + F * (x2[d] - x3[d])

            # binomial crossover + bounds
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if spans[d] == 0.0:
                    u[d] = bounds[d][0]
                    continue
                if random.random() < CR or d == jrand:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]
                lo, hi = bounds[d]
                if u[d] < lo or u[d] > hi:
                    # reflect to keep diversity (often better than clamp-only)
                    if u[d] < lo:
                        u[d] = lo + (lo - u[d])
                    if u[d] > hi:
                        u[d] = hi - (u[d] - hi)
                    # if reflection still out (can happen), clamp
                    u[d] = clamp(u[d], lo, hi)

            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    bestx = u

        # occasional local refinement on the best, increasingly frequent near the end
        if bestx is not None:
            # probability ramps up with time; keep it cheap
            p_ref = 0.08 + 0.35 * frac
            if random.random() < p_ref and time.time() < deadline:
                rx, rf = local_refine(bestx, best, deadline, frac)
                if rf < best:
                    best = rf
                    bestx = rx

                # inject refined best into population by replacing worst (elitism)
                worst_i = 0
                worst_f = fit[0]
                for j in range(1, pop_size):
                    if fit[j] > worst_f:
                        worst_f = fit[j]
                        worst_i = j
                pop[worst_i] = list(bestx)
                fit[worst_i] = best

        # if plenty of time remains, allow a tiny random injection to avoid stagnation
        if frac < 0.85 and random.random() < 0.10:
            j = random.randrange(pop_size)
            x = make_random_point()
            f = eval_f(x)
            pop[j] = x
            fit[j] = f
            if f < best:
                best = f
                bestx = x

    return best
