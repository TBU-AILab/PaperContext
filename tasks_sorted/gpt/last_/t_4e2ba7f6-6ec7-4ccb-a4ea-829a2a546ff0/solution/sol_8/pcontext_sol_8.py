import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs prior version:
      - Uses DE (current-to-best/1/bin) as a strong global explorer on bounded domains.
      - Keeps a small population for speed; auto-fits to time by shrinking/growing eval work.
      - Adds a robust, budgeted local improvement step (adaptive coordinate search)
        only on the incumbent to polish.
      - Strict time checks everywhere; no heavy caching overhead.
      - Bound handling: reflection with final clamp.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if any(s <= 0.0 for s in spans):
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # -------------------- helpers --------------------
    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect until in range (handles large steps)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            else:
                v = hi - (v - hi)
        return clamp(v, lo, hi)

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # Keep it minimal: user func is black-box; just guard time upstream.
        return float(func(x))

    # Simple Gaussian
    def randn():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2.0 * math.pi * u2)

    # -------------------- incumbent init --------------------
    best = float("inf")
    best_x = None

    # initial sampling (few points, time-safe)
    init_n = max(8, min(40, 4 * dim + 8))
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = rand_vec()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    # -------------------- local polish (coordinate) --------------------
    def local_coordinate(x0, f0, max_passes=2):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        # adaptive initial step ~10% span, shrink quickly
        step = [0.1 * spans[i] for i in range(dim)]
        min_step = [1e-12 * spans[i] + 1e-15 for i in range(dim)]

        for _ in range(max_passes):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if time.time() >= deadline:
                    break
                if step[j] <= min_step[j]:
                    continue

                sj = step[j]
                xj = x[j]

                # try +, then -
                xp = x[:]
                xp[j] = reflect(xj + sj, lows[j], highs[j])
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    step[j] = min(spans[j], sj * 1.4)
                    continue

                xm = x[:]
                xm[j] = reflect(xj - sj, lows[j], highs[j])
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    step[j] = min(spans[j], sj * 1.4)
                    continue

                step[j] = sj * 0.55

            if not improved:
                # global shrink and stop if tiny
                tiny = True
                for j in range(dim):
                    step[j] *= 0.6
                    if step[j] > min_step[j]:
                        tiny = False
                if tiny:
                    break
        return x, fx

    # quick initial polish
    if time.time() < deadline and best_x is not None:
        bx, bf = local_coordinate(best_x, best, max_passes=1)
        if bf < best:
            best, best_x = bf, bx

    # -------------------- Differential Evolution (DE) --------------------
    # Small population for speed, but at least 6
    NP = max(6, min(30, 5 + 2 * dim))
    pop = []
    fit = []

    # seed population around best_x + some randoms
    for i in range(NP):
        if time.time() >= deadline:
            return best
        if i == 0 and best_x is not None:
            x = best_x[:]
        else:
            if best_x is not None and random.random() < 0.6:
                # biased around best
                x = [reflect(best_x[j] + 0.15 * spans[j] * randn(), lows[j], highs[j]) for j in range(dim)]
            else:
                x = rand_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # DE control parameters (dithered F helps robustness)
    CR = 0.9
    Fmin, Fmax = 0.4, 0.95

    gen = 0
    no_improve = 0
    last_best = best

    while time.time() < deadline:
        gen += 1

        # occasional polish, but keep it rare
        if (gen % 8 == 0 or no_improve >= 12) and best_x is not None and time.time() < deadline:
            bx, bf = local_coordinate(best_x, best, max_passes=2 if dim <= 15 else 1)
            if bf < best:
                best, best_x = bf, bx
            no_improve = 0

        # one DE generation
        # (iterate individuals; strict time check)
        for i in range(NP):
            if time.time() >= deadline:
                return best

            # choose r1,r2 distinct and != i
            # also use best (current-to-best/1)
            candidates = list(range(NP))
            # lightweight selection without full shuffle
            r1 = i
            while r1 == i:
                r1 = random.randrange(NP)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(NP)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]
            xb = best_x if best_x is not None else pop[random.randrange(NP)]

            F = Fmin + (Fmax - Fmin) * random.random()

            # mutation: v = xi + F*(best - xi) + F*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + F * (xb[j] - xi[j]) + F * (xr1[j] - xr2[j])
                v[j] = vj  # bounds handled after crossover

            # binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]  # start as target
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = reflect(v[j], lows[j], highs[j])

            fu = eval_f(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # stagnation / restart handling
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            no_improve = 0
        else:
            no_improve += 1

        # soft restart: re-inject diversity if stuck
        if no_improve >= 25 and time.time() < deadline:
            no_improve = 0
            # keep best, refresh half population around best + random
            for k in range(1, NP):
                if time.time() >= deadline:
                    return best
                if random.random() < 0.7 and best_x is not None:
                    x = [reflect(best_x[j] + 0.35 * spans[j] * randn(), lows[j], highs[j]) for j in range(dim)]
                else:
                    x = rand_vec()
                fx = eval_f(x)
                pop[k] = x
                fit[k] = fx
                if fx < best:
                    best, best_x = fx, x[:]

    return best
