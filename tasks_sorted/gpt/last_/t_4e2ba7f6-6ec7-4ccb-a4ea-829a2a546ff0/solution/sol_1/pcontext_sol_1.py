import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization.
    Algorithm: Differential Evolution (DE/rand/1/bin) + occasional local refinement.
    - DE is strong for black-box continuous optimization with bounds.
    - Self-adapts via "jDE"-style per-individual F and CR updates.
    - Periodic coordinate/local search around the current best.

    Returns: best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    if dim <= 0:
        return float("inf")

    # --- population size (keep modest but effective) ---
    # Common rule-of-thumb: 5..10 * dim. Cap to avoid large overhead.
    NP = max(20, min(12 * dim, 120))

    # jDE parameters
    tau1 = 0.1   # prob to change F
    tau2 = 0.1   # prob to change CR
    Fl, Fu = 0.15, 0.95

    # Binomial crossover rate minimum
    CRmin = 0.0

    # --- initialize population ---
    pop = [rand_vec() for _ in range(NP)]
    fit = [None] * NP
    F = [0.5] * NP
    CR = [0.9] * NP

    best = float("inf")
    best_x = None

    # Evaluate initial population
    for i in range(NP):
        if time.time() >= deadline:
            return best
        fi = eval_f(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    # --- helpers for DE ---
    def pick3_excluding(excl):
        # pick 3 distinct indices not equal to excl
        a = excl
        while a == excl:
            a = random.randrange(NP)
        b = a
        while b == excl or b == a:
            b = random.randrange(NP)
        c = b
        while c == excl or c == a or c == b:
            c = random.randrange(NP)
        return a, b, c

    # --- occasional local refinement (coordinate + small gaussian) ---
    def local_refine(x0, f0, budget_steps=25):
        x = x0[:]
        fx = f0
        # start step sizes relative to span
        step = [0.12 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
        for _ in range(budget_steps):
            if time.time() >= deadline:
                break
            improved = False

            # Coordinate tries (+/-)
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if time.time() >= deadline:
                    break
                if spans[j] <= 0:
                    continue

                s = step[j]
                if s <= 0:
                    continue

                # Try plus
                xp = x[:]
                xp[j] = clamp(xp[j] + s, j)
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                # Try minus
                xm = x[:]
                xm[j] = clamp(xm[j] - s, j)
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            # Small random perturbation around current
            if time.time() < deadline:
                xr = x[:]
                for j in range(dim):
                    if spans[j] > 0 and random.random() < 0.35:
                        xr[j] = clamp(xr[j] + random.gauss(0.0, 0.05 * spans[j]), j)
                fr = eval_f(xr)
                if fr < fx:
                    x, fx = xr, fr
                    improved = True

            # Step control
            if not improved:
                for j in range(dim):
                    step[j] *= 0.6
                    # stop shrinking too far
                    if step[j] < 1e-14 * (spans[j] if spans[j] > 0 else 1.0):
                        step[j] = 0.0
                # if all steps are tiny, stop
                if all(s == 0.0 for s in step):
                    break
            else:
                # slight expansion if we are making progress
                for j in range(dim):
                    step[j] = min(step[j] * 1.1, 0.35 * (spans[j] if spans[j] > 0 else 1.0))
        return x, fx

    # --- main DE loop ---
    gen = 0
    refine_every = 8  # refine best every N generations (if time allows)

    while time.time() < deadline:
        gen += 1
        # iterate individuals
        for i in range(NP):
            if time.time() >= deadline:
                return best

            # jDE update
            Fi = F[i]
            CRi = CR[i]
            if random.random() < tau1:
                Fi = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CRi = CRmin + random.random() * (1.0 - CRmin)

            a, b, c = pick3_excluding(i)

            xa, xb, xc = pop[a], pop[b], pop[c]
            xi = pop[i]

            # mutation: v = xa + Fi*(xb-xc)
            v = [0.0] * dim
            for j in range(dim):
                vj = xa[j] + Fi * (xb[j] - xc[j])
                # clamp (simple bound handling)
                v[j] = clamp(vj, j)

            # crossover (binomial)
            u = xi[:]
            jrand = random.randrange(dim)  # ensure at least one from v
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]

            fu = eval_f(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                F[i] = Fi
                CR[i] = CRi

                if fu < best:
                    best = fu
                    best_x = u[:]
            else:
                # keep old, but also keep previous F/CR (classic jDE)
                pass

        # occasional targeted refinement on best
        if best_x is not None and (gen % refine_every == 0) and time.time() < deadline:
            bx, bf = local_refine(best_x, best, budget_steps=18)
            if bf < best:
                best, best_x = bf, bx

        # mild restart pressure if stagnating: random re-injection
        # (keeps diversity with low overhead)
        if gen % 15 == 0 and time.time() < deadline:
            # replace a few worst with random points
            # find indices of a few worst
            k = max(1, NP // 12)
            worst = sorted(range(NP), key=lambda idx: fit[idx], reverse=True)[:k]
            for idx in worst:
                if time.time() >= deadline:
                    break
                pop[idx] = rand_vec()
                fit[idx] = eval_f(pop[idx])
                F[idx] = 0.5
                CR[idx] = 0.9
                if fit[idx] < best:
                    best = fit[idx]
                    best_x = pop[idx][:]

    return best
