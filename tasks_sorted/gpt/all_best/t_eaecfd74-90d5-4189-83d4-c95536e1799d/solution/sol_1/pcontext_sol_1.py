import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization:
      - Differential Evolution "current-to-best/1" (faster convergence than rand/1)
      - jDE-style self-adaptation of F and CR per individual
      - Opposition-based initialization (often better start than purely random)
      - Lightweight coordinate/local search around the current best (exploit)
      - Stagnation-triggered partial restart to escape traps (explore)

    Returns:
        best (float): best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low<=high).")

    # --- helpers ---
    def now():
        return time.time()

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clip_val(x, i):
        lo = lows[i]
        hi = highs[i]
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def ensure_bounds(x):
        for i in range(dim):
            x[i] = clip_val(x[i], i)
        return x

    def opp_vec(x):
        # Opposition point across the center of the bounds box: x' = lo+hi-x
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # Evaluate with safety against exceptions / non-finite
    def eval_f(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # --- population size (keep moderate but scale with dim) ---
    pop_size = max(12, min(60, 10 + 6 * dim))

    # --- jDE parameters ---
    tau1 = 0.1  # prob to resample F
    tau2 = 0.1  # prob to resample CR
    Fl, Fu = 0.1, 0.9

    # --- initialization: opposition-based ---
    pop = []
    fit = []
    F_i = []
    CR_i = []

    best = float("inf")
    best_x = None

    # create candidates in pairs (x and opposite) and keep the better
    for _ in range(pop_size):
        if now() >= deadline:
            return best if best < float("inf") else best

        x = rand_vec()
        xo = opp_vec(x)

        fx = eval_f(x)
        fo = eval_f(xo)

        if fo < fx:
            x, fx = xo, fo

        pop.append(x)
        fit.append(fx)

        # init individual control params
        F_i.append(0.5 + 0.3 * (random.random() - 0.5))   # around 0.5
        CR_i.append(0.9 + 0.2 * (random.random() - 0.5))  # around 0.9
        if F_i[-1] < Fl: F_i[-1] = Fl
        if F_i[-1] > Fu: F_i[-1] = Fu
        if CR_i[-1] < 0.0: CR_i[-1] = 0.0
        if CR_i[-1] > 1.0: CR_i[-1] = 1.0

        if fx < best:
            best = fx
            best_x = x[:]

    # --- local search around best (coordinate + small gaussian) ---
    # Designed to be very cheap and time-aware.
    def local_refine(best_x, best_f, budget_evals):
        if best_x is None:
            return best_x, best_f
        x = best_x[:]
        f = best_f

        # initial step sizes relative to spans
        steps = [0.15 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]
        min_steps = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

        evals = 0
        while evals < budget_evals and now() < deadline:
            improved = False

            # coordinate perturbations
            for i in range(dim):
                if evals >= budget_evals or now() >= deadline:
                    break
                if spans[i] == 0:
                    continue

                s = steps[i]
                if s < min_steps[i]:
                    continue

                # try +s
                xp = x[:]
                xp[i] = clip_val(xp[i] + s, i)
                fp = eval_f(xp); evals += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if evals >= budget_evals or now() >= deadline:
                    break

                # try -s
                xm = x[:]
                xm[i] = clip_val(xm[i] - s, i)
                fm = eval_f(xm); evals += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

            # a couple of small gaussian kicks around current point (helps in narrow valleys)
            if evals < budget_evals and now() < deadline:
                kicks = 2
                for _ in range(kicks):
                    if evals >= budget_evals or now() >= deadline:
                        break
                    xt = x[:]
                    for i in range(dim):
                        if spans[i] == 0:
                            continue
                        # gaussian step scaled by steps[i]
                        xt[i] = clip_val(xt[i] + random.gauss(0.0, 0.35) * steps[i], i)
                    ft = eval_f(xt); evals += 1
                    if ft < f:
                        x, f = xt, ft
                        improved = True

            if not improved:
                # reduce steps
                for i in range(dim):
                    steps[i] *= 0.5
        return x, f

    # --- main DE loop ---
    stagn = 0
    last_best = best

    while now() < deadline:
        # occasional quick refine of current best (more as we get closer to end)
        remaining = deadline - now()
        if remaining > 0:
            # small, adaptive budget: a few evals; more near the end to exploit
            budget = 2 * dim if remaining > 0.3 * max_time else 4 * dim
            bx, bf = local_refine(best_x, best, budget_evals=budget)
            if bf < best:
                best, best_x = bf, bx

        for i in range(pop_size):
            if now() >= deadline:
                return best

            # jDE self-adaptation
            Fi = F_i[i]
            CRi = CR_i[i]
            if random.random() < tau1:
                Fi = Fl + random.random() * (Fu - Fl)
            if random.random() < tau2:
                CRi = random.random()

            # select r1, r2 distinct and != i
            # (and we'll use current-to-best/1 with best_x)
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)

            xi = pop[i]
            x1 = pop[r1]
            x2 = pop[r2]

            # mutation: current-to-best/1
            # v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (best_x[d] - xi[d]) + Fi * (x1[d] - x2[d])
                v[d] = clip_val(v[d], d)

            # binomial crossover
            u = [0.0] * dim
            jrand = random.randrange(dim) if dim > 0 else 0
            for d in range(dim):
                if d == jrand or random.random() < CRi:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]

            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                F_i[i] = Fi
                CR_i[i] = CRi

                if fu < best:
                    best = fu
                    best_x = u[:]

        # stagnation / restart logic
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            stagn = 0
            last_best = best
        else:
            stagn += 1

        # partial restart if stagnating and time remains
        if stagn >= 15 and (deadline - now()) > 0.15 * max_time:
            stagn = 0
            # reinitialize worst half around best (biased) + some random
            # rank indices by fitness descending
            idx = list(range(pop_size))
            idx.sort(key=lambda k: fit[k], reverse=True)
            worst = idx[: pop_size // 2]

            for k in worst:
                if now() >= deadline:
                    break
                if random.random() < 0.7 and best_x is not None:
                    # sample around best with decreasing radius
                    xnew = best_x[:]
                    for d in range(dim):
                        if spans[d] == 0:
                            continue
                        # radius ~ 20% span
                        xnew[d] = clip_val(xnew[d] + random.gauss(0.0, 0.2) * spans[d], d)
                else:
                    xnew = rand_vec()

                fnew = eval_f(xnew)
                pop[k] = xnew
                fit[k] = fnew
                F_i[k] = Fl + random.random() * (Fu - Fl)
                CR_i[k] = random.random()
                if fnew < best:
                    best = fnew
                    best_x = xnew[:]

    return best
