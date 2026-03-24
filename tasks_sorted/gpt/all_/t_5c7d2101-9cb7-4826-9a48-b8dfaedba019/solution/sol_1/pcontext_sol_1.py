import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer.

    Hybrid strategy:
      1) Fast quasi-random (Halton) initialization (better coverage than pure random)
      2) Adaptive Differential Evolution, current-to-best/1 (strong exploitation)
      3) Occasional restarts + shrinking local search around the best (intensification)
      4) Budget-aware: checks time frequently and returns best fitness found

    Returns:
        best (float): fitness of the best found solution
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ------------------ helpers ------------------
    def now():
        return time.time()

    def clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def ensure_bounds(vec):
        out = vec[:]
        for i in range(dim):
            lo, hi = bounds[i]
            out[i] = clip(out[i], lo, hi)
        return out

    def eval_f(vec):
        return float(func(vec))

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ------------------ Halton sequence (low-discrepancy) ------------------
    # No external libs; simple prime generation + van der Corput radical inverse.
    def primes_upto_n(n):
        ps = []
        x = 2
        while len(ps) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(x)
            x += 1
        return ps

    def vdc(n, base):
        # van der Corput in [0,1)
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    primes = primes_upto_n(max(1, dim))

    halton_index = 1
    def halton_vec():
        nonlocal halton_index
        h = []
        idx = halton_index
        halton_index += 1
        for i in range(dim):
            u = vdc(idx, primes[i])
            lo, hi = bounds[i]
            h.append(lo + (hi - lo) * u)
        return h

    # ------------------ edge cases ------------------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # ------------------ algorithm parameters ------------------
    # Population size tuned for speed; DE benefits from >= 6..10*dim, but time-bounded.
    pop_size = max(12, min(60, 8 * dim))
    # Adaptive DE parameters (jDE style)
    F_min, F_max = 0.2, 0.9
    CR_min, CR_max = 0.05, 0.98
    tau1, tau2 = 0.1, 0.1  # probabilities to resample F/CR

    # Restart + local search schedule
    restart_every = max(3, int(40 / max(1, dim)))  # in "generations"
    local_tries_base = 8                           # number of local samples per gen
    shrink = 0.985                                 # local step decay

    # Precompute ranges for scaling local steps
    span = [(bounds[i][1] - bounds[i][0]) for i in range(dim)]

    # ------------------ initialize population ------------------
    pop = []
    fit = []
    Fs = []
    CRs = []

    best = float("inf")
    best_x = None

    # mix Halton and random for robustness
    for i in range(pop_size):
        if now() >= deadline:
            return best
        if i < (pop_size * 2) // 3:
            x = halton_vec()
        else:
            x = rand_uniform_vec()
        fx = eval_f(x)
        pop.append(x)
        fit.append(fx)
        Fs.append(random.uniform(F_min, F_max))
        CRs.append(random.uniform(CR_min, CR_max))
        if fx < best:
            best = fx
            best_x = x[:]

    # ------------------ main loop ------------------
    gen = 0
    local_scale = 0.15  # fraction of range for local perturbations (shrinks over time)

    while True:
        if now() >= deadline:
            return best

        # Identify current best index for "current-to-best/1"
        best_idx = 0
        best_fit = fit[0]
        for i in range(1, pop_size):
            if fit[i] < best_fit:
                best_fit = fit[i]
                best_idx = i
        xb = pop[best_idx]

        # One DE "generation"
        for i in range(pop_size):
            if now() >= deadline:
                return best

            # jDE: occasionally resample control parameters per individual
            Fi = Fs[i]
            CRi = CRs[i]
            if random.random() < tau1:
                Fi = random.uniform(F_min, F_max)
            if random.random() < tau2:
                CRi = random.uniform(CR_min, CR_max)

            # Choose r1, r2 distinct and different from i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)
            r2 = r1
            while r2 == i or r2 == r1:
                r2 = random.randrange(pop_size)

            xi = pop[i]
            xr1 = pop[r1]
            xr2 = pop[r2]

            # current-to-best/1 mutation:
            # v = xi + Fi*(xb - xi) + Fi*(xr1 - xr2)
            v = [0.0] * dim
            for j in range(dim):
                vj = xi[j] + Fi * (xb[j] - xi[j]) + Fi * (xr1[j] - xr2[j])
                lo, hi = bounds[j]
                # reflect/clip to keep in bounds; reflection often better than hard clip
                if vj < lo:
                    vj = lo + (lo - vj)
                    if vj > hi:
                        vj = lo
                elif vj > hi:
                    vj = hi - (vj - hi)
                    if vj < lo:
                        vj = hi
                v[j] = vj

            # Binomial crossover, ensure at least one dim from mutant
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    u[j] = v[j]
                else:
                    u[j] = xi[j]

            fu = eval_f(u)

            # Selection + record improved parameters
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                Fs[i] = Fi
                CRs[i] = CRi
                if fu < best:
                    best = fu
                    best_x = u[:]

        gen += 1

        # ------------------ local search around global best ------------------
        # Cheap intensification: sample gaussian-like steps with decaying scale.
        if best_x is not None:
            # number of tries grows mildly with dimension but remains small
            local_tries = local_tries_base + (dim // 5)
            for _ in range(local_tries):
                if now() >= deadline:
                    return best

                # "Gaussian-ish" step via sum of uniforms (CLT), no random.gauss needed
                x = best_x[:]
                for j in range(dim):
                    # approx N(0,1): sum of 6 uniforms - 3
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random()) - 3.0
                    step = g * (span[j] * local_scale)
                    x[j] = clip(x[j] + step, bounds[j][0], bounds[j][1])

                fx = eval_f(x)
                if fx < best:
                    best = fx
                    best_x = x[:]

            local_scale *= shrink
            if local_scale < 1e-6:
                local_scale = 1e-6

        # ------------------ partial restart to escape stagnation ------------------
        # Every few gens, replace worst fraction with new Halton/random points
        if gen % restart_every == 0:
            if now() >= deadline:
                return best

            # Find indices sorted by fitness descending (worst first) without heavy sorting
            # We'll select k worst by repeated scan (k small)
            k = max(2, pop_size // 5)
            worst = []
            used = set()
            for _ in range(k):
                wi = None
                wf = -float("inf")
                for idx in range(pop_size):
                    if idx in used:
                        continue
                    f = fit[idx]
                    if f > wf:
                        wf = f
                        wi = idx
                used.add(wi)
                worst.append(wi)

            # Re-seed these individuals around best (some) and globally (some)
            for cnt, idx in enumerate(worst):
                if now() >= deadline:
                    return best
                if best_x is not None and cnt < len(worst) // 2:
                    # around best with moderate radius
                    x = best_x[:]
                    rad = 0.35
                    for j in range(dim):
                        x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * span[j] * rad,
                                    bounds[j][0], bounds[j][1])
                else:
                    # global reseed (Halton + small random jitter)
                    x = halton_vec()
                    for j in range(dim):
                        x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * span[j] * 0.01,
                                    bounds[j][0], bounds[j][1])

                fx = eval_f(x)
                pop[idx] = x
                fit[idx] = fx
                Fs[idx] = random.uniform(F_min, F_max)
                CRs[idx] = random.uniform(CR_min, CR_max)
                if fx < best:
                    best = fx
                    best_x = x[:]
