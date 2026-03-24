import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no external libs).

    Main upgrades vs. the provided solutions:
      - Stronger global exploration: Halton + opposition points + light crossover
      - Maintains an elite pool + short tabu of visited (coarse) cells
      - Local search is upgraded to an adaptive (1+1)-ES style Gaussian search
        with 1/5 success rule + occasional coordinate probes (cheap)
      - Restarts are driven by stagnation + diversity, with time-slicing

    Returns:
        best (float): best objective value found within time limit
    """

    # ------------------------- basic utilities ------------------------- #
    if dim <= 0:
        return float("inf")
    if len(bounds) != dim:
        raise ValueError("bounds length must match dim")

    def sanitize_bounds(b):
        out = []
        for lo, hi in b:
            lo = float(lo); hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    bnds = sanitize_bounds(bounds)
    rngs = [bnds[i][1] - bnds[i][0] for i in range(dim)]

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def safe_eval(x):
        xx = [clamp(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    # ------------------------- Halton sequence ------------------------- #
    def first_primes(k):
        ps = []
        n = 2
        while len(ps) < k:
            is_p = True
            r = int(n ** 0.5)
            for p in ps:
                if p > r: break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(n)
            n += 1
        return ps

    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_u(index, bases):
        return [vdc(index, b) for b in bases]

    def u_to_x(u):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi == lo:
                x.append(lo)
            else:
                x.append(lo + u[i] * (hi - lo))
        return x

    def rand_x():
        return [random.uniform(bnds[i][0], bnds[i][1]) for i in range(dim)]

    # ------------------------- novelty / tabu keying ------------------------- #
    # Coarse cell hashing in normalized space; discourages resampling the same area too much.
    # Resolution scales mildly with dim to remain stable.
    cell_res = 900 if dim <= 5 else 600 if dim <= 15 else 420

    def cell_key(x):
        k = []
        for i in range(dim):
            lo, hi = bnds[i]
            r = hi - lo
            if r <= 0:
                k.append(0)
            else:
                u = (x[i] - lo) / r
                if u < 0.0: u = 0.0
                elif u > 1.0: u = 1.0
                k.append(int(u * cell_res + 0.5))
        return tuple(k)

    # ------------------------- elite pool ------------------------- #
    elite_k = max(8, min(26, 6 + 2 * dim))
    elite = []  # list of (f, x)
    elite_keys = set()

    def elite_add(f, x):
        nonlocal elite, elite_keys
        k = cell_key(x)
        # If already present and not a meaningful improvement, skip
        if k in elite_keys and (elite and f >= elite[-1][0]):
            return
        elite.append((f, x))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_k:
            elite = elite[:elite_k]
        elite_keys = set(cell_key(xx) for _, xx in elite)

    # ------------------------- local search: (1+1)-ES + coord probes ------------------------- #
    def local_es(x0, f0, t_end):
        x = x0[:]
        f = f0

        # Initial sigma proportional to domain; per-dimension scale factors
        base = 0.22
        sigma = [base * r for r in rngs]
        # handle degenerate dimensions
        for i in range(dim):
            if rngs[i] <= 0:
                sigma[i] = 0.0

        # Minimum sigma (avoid wasting evaluations with no movement)
        min_sigma = [0.0 if rngs[i] <= 0 else max(1e-12, 2e-8 * rngs[i]) for i in range(dim)]
        max_sigma = [0.0 if rngs[i] <= 0 else 0.65 * rngs[i] for i in range(dim)]

        # 1/5 success rule parameters
        succ = 0
        trials = 0

        # Keep it short, time is the real limiter
        eval_budget = 140 + 20 * dim
        evals = 0

        # A tiny set of coordinate indices to probe occasionally
        coord_probe_count = min(dim, 6)

        while evals < eval_budget and time.time() < t_end:
            # --- ES mutation ---
            cand = x[:]
            moved = False
            for i in range(dim):
                si = sigma[i]
                if si > 0.0:
                    # Gaussian step
                    cand[i] = clamp(cand[i] + random.gauss(0.0, si), bnds[i][0], bnds[i][1])
                    moved = True
            if not moved:
                break

            fc, cand = safe_eval(cand)
            evals += 1
            trials += 1

            if fc < f:
                x, f = cand, fc
                succ += 1
            else:
                # occasional cheap coordinate probe to escape shallow ridges
                # (try a few coords with +/- small steps based on sigma)
                if coord_probe_count > 0 and time.time() < t_end and evals < eval_budget:
                    # choose coordinates biased toward larger sigmas
                    idxs = list(range(dim))
                    idxs.sort(key=lambda j: sigma[j], reverse=True)
                    idxs = idxs[:coord_probe_count]
                    random.shuffle(idxs)
                    for j in idxs:
                        if time.time() >= t_end or evals >= eval_budget:
                            break
                        if sigma[j] <= 0.0:
                            continue
                        step = 0.85 * sigma[j]
                        if step <= 0.0:
                            continue
                        lo, hi = bnds[j]
                        # + step
                        xp = x[:]
                        xp[j] = clamp(xp[j] + step, lo, hi)
                        fp, xp = safe_eval(xp); evals += 1
                        if fp < f:
                            x, f = xp, fp
                            succ += 1
                            trials += 1
                            break
                        # - step
                        if time.time() >= t_end or evals >= eval_budget:
                            break
                        xm = x[:]
                        xm[j] = clamp(xm[j] - step, lo, hi)
                        fm, xm = safe_eval(xm); evals += 1
                        trials += 1
                        if fm < f:
                            x, f = xm, fm
                            succ += 1
                            break

            # --- adapt sigma via 1/5 rule every small window ---
            if trials >= 12:
                rate = succ / float(trials)
                # If too successful -> increase, else decrease
                if rate > 0.22:
                    mult = 1.22
                else:
                    mult = 0.80
                for i in range(dim):
                    if sigma[i] > 0.0:
                        sigma[i] = min(max_sigma[i], max(min_sigma[i], sigma[i] * mult))
                succ = 0
                trials = 0

            # early exit if sigmas are all tiny
            tiny = True
            for i in range(dim):
                if sigma[i] > min_sigma[i] * 1.01:
                    tiny = False
                    break
            if tiny:
                break

        return x, f

    # ------------------------- global operators (explore) ------------------------- #
    def opposition_x(x):
        # Mirror around center of bounds per dimension
        xo = []
        for i in range(dim):
            lo, hi = bnds[i]
            xo.append(lo + hi - x[i])
        return xo

    def crossover(a, b):
        # Blend/crossover in normalized space, then map back
        child = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi == lo:
                child.append(lo)
                continue
            t = random.random()
            # slightly biased toward best parent coordinate, but still mixes
            if random.random() < 0.55:
                t = t * t
            v = t * a[i] + (1.0 - t) * b[i]
            child.append(clamp(v, lo, hi))
        return child

    # ------------------------- main loop scheduling ------------------------- #
    start = time.time()
    deadline = start + float(max_time)

    bases = first_primes(dim)
    best = float("inf")
    best_x = None

    # tabu set limited size
    tabu = set()
    tabu_cap = 1200 if dim <= 20 else 2200

    def tabu_add(x):
        nonlocal tabu
        tabu.add(cell_key(x))
        if len(tabu) > tabu_cap:
            # random pruning (cheap)
            for _ in range(len(tabu) - tabu_cap):
                tabu.pop()

    # Phase A: fast global seeding with Halton + opposition
    idx = 1
    seed_cap = 70 + 22 * dim
    while idx <= seed_cap and time.time() < deadline:
        u = halton_u(idx, bases)
        x = u_to_x(u)
        f, x = safe_eval(x)
        if f < best:
            best, best_x = f, x
        elite_add(f, x)
        tabu_add(x)

        # opposition point often helps when optimum is away from early LDS points
        xo = opposition_x(x)
        fo, xo = safe_eval(xo)
        if fo < best:
            best, best_x = fo, xo
        elite_add(fo, xo)
        tabu_add(xo)

        idx += 1

    # Phase B: iterate until time runs out
    no_best_improve = 0
    last_best = best
    it = 0

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        it += 1

        # --- pick a seed (mix exploit/explore) ---
        r = random.random()
        x0 = None

        if elite and r < 0.60:
            # pick among best with quadratic bias
            j = int((random.random() ** 2.3) * len(elite))
            x0 = elite[j][1][:]
            # jitter depends on stagnation
            jit = 0.010 + 0.045 * min(1.0, no_best_improve / 10.0)
            for i in range(dim):
                if rngs[i] > 0:
                    x0[i] = clamp(x0[i] + random.uniform(-jit, jit) * rngs[i], bnds[i][0], bnds[i][1])

        elif elite and r < 0.82:
            # crossover of two elites + occasional opposition
            a = elite[int((random.random() ** 1.8) * len(elite))][1]
            b = elite[random.randrange(len(elite))][1]
            x0 = crossover(a, b)
            if random.random() < 0.25:
                x0 = opposition_x(x0)

        elif r < 0.92:
            # new Halton
            u = halton_u(idx, bases)
            idx += 1
            x0 = u_to_x(u)
        else:
            # random
            x0 = rand_x()

        # Avoid spending too many evals in the same coarse cell: if repeated, force exploration
        ck = cell_key(x0)
        if ck in tabu and random.random() < 0.75:
            if random.random() < 0.60:
                u = halton_u(idx, bases); idx += 1
                x0 = u_to_x(u)
            else:
                x0 = rand_x()

        f0, x0 = safe_eval(x0)
        tabu_add(x0)
        if f0 < best:
            best, best_x = f0, x0
        elite_add(f0, x0)

        # --- local search time slice ---
        # short slices, more when stagnating; never hog the whole budget
        stagn = min(1.0, no_best_improve / 12.0)
        slice_sec = (0.08 + 0.035 * dim) * (1.0 + 0.9 * stagn)
        slice_sec = min(slice_sec, 0.35 * remaining)
        slice_sec = max(0.01, slice_sec)
        t_end = min(deadline, time.time() + slice_sec)

        x1, f1 = local_es(x0, f0, t_end)
        tabu_add(x1)
        if f1 < best:
            best, best_x = f1, x1
        elite_add(f1, x1)

        # --- occasional "big jump" around best to escape basins ---
        if (it % 7 == 0) and best_x is not None and time.time() < deadline:
            jump = 0.10 + 0.22 * random.random()  # 10%..32% range
            xj = best_x[:]
            for i in range(dim):
                if rngs[i] > 0:
                    xj[i] = clamp(xj[i] + random.uniform(-jump, jump) * rngs[i], bnds[i][0], bnds[i][1])
            fj, xj = safe_eval(xj)
            tabu_add(xj)
            if fj < best:
                best, best_x = fj, xj
            elite_add(fj, xj)

        # --- stagnation bookkeeping ---
        if best < last_best - 1e-15:
            last_best = best
            no_best_improve = 0
        else:
            no_best_improve += 1

        # if heavily stagnating, inject extra exploration samples (cheap)
        if no_best_improve >= 11 and time.time() < deadline:
            for _ in range(3):
                if time.time() >= deadline:
                    break
                if random.random() < 0.55:
                    u = halton_u(idx, bases); idx += 1
                    xe = u_to_x(u)
                else:
                    xe = rand_x()
                fe, xe = safe_eval(xe)
                tabu_add(xe)
                if fe < best:
                    best, best_x = fe, xe
                elite_add(fe, xe)
            no_best_improve = 7  # reduce, not reset

    return best
