import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization (no external libs) using:
      1) Low-discrepancy (Halton) + random init sampling
      2) Elitist (mu,lambda)-ES with self-adapting step-size (sigma)
      3) Periodic coordinate-wise pattern search around the current best
      4) Lightweight restarts when progress stalls

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    # ---------- bounds / helpers ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0  # treat as fixed in practice; still safe for scaling

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def clip_vec(x):
        return [clamp(x[i], i) for i in range(dim)]

    def eval_x(x):
        return float(func(x))

    # ---------- Halton sequence (deterministic, cheap, no numpy) ----------
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

    primes = first_primes(dim)

    def van_der_corput(index, base):
        # index >= 1
        vdc = 0.0
        denom = 1.0
        n = index
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(k):
        # k starts at 1
        x = []
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x.append(lows[i] + u * (highs[i] - lows[i]))
        return x

    # ---------- initial sampling ----------
    best = float("inf")
    best_x = None

    # number of init points: scale with dim but remain modest
    n_init = max(24, min(250, 20 * dim))

    # Mix Halton + random (helps when objective aligns poorly with Halton)
    for k in range(1, n_init + 1):
        if time.time() >= deadline - eps_time:
            return best
        if k % 3 == 0:
            x = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]
        else:
            x = halton_point(k)
        fx = eval_x(x)
        if fx < best:
            best = fx
            best_x = x[:]

    if best_x is None:
        return best

    # ---------- ES parameters ----------
    # population sizes tuned for speed; adjust with dimension
    mu = max(4, min(20, 2 + dim // 2))
    lam = max(12, min(60, 6 + 3 * dim))
    elite_keep = max(2, mu // 2)

    # global step-size sigma scaled to bounds
    avg_span = sum(spans) / float(dim)
    sigma = 0.25 * avg_span
    sigma_min = 1e-12 * avg_span + 1e-18
    sigma_max = 2.0 * avg_span

    # evolution path-ish success control (simple 1/5 rule variant)
    success = 0
    trials = 0

    # Keep a small elite set
    elites = [(best, best_x[:])]

    # ---------- local pattern search around best ----------
    def pattern_search(x0, f0, base_scale):
        # coordinate search with decreasing step; very few evaluations
        x = x0[:]
        fx = f0
        step = [base_scale * s for s in spans]
        # limit inner work to keep within budget
        for _round in range(2):
            improved_any = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                if si <= 0:
                    continue
                # try +, then -
                xi = x[i]
                cand = x[:]
                cand[i] = clamp(xi + si, i)
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved_any = True
                    continue
                cand = x[:]
                cand[i] = clamp(xi - si, i)
                fc = eval_x(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved_any = True
                    continue
            # shrink steps
            step = [v * (0.5 if improved_any else 0.35) for v in step]
            if not improved_any:
                break
        return fx, x

    # ---------- main loop ----------
    stall = 0
    restart_after = max(80, 15 * dim)

    while time.time() < deadline - eps_time:
        # Build parent pool from elites + jittered best
        elites.sort(key=lambda t: t[0])
        elites = elites[:max(mu, elite_keep)]
        parents = [x for (_, x) in elites]
        if best_x not in parents:
            parents.append(best_x[:])

        # Generate offspring
        offspring = []
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                break
            p = parents[random.randrange(len(parents))]
            # mutate with Gaussian-like noise using sum of uniforms (no random.gauss dependency)
            # delta ~ approx N(0,1) by CLT
            child = p[:]
            for i in range(dim):
                # 12 uniforms - 6 ~ approx standard normal
                z = 0.0
                for _k in range(12):
                    z += random.random()
                z -= 6.0
                child[i] = clamp(child[i] + z * sigma, i)
            fchild = eval_x(child)
            offspring.append((fchild, child))

            trials += 1
            if fchild < best:
                best = fchild
                best_x = child[:]
                success += 1
                stall = 0
            else:
                stall += 1

        if not offspring:
            break

        # Select next elites from offspring + old elites
        pool = elites + offspring
        pool.sort(key=lambda t: t[0])
        elites = pool[:mu]

        # Update best from elites (safety)
        if elites[0][0] < best:
            best, best_x = elites[0][0], elites[0][1][:]
            stall = 0

        # Step-size adaptation (1/5-ish): adjust every so often
        if trials >= max(20, 3 * lam):
            rate = success / float(trials)
            # if too many successes -> increase sigma, else decrease
            if rate > 0.25:
                sigma = min(sigma_max, sigma * 1.25)
            elif rate < 0.18:
                sigma = max(sigma_min, sigma * 0.75)
            else:
                sigma = max(sigma_min, min(sigma_max, sigma * 0.98))
            success = 0
            trials = 0

        # Occasionally do a deterministic-ish local improvement around best
        if time.time() < deadline - eps_time and (stall % max(25, 4 * dim) == 0):
            f2, x2 = pattern_search(best_x, best, base_scale=0.08)
            if f2 < best:
                best, best_x = f2, x2[:]
                elites.append((best, best_x[:]))
                elites.sort(key=lambda t: t[0])
                elites = elites[:mu]
                stall = 0

        # Restart if very stagnant: re-seed around best + a few Halton points
        if stall >= restart_after:
            stall = 0
            sigma = max(sigma_min, min(sigma_max, 0.18 * avg_span))

            # keep current best, add a few diversified points
            new_elites = [(best, best_x[:])]
            for j in range(max(4, mu - 1)):
                if time.time() >= deadline - eps_time:
                    break
                if j % 2 == 0:
                    x = halton_point(random.randint(1, 3 * n_init + 50))
                else:
                    x = [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]
                # small bias towards best (blend)
                if random.random() < 0.6:
                    a = 0.25 + 0.5 * random.random()
                    x = [clamp(a * best_x[i] + (1.0 - a) * x[i], i) for i in range(dim)]
                fx = eval_x(x)
                if fx < best:
                    best, best_x = fx, x[:]
                new_elites.append((fx, x))
            new_elites.sort(key=lambda t: t[0])
            elites = new_elites[:mu]

    return best
