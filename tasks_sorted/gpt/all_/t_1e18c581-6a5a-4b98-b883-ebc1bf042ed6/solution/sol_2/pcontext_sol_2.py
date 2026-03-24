import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved hybrid optimizer (no external libraries):
      - Sobol-like (Halton) low-discrepancy global sampling for strong early coverage
      - (1+λ) Evolution Strategy with mirrored sampling
      - Diagonal covariance adaptation (CMA-ES-inspired, lightweight)
      - Success-based global step-size control (1/5-ish rule)
      - Periodic local pattern-search polishing around the incumbent
      - Stagnation-triggered restarts + shrinking global-explore budget over time

    Returns:
        best (float): best objective value found within max_time.
    """
    start = time.time()
    deadline = start + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s > 0 else 1.0 for s in spans]

    # ---------- utilities ----------
    def clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def project(x):
        return [clamp(x[i], lows[i], highs[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # ---------- Halton sequence (low discrepancy) ----------
    # Good global exploration without numpy.
    def _nth_prime(n):
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
        return primes[-1]

    bases = [_nth_prime(i + 1) for i in range(dim)]
    halton_index = 1
    halton_shift = [random.random() for _ in range(dim)]  # Cranley-Patterson rotation

    def _radical_inverse(k, base):
        f = 1.0
        r = 0.0
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    def halton_vec():
        nonlocal halton_index
        k = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (_radical_inverse(k, bases[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- initialization: global sampling ----------
    # Spend a small fixed effort up front; then rely on the ES core.
    init_n = max(16, 8 * dim)
    best_x = halton_vec()
    best = evaluate(best_x)

    for _ in range(init_n - 1):
        if time.time() >= deadline:
            return best
        x = halton_vec() if random.random() < 0.85 else rand_vec()
        f = evaluate(x)
        if f < best:
            best, best_x = f, x

    # ---------- ES core state ----------
    mean = best_x[:]

    # global step size relative to box; start moderately small for stability
    sigma = 0.20

    # diagonal "covariance" / per-dimension scales (absolute units)
    diag = [0.25 * spans_safe[i] for i in range(dim)]
    min_diag = [1e-15 * spans_safe[i] for i in range(dim)]
    max_diag = [2.5 * spans_safe[i] for i in range(dim)]

    # population size; mirrored sampling doubles evaluations efficiency
    lam = max(10, 6 + int(4.0 * math.log(dim + 1.0)))
    if lam % 2 == 1:
        lam += 1
    mu = max(2, lam // 2)

    # recombination weights (log), normalized
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    # evolution path for diagonal adaptation
    path = [0.0] * dim
    c_path = 0.65  # path smoothing (higher -> more memory)

    # success-based step-size control
    succ_ema = 0.0
    succ_beta = 0.15  # smoothing
    target_succ = 0.20

    # restarts / stagnation
    no_improve = 0
    best_at_restart = best
    patience = max(80, 30 * dim)

    # local polish schedule
    polish_every = max(25, 6 * dim)
    it = 0

    # ---------- main loop ----------
    while time.time() < deadline:
        it += 1

        # Generate candidates with mirrored Gaussian steps.
        candidates = []  # (f, x, z)
        improvements_in_gen = 0

        half = lam // 2
        for _ in range(half):
            if time.time() >= deadline:
                break
            z = [random.gauss(0.0, 1.0) for _ in range(dim)]

            x1 = [mean[i] + sigma * diag[i] * z[i] for i in range(dim)]
            x2 = [mean[i] - sigma * diag[i] * z[i] for i in range(dim)]
            x1 = project(x1)
            x2 = project(x2)

            f1 = evaluate(x1)
            candidates.append((f1, x1, z))
            if f1 < best:
                best, best_x = f1, x1
                improvements_in_gen += 1

            if time.time() >= deadline:
                break

            f2 = evaluate(x2)
            candidates.append((f2, x2, [-zi for zi in z]))
            if f2 < best:
                best, best_x = f2, x2
                improvements_in_gen += 1

        if not candidates:
            break

        candidates.sort(key=lambda t: t[0])
        elites = candidates[:mu]

        # Update mean (recombination)
        new_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * elites[k][1][i]
            new_mean[i] = s
        mean = project(new_mean)

        # Update z-mean (normalized direction) and evolution path
        z_mean = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for k in range(mu):
                s += weights[k] * elites[k][2][i]
            z_mean[i] = s

        for i in range(dim):
            path[i] = c_path * path[i] + (1.0 - c_path) * z_mean[i]

        # Diagonal scale adaptation: expand along persistent directions, contract otherwise
        # Bounded, gentle to avoid instability across unknown objective landscapes.
        for i in range(dim):
            t = abs(path[i])
            # map t to multiplier; ~[0.97..1.06] typical
            mult = 1.0 + 0.10 * (t - 0.7) / (1.0 + t)
            if mult < 0.85:
                mult = 0.85
            elif mult > 1.20:
                mult = 1.20
            di = diag[i] * mult
            if di < min_diag[i]:
                di = min_diag[i]
            elif di > max_diag[i]:
                di = max_diag[i]
            diag[i] = di

        # Success-based sigma control using EMA of "any improvement happened"
        gen_success = 1.0 if improvements_in_gen > 0 else 0.0
        succ_ema = (1.0 - succ_beta) * succ_ema + succ_beta * gen_success

        # Adjust sigma: if success > target, expand slightly; else contract.
        if succ_ema > target_succ:
            sigma *= 1.03
        else:
            sigma *= 0.97

        # Keep sigma sane
        if sigma < 1e-15:
            sigma = 1e-15
        if sigma > 2.0:
            sigma = 2.0

        # Stagnation bookkeeping
        if best < best_at_restart:
            best_at_restart = best
            no_improve = 0
        else:
            no_improve += 1

        # Periodic local polish around best (coordinate pattern search with decreasing steps)
        if it % polish_every == 0 and time.time() < deadline:
            # Start with a small step derived from current exploration scale
            for j in range(dim):
                if time.time() >= deadline:
                    break
                step = (0.75 * sigma) * diag[j]
                if step <= 0.0:
                    continue

                # try a few step halvings for robust local improvement
                x0 = best_x[:]
                f0 = best
                improved = False
                for _ in range(3):
                    xp = x0[:]
                    xm = x0[:]
                    xp[j] = clamp(xp[j] + step, lows[j], highs[j])
                    xm[j] = clamp(xm[j] - step, lows[j], highs[j])

                    fp = evaluate(xp)
                    if fp < f0:
                        f0, x0 = fp, xp
                        improved = True

                    if time.time() >= deadline:
                        break

                    fm = evaluate(xm)
                    if fm < f0:
                        f0, x0 = fm, xm
                        improved = True

                    if not improved:
                        step *= 0.5
                    else:
                        # if improved, try same step again once (greedy)
                        improved = False

                    if time.time() >= deadline:
                        break

                if f0 < best:
                    best, best_x = f0, x0
                    mean = best_x[:]

        # Restart when stuck: mix global Halton point with incumbent region
        if no_improve >= patience and time.time() < deadline:
            no_improve = 0
            best_at_restart = best

            # Choose restart anchor
            if random.random() < 0.6:
                anchor = halton_vec()
            else:
                anchor = rand_vec()

            # Blend towards best with some probability (keeps good regions)
            if random.random() < 0.5:
                a = 0.3 + 0.4 * random.random()
                anchor = [clamp(a * best_x[i] + (1.0 - a) * anchor[i], lows[i], highs[i]) for i in range(dim)]

            f_anchor = evaluate(anchor)
            if f_anchor < best:
                best, best_x = f_anchor, anchor
                mean = best_x[:]
            else:
                mean = anchor[:]

            # Reset exploration scales (slightly smaller after many restarts is ok)
            sigma = 0.25
            diag = [0.30 * spans_safe[i] for i in range(dim)]
            path = [0.0] * dim
            succ_ema = 0.0

    return best
