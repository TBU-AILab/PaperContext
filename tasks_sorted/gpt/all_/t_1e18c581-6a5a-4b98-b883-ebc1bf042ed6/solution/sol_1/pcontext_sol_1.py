import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved optimizer: hybrid of
      - low-discrepancy-ish global sampling (stratified / quasi-random per-dim)
      - CMA-ES-like diagonal adaptation (no numpy)
      - (1+λ) evolution strategy with mirrored sampling
      - occasional coordinate pattern search polish
      - adaptive restarts based on stagnation

    Returns: best (float) best objective value found within max_time seconds.
    """
    start = time.time()
    deadline = start + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans_safe = [s if s != 0 else 1.0 for s in spans]

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def project(x):
        return [clamp(x[i], lows[i], highs[i]) for i in range(dim)]

    def evaluate(x):
        return float(func(x))

    # --- quasi-random / stratified sampler (cheap, deterministic-ish sequence) ---
    # Use per-dimension irrational increments (golden ratio variants) to reduce clustering.
    # No external libs.
    phi = 0.6180339887498949
    inc = [(phi + (i + 1) * 0.1234567) % 1.0 for i in range(dim)]
    seq_state = [random.random() for _ in range(dim)]
    seq_k = 0

    def quasi_vec():
        nonlocal seq_k
        seq_k += 1
        x = [0.0] * dim
        # Cranley-Patterson rotation + wrap
        for i in range(dim):
            seq_state[i] = (seq_state[i] + inc[i]) % 1.0
            u = (seq_state[i] + 0.37 * ((seq_k * (i + 1)) % 17) / 17.0) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --- initialization: sample a small population, keep best ---
    # Budget some early global exploration but keep it time-aware.
    init_n = max(10, 6 * dim)
    best_x = quasi_vec()
    best = evaluate(best_x)

    for _ in range(init_n - 1):
        if time.time() >= deadline:
            return best
        x = quasi_vec() if random.random() < 0.7 else rand_vec()
        f = evaluate(x)
        if f < best:
            best, best_x = f, x

    # --- Diagonal "CMA-ES-like" parameters ---
    # Mean starts at best_x, sigma relative to span, per-dimension scales adapt.
    mean = best_x[:]
    sigma = 0.25  # global step multiplier (relative)
    # diagonal std devs in absolute coordinate units
    diag = [0.3 * spans_safe[i] for i in range(dim)]
    min_diag = [1e-12 * spans_safe[i] for i in range(dim)]
    max_diag = [2.0 * spans_safe[i] for i in range(dim)]

    # population size (small for speed); mirrored sampling improves efficiency
    lam = max(8, 4 + int(3.0 * math.log(dim + 1.0)))
    mu = max(2, lam // 2)

    # recombination weights (log)
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # evolution paths (diagonal)
    path = [0.0] * dim
    path_decay = 0.8  # smoother path => more stable adaptation

    # stagnation / restart
    no_improve = 0
    best_seen_at_restart = best
    restart_patience = max(60, 25 * dim)

    # coordinate polish frequency
    polish_every = max(20, 5 * dim)

    # --- main loop ---
    it = 0
    while time.time() < deadline:
        it += 1

        # Generate candidates around mean using diagonal Gaussian.
        # Use mirrored pairs to reduce sampling noise.
        candidates = []
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
            if f1 < best:
                best, best_x = f1, x1
                no_improve = 0
            candidates.append((f1, x1, z))
            if time.time() >= deadline:
                break
            f2 = evaluate(x2)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0
            candidates.append((f2, x2, [-zi for zi in z]))

        # If we couldn't evaluate anything, exit.
        if not candidates:
            break

        # Sort and select top mu
        candidates.sort(key=lambda t: t[0])
        elites = candidates[:mu]

        # Update mean using elites
        new_mean = mean[:]
        for i in range(dim):
            acc = 0.0
            for k in range(mu):
                acc += weights[k] * elites[k][1][i]
            new_mean[i] = acc

        # Update evolution path using weighted z (direction in normalized space)
        # path <- decay*path + (1-decay)*z_mean
        z_mean = [0.0] * dim
        for i in range(dim):
            acc = 0.0
            for k in range(mu):
                acc += weights[k] * elites[k][2][i]
            z_mean[i] = acc

        for i in range(dim):
            path[i] = path_decay * path[i] + (1.0 - path_decay) * z_mean[i]

        # Adapt diagonal scales: increase along persistent directions, decrease otherwise
        # A gentle update to remain robust.
        for i in range(dim):
            # target multiplier based on |path|
            t = abs(path[i])
            # map t in [0, ~] to scale factor in [~0.97, ~1.03]
            mult = 1.0 + 0.06 * (t - 1.0) / (1.0 + t)  # bounded
            diag[i] *= mult
            if diag[i] < min_diag[i]:
                diag[i] = min_diag[i]
            elif diag[i] > max_diag[i]:
                diag[i] = max_diag[i]

        # Adapt global sigma based on success of elites vs median-ish
        # If the best improved recently, allow slightly larger sigma; else shrink.
        if candidates[0][0] <= best:
            sigma *= 1.01
        else:
            sigma *= 0.995

        # Keep sigma in sane range
        if sigma < 1e-12:
            sigma = 1e-12
        if sigma > 2.0:
            sigma = 2.0

        mean = project(new_mean)

        # Track stagnation
        if best < best_seen_at_restart:
            best_seen_at_restart = best
            no_improve = 0
        else:
            no_improve += 1

        # Occasional coordinate pattern search polish around best_x
        if it % polish_every == 0 and time.time() < deadline:
            # small step based on current diag*sigma
            for j in range(dim):
                if time.time() >= deadline:
                    break
                step = sigma * diag[j]
                if step <= 0:
                    continue
                x = best_x[:]
                x[j] = clamp(x[j] + step, lows[j], highs[j])
                fx = evaluate(x)
                if fx < best:
                    best, best_x = fx, x
                    mean = best_x[:]
                    continue
                x = best_x[:]
                x[j] = clamp(x[j] - step, lows[j], highs[j])
                fx = evaluate(x)
                if fx < best:
                    best, best_x = fx, x
                    mean = best_x[:]

        # Restart if stuck: new mean from quasi-random, reset scales moderately
        if no_improve >= restart_patience and time.time() < deadline:
            no_improve = 0
            best_seen_at_restart = best

            # choose restart point: sometimes best-centered jitter, sometimes global
            if random.random() < 0.5:
                mean = quasi_vec()
            else:
                mean = best_x[:]
                # jitter mean a bit
                for i in range(dim):
                    mean[i] = clamp(mean[i] + random.gauss(0.0, 0.15 * spans_safe[i]), lows[i], highs[i])

            # reset step sizes
            sigma = 0.3
            diag = [0.35 * spans_safe[i] for i in range(dim)]
            path = [0.0] * dim

            # also try a couple of global samples immediately
            for _ in range(3):
                if time.time() >= deadline:
                    break
                x = quasi_vec()
                f = evaluate(x)
                if f < best:
                    best, best_x = f, x
                    mean = best_x[:]

    return best
