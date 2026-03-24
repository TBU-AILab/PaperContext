import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs):
      - Sobol-like quasi-random init replaced with fast LCG + Cranley-Patterson rotation
      - Multi-start: keep a small elite set of best points
      - Local refinement per elite using:
          * coordinate search + quadratic interpolation (1D parabolic fit)
          * occasional random direction pattern move
      - Adaptive trust radii and restarts

    Returns:
        best (float): best (minimum) function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time
    if max_time <= 0 or dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # Handle degenerate spans
    for i in range(dim):
        if spans[i] < 0:
            lo, hi = highs[i], lows[i]
            lows[i], highs[i] = lo, hi
            spans[i] = highs[i] - lows[i]

    def clamp(v, i):
        lo, hi = lows[i], highs[i]
        if v < lo: return lo
        if v > hi: return hi
        return v

    def eval_f(x):
        return float(func(list(x)))

    # --- Fast quasi-random generator (deterministic-ish but seeded) ---
    # LCG for speed; Cranley-Patterson rotation for each dimension to reduce lattice artifacts.
    rng_state = random.getrandbits(64) ^ (int(time.time() * 1e9) & ((1 << 64) - 1))
    def lcg_u01():
        nonlocal rng_state
        rng_state = (6364136223846793005 * rng_state + 1442695040888963407) & ((1 << 64) - 1)
        # Use top 53 bits as double-like mantissa
        return ((rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    rot = [lcg_u01() for _ in range(dim)]

    def qrand_point(k):
        # Van der Corput per dimension with different bases (small primes),
        # then rotate. This is cheap and more space-filling than pure random.
        # Precomputed primes for up to 64 dims; if dim larger, fallback to LCG.
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                  59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                  191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
                  257, 263, 269, 271, 277, 281, 283, 293, 307, 311]

        x = [0.0] * dim
        for i in range(dim):
            if spans[i] == 0.0:
                x[i] = lows[i]
                continue

            if i < len(primes):
                base = primes[i]
                n = k + 1
                f = 1.0 / base
                r = 0.0
                while n > 0:
                    r += (n % base) * f
                    n //= base
                    f /= base
                u = r
            else:
                u = lcg_u01()

            u = (u + rot[i]) % 1.0
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_point():
        return [lows[i] + lcg_u01() * spans[i] if spans[i] != 0.0 else lows[i] for i in range(dim)]

    # --- Elite set (small) ---
    elite_size = max(3, min(12, 2 + int(math.sqrt(dim))))
    elites = []  # list of (f, x)

    def add_elite(fx, x):
        nonlocal elites
        # insert keeping sorted
        inserted = False
        for j in range(len(elites)):
            if fx < elites[j][0]:
                elites.insert(j, (fx, x))
                inserted = True
                break
        if not inserted:
            elites.append((fx, x))
        if len(elites) > elite_size:
            elites.pop()

    best = float("inf")
    best_x = None

    # --- Initial exploration budget based on time ---
    # Try to spend ~25-40% time on exploration; but keep count bounded.
    # We'll just run until a small time fraction elapses.
    explore_end = t0 + max_time * 0.35

    k = 0
    while time.time() < min(explore_end, deadline):
        # Mix quasi-random and random (helps on weird landscapes)
        if (k & 3) != 0:
            x = qrand_point(k)
        else:
            x = rand_point()
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        add_elite(fx, x)
        k += 1

    if best_x is None:
        x = rand_point()
        best_x = x
        best = eval_f(x)
        add_elite(best, best_x)

    # --- Local refinement tools ---
    def parabolic_min_1d(x0, i, h):
        """
        1D parabolic interpolation along coordinate i around x0 using points:
            x0-h, x0, x0+h  (clamped)
        Returns (x_new, f_new) possibly improved; otherwise (None, None).
        """
        xi = x0[i]
        if h <= 0.0:
            return None, None

        xL = list(x0); xR = list(x0)
        xL[i] = clamp(xi - h, i)
        xR[i] = clamp(xi + h, i)

        # If clamping collapsed points, skip
        if xL[i] == xi and xR[i] == xi:
            return None, None

        f0 = eval_f(x0)
        fL = eval_f(xL)
        fR = eval_f(xR)

        # Fit parabola through (-h,fL), (0,f0), (+h,fR) assuming symmetric spacing h.
        # Vertex at t* = h*(fL - fR) / (2*(fL - 2f0 + fR))
        denom = (fL - 2.0 * f0 + fR)
        if denom == 0.0:
            return None, None

        t = 0.5 * h * (fL - fR) / denom
        # Limit to a reasonable range to avoid wild steps
        t = max(-2.0 * h, min(2.0 * h, t))

        xN = list(x0)
        xN[i] = clamp(xi + t, i)
        if xN[i] == xi:
            return None, None
        fN = eval_f(xN)
        if fN < f0:
            return xN, fN
        return None, None

    def try_pattern_move(x0, fx0, radii):
        """
        Random direction move using normalized direction and adaptive step.
        """
        # Build a random direction
        d = [lcg_u01() * 2.0 - 1.0 for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in d))
        if norm == 0.0:
            return None, None
        inv = 1.0 / norm

        x1 = list(x0)
        for i in range(dim):
            if spans[i] == 0.0:
                continue
            step = radii[i] * d[i] * inv
            x1[i] = clamp(x1[i] + step, i)

        f1 = eval_f(x1)
        if f1 < fx0:
            # Try an "expand" step in same direction
            x2 = list(x0)
            for i in range(dim):
                if spans[i] == 0.0:
                    continue
                step = 2.0 * radii[i] * d[i] * inv
                x2[i] = clamp(x2[i] + step, i)
            f2 = eval_f(x2)
            if f2 < f1:
                return x2, f2
            return x1, f1
        return None, None

    # --- Main refine loop over elites (cycling) ---
    # Initialize per-dimension radii (trust region)
    base_r = [0.25 * s if s > 0 else 0.0 for s in spans]
    min_r = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    elite_idx = 0
    stall = 0

    while time.time() < deadline:
        if not elites:
            # Shouldn't happen, but just in case:
            x = rand_point()
            fx = eval_f(x)
            add_elite(fx, x)

        elite_idx %= len(elites)
        fx, x = elites[elite_idx]
        elite_idx += 1

        # Local radii for this start: scaled by rank (better -> smaller, worse -> larger)
        rank = min(len(elites) - 1, max(0, elite_idx - 1))
        scale = 1.0 + 0.5 * (rank / max(1, len(elites) - 1))
        radii = [max(min_r[i], base_r[i] * scale) for i in range(dim)]

        improved_any = False

        # A few coordinate sweeps
        sweeps = 2 if dim <= 10 else 1
        for _ in range(sweeps):
            # random order
            coords = list(range(dim))
            # Fisher-Yates using our LCG
            for j in range(dim - 1, 0, -1):
                r = int(lcg_u01() * (j + 1))
                coords[j], coords[r] = coords[r], coords[j]

            for i in coords:
                if time.time() >= deadline:
                    return best
                if spans[i] == 0.0:
                    continue

                h = radii[i]
                # First try a cheap coordinate +/- check
                xi = x[i]
                xP = list(x); xM = list(x)
                xP[i] = clamp(xi + h, i)
                xM[i] = clamp(xi - h, i)
                fP = eval_f(xP)
                fM = eval_f(xM)

                if fP < fx or fM < fx:
                    if fP <= fM:
                        x, fx = xP, fP
                    else:
                        x, fx = xM, fM
                    improved_any = True
                    # slightly expand this radius
                    radii[i] = min(0.5 * spans[i], radii[i] * 1.5) if spans[i] > 0 else radii[i]
                else:
                    # If no improvement, try parabolic interpolation (can jump to better point)
                    xN, fN = parabolic_min_1d(x, i, h)
                    if xN is not None and fN < fx:
                        x, fx = xN, fN
                        improved_any = True
                        radii[i] = min(0.5 * spans[i], radii[i] * 1.2) if spans[i] > 0 else radii[i]
                    else:
                        radii[i] = max(min_r[i], radii[i] * 0.6)

        # Occasional random-direction pattern move to escape coordinate-wise traps
        if time.time() < deadline and (lcg_u01() < 0.35):
            xN, fN = try_pattern_move(x, fx, radii)
            if xN is not None and fN < fx:
                x, fx = xN, fN
                improved_any = True

        # Update global best / elites
        if fx < best:
            best, best_x = fx, x

        add_elite(fx, x)

        if improved_any:
            stall = 0
            # Tighten base radii slowly when things improve (focus)
            for i in range(dim):
                base_r[i] = max(min_r[i], base_r[i] * 0.98)
        else:
            stall += 1
            # On stall, diversify: either broaden radii or inject new candidate
            if stall % 7 == 0:
                for i in range(dim):
                    base_r[i] = min(0.5 * spans[i], base_r[i] * 1.25) if spans[i] > 0 else base_r[i]
            if stall % 5 == 0:
                # Inject a new exploration point (sometimes near best, sometimes global)
                if lcg_u01() < 0.6 and best_x is not None:
                    y = list(best_x)
                    for i in range(dim):
                        if spans[i] == 0.0:
                            continue
                        # jitter proportional to base_r
                        y[i] = clamp(y[i] + (lcg_u01() * 2.0 - 1.0) * max(base_r[i], 0.02 * spans[i]), i)
                else:
                    y = qrand_point(k)
                    k += 1
                fy = eval_f(y)
                if fy < best:
                    best, best_x = fy, y
                add_elite(fy, y)

    return best
