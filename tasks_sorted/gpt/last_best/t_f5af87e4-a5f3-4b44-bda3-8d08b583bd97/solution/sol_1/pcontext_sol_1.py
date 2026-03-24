import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (no external libs):
      - Opposition-based + random initialization
      - Multi-start (keeps a small elite set)
      - (1+1)-ES style local search per elite with Rechenberg step-size control
      - Occasional heavy-tailed/global jumps for escaping local minima

    Returns
    -------
    best : float
        Best (minimum) objective value found within the time budget.
    """

    # -------- basic setup --------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # Guard against degenerate spans
    safe_spans = [s if s != 0.0 else 1.0 for s in spans]

    deadline = time.perf_counter() + float(max_time)

    def clamp_inplace(x):
        for i in range(dim):
            lo = lows[i]; hi = highs[i]
            v = x[i]
            if v < lo:
                x[i] = lo
            elif v > hi:
                x[i] = hi

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite_point(x):
        # reflection around middle of bounds: x' = lo + hi - x
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # Fast normal approx (sum of uniforms), no math.sin/cos/log needed
    def randn():
        # approx N(0,1)
        return (random.random() + random.random() + random.random() + random.random() - 2.0)

    # Cauchy-like heavy tail using tan(pi*(u-0.5))
    def cauchy():
        u = random.random()
        # avoid tan blow-up at exactly 0.5 +/- 0.5
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # -------- elite management --------
    # Keep small elite to diversify local searches
    elite_size = max(3, min(10, 2 + dim // 2))
    elites = []  # list of dicts: {'x':..., 'fx':..., 'sigma':..., 'succ':..., 'tries':...}

    best = float("inf")
    best_x = None

    def consider(x, fx):
        nonlocal best, best_x, elites
        if fx < best:
            best = fx
            best_x = list(x)

        # insert into elites if good or if elites not full
        if len(elites) < elite_size or fx < elites[-1]['fx']:
            # initial sigma proportional to search size
            base_sigma = 0.2
            sigma = base_sigma
            elites.append({
                'x': list(x),
                'fx': fx,
                'sigma': sigma,
                'succ': 0,
                'tries': 0
            })
            elites.sort(key=lambda d: d['fx'])
            if len(elites) > elite_size:
                elites.pop()

    # -------- initialization (opposition-based + random) --------
    # A bit larger init helps a lot for rugged landscapes
    init_n = max(20, 12 * dim)
    for _ in range(init_n):
        if time.perf_counter() >= deadline:
            return best
        x = rand_point()
        fx = eval_f(x)
        consider(x, fx)

        # opposition point often gives a free improvement
        xo = opposite_point(x)
        clamp_inplace(xo)
        fxo = eval_f(xo)
        consider(xo, fxo)

    if best_x is None:
        return best

    # -------- main loop: multi-start local search over elites --------
    # Per-elite sigma is in normalized units; convert to actual via spans when stepping.
    # Rechenberg: every window steps, if success_rate > 1/5 increase sigma else decrease.
    window = 25
    # floor/ceiling for sigma in normalized space
    sigma_min = 1e-6
    sigma_max = 0.5

    # used to cycle elites fairly
    elite_idx = 0

    while True:
        if time.perf_counter() >= deadline:
            return best

        if not elites:
            # should not happen, but keep going
            x = rand_point()
            consider(x, eval_f(x))
            continue

        elite = elites[elite_idx]
        elite_idx = (elite_idx + 1) % len(elites)

        x0 = elite['x']
        fx0 = elite['fx']
        sigma = elite['sigma']

        # Decide move type:
        # - mostly Gaussian local
        # - sometimes coordinate perturbation
        # - sometimes heavy-tailed/global jump (more if stagnating)
        # Stagnation proxy: if best hasn't improved in a while, elites will tend to have low success rates.
        success_rate = (elite['succ'] / elite['tries']) if elite['tries'] > 0 else 0.0
        p_global = 0.03 + (0.10 if elite['tries'] > window and success_rate < 0.08 else 0.0)
        p_coord = 0.25

        r = random.random()
        if r < p_global:
            # global/heavy-tailed jump around best or random
            if random.random() < 0.7 and best_x is not None:
                base = best_x
            else:
                base = x0
            x = list(base)
            # heavy-tailed step with moderate scale
            scale = 0.25
            for i in range(dim):
                if spans[i] == 0.0:
                    continue
                step = scale * safe_spans[i] * cauchy()
                x[i] += step
            clamp_inplace(x)
        elif r < p_global + p_coord:
            # coordinate search: change a few coordinates strongly, others slightly
            x = list(x0)
            k = 1 if dim <= 3 else 1 + (random.randrange(min(dim, 4)))
            for _ in range(k):
                j = random.randrange(dim)
                if spans[j] == 0.0:
                    continue
                # stronger step in one coordinate
                x[j] += (randn() * 0.75) * (sigma * safe_spans[j])
            # light noise on the rest
            for i in range(dim):
                if spans[i] == 0.0:
                    continue
                if random.random() < 0.2:
                    x[i] += (randn() * 0.25) * (sigma * safe_spans[i])
            clamp_inplace(x)
        else:
            # (1+1)-ES: isotropic gaussian in normalized coordinates scaled by spans
            x = list(x0)
            for i in range(dim):
                if spans[i] == 0.0:
                    continue
                x[i] += randn() * (sigma * safe_spans[i])
            clamp_inplace(x)

        fx = eval_f(x)

        # accept if improved for that elite
        elite['tries'] += 1
        if fx < fx0:
            elite['x'] = x
            elite['fx'] = fx
            elite['succ'] += 1
            consider(x, fx)  # also updates global best/elite pool

        # step-size adaptation (Rechenberg 1/5 rule)
        if elite['tries'] >= window and (elite['tries'] % window) == 0:
            sr = elite['succ'] / float(window)
            # reset counters for next window
            elite['succ'] = 0

            if sr > 0.2:
                sigma *= 1.25
            else:
                sigma *= 0.82

            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            elite['sigma'] = sigma

        # Keep elites sorted by fitness occasionally (cheap)
        # Doing this every iteration can be a bit wasteful; do sporadically.
        if random.random() < 0.1:
            elites.sort(key=lambda d: d['fx'])
            if len(elites) > elite_size:
                elites = elites[:elite_size]

        # Inject fresh blood occasionally
        if random.random() < 0.02:
            xr = rand_point()
            consider(xr, eval_f(xr))
            xo = opposite_point(xr)
            clamp_inplace(xo)
            consider(xo, eval_f(xo))
