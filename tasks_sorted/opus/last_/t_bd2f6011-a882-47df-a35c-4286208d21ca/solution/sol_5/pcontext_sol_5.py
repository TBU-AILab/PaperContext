#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    def time_left():
        return max_time - (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(200, max(50, dim * 10))
    for i in range(n_init):
        if time_left() < 0.1:
            return best
        x = np.array([np.random.uniform(l, h) for l, h in bounds])
        evaluate(x)

    # Phase 2: CMA-ES inspired search
    def cmaes_search(x0, sigma0, budget_fraction=0.4):
        nonlocal best, best_params
        pop_size = max(10, 4 + int(3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        deadline = time_left() * budget_fraction
        t0 = (datetime.now() - start).total_seconds()

        gen = 0
        while True:
            elapsed_phase = (datetime.now() - start).total_seconds() - t0
            if elapsed_phase > deadline or time_left() < 0.2:
                break

            try:
                sqrtC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sqrtC = np.eye(dim)

            solutions = []
            for k in range(pop_size):
                z = np.random.randn(dim)
                x = mean + sigma * sqrtC.dot(z)
                x = clip(x)
                f = evaluate(x)
                solutions.append((f, x, z))

            solutions.sort(key=lambda s: s[0])
            selected = solutions[:mu]

            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i, (f, x, z) in enumerate(selected):
                mean += weights[i] * x
            mean = clip(mean)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.solve(sqrtC, (mean - old_mean) / sigma)
            h_sig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
            pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

            artmp = np.array([(x - old_mean) / sigma for (f, x, z) in selected])
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(artmp[i], artmp[i])

            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, np.max(upper - lower))
            gen += 1

    if best_params is not None:
        sigma0 = np.mean(upper - lower) * 0.3
        cmaes_search(best_params.copy(), sigma0, budget_fraction=0.5)

    # Phase 3: Coordinate descent refinement
    def coordinate_descent(x0, step_factor=0.01):
        nonlocal best, best_params
        x = x0.copy()
        fx = evaluate(x)
        steps = (upper - lower) * step_factor

        while time_left() > 0.1:
            improved = False
            for i in range(dim):
                if time_left() < 0.05:
                    return
                for direction in [1, -1]:
                    x_new = x.copy()
                    x_new[i] = clip(x_new)[i] + direction * steps[i]
                    x_new = clip(x_new)
                    f_new = evaluate(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved = True
                        break
            if not improved:
                steps *= 0.5
                if np.max(steps) < 1e-12:
                    break

    if best_params is not None:
        coordinate_descent(best_params.copy(), step_factor=0.01)

    # Phase 4: Use remaining time for random restarts with local search
    while time_left() > 0.5:
        x = best_params + np.random.randn(dim) * (upper - lower) * 0.1
        x = clip(x)
        evaluate(x)
        coordinate_descent(x, step_factor=0.005)

    return best
