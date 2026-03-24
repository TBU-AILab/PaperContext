#The best-performing algorithm so far is **iL-SHADE-OBL** (Algorithm 1/6), achieving an output of **1.27**. It significantly outperformed standard approaches by using Opposition-Based Learning (OBL) for initialization.
#
#To further improve performance and aim for a value closer to the global optimum (0), I propose **iL-SHADE-GW (Improved L-SHADE with Gaussian Walk Exploitation)**.
#
#**Key Improvements:**
#1.  **Gaussian Walk Local Search (GW)**: While standard DE is excellent at global search, it can struggle to refine solutions to high precision in the final stages. I added a lightweight **Gaussian Walk** mechanism. If the global best solution does not improve during a generation (in the latter half of the runtime), the algorithm generates a single mutant near the best solution with a shrinking standard deviation ($ \sigma $). This effectively acts as a local hill-climber to polish the best result without expensive gradient calculations.
#2.  **Weighted Mutation (jSO-inspired)**: The mutation strategy is refined to include a time-dependent weight ($F_w$) applied to the difference vector towards the best individual ($x_{pbest} - x_i$). Early in the search, $F_w$ is scaled down (0.7x - 0.8x) to prevent the population from collapsing onto the current best too quickly (premature convergence).
#3.  **Preserved OBL Initialization**: Retained the Opposition-Based Learning initialization phase as it proved highly effective in the previous iteration for generating a superior starting population.
#
#Here is the fully operational and self-contained Python code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # -------------------------------------------------------------------------
    # Initialization & Time Management
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    # Safety buffer: stop 100ms before max_time to ensure return
    time_limit = timedelta(seconds=max_time) - timedelta(milliseconds=100)

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Population Size
    # Large initial population for OBL to filter best candidates
    pop_size_init = int(30 * dim)
    pop_size_init = max(50, min(300, pop_size_init))
    pop_size_min = 5

    # Archive Size: 2.5x population size for diversity history
    arc_rate = 2.5
    
    # Memory Size for parameter adaptation
    h_mem_size = 6
    
    # Dynamic p-best parameters
    # Linearly decrease from p_max (exploration) to p_min (exploitation)
    p_max = 0.20
    p_min = 0.05

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Memory for adaptive parameters (initialized to 0.5)
    mem_cr = np.full(h_mem_size, 0.5)
    mem_f = np.full(h_mem_size, 0.5)
    k_mem = 0

    best_val = float('inf')
    best_sol = None
    archive = []

    # -------------------------------------------------------------------------
    # 1. Opposition-Based Learning (OBL) Initialization
    # -------------------------------------------------------------------------
    # A. Generate Random Population
    pop_rand = min_b + np.random.rand(pop_size_init, dim) * diff_b
    fit_rand = np.full(pop_size_init, float('inf'))

    for i in range(pop_size_init):
        if check_time(): return best_val
        val = func(pop_rand[i])
        fit_rand[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_rand[i].copy()

    # B. Generate Opposite Population
    # x' = lower + upper - x
    pop_opp = min_b + max_b - pop_rand
    pop_opp = np.clip(pop_opp, min_b, max_b)
    fit_opp = np.full(pop_size_init, float('inf'))

    for i in range(pop_size_init):
        if check_time(): return best_val
        val = func(pop_opp[i])
        fit_opp[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_opp[i].copy()

    # C. Select Best N Individuals
    total_pop = np.vstack((pop_rand, pop_opp))
    total_fit = np.concatenate((fit_rand, fit_opp))
    
    sorted_idx = np.argsort(total_fit)
    pop_size = pop_size_init
    population = total_pop[sorted_idx[:pop_size]]
    fitness = total_fit[sorted_idx[:pop_size]]

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        if check_time(): return best_val

        # --- Calculate Progress & Update LPSR ---
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress >= 1.0: return best_val

        # Linear Population Size Reduction
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Resize: Population is sorted, keep top 'plan_pop_size'
            pop_size = plan_pop_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            
            # Archive Resize
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                rm_idx = np.random.randint(0, len(archive))
                archive.pop(rm_idx)

        # --- Dynamic Strategy Parameters ---
        # p-best selection range
        current_p = p_max - (p_max - p_min) * progress
        p_num = max(2, int(current_p * pop_size))
        
        # Weighted Mutation Factor (jSO strategy)
        # Prevents premature convergence by scaling down attraction to p-best early on
        if progress < 0.2:
            fw_scale = 0.7
        elif progress < 0.4:
            fw_scale = 0.8
        else:
            fw_scale = 1.0

        # --- Generation Loop ---
        new_pop = np.empty_like(population)
        new_fit = np.empty_like(fitness)
        
        s_cr = []
        s_f = []
        s_imp = []
        
        best_improved_in_gen = False

        for i in range(pop_size):
            if check_time(): return best_val

            # A. Parameter Generation
            r_idx = np.random.randint(0, h_mem_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]

            # CR ~ Normal(mu_cr, 0.1)
            if mu_cr == -1:
                cr = 0.0
            else:
                cr = np.random.normal(mu_cr, 0.1)
                cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            while True:
                f_val = mu_f + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    if f_val > 1: f_val = 1.0
                    break
            
            # B. Mutation: current-to-pbest-weighted/1
            # v = x_i + Fw * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            n_arch = len(archive)
            r2 = np.random.randint(0, pop_size + n_arch)
            while True:
                if r2 < pop_size:
                    if r2 == i or r2 == r1:
                        r2 = np.random.randint(0, pop_size + n_arch)
                        continue
                break
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]

            # Apply weighted mutation
            mutant = population[i] + (f_val * fw_scale) * (x_pbest - population[i]) + f_val * (x_r1 - x_r2)
            
            # C. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, population[i])
            
            # D. Bound Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            val_trial = func(trial)
            
            if val_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = val_trial
                
                # Update Global Best
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial.copy()
                    best_improved_in_gen = True
                
                # Success History
                if val_trial < fitness[i]:
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(fitness[i] - val_trial)
                    archive.append(population[i].copy())
            else:
                new_pop[i] = population[i]
                new_fit[i] = fitness[i]

        population = new_pop
        fitness = new_fit

        # --- Gaussian Walk (Exploitation) ---
        # If best solution stagnated this generation (and we are past 50% time),
        # try a single Gaussian perturbation on the best solution.
        if not best_improved_in_gen and progress > 0.5:
            if check_time(): return best_val
            
            # Sigma shrinks as time progresses
            sigma_gw = 0.02 * (1.0 - progress)
            
            # Only perform if sigma is significant
            if sigma_gw > 1e-9:
                # Generate mutant around best_sol
                # Scaled by domain width (diff_b)
                gw_trial = best_sol + np.random.normal(0, sigma_gw, dim) * diff_b
                gw_trial = np.clip(gw_trial, min_b, max_b)
                
                gw_val = func(gw_trial)
                
                if gw_val < best_val:
                    best_val = gw_val
                    best_sol = gw_trial.copy()
                    # Inject improved solution into population (replace worst)
                    worst_idx = np.argmax(fitness)
                    population[worst_idx] = gw_trial
                    fitness[worst_idx] = gw_val

        # --- Archive Maintenance ---
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            idx = np.random.randint(0, len(archive))
            archive.pop(idx)

        # --- Memory Update (Weighted Lehmer Mean) ---
        if len(s_cr) > 0:
            s_cr_np = np.array(s_cr)
            s_f_np = np.array(s_f)
            s_imp_np = np.array(s_imp)
            
            # Weights proportional to fitness improvement
            weights = s_imp_np / np.sum(s_imp_np)
            
            # Update CR
            if np.max(mem_cr) != -1:
                mean_cr = np.sum(weights * s_cr_np)
                mem_cr[k_mem] = mean_cr
            
            # Update F
            sum_wf = np.sum(weights * s_f_np)
            if sum_wf > 0:
                mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                mem_f[k_mem] = mean_f
            
            k_mem = (k_mem + 1) % h_mem_size

        # --- Sort Population ---
        # Required for p-best selection in next generation
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_val
