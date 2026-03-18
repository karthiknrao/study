# Wolf Pack Simulation: Deep Dive

A comprehensive overview of wolf pack simulation research, algorithms, and code implementations.

---

## Foundational Academic Papers

### 1. Muro et al. (2011) - The Seminal Paper

**"Wolf-pack (Canis lupus) hunting strategies emerge from simple rules in computational simulations"**

Published in *Behavioural Processes*

**Key Finding**: Two simple decentralized rules can reproduce wolf-pack hunting behavior:
1. Move towards the prey until a minimum safe distance `d_c` is reached
2. When at safe distance `d_c`, move away from other wolves within the safe area

**Emergent behaviors**: tracking, pursuit, encircling, cursorial pursuit, relay running, ambushing

**Important insight**: Complex hunting patterns emerge **without** hierarchical social structure or high cognitive abilities

- [PubMed](https://pubmed.ncbi.nlm.nih.gov/21963347/)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0376635711001884)

### 2. Escobedo et al. (2014) - Extended Model

**"Group size, individual role differentiation and effectiveness of cooperation in a homogeneous group of hunters"**

Published in *Journal of the Royal Society Interface*

Uses particle-based computational model with attractive/repulsive forces.

**Key discoveries**:
- Critical pack size `N* = 5` - below this, stable configurations form regular polygons
- Above `N* = 5`, complex behaviors emerge: multiple orbits, traveling formations, rotating formations, periodic oscillations, and chaos
- Explains why hunting success peaks at small pack sizes
- **Role differentiation emerges spontaneously** from spatial dynamics alone

- [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4006263/)

---

## Mathematical Model

The model uses Newton's second law with interaction forces:

```
For each wolf Wi:
m_i * dv_i/dt = F_i,p(t) + Sum(F_i,j(t)) - v_i * v_i

Where:
- F_i,p = attraction to prey (long-range) + repulsion (short-range)
- F_i,j = repulsion from other wolves
- v_i = friction coefficient
```

### Interaction Functions

- `g_a(u) = -u/||u||^2` (attraction)
- `g_r(u) = u/||u|| * exp(-||u||/d_c)` (repulsion)

### Critical Distances

- `d_c` = safe distance from prey (avoid horns/kicks)
- `d_a` = distance at which wolves repulse each other (`d_a > d_c`)

### Model Parameters (Normalized Units)

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `m_p` | mass of prey (350-400 kg) | 1 |
| `m_i` | mass of wolf (35-40 kg) | 0.1 |
| `v_p` | prey friction coefficient | 2 |
| `v_i` | wolf friction coefficient | 1 |
| `c_wp` | prey-to-wolf force coefficient | 2 |
| `c_ww` | wolf-to-wolf interaction coefficient | 0.5 |
| `c_pw` | wolf-to-prey force coefficient | 0.2 |
| `d_c` | safe distance from prey | 1 |
| `d_a` | wolf repulsion distance | 1.5 |
| `c_w` | Gaussian width coefficient | 0.5 |
| `v_max` | max prey velocity | 0.1 |

---

## Optimization Algorithms

### 1. Wolf Pack Algorithm (WPA)

Proposed by Wu et al. (2014), inspired by wolf hunting behavior.

| Behavior | Description |
|----------|-------------|
| **Scouting** | Scout wolves explore for prey |
| **Calling** | Lead wolf howls to summon pack |
| **Round-up** | Pack encircles prey |
| **Siege** | Final attack on prey |

**Key Parameters**:
- `L_near`: Distance determinant coefficient
- `step`: Movement step size
- Population update: "survival of the fittest"

**Algorithm Flow**:
1. Initialize wolf population randomly
2. Calculate fitness for each wolf
3. Select lead wolf (best fitness)
4. Scout wolves explore neighborhood
5. Calling behavior updates positions toward lead wolf
6. Round-up behavior encircles best position
7. Update population (survival of fittest)
8. Repeat until convergence

- [Original Paper](https://www.hindawi.com/journals/mpe/2014/465082/)

### 2. Grey Wolf Optimizer (GWO)

Proposed by Mirjalili et al. (2014).

**Social Hierarchy**:
```
alpha -> Best solution
beta  -> Second best
delta -> Third best
omega -> Remaining wolves
```

**Hunting Phases**:
1. **Tracking/Searching** - omega wolves follow alpha, beta, delta
2. **Encircling** - Position updates around prey
3. **Attacking** - Exploitation phase

**Position Update Formula**:
```python
D_alpha = |C1 * X_alpha - X|
D_beta  = |C2 * X_beta - X|
D_delta = |C3 * X_delta - X|

X1 = X_alpha - A1 * D_alpha
X2 = X_beta  - A2 * D_beta
X3 = X_delta - A3 * D_delta

X(t+1) = (X1 + X2 + X3) / 3
```

Where:
- `A = 2a * r1 - a` (a decreases from 2 to 0 over iterations)
- `C = 2 * r2` (r1, r2 are random [0,1])

- [Official GWO Page](https://seyedalimirjalili.com/gwo)
- [Original Paper](https://www.sciencedirect.com/science/article/abs/pii/S096997813001853)

### 3. Improved Variants

- **OGL-WPA**: Opposition-based learning + Genetic Algorithm + Levy flight
- **VW-GWO**: Variable weights GWO
- **EBGWO**: Elite Inheritance + Balance Search
- **CG-GWO**: Cauchy-Gaussian mutation

---

## Code Implementations

### Python Libraries & Repositories

| Repository | Description | Link |
|------------|-------------|------|
| **Substancia/Wolfpack-Search-Algorithm** | WPA + GWO hybrid | [GitHub](https://github.com/Substancia/Wolfpack-Search-Algorithm) |
| **Tomatohust/Wolf-Pack-Algorithm** | WPA demo | [GitHub](https://github.com/Tomatohust/Wolf-Pack-Algorithm) |
| **7ossam81/EvoloPy** | Multi-algorithm toolbox (includes GWO) | [GitHub](https://github.com/7ossam81/EvoloPy) |
| **RenatoFFilho/Grey_Wolf_Optimizer** | GWO implementation | [GitHub](https://github.com/RenatoFFilho/Grey_Wolf_Optimizer) |
| **xzltc/WPA** | Wolf Pack Algorithm | [GitHub](https://github.com/xzltc/WPA) |
| **chengfeng2021/OGL-WPA** | Improved WPA | [GitHub](https://github.com/chengfeng2021/OGL-WPA) |

### Python Package Installation

```bash
# Grey Wolf Optimizer
pip install gwo-package

# MEALPY (includes GWO and variants)
pip install mealpy

# EvoloPy toolbox
git clone https://github.com/7ossam81/EvoloPy.git
```

### Basic GWO Usage Example

```python
import numpy as np

def objective_function(x):
    return np.sum(x**2)

class GreyWolfOptimizer:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

    def optimize(self, lb, ub):
        # Initialize population
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([self.obj_func(ind) for ind in population])

        # Sort and select alpha, beta, delta
        sorted_idx = np.argsort(fitness)
        alpha = population[sorted_idx[0]].copy()
        beta = population[sorted_idx[1]].copy()
        delta = population[sorted_idx[2]].copy()

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Decreases from 2 to 0

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * alpha[j] - population[i,j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * beta[j] - population[i,j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * delta[j] - population[i,j])
                    X3 = delta[j] - A3 * D_delta

                    population[i,j] = (X1 + X2 + X3) / 3

            # Update fitness and leaders
            fitness = np.array([self.obj_func(ind) for ind in population])
            sorted_idx = np.argsort(fitness)
            alpha = population[sorted_idx[0]].copy()
            beta = population[sorted_idx[1]].copy()
            delta = population[sorted_idx[2]].copy()

        return alpha, self.obj_func(alpha)
```

---

## Agent-Based Simulation Platforms

### NetLogo

The classic platform for agent-based wolf-sheep predation models.

| Model | Description |
|-------|-------------|
| **Wolf Sheep Predation** | Standard predator-prey ABM |
| **Wolf Sheep Predation (Docked Hybrid)** | Compares agent-based vs aggregate models |
| **Wolf Sheep Predation (System Dynamics)** | Lotka-Volterra equations |

- [NetLogo Models Library](https://ccl.northwestern.edu/netlogo/models/WolfSheepPredation)
- [EduTech Wiki Guide](https://edutechwiki.unige.ch/en/NetLogo_Wolf_Sheep_Predation_model)

### Mesa (Python)

Python framework equivalent to NetLogo.

```python
# Install
pip install mesa

# Wolf-Sheep example included
# https://mesa.readthedocs.io/stable/examples/advanced/wolf_sheep.html
```

- [Mesa Documentation](https://mesa.readthedocs.io/stable/examples/advanced/wolf_sheep.html)

---

## Simulation Projects

### Notable GitHub Projects

| Project | Description | Link |
|---------|-------------|------|
| **Wolf-Pack-Simulator** | Terminal-based survival simulation | [GitHub](https://github.com/WolfBot2026/wolf-pack-simulator) |
| **Fenrir-wolfpack-simulator** | JavaScript simulation with data trees | [GitHub](https://github.com/skywarth/Fenrir-wolfpack-simulator) |
| **gym-wolfpack** | OpenAI Gym environment for RL | [GitHub](https://github.com/dkkim93/gym-wolfpack) |
| **PackGen** | Pack management game with genetics/territory | [GitHub](https://github.com/listst/PackGen) |

---

## Key Research Insights

### Pack Size Effects

```
Pack Size (N) | Behavior
--------------|------------------
N <= 5        | Stable regular polygon (SRP)
N > 5         | Multi-orbit configurations
N >> 5        | Chaos, oscillations, role "differentiation"
```

### Emergent Patterns (from simple rules)

| Pattern | Description |
|---------|-------------|
| **Cursorial pursuit** | Extended chase over distance |
| **Encircling** | Surrounding prey from multiple angles |
| **Relay running** | Taking turns leading the pursuit |
| **Ambushing** | Strategic positioning ahead of prey |
| **Flocking** | Coordinated group movement |
| **Milling** | Rotating formations around prey |

### Why Hunting Success Peaks at Small Pack Sizes

1. **N <= 5**: Wolves form stable regular polygon around prey - optimal coordination
2. **N > 5**: Bifurcation occurs - wolves distributed across multiple orbits
3. **N >> 5**: Complex dynamics (oscillations, chaos) disrupt coordination
4. **Freeriding**: Wolves in outer orbits can "hold back" while others hunt

---

## Applications

| Domain | Application |
|--------|-------------|
| **Robotics** | Multi-robot coordination, UAV swarms |
| **Optimization** | Engineering design, path planning, scheduling |
| **Ecology** | Predator-prey dynamics, conservation modeling |
| **Machine Learning** | Hyperparameter tuning, neural network training |
| **Game Development** | AI behavior for wildlife simulation |
| **Economics** | Market dynamics, resource allocation |

---

## References

### Core Papers

1. Muro, C., Escobedo, R., Spector, L., & Coppinger, R. P. (2011). Wolf-pack (Canis lupus) hunting strategies emerge from simple rules in computational simulations. *Behavioural Processes*, 88, 192-197.

2. Escobedo, R., Muro, C., Spector, L., & Coppinger, R. P. (2014). Group size, individual role differentiation and effectiveness of cooperation in a homogeneous group of hunters. *Journal of the Royal Society Interface*, 11(95), 20140204.

3. Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46-61.

4. Wu, H., Zhang, F., & Wu, H. (2014). Wolf pack algorithm for unconstrained global optimization. *Mathematical Problems in Engineering*, 2014, 465082.

### Additional Resources

- [GeeksforGeeks GWO Implementation](https://www.geeksforgeeks.org/machine-learning/implementation-of-grey-wolf-optimization-gwo-algorithm/)
- [PLOS One: Improved WPA](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254239)
- [Nature: Wolf Pack Algorithm Overview](https://link.springer.com/chapter/10.1007/978-981-96-0795-2_8)
