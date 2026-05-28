import numpy as np
import matplotlib.pyplot as plt

seed = 71
rng = np.random.default_rng(seed)

IEEE754_bits = 32
num_genes = 3
gnotyp_length = num_genes * IEEE754_bits  # 96
rastrigin_range = (-5.12, 5.12)


class Subject:
    def __init__(self, x, y, z):
        self.x_1 = self.to_ieee754(x)
        self.x_2 = self.to_ieee754(y)
        self.x_3 = self.to_ieee754(z)
        self.fitness = None

    @staticmethod
    def clip(value):
        return np.clip(np.float32(value), rastrigin_range[0], rastrigin_range[1]).astype(np.float32)

    @staticmethod
    def to_ieee754(value):
        value = np.float32(value)
        uint_val = np.frombuffer(value.tobytes(), dtype=np.uint32)[0]
        return np.unpackbits(np.array([uint_val], dtype='>u4').view(np.uint8)).astype(np.uint8)

    @staticmethod
    def from_ieee754(bits):
        packed = np.packbits(bits)
        val = packed.view('>f4')[0].astype(np.float32)
        if np.isnan(val) or np.isinf(val):
            return np.float32(0.0)
        return val

    @property
    def x1(self): return self.from_ieee754(self.x_1)
    @property
    def x2(self): return self.from_ieee754(self.x_2)
    @property
    def x3(self): return self.from_ieee754(self.x_3)

    def get_genotype(self):
        return np.concatenate((self.x_1, self.x_2, self.x_3)).astype(np.uint8)

    @classmethod
    def from_genotype(cls, genotype):
        genotype = np.asarray(genotype, dtype=np.uint8)
        obj = cls.__new__(cls)
        obj.x_1 = Subject.to_ieee754(Subject.clip(Subject.from_ieee754(genotype[:32].copy())))
        obj.x_2 = Subject.to_ieee754(Subject.clip(Subject.from_ieee754(genotype[32:64].copy())))
        obj.x_3 = Subject.to_ieee754(Subject.clip(Subject.from_ieee754(genotype[64:96].copy())))
        obj.fitness = None
        return obj

    def evaluate_fitness(self):
        x, y, z = self.x1, self.x2, self.x3
        if any(np.isnan(v) or np.isinf(v) for v in [x, y, z]):
            self.fitness = -1e9
            return
        result = 30 + (x**2 - 10*np.cos(2*np.pi*x)) + \
                      (y**2 - 10*np.cos(2*np.pi*y)) + \
                      (z**2 - 10*np.cos(2*np.pi*z))
        self.fitness = float(-result)

    def __repr__(self):
        return f'Subject(x1={self.x1:.4f}, x2={self.x2:.4f}, x3={self.x3:.4f}, fitness={self.fitness})'


def _multi_cut_swap(bits1, bits2, cuts_count):
    bits1 = np.asarray(bits1, dtype=np.uint8)
    bits2 = np.asarray(bits2, dtype=np.uint8)
    cut_points = np.sort(rng.choice(np.arange(1, bits1.size), size=cuts_count, replace=False))
    bounds = np.concatenate(([0], cut_points, [bits1.size]))
    child1_parts = []
    child2_parts = []
    use_original = True
    for start, end in zip(bounds[:-1], bounds[1:]):
        if use_original:
            child1_parts.append(bits1[start:end])
            child2_parts.append(bits2[start:end])
        else:
            child1_parts.append(bits2[start:end])
            child2_parts.append(bits1[start:end])
        use_original = not use_original
    return np.concatenate(child1_parts), np.concatenate(child2_parts)


def _crossover_component(bits1, bits2):
    child1 = bits1.copy()
    child2 = bits2.copy()
    # bit znaku
    if bits1[0] == bits2[0]:
        child1[0] = bits1[0]
        child2[0] = bits1[0]
    else:
        sign_bit = np.uint8(rng.integers(0, 2))
        child1[0] = sign_bit
        child2[0] = sign_bit
    # wykładnik
    child1[1:9], child2[1:9] = _multi_cut_swap(bits1[1:9], bits2[1:9], cuts_count=1)
    # mantysa
    child1[9:], child2[9:] = _multi_cut_swap(bits1[9:], bits2[9:], cuts_count=6)
    return child1, child2


def crossover(parent1, parent2):
    child1_x1, child2_x1 = _crossover_component(parent1.x_1, parent2.x_1)
    child1_x2, child2_x2 = _crossover_component(parent1.x_2, parent2.x_2)
    child1_x3, child2_x3 = _crossover_component(parent1.x_3, parent2.x_3)
    child1 = Subject.from_genotype(np.concatenate((child1_x1, child1_x2, child1_x3)))
    child2 = Subject.from_genotype(np.concatenate((child2_x1, child2_x2, child2_x3)))
    return child1, child2


def mutate(subject, mutation_rate=0.05):
    genotype = subject.get_genotype().copy()
    if rng.random() < mutation_rate:
        mutation_index = rng.integers(0, genotype.size)
        genotype[mutation_index] = np.uint8(1 - genotype[mutation_index])
    mutated = Subject.from_genotype(genotype)
    mutated.fitness = subject.fitness
    return mutated


def init_population(size):
    population = []
    for _ in range(size):
        x = rng.uniform(rastrigin_range[0], rastrigin_range[1])
        y = rng.uniform(rastrigin_range[0], rastrigin_range[1])
        z = rng.uniform(rastrigin_range[0], rastrigin_range[1])
        subject = Subject(x, y, z)
        subject.evaluate_fitness()
        population.append(subject)
    return population


def roulette_wheel_selection(population):
    fitness_values = np.array([s.fitness for s in population], dtype=np.float64)
    fitness_values = np.nan_to_num(fitness_values, nan=-1e9, posinf=-1e9, neginf=-1e9)
    max_fitness = np.max(fitness_values)
    weights = (fitness_values - np.min(fitness_values)) + 1e-6
    probabilities = weights / np.sum(weights)
    selected_index = rng.choice(len(population), p=probabilities)
    return population[selected_index]


def run_ga(population_size=50, generations=500, crossover_prob=0.7, mutation_prob=0.05, no_improve_limit=50):
    population = init_population(population_size)
    best_history = []
    avg_history = []
    worst_history = []
    no_improve = 0
    best_ever = -1e9

    for generation in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)

            if rng.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1 = Subject.from_genotype(parent1.get_genotype())
                child2 = Subject.from_genotype(parent2.get_genotype())

            child1 = mutate(child1, mutation_prob)
            child1.evaluate_fitness()
            new_population.append(child1)

            if len(new_population) < population_size:
                child2 = mutate(child2, mutation_prob)
                child2.evaluate_fitness()
                new_population.append(child2)

        elite = max(population, key=lambda s: s.fitness)
        new_population[0] = elite
        population = new_population
        fitness_values = np.array([s.fitness for s in population], dtype=np.float32)
        best_history.append(np.max(fitness_values))
        avg_history.append(np.mean(fitness_values))
        worst_history.append(np.min(fitness_values))

        current_best = np.max(fitness_values)
        if current_best > best_ever:
            best_ever = current_best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= no_improve_limit:
            print(f'Zatrzymano po {generation + 1} generacjach (brak poprawy)')
            break



    best = max(population, key=lambda s: s.fitness)
    print(f'Parameters of the best solution : [{best.x1:.8f} {best.x2:.8f} {best.x3:.8f}]')
    print(f'Fitness value of the best solution = {best.fitness}')
    return best, best_history, avg_history, worst_history, population


def plot_fitness(best_history, avg_history, worst_history):
    generations = np.arange(1, len(best_history) + 1)
    plt.figure(figsize=(11, 6))
    plt.fill_between(generations, best_history, worst_history, color='skyblue', alpha=0.3, label='Range: best to worst')
    plt.plot(generations, avg_history, color='navy', linewidth=2.5, label='Average fitness')
    plt.plot(generations, best_history, color='green', linestyle='--', linewidth=1.5, label='Best fitness')
    plt.plot(generations, worst_history, color='firebrick', linestyle='--', linewidth=1.5, label='Worst fitness')
    plt.title('Historia fitness populacji w kolejnych generacjach')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter(population):
    xs = np.array([s.x1 for s in population])
    ys = np.array([s.x2 for s in population])
    zs = np.array([s.x3 for s in population])
    fitness_vals = np.array([s.fitness for s in population])
    best = max(population, key=lambda s: s.fitness)

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))

    pairs = [(xs, ys, 'x', 'y'), (xs, zs, 'x', 'z'), (ys, zs, 'y', 'z')]
    for ax, (a, b, label_a, label_b) in zip(axes, pairs):
        sc = ax.scatter(a, b, c=fitness_vals, cmap='plasma', s=60, edgecolors='black', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='Fitness')
        ax.scatter(getattr(best, f'x{["x","y","z"].index(label_a)+1}'),
                   getattr(best, f'x{["x","y","z"].index(label_b)+1}'),
                   color='lime', s=300, marker='*', edgecolors='black', label='Najlepsze')
        ax.set_xlabel(label_a, fontsize=12)
        ax.set_ylabel(label_b, fontsize=12)
        ax.set_title(f'{label_a} vs {label_b}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.25)

    plt.suptitle('Populacja końcowa — rzuty 2D', fontsize=16)
    plt.tight_layout()
    plt.show()