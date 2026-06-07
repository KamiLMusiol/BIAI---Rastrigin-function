import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
seed = 42
rng = np.random.default_rng(seed)


"""

Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
x,y pozycje, vx, vy predkosc, kat, , vkat, L czt lewa noga dotknela ziemi, P
"""

N_INPUTS = 8   # liczba obserwacji
"""
Action Space
There are four discrete actions available:

    0: do nothing

    1: fire left orientation engine

    2: fire main engine

    3: fire right orientation engine

"""
N_ACTIONS = 4  # liczba akcji
N_WEIGHTS = N_INPUTS * N_ACTIONS  # 32 wagi na osobnika kazda kolumna to nic,0,1,2
N_EPISODES = 5  # ile epizodów na ocenę osobnika (uśredniamy)

class LunarLander:
    def __init__(self,):
        self.env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5)

    def evaluate(self, weights, render=False):
        weights_matrix = weights.reshape(N_INPUTS, N_ACTIONS) #matrix 8x4
        total_reward = 0

        for _ in range(N_EPISODES):
            obs, _ = self.env.reset() #zaczynamy od nowa
            done = False
            while not done:
                scores = obs @ weights_matrix #mnozy obserwacje przez macierz wag
                action = np.argmax(scores) #wybiera akcje o najwyzszym score
                obs, reward, terminated, truncated, _ = self.env.step(action) #wykonuje akcje i obseracje
                total_reward += reward
                done = terminated or truncated

        self.env.close()
        return total_reward / N_EPISODES

def init_population(pop_size):
    return [rng.standard_normal(N_WEIGHTS) for _ in range(pop_size)]

def select_parents(population, fitnesses):
    fitnesses = np.array(fitnesses)
    weights = (fitnesses - np.min(fitnesses)) + 1e-6
    probabilities = weights / np.sum(weights)
    idx1, idx2 = rng.choice(len(population), size=2, replace=False, p=probabilities)
    return population[idx1], population[idx2]

def crossover(parent1, parent2, crossover_prob=0.7):
    if rng.random() < crossover_prob:
        point = rng.integers(1, N_WEIGHTS)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(weights, mutation_rate=0.1):
    mutated = weights.copy()
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] += rng.standard_normal()
    return mutated

def run_ga(pop_size=20, generations=30):
    lander = LunarLander()
    population = init_population(pop_size)
    best_fitness_history = []

    for generation in range(generations):
        fitnesses = [lander.evaluate(w) for w in population]
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)
        print(f'Generation {generation + 1}: best fitness = {best_fitness:.2f}')

        new_population = [population[best_idx].copy()]  # elityzm

        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    best = population[np.argmax([lander.evaluate(w) for w in population])]
    print(f'\nNajlepsze fitness: {max(fitnesses):.2f}')
    return best, best_fitness_history

def plot_fitness(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='green', linewidth=2)
    plt.axhline(y=200, color='red', linestyle='--', label='Próg rozwiązania (200)')
    plt.title('Historia fitness - Lunar Lander GA')
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()

def show_best(weights):
    env = gym.make("LunarLander-v3", render_mode="human")
    weights_matrix = weights.reshape(N_INPUTS, N_ACTIONS)
    obs, _ = env.reset()
    done = False
    while not done:
        scores = obs @ weights_matrix
        action = np.argmax(scores)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()