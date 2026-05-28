import pygad as pyg

import numpy as np
import matplotlib.pyplot as plt


from rastrigin_main import Rastrigin

class Rastrigin_Pygad:
    def __init__(self, num_generations=50, sol_per_pop=50, num_genes=2):
        self.class_ras = Rastrigin(number=num_genes)
        np.random.seed(42)
        self.last_population = None

        self.ga_instance = pyg.GA(
            num_generations=num_generations,
            num_parents_mating=10,
            fitness_func=self.fit_func,  #nasza funckja z folderu rastrigin function
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=-5.12,
            init_range_high=5.12,
            mutation_percent_genes=15,
            on_generation = self.on_generation
        )



    def fit_func(self, ga_instance, solution, solution_idx):
        return self.class_ras.function(ga_instance, solution, solution_idx)

    def start_alg(self):
        self.ga_instance.run()

    def show_res(self):
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        self.ga_instance.plot_fitness()

    def on_generation(self, ga_instance):
        self.last_population = ga_instance.population.copy()

    def plot_scatter(self):
        x = np.linspace(-5.12, 5.12, 300)
        y = np.linspace(-5.12, 5.12, 300)
        X, Y = np.meshgrid(x, y)
        Z = 20 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))

        solution, _, _ = self.ga_instance.best_solution()

        plt.figure(figsize=(8, 7))
        plt.contourf(X, Y, Z, levels=40, cmap='plasma')
        plt.colorbar(label='f(x1, x2)')
        plt.scatter(
            self.last_population[:, 0],
            self.last_population[:, 1],
            color='white', edgecolors='black', s=40, label='Populacja'
        )
        plt.scatter(
            solution[0], solution[1],
            color='lime', edgecolors='black', s=150, marker='*', label='Najlepsze rozwiązanie'
        )
        plt.title('Populacja końcowa na mapie konturowej Rastrigina')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()

        plt.show()

