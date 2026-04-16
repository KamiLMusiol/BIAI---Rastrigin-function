import pygad as pyg
from rastrigin_function.rastrigin_main import Rastrigin

class Rastrigin_Pygad:
    def __init__(self, num_generations=50, sol_per_pop=50, num_genes=2):
        self.class_ras = Rastrigin(number=num_genes)

        self.ga_instance = pyg.GA(
            num_generations=num_generations,
            num_parents_mating=10,
            fitness_func=self.fit_func,  #nasza funckja z folderu rastrigin function
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=-5.12,
            init_range_high=5.12,
            mutation_percent_genes=15
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


