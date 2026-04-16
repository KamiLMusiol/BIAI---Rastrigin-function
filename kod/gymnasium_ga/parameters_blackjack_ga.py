import numpy as np
import gymnasium as gym
import pygad
class Ga_Parameters_Backjack:
    """
    parametry:
        solutions:
        0 - waga jak szybko mowimy pass
        1 - waga jak bardzo boimy sie krupiera
        2 - waga mniej sie boimy gdy mamy asa (moze byc 1 lub 11)
        3 - bias czyli agresja wieksza oznacza wieksza agresje

        wynik jezeli jest <0 to robimy stand
        sumuje sie to tak:
        (w1 * suma) + (w2 * krupier) + (w3 * as) + w4_bias

    """
    def __init__(self, num_generations=50, sol_per_pop=50, num_genes=4):

        self.env = gym.make("Blackjack-v1")

        self.ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=10,
                               fitness_func=self.fitness_func,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=-5,
                               init_range_high=5,
                               parent_selection_type='sss',
                               keep_parents=2,
                               crossover_type='single_point',
                               mutation_type='random',
                               mutation_percent_genes=25)


    def fitness_func(self, ga_instance, solution, solution_idx):
        #  trzeba zrobic 100 partii zeby usrednic szczescie (wynik)

        num_episodes = 100
        total_reward = 0


        for _ in range(num_episodes):
            # Resetujemy grę na start każdej partii
            observation, info = self.env.reset() #obesrvation nasze wejscie czyli nasze pkt, ktopiera i czy as
            terminated = False # zakonczenie przegralismy/Stand
            truncated = False #zakonczenie zwiazane z bledem np klatki

            while not (terminated or truncated):

                player_sum = observation[0]
                dealer_card = observation[1]
                usable_ace = 1.0 if observation[2] else 0.0  #  True/False na 1/0

                # Obliczamy sumę ważoną na podstawie genów (solution)

                decision_value = (solution[0] * player_sum) + \
                                 (solution[1] * dealer_card) + \
                                 (solution[2] * usable_ace) + \
                                 solution[3]

                #  Jeśli wynik > 0 to dobieramy (Hit=1), inaczej czekamy (Stand=0)
                action = 1 if decision_value > 0 else 0

                # Gymnasium
                observation, reward, terminated, truncated, info = self.env.step(action)


                # 1 - wygrana 0 - remis -1 - przegrana,
                if terminated or truncated:
                    total_reward += reward

       #zwracamy srednia z wszystkich gier
        return total_reward / num_episodes

    def start_alg(self):
        self.ga_instance.run()

    def show_res(self):
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        self.ga_instance.plot_fitness()