"""
Krótki opis czym ta funkcja jest:
Funckja testowa dla algorytmów optymalizacji.
Sprawdza czy algorytm jest w stanie znaleźć minimum globalne w środowisku gdzie jest wiele
minimów lokalnych.

Parametry:
Funkcja przyjmuje jeden wektor (tablicę) liczb rzeczywistych x = (x1 ... xn)

znaczenie parametrow:
x:oceniane rozwiazanei
n:wymiar problemu
xi: konkretna wartosc, musi sie miesici w przedziale -5.12 - 5.12
10n - stala sluzaca do przesuniecia wykresu w gore , tak by jej absolutne minimum wynosilo 0

wynik:
ocena bledu - im mniejszy i blizszy 0 tym rozwiazanie lepsze.
Wartosc 0 wystapi tylko wtedy gdy kazda zmienna w x wynosi 0

"""
import numpy as np
import matplotlib.pyplot as plt

class Rastrigin:
    def __init__(self, number: int = 5):
        self.n = number


    """
    parametry:
    ga_instance - instancja algorytmu PyGAD
    solution: np.ndarray
    solution_idx: int - indeks ocenianego rozwiązania.
    
    
    """

    def function(self, ga_instance, solution: np.ndarray, solution_idx: int ) -> float:
        answer = 10 * self.n + np.sum(solution ** 2 - 10 * np.cos(2 * np.pi * solution))
        return -answer

    def visualization3d(self):

        #najpierw tworzenie prostej linii od lewej do prawej na osiach x i y potem krzyzowanie ich
        x = np.linspace(-5.12, 5.12, 200)
        y = np.linspace(-5.12, 5.12, 200)
        X,Y = np.meshgrid(x, y)

        #funckcja rastrigina
        Z = 20 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        plt.show()


    def visualization2d(self) -> None:
        x = np.linspace(-5.12, 5.12, 1000)

        # 2. Obliczenie wartości funkcji Rastrigina dla jednego wymiaru (n=1)
        # Wzór: 10*1 + x^2 - 10*cos(2*pi*x)
        y = 10 + x ** 2 - 10 * np.cos(2 * np.pi * x)

        # 3. Rysowanie wykresu
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, color='blue', linewidth=1)

        # Kosmetyka wykresu (aby przypominał ten ze zdjęcia)
        plt.title('2D View - Funkcja Rastrigina')
        plt.xlabel('x')
        plt.ylabel('f(x)')

        # Dodanie lekkiej siatki ułatwiającej odczytywanie wartości
        plt.grid(True, linestyle='--', alpha=0.6)

        # Ustawienie limitów osi dla lepszej widoczności
        plt.xlim(-5.12, 5.12)
        plt.ylim(0, 45)

        # Wyświetlenie okna z wykresem
        plt.show()
