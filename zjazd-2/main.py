"""
==========================================
System kontroli rozmytej: Problem stylu jazdy
==========================================

Aby uruchomić program, zainstaluj wymagane pakiety:
pip install scikit-fuzzy
pip install matplotlib
pip install numpy

Niniejszy program demonstruje zastosowanie zasad logiki rozmytej do oceny stylu jazdy
na podstawie prędkości, przyspieszenia oraz siły hamowania. Ten przykład
pokazuje, jak logika rozmyta może umożliwić generowanie skomplikowanego zachowania
na podstawie zwięzłego i intuicyjnego zestawu reguł eksperckich.

**Problem stylu jazdy**
-----------------------

Program ten tworzy system kontroli rozmytej, który modeluje styl jazdy kierowcy
na podstawie jego parametrów jazdy, takich jak prędkość, przyspieszenie oraz siła hamowania.
Styl jazdy może być oceniany jako spokojny, umiarkowany lub agresywny.

Definiujemy problem za pomocą następujących elementów:

* Zmienne wejściowe (Antecedents):
   - `prędkość`
      * Zakres: od 0 do 100 (jednostki arbitralne, np. km/h)
      * Zbiory rozmyte: niska, średnia, wysoka
   - `przyspieszenie`
      * Zakres: od 0 do 10 (jednostki arbitralne)
      * Zbiory rozmyte: powolne, umiarkowane, szybkie
   - `hamowanie`
      * Zakres: od 0 do 10 (jednostki arbitralne)
      * Zbiory rozmyte: słabe, umiarkowane, mocne
* Zmienna wyjściowa (Consequent):
   - `styl jazdy`
      * Zakres: od 0 do 100 (wartości symbolizujące styl jazdy)
      * Zbiory rozmyte: spokojny, umiarkowany, agresywny
* Reguły
   - Jeśli prędkość jest niska *i* przyspieszenie jest powolne *i* hamowanie jest słabe, 
     to styl jazdy będzie spokojny.
   - Jeśli prędkość jest wysoka *i* przyspieszenie jest szybkie *i* hamowanie jest mocne,
     to styl jazdy będzie agresywny.
   - Jeśli prędkość jest średnia i inne wartości są umiarkowane, styl jazdy będzie umiarkowany.

**Tworzenie kontrolera**
------------------------

System kontroli rozmytej definiowany jest przy użyciu pakietu `skfuzzy`. 
Definiujemy zmienne wejściowe i wyjściowe, funkcje przynależności oraz reguły.
Na podstawie wprowadzonych danych obliczany jest wynikowy styl jazdy.

**Symulacja systemu**
---------------------

Symulację można przeprowadzić, wprowadzając wartości dla prędkości, przyspieszenia oraz hamowania,
po czym zostaje wyznaczony styl jazdy. Wartości są przetwarzane w systemie kontroli rozmytej, 
a wynik można wizualizować.

"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definicja kolorów i zakresów zmiennych wejściowych
predkosc = ctrl.Antecedent(np.arange(0, 101, 1), 'predkosc')
przyspieszenie = ctrl.Antecedent(np.arange(0, 11, 1), 'przyspieszenie')
hamowanie = ctrl.Antecedent(np.arange(0, 11, 1), 'hamowanie')

styl_jazdy = ctrl.Consequent(np.arange(0, 101, 1), 'styl_jazdy')

def define_membership():
    """
    Definiuje funkcje przynależności dla zmiennych wejściowych i wyjściowych.
    """
    predkosc['niska'] = fuzz.trimf(predkosc.universe, [0, 0, 50])
    predkosc['srednia'] = fuzz.trimf(predkosc.universe, [30, 50, 80])
    predkosc['wysoka'] = fuzz.trimf(predkosc.universe, [60, 100, 100])

    przyspieszenie['powolne'] = fuzz.trimf(przyspieszenie.universe, [0, 0, 5])
    przyspieszenie['umiarkowane'] = fuzz.trimf(przyspieszenie.universe, [3, 6, 8])
    przyspieszenie['szybkie'] = fuzz.trimf(przyspieszenie.universe, [7, 10, 10])

    hamowanie['slabe'] = fuzz.trimf(hamowanie.universe, [0, 0, 5])
    hamowanie['umiarkowane'] = fuzz.trimf(hamowanie.universe, [3, 5, 8])
    hamowanie['mocne'] = fuzz.trimf(hamowanie.universe, [6, 10, 10])

    styl_jazdy['spokojny'] = fuzz.trimf(styl_jazdy.universe, [0, 0, 50])
    styl_jazdy['umiarkowany'] = fuzz.trimf(styl_jazdy.universe, [40, 55, 80])
    styl_jazdy['agresywny'] = fuzz.trimf(styl_jazdy.universe, [60, 100, 100])

def define_rules():
    """
    Tworzy reguły logiki rozmytej, określające styl jazdy na podstawie prędkości, przyspieszenia i hamowania.
    """
    rules = [
        ctrl.Rule(predkosc['niska'] & przyspieszenie['powolne'] & hamowanie['slabe'], styl_jazdy['spokojny']),
        ctrl.Rule(predkosc['niska'] & przyspieszenie['umiarkowane'] & hamowanie['umiarkowane'], styl_jazdy['spokojny']),
        ctrl.Rule(predkosc['srednia'] & przyspieszenie['powolne'] & hamowanie['slabe'], styl_jazdy['umiarkowany']),
        ctrl.Rule(predkosc['srednia'] & przyspieszenie['umiarkowane'] & hamowanie['umiarkowane'], styl_jazdy['umiarkowany']),
        ctrl.Rule(predkosc['srednia'] & przyspieszenie['szybkie'] & hamowanie['mocne'], styl_jazdy['agresywny']),
        ctrl.Rule(predkosc['wysoka'] & przyspieszenie['powolne'] & hamowanie['slabe'], styl_jazdy['umiarkowany']),
        ctrl.Rule(predkosc['wysoka'] & przyspieszenie['umiarkowane'] & hamowanie['umiarkowane'], styl_jazdy['agresywny']),
        ctrl.Rule(predkosc['wysoka'] & przyspieszenie['szybkie'] & hamowanie['mocne'], styl_jazdy['agresywny']),
        # Domyślna reguła
        ctrl.Rule(predkosc['srednia'] & przyspieszenie['umiarkowane'] & hamowanie['umiarkowane'], styl_jazdy['umiarkowany'])
    ]
    return rules

def create_control_system():
    """
    Inicjuje system kontroli na podstawie zdefiniowanych reguł logiki rozmytej.
    """
    define_membership()
    rules = define_rules()
    styl_jazdy_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(styl_jazdy_ctrl)

def evaluate_driving_style(predkosc_input, przyspieszenie_input, hamowanie_input):
    """
    Przeprowadza symulację stylu jazdy na podstawie wprowadzonych wartości dla prędkości, przyspieszenia i hamowania.
    
    Args:
        predkosc_input (float): Wartość prędkości
        przyspieszenie_input (float): Wartość przyspieszenia
        hamowanie_input (float): Wartość hamowania
    """
    styl_jazdy_sim = create_control_system()
    styl_jazdy_sim.input['predkosc'] = predkosc_input
    styl_jazdy_sim.input['przyspieszenie'] = przyspieszenie_input
    styl_jazdy_sim.input['hamowanie'] = hamowanie_input
    styl_jazdy_sim.compute()

    if 'styl_jazdy' in styl_jazdy_sim.output:
        print(f"Oceniany styl jazdy: {styl_jazdy_sim.output['styl_jazdy']}")
        styl_jazdy.view(sim=styl_jazdy_sim)
    else:
        print("Błąd: Nie udało się wygenerować wyniku dla 'styl_jazdy'.")

if __name__ == "__main__":
    evaluate_driving_style(5, 1, 1)  # Wprowadzenie przykładowych wartości wejściowych
    predkosc.view()
    przyspieszenie.view()
    hamowanie.view()

    plt.show()

    input("Naciśnij Enter, aby zakończyć program...")
