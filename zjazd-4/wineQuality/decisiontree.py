"""
Problem: Klasyfikacja jakości wina na podstawie danych chemicznych z wykorzystaniem modelu drzewa decyzyjnego.

Autorzy: Szymon Rosztajn, Mateusz Lech

Instrukcja użycia:
1. Upewnij się, że plik 'winequality.csv' znajduje się w tym samym katalogu, co ten skrypt.
2. Uruchom skrypt, aby załadować dane, wytrenować model drzewa decyzyjnego, dokonać predykcji i wyświetlić wyniki.
"""

import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import seaborn as sns

warnings.filterwarnings("ignore")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Ładuje dane z pliku CSV.
    """
    return pd.read_csv(file_path, delimiter=';')

def split_data(data: pd.DataFrame) -> tuple:
    """
    Dzieli dane na zestawy treningowe i testowe.

    data (pd.DataFrame): Dane wejściowe.

    Zestawy treningowe i testowe (x_train, x_test, y_train, y_test).
    """
    x = data.drop(columns=["quality"])
    y = data["quality"]
    return train_test_split(x, y, test_size=0.2, random_state=3)

def train_decision_tree(x_train: pd.DataFrame, y_train: pd.Series) -> tree.DecisionTreeClassifier:
    """
    Trenuje model drzewa decyzyjnego.

    x_train (pd.DataFrame): Dane treningowe.
     y_train (pd.Series): Etykiety treningowe.

    DecisionTreeClassifier: Wytrenowany model drzewa decyzyjnego.
    """
    model = tree.DecisionTreeClassifier(random_state=3)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model: tree.DecisionTreeClassifier, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Ocena modelu na podstawie danych testowych.

    model (DecisionTreeClassifier): Wytrenowany model.
    x_test (pd.DataFrame): Dane testowe.
    y_test (pd.Series): Etykiety testowe.

    Wyświetla raport klasyfikacyjny i dokładność modelu.
    """
    predictions = model.predict(x_test)
    print(metrics.classification_report(y_test, predictions))
    print("Dokładność modelu:", metrics.accuracy_score(y_test, predictions))

def visualize_data(data: pd.DataFrame):
    """
    Wizualizuje zależność między cukrem resztkowym a jakością wina.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='residual sugar',
        y='quality',
        data=data,
        hue='quality',
        palette='viridis',
        s=100
    )
    plt.title("Zależność między cukrem resztkowym a jakością wina")
    plt.xlabel("Cukier resztkowy (g/l)")
    plt.ylabel("Jakość wina")
    plt.show()

if __name__ == "__main__":
    data = load_data('winequality.csv')
    x_train, x_test, y_train, y_test = split_data(data)
    model = train_decision_tree(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    visualize_data(data)
