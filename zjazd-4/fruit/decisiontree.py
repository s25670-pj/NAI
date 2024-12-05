"""
Problem: Klasyfikacja danych o owocach za pomocą drzewa decyzyjnego, a następnie wizualizacja wyników po redukcji wymiarowości za pomocą PCA.

Autorzy: Szymon Rosztajn, Mateusz Lech

Instrukcja użycia:
1. Upewnij się, że plik 'fruit.csv' znajduje się w tym samym katalogu co skrypt.
2. Uruchom skrypt, aby załadować dane, wytrenować model, dokonać predykcji i zwizualizować dane.
"""

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Ładuje dane z pliku CSV.
    """
    return pd.read_csv(file_path)


def split_data(data: pd.DataFrame) -> tuple:
    """
    Dzieli dane na zestawy treningowe i testowe.
    data (pd.DataFrame): Dane wejściowe.
    Zestawy treningowe i testowe (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=["Class"])
    y = data["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Trenuje model drzewa decyzyjnego.

    X_train (pd.DataFrame): Dane treningowe.
    y_train (pd.Series): Etykiety treningowe.

    DecisionTreeClassifier: Wytrenowany model drzewa decyzyjnego.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Ocena modelu na podstawie danych testowych.

        model (DecisionTreeClassifier): Wytrenowany model.
        X_test (pd.DataFrame): Dane testowe.
        y_test (pd.Series): Etykiety testowe.

    Wyświetla raport klasyfikacyjny i dokładność modelu.
    """
    predictions = model.predict(X_test)
    print(metrics.classification_report(y_test, predictions))
    print("Dokładność modelu:", metrics.accuracy_score(y_test, predictions))


def visualize_with_pca(data: pd.DataFrame, labels: pd.Series):
    """
    Wizualizuje dane po redukcji wymiarowości za pomocą PCA.

    data (pd.DataFrame): Dane wejściowe.
    labels (pd.Series): Etykiety klasyfikacji.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    # Mapowanie etykiet na liczby dla wizualizacji
    label_mapping = labels.map({'BERHI': 0, 'OtherClass': 1})

    # Wizualizacja
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label_mapping, cmap='viridis', edgecolor='k', s=50)
    plt.title("Wizualizacja Drzewa Decyzyjnego (PCA)")
    plt.xlabel("Pierwsza główna składowa")
    plt.ylabel("Druga główna składowa")
    plt.colorbar(label="Klasa owoców")
    plt.show()


if __name__ == "__main__":
    data = load_data('fruit.csv')
    X_train, X_test, y_train, y_test = split_data(data)

    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test)

    visualize_with_pca(data.drop(columns=["Class"]), data["Class"])
