"""
Problem: Klasyfikacja danych o owocach za pomocą SVM i wizualizacja wyników po redukcji wymiarowości za pomocą PCA.

Autorzy: Szymon Rosztajn, Mateusz Lech

Instrukcja użycia:
1. Upewnij się, że plik 'fruit.csv' znajduje się w tym samym katalogu co skrypt.
2. Uruchom skrypt, aby załadować dane, wytrenować model SVM i zwizualizować dane.
"""

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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


def train_svm(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    """
    Trenuje model SVM z jądrem RBF.

    X_train (pd.DataFrame): Dane treningowe.
    y_train (pd.Series): Etykiety treningowe.

    SVC: Wytrenowany model SVM.
    """
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: SVC, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Ocena modelu na podstawie danych testowych.

    model (SVC): Wytrenowany model.
    X_test (pd.DataFrame): Dane testowe.
    y_test (pd.Series): Etykiety testowe.
    """
    predictions = model.predict(X_test)
    print(metrics.classification_report(y_test, predictions))
    print("Dokładność modelu:", metrics.accuracy_score(y_test, predictions))


def visualize_with_pca(data: pd.DataFrame, labels: pd.Series):
    """
    Wizualizuje dane po redukcji wymiarowości za pomocą PCA.

    Args:
        data (pd.DataFrame): Dane wejściowe.
        labels (pd.Series): Etykiety klasyfikacji.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    label_mapping = labels.map({'BERHI': 0, 'OtherClass': 1})

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label_mapping, cmap='plasma', edgecolor='k', s=50)
    plt.title("Wizualizacja SVM (PCA)")
    plt.xlabel("Pierwsza główna składowa")
    plt.ylabel("Druga główna składowa")
    plt.colorbar(label="Klasa owoców")
    plt.show()


if __name__ == "__main__":
    data = load_data('fruit.csv')
    X_train, X_test, y_train, y_test = split_data(data)

    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test)

    visualize_with_pca(data.drop(columns=["Class"]), data["Class"])
